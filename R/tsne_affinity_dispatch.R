# KNN normalization, sparse-affinity construction, and backend dispatch.

normalize_opentsne_knn_input <- function(indices, distances = NULL, n_neighbors = NULL) {
  knn <- coerce_knn_input(indices, distances)
  n <- nrow(knn$indices)
  available <- knn$n_neighbors
  if (is.null(n_neighbors)) {
    n_neighbors <- available
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) ||
        !is.finite(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
      stop("`n_neighbors` must be a positive integer smaller than the number of rows.", call. = FALSE)
    }
    if (n_neighbors > available) {
      stop("`n_neighbors` is larger than the supplied KNN width.", call. = FALSE)
    }
  }
  materialized <- materialize_knn_range(
    knn$indices,
    knn$distances,
    knn$col_start,
    n_neighbors
  )
  distance_type <- knn$distance_type
  list(
    indices = materialized$indices,
    distances = materialized$distances,
    n = n,
    n_neighbors = as.integer(n_neighbors),
    has_self = isTRUE(knn$has_self),
    input_backend = knn$input_backend,
    distance_type = distance_type
  )
}

fast_knn_opentsne_materialized <- function(indices,
                                           distances,
                                           n_components = 2L,
                                           perplexity = NULL,
                                           theta = 0.5,
                                           early_exaggeration_iter = NULL,
                                           n_iter = NULL,
                                           learning_rate = "auto",
                                           early_exaggeration = "auto",
                                           exaggeration = NULL,
                                           Y_init = NULL,
                                           initial_momentum = 0.8,
                                           final_momentum = 0.8,
                                           min_gain = 0.01,
                                           max_step_norm = "auto",
                                           negative_gradient_method = "auto",
                                           record_costs = FALSE,
                                           n_threads = NULL,
                                           seed = 42L,
                                           verbose = FALSE,
                                           backend = c("cpu", "cuda", "metal"),
                                           auto_config = TRUE,
                                           input_had_self = FALSE,
                                           input_backend = NA_character_,
                                           gpu_knn = NULL,
                                           gpu_n = NULL,
                                           gpu_k = NULL,
                                           cuda_init_data = NULL) {
  backend <- resolve_embedding_backend(backend)
  optimizer_backend <- if (identical(backend, "cpu")) {
    "cpu"
  } else if (identical(backend, "metal")) {
    if (!embedding_metal_available_cpp() || !metal_opentsne_native_available()) {
      stop(
        "Native Metal openTSNE optimizer was requested, but it is not available in this build. ",
        "No CPU fallback is used for Metal-labelled runs.",
        call. = FALSE
      )
    }
    "metal"
  } else if (identical(backend, "cuda")) {
    if (!embedding_cuda_available_cpp() || !cuda_opentsne_native_available()) {
      stop(
        "Native CUDA openTSNE optimizer was requested, but it is not available in this build. ",
        "No CPU fallback is used for CUDA-labelled runs.",
        call. = FALSE
      )
    }
    "cuda"
  } else {
    "cpu"
  }
  gpu_resident_knn <- !is.null(gpu_knn)
  n <- if (isTRUE(gpu_resident_knn)) as.integer(gpu_n) else nrow(indices)
  k <- if (isTRUE(gpu_resident_knn)) as.integer(gpu_k) else ncol(indices)
  Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  negative_gradient_method <- normalize_tsne_negative_gradient_method(negative_gradient_method)
  auto_params <- resolve_opentsne_auto_parameters(
    n = n,
    k = k,
    perplexity = perplexity,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    learning_rate = learning_rate,
    optimizer_backend = optimizer_backend,
    negative_gradient_method = negative_gradient_method,
    auto_config = auto_config
  )
  perplexity <- auto_params$perplexity
  early_exaggeration_iter <- auto_params$early_exaggeration_iter
  n_iter <- auto_params$n_iter
  early_exaggeration_iter <- as.integer(early_exaggeration_iter)
  n_iter <- as.integer(n_iter)
  if (length(early_exaggeration_iter) != 1L || is.na(early_exaggeration_iter) || early_exaggeration_iter < 0L) {
    stop("`early_exaggeration_iter` must be a non-negative integer.", call. = FALSE)
  }
  if (length(n_iter) != 1L || is.na(n_iter) || n_iter < 0L) {
    stop("`n_iter` must be a non-negative integer.", call. = FALSE)
  }
  if (early_exaggeration_iter + n_iter < 1L) {
    stop("At least one optimization iteration is required.", call. = FALSE)
  }
  if (identical(optimizer_backend, "metal")) {
    if (identical(negative_gradient_method, "auto")) {
      negative_gradient_method <- if (n <= 3000L) "exact" else "fft"
    }
  } else if (identical(optimizer_backend, "cuda")) {
    if (identical(negative_gradient_method, "auto")) {
      negative_gradient_method <- "fft"
    }
  } else if (identical(negative_gradient_method, "auto")) {
    negative_gradient_method <- if (n <= 3000L) "exact" else "fft"
  }
  if (identical(optimizer_backend, "cpu") && identical(negative_gradient_method, "fft")) {
    if (n_components != 2L) {
      stop(
        "`negative_gradient_method = \"fft\"` currently supports two output components.",
        call. = FALSE
      )
    }
  }
  init_info <- if (is.null(Y_init)) {
    if (isTRUE(gpu_resident_knn) && identical(optimizer_backend, "cuda") && !is.null(cuda_init_data)) {
      list(
        Y_init = NULL,
        method = "pca_cuda_raft_tsvd_pca_device",
        backend = "cuda_raft_tsvd_device",
        spectral_n_iter = NA_integer_
      )
    } else if (isTRUE(gpu_resident_knn)) {
      list(
        Y_init = NULL,
        method = "cuda_random",
        backend = "cuda",
        spectral_n_iter = NA_integer_
      )
    } else {
      make_opentsne_default_init(
        indices,
        distances,
        n_components,
        seed,
        optimizer_backend,
        negative_gradient_method
      )
    }
  } else {
    list(
      Y_init = Y_init,
      method = attr(Y_init, "fastEmbedR_init_method") %||% "user",
      backend = attr(Y_init, "fastEmbedR_init_backend") %||% NA_character_,
      spectral_n_iter = attr(Y_init, "fastEmbedR_init_spectral_n_iter") %||% NA_integer_
    )
  }
  if (is.null(Y_init)) {
    Y_init <- init_info$Y_init
  }
  args <- check_tsne_neighbor_params(
    n = n,
    n_components = n_components,
    perplexity = perplexity,
    theta = theta,
    max_iter = early_exaggeration_iter + n_iter,
    verbose = verbose,
    Y_init = Y_init,
    momentum = initial_momentum,
    final_momentum = final_momentum
  )
  lr <- normalize_opentsne_learning_rate(learning_rate)
  ex <- normalize_opentsne_exaggeration(early_exaggeration, exaggeration)
  if (isTRUE(auto_params$opt_sne_learning_rate) &&
      is.finite(auto_params$learning_rate_value) &&
      auto_params$learning_rate_value > 0) {
    lr <- list(auto = FALSE, value = auto_params$learning_rate_value)
  }
  min_gain <- as.numeric(min_gain)
  if (length(min_gain) != 1L || is.na(min_gain) || !is.finite(min_gain) || min_gain <= 0) {
    stop("`min_gain` must be a positive number.", call. = FALSE)
  }
  record_costs <- isTRUE(record_costs) || isTRUE(verbose)
  max_step_norm <- if (is.character(max_step_norm) &&
                       length(max_step_norm) == 1L &&
                       identical(tolower(max_step_norm), "auto")) {
    if (identical(optimizer_backend, "metal") &&
        identical(negative_gradient_method, "fft")) {
      0.5
    } else {
      5
    }
  } else if (is.null(max_step_norm) || (length(max_step_norm) == 1L && is.na(max_step_norm))) {
    NA_real_
  } else {
    value <- suppressWarnings(as.numeric(max_step_norm))
    if (length(value) != 1L || is.na(value) || !is.finite(value) || value <= 0) {
      stop("`max_step_norm` must be NULL/NA or a positive number.", call. = FALSE)
    }
    value
  }

  out <- if (identical(optimizer_backend, "metal")) {
    knn_tsne_opentsne_metal_cpp(
      indices,
      distances,
      args$Y_init,
      args$init,
      args$n_components,
      args$perplexity,
      early_exaggeration_iter,
      n_iter,
      ex$early,
      ex$normal,
      lr$value,
      lr$auto,
      args$momentum,
      args$final_momentum,
      min_gain,
      max_step_norm,
      negative_gradient_method,
      as.integer(seed),
      record_costs,
      isTRUE(auto_params$auto_kld_stop),
      auto_params$auto_iter_end
    )
  } else if (identical(optimizer_backend, "cuda") && isTRUE(gpu_resident_knn)) {
    knn_tsne_opentsne_cuda_gpu_cpp(
      gpu_knn,
      as.integer(k),
      args$Y_init,
      args$init,
      if (is.null(cuda_init_data)) NULL else cuda_init_data,
      args$n_components,
      args$perplexity,
      early_exaggeration_iter,
      n_iter,
      ex$early,
      ex$normal,
      lr$value,
      lr$auto,
      args$momentum,
      args$final_momentum,
      min_gain,
      max_step_norm,
      negative_gradient_method,
      as.integer(seed),
      record_costs
    )
  } else if (identical(optimizer_backend, "cuda")) {
    knn_tsne_opentsne_cuda_float_cpp(
      indices,
      distances,
      args$Y_init,
      args$init,
      args$n_components,
      args$perplexity,
      early_exaggeration_iter,
      n_iter,
      ex$early,
      ex$normal,
      lr$value,
      lr$auto,
      args$momentum,
      args$final_momentum,
      min_gain,
      max_step_norm,
      negative_gradient_method,
      as.integer(seed),
      record_costs
    )
  } else {
    knn_tsne_opentsne_float_cpp(
      indices,
      distances,
      args$Y_init,
      args$init,
      args$n_components,
      args$perplexity,
      args$theta,
      early_exaggeration_iter,
      n_iter,
      ex$early,
      ex$normal,
      lr$value,
      lr$auto,
      args$momentum,
      args$final_momentum,
      min_gain,
      max_step_norm,
      negative_gradient_method,
      as.integer(n_threads),
      as.integer(seed),
      args$verbose,
      record_costs,
      isTRUE(auto_params$auto_kld_stop),
      auto_params$auto_iter_end
    )
  }
  return_float32 <- isTRUE(gpu_resident_knn) || is_float32_matrix(distances)
  layout <- finalize_embedding_layout(
    out$Y,
    "openTSNE",
    return_float32 = return_float32
  )
  probabilities <- out$probabilities
  if (is.null(probabilities)) probabilities <- "symmetric_sparse_knn_cpu"
  n_negatives <- out$n_negatives
  if (is.null(n_negatives)) n_negatives <- NA_integer_
  cfg <- list(
    method = "opentsne",
    backend = optimizer_backend,
    n = n,
    n_neighbors = as.integer(k),
    perplexity = args$perplexity,
    theta = args$theta,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    early_exaggeration_iter_actual = out$early_exaggeration_iter_actual %||% early_exaggeration_iter,
    n_iter_actual = out$n_iter_actual %||% n_iter,
    max_iter = early_exaggeration_iter + n_iter,
    max_iter_actual = out$max_iter_actual %||% (early_exaggeration_iter + n_iter),
    learning_rate = if (isTRUE(auto_params$opt_sne_learning_rate)) {
      "auto_opt_sne_n_over_early_exaggeration"
    } else if (isTRUE(lr$auto)) {
      "auto"
    } else {
      lr$value
    },
    learning_rate_early = out$learning_rate_early,
    learning_rate_normal = out$learning_rate_normal,
    early_exaggeration = ex$early,
    exaggeration = ex$normal,
    initial_momentum = args$momentum,
    final_momentum = args$final_momentum,
    min_gain = min_gain,
    max_step_norm = max_step_norm,
    initialization = init_info$method,
    initialization_spectral_n_iter = init_info$spectral_n_iter,
    negative_gradient_method = negative_gradient_method,
    auto_config = isTRUE(auto_params$auto_config),
    auto_config_rule = auto_params$auto_rule,
    auto_kld_stop = isTRUE(out$auto_kld_stop %||% FALSE),
    auto_stop_reason = out$auto_stop_reason %||% "not_reported",
    auto_iter_end = out$auto_iter_end %||% auto_params$auto_iter_end,
    auto_perplexity = auto_params$auto_perplexity,
    auto_n_neighbors = auto_params$auto_n_neighbors,
    record_costs = record_costs,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    precision = out$precision %||% "float32",
    output_precision = if (return_float32) "float32" else "double",
    probabilities = probabilities,
    n_negatives = n_negatives,
    n.cores = out$n_threads,
    input_had_self = isTRUE(input_had_self),
    knn_backend = input_backend,
    knn_residency = if (isTRUE(gpu_resident_knn)) "cuda_device" else "host",
    provenance = if (identical(optimizer_backend, "metal")) {
      "openTSNE_style_native_metal_directed_knn_gpu_probability_optimizer"
    } else if (identical(optimizer_backend, "cuda")) {
      "openTSNE_style_native_cuda_directed_knn_gpu_probability_optimizer"
    } else {
      "openTSNE_native_cpp_two_phase_optimizer_bsd3_informed"
    }
  )
  metal_stage_timing <- out$metal_stage_timing
  if (!is.null(metal_stage_timing) && NROW(metal_stage_timing) > 0L) {
    cfg$metal_stage_timing <- metal_stage_timing
  }
  attr(layout, "fastEmbedR_config") <- cfg
  attr(layout, "costs") <- out$costs
  attr(layout, "itercosts") <- out$itercosts
  attr(layout, "itercost_iterations") <- out$itercost_iterations
  attr(layout, "metal_trace") <- out$metal_trace
  if (!is.null(metal_stage_timing) && NROW(metal_stage_timing) > 0L) {
    attr(layout, "metal_stage_timing") <- metal_stage_timing
  }
  layout
}

fast_knn_opentsne_core <- function(indices,
                                   distances = NULL,
                                   n_components = 2L,
                                   perplexity = NULL,
                                   theta = 0.5,
                                   early_exaggeration_iter = NULL,
                                   n_iter = NULL,
                                   learning_rate = "auto",
                                   early_exaggeration = "auto",
                                   exaggeration = NULL,
                                   Y_init = NULL,
                                   initial_momentum = 0.8,
                                   final_momentum = 0.8,
                                   min_gain = 0.01,
                                   max_step_norm = 5,
                                   negative_gradient_method = "auto",
                                   record_costs = FALSE,
                                   n_threads = NULL,
                                   seed = 42L,
                                   verbose = FALSE,
                                   backend = c("cpu", "cuda", "metal"),
                                   auto_config = TRUE) {
  backend <- resolve_embedding_backend(backend)
  if (inherits(indices, "fastEmbedR_opentsne_prepared")) {
    if (!is.null(distances)) {
      stop("Do not pass `distances` when `indices` is a prepared openTSNE object.", call. = FALSE)
    }
    knn <- indices$knn
  } else {
    knn <- normalize_opentsne_knn_input(indices, distances)
  }
  Y_init <- resolve_opentsne_y_init(Y_init, knn$n, n_components)
  fast_knn_opentsne_materialized(
    knn$indices,
    knn$distances,
    n_components = n_components,
    perplexity = perplexity,
    theta = theta,
    early_exaggeration_iter = early_exaggeration_iter,
    n_iter = n_iter,
    learning_rate = learning_rate,
    early_exaggeration = early_exaggeration,
    exaggeration = exaggeration,
    Y_init = Y_init,
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    min_gain = min_gain,
    max_step_norm = max_step_norm,
    negative_gradient_method = negative_gradient_method,
    record_costs = record_costs,
    n_threads = n_threads,
    seed = seed,
    verbose = verbose,
    backend = backend,
    auto_config = auto_config,
    input_had_self = knn$has_self,
    input_backend = knn$input_backend
  )
}

#' Precompute reusable openTSNE KNN state
#'
#' `prepare_opentsne_knn()` strips self-neighbors, trims to the requested
#' non-self width, and stores compact KNN matrices once. Pass the returned
#' object to [opentsne_knn()] for repeated seeds or backend comparisons without
#' repeating KNN normalization/materialization.
#'
#' @inheritParams opentsne_knn
#' @return A prepared openTSNE KNN object.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 15L
#' idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
#'   nrow = nrow(d), byrow = TRUE)
#' prep <- prepare_opentsne_knn(idx, dst, perplexity = 5)
#' y1 <- opentsne_knn(prep, seed = 1, early_exaggeration_iter = 5, n_iter = 10)
#' y2 <- opentsne_knn(prep, seed = 2, early_exaggeration_iter = 5, n_iter = 10)
#' @export
prepare_opentsne_knn <- function(indices,
                                 distances = NULL,
                                 n_neighbors = NULL,
                                 perplexity = NULL) {
  knn0 <- coerce_knn_input(indices, distances)
  policy <- opentsne_neighbor_policy(
    nrow(knn0$indices),
    perplexity = perplexity,
    available = knn0$n_neighbors
  )
  if (is.null(n_neighbors)) {
    n_neighbors <- policy$n_neighbors
  }
  knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
  out <- list(
    knn = knn,
    perplexity = policy$perplexity,
    n_neighbors = as.integer(n_neighbors),
    affinity_state = "knn_materialized_affinity_builder_internal"
  )
  class(out) <- c("fastEmbedR_opentsne_prepared", "list")
  out
}
