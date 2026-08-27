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
      stop(
        "`n_neighbors` must be a positive integer smaller than ",
        "the number of rows.",
        call. = FALSE
      )
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
                                           backend = NULL,
                                           auto_config = TRUE,
                                           input_had_self = FALSE,
                                           input_backend = NA_character_,
                                           gpu_knn = NULL,
                                           gpu_n = NULL,
                                           gpu_k = NULL,
                                           cuda_init_data = NULL) {
  backend <- resolve_embedding_backend(backend)
  optimizer_backend <- resolve_opentsne_optimizer_backend(backend)
  gpu_resident_knn <- !is.null(gpu_knn)
  n <- if (gpu_resident_knn) as.integer(gpu_n) else nrow(indices)
  k <- if (gpu_resident_knn) as.integer(gpu_k) else ncol(indices)
  Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }

  negative_gradient_method <- normalize_tsne_negative_gradient_method(
    negative_gradient_method
  )
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
  iterations <- validate_opentsne_iteration_counts(auto_params)
  negative_gradient_method <- resolve_opentsne_gradient_method(
    negative_gradient_method,
    optimizer_backend,
    n,
    n_components
  )
  init_info <- prepare_opentsne_initialization(
    Y_init,
    gpu_resident_knn,
    cuda_init_data,
    optimizer_backend,
    indices,
    distances,
    n_components,
    seed,
    negative_gradient_method
  )
  controls <- resolve_opentsne_controls(
    n = n,
    n_components = n_components,
    perplexity = auto_params$perplexity,
    theta = theta,
    early_iter = iterations$early,
    normal_iter = iterations$normal,
    verbose = verbose,
    Y_init = init_info$Y_init,
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    learning_rate = learning_rate,
    early_exaggeration = early_exaggeration,
    exaggeration = exaggeration,
    auto_params = auto_params,
    min_gain = min_gain,
    max_step_norm = max_step_norm,
    optimizer_backend = optimizer_backend,
    negative_gradient_method = negative_gradient_method,
    record_costs = record_costs
  )
  out <- run_opentsne_native_optimizer(
    optimizer_backend,
    indices,
    distances,
    gpu_knn,
    gpu_resident_knn,
    cuda_init_data,
    k,
    controls,
    iterations$early,
    iterations$normal,
    negative_gradient_method,
    auto_params,
    n_threads,
    seed
  )
  finalize_opentsne_native_result(
    out,
    optimizer_backend,
    gpu_resident_knn,
    distances,
    n,
    k,
    controls,
    auto_params,
    init_info,
    negative_gradient_method,
    iterations$early,
    iterations$normal,
    input_had_self,
    input_backend
  )
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
                                   backend = NULL,
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
                                 perplexity = NULL,
                                 affinity_support = c("standard", "compact")) {
  affinity_support <- normalize_opentsne_affinity_support(affinity_support)
  knn0 <- coerce_knn_input(indices, distances)
  policy <- opentsne_neighbor_policy(
    nrow(knn0$indices),
    perplexity = perplexity,
    available = knn0$n_neighbors,
    affinity_support = affinity_support
  )
  if (is.null(n_neighbors)) {
    n_neighbors <- policy$n_neighbors
  }
  required_k <- opentsne_support_width(policy$perplexity, affinity_support)
  if (n_neighbors < required_k) {
    stop(
      "`n_neighbors` is too small for `affinity_support = \"",
      affinity_support, "\"`; need at least ", required_k, ".",
      call. = FALSE
    )
  }
  knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
  out <- list(
    knn = knn,
    perplexity = policy$perplexity,
    n_neighbors = as.integer(n_neighbors),
    affinity_support = affinity_support,
    affinity_state = "knn_materialized_affinity_builder_internal"
  )
  class(out) <- c("fastEmbedR_opentsne_prepared", "list")
  out
}
