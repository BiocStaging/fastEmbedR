#' Transform query points into an existing t-SNE embedding
#'
#' `transform_tsne()` places query observations into a fixed reference t-SNE
#' embedding. It follows the openTSNE transform design at the algorithm level:
#' query-to-reference affinities are computed from KNN distances, query points
#' are initialized from nearby reference points, and optimization moves only the
#' query points.
#'
#' @param reference_layout Numeric reference embedding matrix, or a
#'   `fastEmbedR_embedding` object.
#' @param knn Optional query-to-reference KNN list with `indices` and
#'   `distances`.
#' @param reference_data Reference observations in the same preprocessing space
#'   used to fit `reference_layout`. Required only when `knn` is `NULL`.
#' @param new_data Query observations in the same preprocessing space as
#'   `reference_data`. Required only when `knn` is `NULL`.
#' @param k Number of reference neighbors used for transform KNN. If `NULL`,
#'   uses at least `3 * perplexity`, matching the usual t-SNE affinity width.
#' @param perplexity Transform perplexity. The openTSNE default is 5.
#' @param initialization Initial query placement: `"median"` from reference
#'   neighbors, inverse-distance `"weighted"`, or `"random"`.
#' @param Y_init Optional initial query embedding matrix. When supplied, it
#'   overrides `initialization`.
#' @param n_iter Number of normal transform optimization iterations.
#' @param early_exaggeration_iter Number of early-exaggeration iterations.
#' @param learning_rate Transform learning rate.
#' @param early_exaggeration Early exaggeration multiplier.
#' @param exaggeration Normal transform exaggeration multiplier.
#' @param initial_momentum Momentum used during early exaggeration.
#' @param final_momentum Momentum used during normal optimization.
#' @param max_grad_norm Maximum per-query gradient norm. Use `Inf` to disable.
#' @param max_step_norm Maximum per-query step norm. Use `Inf` to disable.
#' @param n_negatives Number of sampled reference points for repulsion when
#'   the reference set is larger than `exact_repulsion_threshold`. GPU sampled
#'   repulsion is native and experimental for large reference sets.
#' @param exact_repulsion_threshold Use exact query-reference repulsion at or
#'   below this reference count.
#' @param n.cores Number of CPU cores for the native optimizer.
#' @param seed Random seed.
#' @param backend Backend used for query KNN when `knn` is `NULL`; `"metal"`
#'   and `"cuda"` run their native fixed-reference transform optimizers when
#'   those backends were compiled. Explicit unavailable GPU requests fail; CPU
#'   work is never reported as Metal or CUDA.
#' @param verbose Print native optimizer progress.
#' @return A numeric matrix with one row per query observation.
#' @examples
#' reference <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), 4L, 2L, byrow = TRUE)
#' query_knn <- list(
#'   indices = matrix(c(1L, 2L, 3L, 4L), 2L, 2L, byrow = TRUE),
#'   distances = matrix(c(0.1, 0.2, 0.15, 0.25), 2L, 2L, byrow = TRUE)
#' )
#' query_layout <- transform_tsne(
#'   reference, knn = query_knn, perplexity = 0.5,
#'   n_iter = 2, exact_repulsion_threshold = 10, seed = 1
#' )
#' @export
transform_tsne <- function(reference_layout,
                           knn = NULL,
                           reference_data = NULL,
                           new_data = NULL,
                           k = NULL,
                           perplexity = 5,
                           initialization = c("median", "weighted", "random"),
                           Y_init = NULL,
                           n_iter = 250L,
                           early_exaggeration_iter = 0L,
                           learning_rate = 0.1,
                           early_exaggeration = 4,
                           exaggeration = 1.5,
                           initial_momentum = 0.8,
                           final_momentum = 0.8,
                           max_grad_norm = 0.25,
                           max_step_norm = Inf,
                           n_negatives = NULL,
                           exact_repulsion_threshold = 4096L,
                           n.cores = NULL,
                           seed = 4L,
                           backend = NULL,
                           verbose = FALSE) {
  n_threads <- n.cores
  if (inherits(reference_layout, "fastEmbedR_embedding")) {
    reference_layout <- reference_layout$layout
  }
  initialization <- match.arg(initialization)
  backend <- resolve_embedding_backend(backend)
  optimizer_backend <- resolve_tsne_transform_backend(backend)
  reference_layout <- transform_embedding_matrix(
    reference_layout,
    "reference_layout",
    min_rows = 1L
  )
  perplexity <- as.numeric(perplexity)
  if (length(perplexity) != 1L || is.na(perplexity) || !is.finite(perplexity) || perplexity <= 0) {
    stop("`perplexity` must be a positive number.", call. = FALSE)
  }

  raw_backend <- "precomputed"
  exact <- NA
  if (is.null(knn)) {
    if (is.null(reference_data) || is.null(new_data)) {
      stop(
        "Supply either `knn`, or both `reference_data` and `new_data`.",
        call. = FALSE
      )
    }
    reference_data <- transform_embedding_matrix(
      reference_data,
      "reference_data",
      min_rows = 1L
    )
    new_data <- transform_embedding_matrix(new_data, "new_data", min_rows = 1L)
    if (nrow(reference_data) != nrow(reference_layout)) {
      stop(
        "`reference_data` and `reference_layout` must have the same number of rows.",
        call. = FALSE
      )
    }
    if (ncol(reference_data) != ncol(new_data)) {
      stop(
        "`reference_data` and `new_data` must have the same number of columns.",
        call. = FALSE
      )
    }
    if (is.null(k)) {
      k <- min(nrow(reference_data), max(25L, ceiling(3 * perplexity)))
    }
    k <- transform_embedding_k(k, nrow(reference_data))
    query_policy <- fastembedr_query_nn_policy(
      optimizer_backend,
      n_reference = nrow(reference_data),
      n_query = nrow(new_data),
      p = ncol(reference_data)
    )
    raw_knn <- fastembedr_native_query_knn(
      reference_data,
      new_data,
      k = k,
      metric = "euclidean",
      n_threads = n_threads,
      target_recall = query_policy$target_recall,
      backend = query_policy$backend,
      method = query_policy$method,
      keep_gpu = FALSE
    )
    projection <- transform_projection_knn(
      raw_knn,
      n_reference = nrow(reference_layout),
      k = k
    )
    raw_backend <- attr(raw_knn, "backend")
    exact <- attr(raw_knn, "exact")
  } else {
    projection <- transform_projection_knn(
      knn,
      n_reference = nrow(reference_layout),
      k = k
    )
    raw_backend <- attr(knn, "backend")
    exact <- attr(knn, "exact")
  }
  if (is.null(raw_backend) || length(raw_backend) == 0L || is.na(raw_backend)) {
    raw_backend <- "precomputed"
  }

  if (is.null(n_threads)) {
    n_threads <- default_tsne_threads()
  }
  n_threads <- as.integer(n_threads)
  if (length(n_threads) != 1L || is.na(n_threads) || !is.finite(n_threads) || n_threads < 0L) {
    stop("`n.cores` must be NULL or a non-negative integer.", call. = FALSE)
  }
  n_iter <- as.integer(n_iter)
  early_exaggeration_iter <- as.integer(early_exaggeration_iter)
  exact_repulsion_threshold <- as.integer(exact_repulsion_threshold)
  if (length(n_iter) != 1L || is.na(n_iter) || n_iter < 0L ||
      length(early_exaggeration_iter) != 1L || is.na(early_exaggeration_iter) ||
      early_exaggeration_iter < 0L ||
      n_iter + early_exaggeration_iter < 1L) {
    stop("Transform iteration counts must be non-negative and sum to at least one.", call. = FALSE)
  }
  if (length(exact_repulsion_threshold) != 1L || is.na(exact_repulsion_threshold)) {
    exact_repulsion_threshold <- 4096L
  }
  if (is.null(n_negatives)) {
    n_negatives <- if (nrow(reference_layout) <= exact_repulsion_threshold) {
      nrow(reference_layout)
    } else {
      min(256L, nrow(reference_layout))
    }
  }
  n_negatives <- as.integer(n_negatives)
  if (length(n_negatives) != 1L || is.na(n_negatives) || n_negatives < 1L) {
    stop("`n_negatives` must be NULL or a positive integer.", call. = FALSE)
  }
  init <- !is.null(Y_init)
  y_init <- if (init) {
    y_init <- transform_embedding_matrix(Y_init, "Y_init", min_rows = nrow(projection$indices))
    if (nrow(y_init) != nrow(projection$indices) || ncol(y_init) != ncol(reference_layout)) {
      stop("`Y_init` must have one row per query and the same columns as `reference_layout`.", call. = FALSE)
    }
    y_init
  } else {
    matrix(0, 0L, 0L)
  }

  out <- if (identical(optimizer_backend, "metal")) {
    if (ncol(reference_layout) != 2L) {
      stop("Metal t-SNE transform currently supports only two-dimensional reference layouts.", call. = FALSE)
    }
    transform_tsne_metal_cpp(
      reference_layout,
      projection$indices,
      projection$distances,
      y_init,
      init,
      initialization,
      perplexity,
      n_iter,
      early_exaggeration_iter,
      as.numeric(learning_rate),
      as.numeric(early_exaggeration),
      as.numeric(exaggeration),
      as.numeric(initial_momentum),
      as.numeric(final_momentum),
      as.numeric(max_grad_norm),
      as.numeric(max_step_norm),
      n_negatives,
      exact_repulsion_threshold,
      as.integer(seed)
    )
  } else if (identical(optimizer_backend, "cuda")) {
    if (ncol(reference_layout) != 2L) {
      stop("CUDA t-SNE transform currently supports only two-dimensional reference layouts.", call. = FALSE)
    }
    transform_tsne_cuda_cpp(
      reference_layout,
      projection$indices,
      projection$distances,
      y_init,
      init,
      initialization,
      perplexity,
      n_iter,
      early_exaggeration_iter,
      as.numeric(learning_rate),
      as.numeric(early_exaggeration),
      as.numeric(exaggeration),
      as.numeric(initial_momentum),
      as.numeric(final_momentum),
      as.numeric(max_grad_norm),
      as.numeric(max_step_norm),
      n_negatives,
      exact_repulsion_threshold,
      as.integer(seed)
    )
  } else {
    transform_tsne_cpp(
      reference_layout,
      projection$indices,
      projection$distances,
      y_init,
      init,
      initialization,
      perplexity,
      n_iter,
      early_exaggeration_iter,
      as.numeric(learning_rate),
      as.numeric(early_exaggeration),
      as.numeric(exaggeration),
      as.numeric(initial_momentum),
      as.numeric(final_momentum),
      as.numeric(max_grad_norm),
      as.numeric(max_step_norm),
      n_negatives,
      exact_repulsion_threshold,
      n_threads,
      as.integer(seed),
      isTRUE(verbose)
    )
  }
  layout <- out$Y
  colnames(layout) <- colnames(reference_layout)
  attr(layout, "backend") <- optimizer_backend
  attr(layout, "nn_backend") <- raw_backend
  attr(layout, "exact") <- if (is.null(exact)) NA else isTRUE(exact)
  attr(layout, "k") <- as.integer(ncol(projection$indices))
  attr(layout, "transform") <- "opentsne_style_fixed_reference"
  attr(layout, "fastEmbedR_config") <- list(
    method = "transform_tsne",
    backend = optimizer_backend,
    backend_requested = backend,
    nn_backend = raw_backend,
    n_reference = nrow(reference_layout),
    n_query = nrow(layout),
    k = as.integer(ncol(projection$indices)),
    perplexity = perplexity,
    n_iter = n_iter,
    early_exaggeration_iter = early_exaggeration_iter,
    learning_rate = as.numeric(learning_rate),
    early_exaggeration = as.numeric(early_exaggeration),
    exaggeration = as.numeric(exaggeration),
    initial_momentum = as.numeric(initial_momentum),
    final_momentum = as.numeric(final_momentum),
    max_grad_norm = as.numeric(max_grad_norm),
    max_step_norm = as.numeric(max_step_norm),
    n_negatives = out$n_negatives,
    exact_repulsion_threshold = exact_repulsion_threshold,
    initialization = out$initialization,
    optimizer = out$optimizer,
    repulsion = out$repulsion,
    affinities = out$affinities,
    affinity_storage = out$affinity_storage,
    transform_batch_size = out$transform_batch_size,
    transform_batches = out$transform_batches,
    n.cores = if (is.null(out$n_threads)) NA_integer_ else out$n_threads,
    seed = as.integer(seed),
    provenance = "openTSNE_transform_design_native_cpp"
  )
  layout
}

resolve_tsne_transform_backend <- function(backend) {
  backend <- resolve_embedding_backend(backend)
  if (identical(backend, "cpu")) {
    return("cpu")
  }
  if (identical(backend, "metal")) {
    if (!metal_metric_available()) {
      stop("Metal t-SNE transform backend is not available on this system.", call. = FALSE)
    }
    return("metal")
  }
  if (identical(backend, "cuda")) {
    if (!cuda_metric_available()) {
      stop("CUDA t-SNE transform backend is not available on this system.", call. = FALSE)
    }
    return("cuda")
  }
  stop("Unsupported t-SNE transform backend: ", backend, ".", call. = FALSE)
}

landmark_projection_mode <- function() {
  mode <- getOption("fastEmbedR.landmark_projection", "auto")
  mode <- tolower(as.character(mode)[1L])
  if (!mode %in% c("auto", "exact", "approx")) {
    mode <- "auto"
  }
  mode
}

landmark_projection_min_work <- function() {
  value <- suppressWarnings(as.numeric(getOption(
    "fastEmbedR.landmark_projection_min_work",
    1e10
  )))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 0) {
    value <- 1e10
  }
  value
}

landmark_projection_min_rows <- function() {
  value <- suppressWarnings(as.integer(getOption(
    "fastEmbedR.landmark_projection_min_rows",
    2000L
  )))
  if (length(value) != 1L || is.na(value) || !is.finite(value) || value < 1L) {
    value <- 2000L
  }
  value
}

landmark_projection_approx_params <- function(n_landmarks, k) {
  n_landmarks <- as.integer(n_landmarks)
  k <- as.integer(k)
  n_projections <- getOption("fastEmbedR.landmark_projection_n_projections", NULL)
  if (is.null(n_projections)) {
    n_projections <- max(8L, min(24L, 2L * ceiling(log2(max(2L, n_landmarks)))))
  } else {
    n_projections <- suppressWarnings(as.integer(n_projections))
    if (length(n_projections) != 1L || is.na(n_projections) || !is.finite(n_projections)) {
      n_projections <- max(8L, min(24L, 2L * ceiling(log2(max(2L, n_landmarks)))))
    }
  }
  n_projections <- as.integer(max(1L, min(64L, n_projections)))

  window <- getOption("fastEmbedR.landmark_projection_window", NULL)
  if (is.null(window)) {
    window <- max(
      64L,
      20L * k,
      ceiling(0.015 * n_landmarks),
      ceiling(sqrt(max(1L, n_landmarks)))
    )
  } else {
    window <- suppressWarnings(as.integer(window))
    if (length(window) != 1L || is.na(window) || !is.finite(window)) {
      window <- max(
        64L,
        20L * k,
        ceiling(0.015 * n_landmarks),
        ceiling(sqrt(max(1L, n_landmarks)))
      )
    }
  }
  window <- as.integer(max(k, min(n_landmarks, window)))

  list(n_projections = n_projections, window = window)
}

landmark_reference_knn <- function(x_landmarks,
                                   k,
                                   backend,
                                   n_threads,
                                   metric = "euclidean") {
  knn_backend <- embedding_knn_backend(backend)
  fastembedr_nn_without_self(
    x_landmarks,
    k = k,
    backend = knn_backend,
    method = "auto",
    metric = metric,
    output = fastembedr_knn_output_type(x_landmarks, knn_backend),
    n_threads = n_threads,
    tuning = "auto",
    target_recall = 0.99,
    keep_gpu = identical(knn_backend, "cuda")
  )
}

landmark_projection_knn <- function(x_landmarks,
                                    x_query,
                                    k,
                                    backend,
                                    seed,
                                    n_threads = NULL,
                                    landmark_layout = NULL,
                                    all_data = NULL,
                                    landmark_indices = NULL,
                                    query_rows = NULL) {
  backend <- as.character(backend)[1L]
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
    backend <- "cpu"
  }
  n_threads <- normalize_nn_threads(n_threads)
  policy <- fastembedr_query_nn_policy(
    backend,
    n_reference = nrow(x_landmarks),
    n_query = nrow(x_query),
    p = ncol(x_landmarks)
  )
  result <- fastembedr_native_query_knn(
    x_landmarks,
    x_query,
    k = k,
    metric = "euclidean",
    output = fastembedr_knn_output_type(x_landmarks, policy$backend),
    n_threads = n_threads,
    target_recall = policy$target_recall,
    backend = policy$backend,
    method = policy$method,
    keep_gpu = identical(policy$backend, "cuda")
  )
  attr(result, "approximation") <- list(
    strategy = paste0(
      "native_", policy$method, "_reference_query_knn"
    ),
    backend = policy$backend,
    target_recall = policy$target_recall,
    seed = as.integer(seed)
  )
  result
}

landmark_affine_projection <- function(x_landmarks,
                                       x_query,
                                       landmark_layout,
                                       projection_knn,
                                       max_neighbors = NULL,
                                       ridge = 1e-3,
                                       max_extrapolation = 2.5,
                                       n_threads = NULL,
                                       backend = "auto") {
  if (is.null(max_neighbors)) {
    max_neighbors <- min(12L, ncol(projection_knn$indices))
  }
  max_neighbors <- as.integer(max_neighbors)
  if (length(max_neighbors) != 1L || is.na(max_neighbors) || max_neighbors < 3L) {
    max_neighbors <- min(12L, ncol(projection_knn$indices))
  }
  backend <- as.character(backend)[1L]
  metal_work <- as.double(nrow(x_query)) * as.double(ncol(x_query))
  use_metal <- ncol(landmark_layout) == 2L &&
    backend %in% c("metal", "gpu") &&
    isTRUE(embedding_metal_available_cpp()) &&
    metal_work >= 5e6
  if (use_metal) {
    out <- tryCatch(
      project_embedding_affine_metal_cpp(
        x_landmarks,
        x_query,
        landmark_layout,
        projection_knn$indices,
        projection_knn$distances,
        as.integer(max_neighbors),
        as.numeric(ridge),
        as.numeric(max_extrapolation)
      ),
      error = function(e) {
        if (identical(backend, "metal")) {
          stop("Metal affine landmark projection failed: ", conditionMessage(e), call. = FALSE)
        }
        NULL
      }
    )
    if (!is.null(out) && !is.null(out$layout)) {
      layout <- out$layout
      attr(layout, "projection_method") <- out$method %||% "local_affine_knn_projection_metal"
      attr(layout, "projection_backend") <- out$backend %||% "metal"
      attr(layout, "projection_confidence") <- out$confidence
      attr(layout, "projection_fallback") <- out$fallback
      attr(layout, "projection_max_neighbors") <- out$max_neighbors
      attr(layout, "projection_ridge") <- out$ridge
      attr(layout, "projection_max_extrapolation") <- out$max_extrapolation
      attr(layout, "projection_threads") <- NA_integer_
      return(layout)
    }
  }
  n_threads <- normalize_nn_threads(n_threads)
  use_parallel <- exists(
    "project_embedding_affine_parallel_cpp",
    envir = asNamespace("fastEmbedR"),
    inherits = FALSE
  )
  affine_fun <- if (use_parallel) {
    project_embedding_affine_parallel_cpp
  } else {
    project_embedding_affine_cpp
  }
  out <- tryCatch(
    if (use_parallel) {
      affine_fun(
        x_landmarks,
        x_query,
        landmark_layout,
        projection_knn$indices,
        projection_knn$distances,
        as.integer(max_neighbors),
        as.numeric(ridge),
        as.numeric(max_extrapolation),
        as.integer(n_threads)
      )
    } else {
      affine_fun(
        x_landmarks,
        x_query,
        landmark_layout,
        projection_knn$indices,
        projection_knn$distances,
        as.integer(max_neighbors),
        as.numeric(ridge),
        as.numeric(max_extrapolation)
      )
    },
    error = function(e) NULL
  )
  if (is.null(out) || is.null(out$layout)) {
    layout <- project_embedding_knn_cpp(
      landmark_layout,
      projection_knn$indices,
      projection_knn$distances
    )
    attr(layout, "projection_method") <- "weighted_knn_fallback"
    attr(layout, "projection_backend") <- "cpu"
    return(layout)
  }
  layout <- out$layout
  attr(layout, "projection_method") <- out$method %||% "local_affine_knn_projection"
  attr(layout, "projection_backend") <- "cpu"
  attr(layout, "projection_confidence") <- out$confidence
  attr(layout, "projection_fallback") <- out$fallback
  attr(layout, "projection_max_neighbors") <- out$max_neighbors
  attr(layout, "projection_ridge") <- out$ridge
  attr(layout, "projection_max_extrapolation") <- out$max_extrapolation
  attr(layout, "projection_threads") <- out$n_threads %||% 1L
  layout
}

zero_proc_time <- function() {
  structure(rep(0, 5), names = names(system.time({})))
}

scalar_or_default <- function(values, name, default) {
  value <- values[[name]]
  if (is.null(value)) return(default)
  value
}

scalar_numeric_or_default <- function(values, name, default) {
  value <- scalar_or_default(values, name, default)
  if (length(value) != 1L || is.na(value)) return(default)
  if (is.character(value) && identical(tolower(value), "auto")) return(default)
  out <- suppressWarnings(as.numeric(value))
  if (length(out) != 1L || is.na(out) || !is.finite(out)) default else out
}

scalar_integer_or_default <- function(values, name, default) {
  value <- scalar_numeric_or_default(values, name, default)
  out <- suppressWarnings(as.integer(value))
  if (length(out) != 1L || is.na(out)) as.integer(default) else out
}

resident_transform_backend <- function(backend, k, keep_knn) {
  if (isTRUE(keep_knn)) return(NA_character_)
  backend <- as.character(backend)[1L]
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
    backend <- "auto"
  }
  if (k > 128L) return(NA_character_)
  if (backend %in% c("metal", "gpu") &&
      isTRUE(embedding_metal_available_cpp())) {
    return("metal")
  }
  if (backend %in% c("cuda", "gpu") &&
      isTRUE(embedding_cuda_available_cpp())) {
    return("cuda")
  }
  NA_character_
}

resident_projected_layout <- function(resident,
                                      backend,
                                      backend_requested,
                                      n_reference,
                                      k,
                                      perplexity,
                                      n_iter,
                                      early_exaggeration_iter,
                                      learning_rate,
                                      early_exaggeration,
                                      exaggeration,
                                      initial_momentum,
                                      final_momentum,
                                      max_grad_norm,
                                      max_step_norm,
                                      exact_repulsion_threshold,
                                      seed,
                                      reference_layout) {
  layout <- resident$Y
  colnames(layout) <- colnames(reference_layout)
  attr(layout, "backend") <- backend
  attr(layout, "nn_backend") <- paste0(backend, "_resident_projection")
  attr(layout, "exact") <- TRUE
  attr(layout, "k") <- as.integer(k)
  attr(layout, "transform") <- "opentsne_style_fixed_reference"
  attr(layout, "fastEmbedR_config") <- list(
    method = "transform_tsne",
    backend = backend,
    backend_requested = backend_requested,
    nn_backend = paste0(backend, "_resident_projection"),
    n_reference = as.integer(n_reference),
    n_query = nrow(layout),
    k = as.integer(k),
    perplexity = perplexity,
    n_iter = as.integer(n_iter),
    early_exaggeration_iter = as.integer(early_exaggeration_iter),
    learning_rate = as.numeric(learning_rate),
    early_exaggeration = as.numeric(early_exaggeration),
    exaggeration = as.numeric(exaggeration),
    initial_momentum = as.numeric(initial_momentum),
    final_momentum = as.numeric(final_momentum),
    max_grad_norm = as.numeric(max_grad_norm),
    max_step_norm = as.numeric(max_step_norm),
    n_negatives = resident$n_negatives,
    exact_repulsion_threshold = as.integer(exact_repulsion_threshold),
    initialization = resident$initialization,
    optimizer = resident$optimizer,
    repulsion = resident$repulsion,
    affinities = resident$affinities,
    affinity_storage = resident$affinity_storage,
    transform_batch_size = NA_integer_,
    transform_batches = NA_integer_,
    n.cores = NA_integer_,
    seed = as.integer(seed),
    resident = TRUE,
    returned_intermediates = resident$returned_intermediates,
    provenance = "openTSNE_transform_design_native_cpp_gpu_resident"
  )
  layout
}

resident_projection_result <- function(backend, k) {
  out <- list(
    indices = matrix(integer(0L), nrow = 0L, ncol = 0L),
    distances = matrix(numeric(0L), nrow = 0L, ncol = 0L)
  )
  result <- finish_nn_result(out, paste0(backend, "_resident_projection"), k, FALSE, exact = TRUE)
  attr(result, "approximation") <- list(
    strategy = "gpu_resident_landmark_projection_transform",
    backend = backend,
    returned_intermediates = "final_layout_only"
  )
  result
}

#' Landmark t-SNE with openTSNE-style transform
#'
#' `landmark_tsne()` embeds a subset of observations with [opentsne()], then
#' places the remaining observations with `transform_tsne()`. This mirrors the
#' practical landmark workflow exposed by openTSNE while keeping the
#' implementation native to this package. Projection KNN searches only the
#' fixed landmark reference: CPU uses native HNSW, Metal uses native exact or
#' recall-tuned IVF-Flat, and CUDA keeps native exact or IVF-Flat results
#' resident on the device.
#'
#' @param data Numeric matrix/data frame with observations in rows.
#' @param landmarks `TRUE` for an automatic subset, a fraction such as `0.5`, a
#'   landmark count, or explicit row indices.
#' @param reference_method Kept for compatibility. Only `"opentsne"` is
#'   accepted in the cleaned package API.
#' @inheritParams opentsne
#' @param transform_k Number of landmark neighbors used to place non-landmarks.
#' @param transform_perplexity Perplexity used by `transform_tsne()`.
#' @param transform_iter Number of normal transform iterations. Use `0` for
#'   projection-only landmarking with no transform refinement.
#' @param n_neighbors Number of non-self neighbors used to embed the landmark
#'   reference set. If `NULL`, it follows the same neighbor policy as
#'   [opentsne()]: `ceiling(perplexity)` when perplexity is supplied.
#' @param perplexity t-SNE perplexity for the landmark reference embedding. If
#'   `NULL`, the optimizer chooses a safe value from the reference KNN width and
#'   sample size.
#' @param transform_early_exaggeration_iter Number of transform early
#'   exaggeration iterations.
#' @param transform_n_negatives Number of sampled reference negatives used by
#'   `transform_tsne()` on large landmark sets. GPU sampled transform repulsion
#'   is native and experimental for large reference sets.
#' @param initialization Initial placement for transformed observations.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores used by CPU KNN and CPU
#'   transform optimization. Native GPU stages ignore this argument.
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' fit <- landmark_tsne(
#'   as.matrix(iris[, 1:4]), landmarks = 0.5, perplexity = 5,
#'   early_exaggeration_iter = 5, n_iter = 10,
#'   transform_iter = 5, seed = 1
#' )
#' plot(fit, labels = iris$Species)
#' @export
landmark_tsne <- function(data,
                          landmarks = TRUE,
                          reference_method = c("opentsne"),
                          n_neighbors = NULL,
                          perplexity = NULL,
                          n_components = 2L,
                          standardize = TRUE,
                          pca_dims = NULL,
                          seed = 4L,
                          backend = NULL,
                          transform_k = NULL,
                          transform_perplexity = 5,
                          transform_iter = 250L,
                          transform_early_exaggeration_iter = 0L,
                          transform_n_negatives = NULL,
                          initialization = c("median", "weighted", "random"),
                          keep_knn = FALSE,
                          verbose = FALSE,
                          n.cores = NULL,
                          ...) {
  n_threads <- n.cores
  dots <- list(...)
  reference_method <- match.arg(reference_method)
  initialization <- match.arg(initialization)
  backend <- resolve_embedding_backend(backend)
  preprocess_time <- system.time({
    prepared <- prepare_embedding_data(
      data,
      standardize,
      pca_dims,
      seed,
      backend = resolve_preprocess_backend(backend)
    )
  })
  x <- prepared$data
  n <- nrow(x)
  if (is.null(n_neighbors)) {
    n_neighbors <- opentsne_neighbor_policy(
      n,
      perplexity = perplexity
    )$n_neighbors
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || n_neighbors < 1L || n_neighbors >= n) {
      stop("`n_neighbors` must be a positive integer smaller than `nrow(data)`.", call. = FALSE)
    }
  }

  landmark_selection <- select_landmarks(
    x,
    landmarks,
    seed = seed,
    n.cores = n_threads
  )
  landmark_indices <- landmark_selection$indices
  if (length(landmark_selection$query_indices) == 0L) {
    return(opentsne(
      x,
      perplexity = perplexity,
      n_components = n_components,
      standardize = FALSE,
      pca_dims = NULL,
      seed = seed,
      backend = backend,
      keep_knn = keep_knn,
      verbose = verbose,
      n.cores = n_threads,
      ...
    ))
  }

  non_landmarks <- landmark_selection$query_indices
  partition <- split_landmark_data(
    x,
    landmark_indices,
    non_landmarks,
    n_threads = n_threads
  )
  x_landmarks <- partition$landmarks
  x_query <- partition$query
  n_landmarks <- nrow(x_landmarks)
  landmark_neighbors <- min(n_neighbors, n_landmarks - 1L)
  reference_time <- system.time({
    reference_knn <- landmark_reference_knn(
      x_landmarks,
      k = landmark_neighbors,
      backend = backend,
      n_threads = n_threads
    )
    reference_layout <- opentsne_knn(
      reference_knn,
      n_neighbors = landmark_neighbors,
      perplexity = perplexity,
      n_components = n_components,
      init_data = x_landmarks,
      seed = seed,
      backend = backend,
      verbose = verbose,
      n.cores = n_threads,
      ...
    )
  })
  reference_cfg <- attr(reference_layout, "fastEmbedR_config")
  reference_knn_backend <- attr(reference_knn, "backend")
  if (is.null(reference_knn_backend)) reference_knn_backend <- "supplied"
  reference_fit <- list(
    layout = reference_layout,
    metrics = data.frame(
      method = "opentsne",
      n = n_landmarks,
      p = ncol(x_landmarks),
      n_neighbors = landmark_neighbors,
      perplexity = reference_cfg$perplexity,
      elapsed = reference_time[["elapsed"]],
      preprocess_elapsed = 0,
      knn_elapsed = NA_real_,
      embedding_elapsed = NA_real_,
      stringsAsFactors = FALSE
    ),
    parameters = c(
      list(
        method = "opentsne",
        input = "knn",
        n = n_landmarks,
        p = ncol(x_landmarks),
        n_neighbors = landmark_neighbors,
        k = landmark_neighbors + 1L,
        n_components = as.integer(n_components),
        seed = as.integer(seed),
        nn_backend = reference_knn_backend,
        keep_knn = keep_knn
      ),
      reference_cfg,
      list(preprocess = "none_precomputed_knn")
    ),
    knn = if (isTRUE(keep_knn)) reference_knn else NULL
  )

  if (is.null(transform_k)) {
    transform_k <- min(n_landmarks, max(25L, ceiling(3 * transform_perplexity)))
  }
  transform_k <- transform_embedding_k(transform_k, n_landmarks)
  projection_knn <- NULL
  projection_time <- zero_proc_time()
  transform_time <- zero_proc_time()
  projected <- NULL
  transform_iter <- as.integer(transform_iter)
  if (length(transform_iter) != 1L || is.na(transform_iter) || transform_iter < 0L) {
    stop("`transform_iter` must be a non-negative integer.", call. = FALSE)
  }
  resident_backend <- NA_character_
  resident_exact_repulsion_threshold <- scalar_integer_or_default(
    dots,
    "exact_repulsion_threshold",
    4096L
  )
  if (length(resident_exact_repulsion_threshold) != 1L ||
      is.na(resident_exact_repulsion_threshold)) {
    resident_exact_repulsion_threshold <- 4096L
  }
  resident_n_negatives <- transform_n_negatives
  if (is.null(resident_n_negatives)) {
    resident_n_negatives <- if (n_landmarks <= resident_exact_repulsion_threshold) {
      n_landmarks
    } else {
      min(256L, n_landmarks)
    }
  }
  resident_n_negatives <- as.integer(resident_n_negatives)
  resident_learning_rate <- scalar_numeric_or_default(dots, "transform_learning_rate", 0.1)
  resident_early_exaggeration <- scalar_numeric_or_default(dots, "transform_early_exaggeration", 4)
  resident_exaggeration <- scalar_numeric_or_default(dots, "transform_exaggeration", 1.5)
  resident_initial_momentum <- scalar_numeric_or_default(dots, "transform_initial_momentum", 0.8)
  resident_final_momentum <- scalar_numeric_or_default(dots, "transform_final_momentum", 0.8)
  resident_max_grad_norm <- scalar_numeric_or_default(dots, "transform_max_grad_norm", 0.25)
  resident_max_step_norm <- scalar_numeric_or_default(dots, "transform_max_step_norm", Inf)

  if (!is.na(resident_backend)) {
    transform_time <- system.time({
      resident <- tryCatch(
        if (identical(resident_backend, "metal")) {
          landmark_tsne_transform_resident_metal_cpp(
            x_landmarks,
            x_query,
            reference_fit$layout,
            as.integer(transform_k),
            as.numeric(transform_perplexity),
            as.integer(transform_iter),
            as.integer(transform_early_exaggeration_iter),
            resident_learning_rate,
            resident_early_exaggeration,
            resident_exaggeration,
            resident_initial_momentum,
            resident_final_momentum,
            resident_max_grad_norm,
            resident_max_step_norm,
            as.integer(resident_n_negatives),
            as.integer(resident_exact_repulsion_threshold),
            as.integer(seed + 1009L)
          )
        } else {
          landmark_tsne_transform_resident_cuda_cpp(
            x_landmarks,
            x_query,
            reference_fit$layout,
            as.integer(transform_k),
            as.numeric(transform_perplexity),
            as.integer(transform_iter),
            as.integer(transform_early_exaggeration_iter),
            resident_learning_rate,
            resident_early_exaggeration,
            resident_exaggeration,
            resident_initial_momentum,
            resident_final_momentum,
            resident_max_grad_norm,
            resident_max_step_norm,
            as.integer(resident_n_negatives),
            as.integer(resident_exact_repulsion_threshold),
            as.integer(seed + 1009L)
          )
        },
        error = function(e) {
          attr(e, "fastEmbedR_resident_backend") <- resident_backend
          if (isTRUE(verbose)) message(conditionMessage(e))
          NULL
        }
      )
      if (!is.null(resident)) {
        projected <- resident_projected_layout(
          resident,
          backend = resident_backend,
          backend_requested = backend,
          n_reference = n_landmarks,
          k = transform_k,
          perplexity = transform_perplexity,
          n_iter = transform_iter,
          early_exaggeration_iter = transform_early_exaggeration_iter,
          learning_rate = resident_learning_rate,
          early_exaggeration = resident_early_exaggeration,
          exaggeration = resident_exaggeration,
          initial_momentum = resident_initial_momentum,
          final_momentum = resident_final_momentum,
          max_grad_norm = resident_max_grad_norm,
          max_step_norm = resident_max_step_norm,
          exact_repulsion_threshold = resident_exact_repulsion_threshold,
          seed = seed + 1009L,
          reference_layout = reference_fit$layout
        )
        projection_knn <- resident_projection_result(resident_backend, transform_k)
      }
    })
  }

  if (is.null(projected)) {
    projection_init_backend <- NA_character_
    projection_init_method <- NA_character_
    projection_time <- system.time({
      projection_knn <- landmark_projection_knn(
        x_landmarks,
        x_query,
        k = transform_k,
        backend = backend,
        seed = seed + 503L,
        n_threads = n_threads,
        landmark_layout = reference_fit$layout,
        all_data = x,
        landmark_indices = landmark_indices,
        query_rows = non_landmarks
      )
    })
    cuda_resident_projection <- identical(backend, "cuda") &&
      fastembedr_is_gpu_knn(projection_knn)
    if (cuda_resident_projection) {
      transform_time <- system.time({
        resident <- landmark_tsne_transform_cuda_gpu_cpp(
          projection_knn,
          x_landmarks,
          x_query,
          reference_fit$layout,
          as.numeric(transform_perplexity),
          as.integer(transform_iter),
          as.integer(transform_early_exaggeration_iter),
          resident_learning_rate,
          resident_early_exaggeration,
          resident_exaggeration,
          resident_initial_momentum,
          resident_final_momentum,
          resident_max_grad_norm,
          resident_max_step_norm,
          as.integer(resident_n_negatives),
          as.integer(resident_exact_repulsion_threshold),
          as.integer(seed + 1009L),
          12L,
          1e-3,
          2.5
        )
        projected <- resident$Y
        attr(projected, "backend") <- "cuda"
        attr(projected, "fastEmbedR_config") <- resident$config
      })
      projection_init_backend <- "cuda"
      projection_init_method <- "local_affine_knn_projection_cuda_resident"
    } else {
      projection_y_init <- attr(projection_knn, "projected_layout", exact = TRUE)
      affine_y_init <- landmark_affine_projection(
        x_landmarks,
        x_query,
        reference_fit$layout,
        projection_knn,
        n_threads = n_threads,
        backend = backend
      )
      if (embedding_layout_dims_match(
        affine_y_init, length(non_landmarks), n_components
      )) {
        projection_y_init <- affine_y_init
        projection_init_backend <- attr(affine_y_init, "projection_backend") %||% NA_character_
        projection_init_method <- attr(affine_y_init, "projection_method") %||% NA_character_
      } else if (!embedding_layout_dims_match(
        projection_y_init, length(non_landmarks), n_components
      )) {
        projection_y_init <- NULL
      }
      if (transform_iter == 0L) {
        if (is.null(projection_y_init)) {
          projection_y_init <- project_embedding_knn_cpp(
            reference_fit$layout,
            projection_knn$indices,
            projection_knn$distances
          )
          projection_init_backend <- "cpu"
          projection_init_method <- "weighted_knn_fallback"
        }
        projected <- projection_y_init
        attr(projected, "backend") <- attr(projection_knn, "backend") %||% backend
        attr(projected, "fastEmbedR_config") <- list(
          optimizer = "projection_only",
          repulsion = "none",
          n_negatives = 0L,
          initialization = initialization,
          backend = attr(projected, "backend")
        )
      } else {
        transform_time <- system.time({
          projected <- transform_tsne(
            reference_fit$layout,
            knn = projection_knn,
            perplexity = transform_perplexity,
            initialization = initialization,
            Y_init = projection_y_init,
            n_iter = transform_iter,
            early_exaggeration_iter = transform_early_exaggeration_iter,
            learning_rate = resident_learning_rate,
            early_exaggeration = resident_early_exaggeration,
            exaggeration = resident_exaggeration,
            initial_momentum = resident_initial_momentum,
            final_momentum = resident_final_momentum,
            max_grad_norm = resident_max_grad_norm,
            max_step_norm = resident_max_step_norm,
            n_negatives = transform_n_negatives,
            n.cores = n_threads,
            seed = seed + 1009L,
            backend = backend,
            verbose = verbose
          )
        })
      }
    }
  }
  if (!exists("projection_init_backend", inherits = FALSE)) {
    projection_init_backend <- NA_character_
  }
  if (!exists("projection_init_method", inherits = FALSE)) {
    projection_init_method <- NA_character_
  }

  layout <- assemble_landmark_layout(
    reference_fit$layout,
    projected,
    landmark_indices,
    non_landmarks,
    n,
    prefix = "openTSNE",
    return_float32 = is_float32_matrix(x)
  )

  timings <- rbind(
    preprocess = preprocess_time,
    reference_embedding = reference_time,
    landmark_projection_knn = projection_time,
    transform = transform_time
  )
  transform_cfg <- attr(projected, "fastEmbedR_config")
  reference_params <- reference_fit$parameters
  projection_approximation <- attr(projection_knn, "approximation", exact = TRUE)
  projection_strategy <- if (is.null(projection_approximation$strategy)) {
    if (isTRUE(attr(projection_knn, "exact"))) "exact"
    else NA_character_
  } else {
    as.character(projection_approximation$strategy)
  }
  selection_method <- landmark_selection$method
  reference_metrics <- reference_fit$metrics
  reference_total_elapsed <- if ("elapsed" %in% names(reference_metrics)) {
    reference_metrics$elapsed[[1L]]
  } else {
    reference_time[["elapsed"]]
  }
  reference_knn_elapsed <- if ("knn_elapsed" %in% names(reference_metrics)) {
    reference_metrics$knn_elapsed[[1L]]
  } else {
    NA_real_
  }
  reference_optimizer_elapsed <- if ("embedding_elapsed" %in% names(reference_metrics)) {
    reference_metrics$embedding_elapsed[[1L]]
  } else {
    NA_real_
  }
  metrics <- data.frame(
    method = "landmark_tsne",
    reference_method = reference_method,
    n = n,
    p = ncol(x),
    n_neighbors = n_neighbors,
    perplexity = if (is.null(perplexity)) reference_params$perplexity else perplexity,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time["elapsed"],
    reference_embedding_elapsed = reference_time["elapsed"],
    reference_total_elapsed = reference_total_elapsed,
    reference_knn_elapsed = reference_knn_elapsed,
    reference_optimizer_elapsed = reference_optimizer_elapsed,
    landmark_projection_knn_elapsed = projection_time["elapsed"],
    transform_elapsed = transform_time["elapsed"],
    landmark = TRUE,
    n_landmarks = n_landmarks,
    landmark_fraction = n_landmarks / n,
    transform_k = transform_k,
    stringsAsFactors = FALSE
  )
  parameters <- c(
    list(
      method = "landmark_tsne",
      reference_method = reference_method,
      n = n,
      p = ncol(x),
      n_neighbors = n_neighbors,
      k = n_neighbors + 1L,
      n_components = as.integer(n_components),
      seed = as.integer(seed),
      nn_backend = reference_params$nn_backend,
      projection_nn_backend = attr(projection_knn, "backend"),
      projection_strategy = projection_strategy,
      projection_init_backend = projection_init_backend,
      projection_init_method = projection_init_method,
      backend = attr(projected, "backend"),
      transform_backend = attr(projected, "backend"),
      n.cores = normalize_nn_threads(n_threads),
      landmark = TRUE,
      n_landmarks = n_landmarks,
      landmark_fraction = n_landmarks / n,
      landmark_selection = selection_method,
      transform_k = transform_k,
      transform_perplexity = transform_perplexity,
      transform_iter = as.integer(transform_iter),
      transform_early_exaggeration_iter = as.integer(transform_early_exaggeration_iter),
      transform_optimizer = transform_cfg$optimizer,
      transform_repulsion = transform_cfg$repulsion,
      transform_n_negatives = transform_cfg$n_negatives,
      transform_initialization = transform_cfg$initialization,
      keep_knn = keep_knn,
      provenance = "openTSNE_landmark_transform_design_native_cpp"
    ),
    prepared$preprocess
  )
  out <- list(
    layout = layout,
    labels = NULL,
    method = "landmark_tsne",
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    knn = NULL,
    landmarks = list(
      indices = landmark_indices,
      layout = reference_fit$layout,
      reference_fit = reference_fit,
      projection_knn = if (isTRUE(keep_knn)) projection_knn else NULL,
      transform = attr(projected, "fastEmbedR_config")
    ),
    preprocess = prepared$preprocess
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
