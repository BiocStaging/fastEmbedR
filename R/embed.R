#' Choose a default neighborhood size
#'
#' @param x A data matrix/data frame, or an integer row count.
#' @param include_self If `TRUE`, return the value to request from `nn()` when
#'   the query points are the data itself.
#' @return An integer neighborhood size.
#' @noRd
auto_k <- function(x, include_self = FALSE) {
  n <- if (length(x) == 1L && is.numeric(x)) {
    as.integer(x)
  } else {
    nrow(x)
  }
  if (length(n) != 1L || is.na(n) || !is.finite(n) || n < 2L) {
    stop("`x` must describe at least two observations.", call. = FALSE)
  }

  k <- if (n < 500L) {
    15L
  } else if (n < 10000L) {
    30L
  } else {
    50L
  }
  k <- max(1L, min(k, n - 1L))
  if (isTRUE(include_self)) k + 1L else k
}

auto_embedding_k <- function(x, method = "opentsne", include_self = FALSE) {
  n <- if (length(x) == 1L && is.numeric(x)) {
    as.integer(x)
  } else {
    nrow(x)
  }
  auto_k(n, include_self = include_self)
}

resolve_embedding_metric <- function(metric, data = NULL) {
  match.arg(metric, c("euclidean", "cosine", "correlation", "inner_product"))
}

prepare_embedding_data <- function(data,
                                   standardize,
                                   pca_dims,
                                   seed,
                                   backend = "cpu") {
  keep_float32 <- is_float32_matrix(data) &&
    !isTRUE(standardize) &&
    is.null(pca_dims)
  x <- if (keep_float32) {
    data
  } else {
    x <- as.matrix(data)
    storage.mode(x) <- "double"
    x
  }
  if (nrow(x) < 2L || ncol(x) < 1L) {
    stop("`data` must have at least two rows and one column.", call. = FALSE)
  }
  finite_input <- if (keep_float32) {
    float32_all_finite_cpp(x)
  } else {
    all(is.finite(x))
  }
  if (!isTRUE(finite_input)) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }

  preprocess <- list(
    standardize = isTRUE(standardize),
    pca_dims = NA_integer_,
    standardize_backend = if (isTRUE(standardize)) "cpu" else "none",
    pca_backend = "none",
    pca_method = "none",
    pca_oversample = NA_integer_,
    pca_power = NA_integer_,
    pca_backend_reason = NA_character_,
    preprocess_backend = if (isTRUE(standardize)) "cpu" else "none",
    preprocess_backend_reason = if (keep_float32) "float32_input_preserved_without_preprocessing" else NA_character_
  )
  if (isTRUE(standardize)) {
    used_native <- FALSE
    if (identical(backend, "cuda") && cuda_metric_available()) {
      standardized <- tryCatch(
        standardize_cuda_cpp(x),
        error = function(e) {
          preprocess$preprocess_backend_reason <<- conditionMessage(e)
          NULL
        }
      )
      if (!is.null(standardized)) {
        x <- standardized$data
        used_native <- TRUE
        preprocess$standardize_backend <- "cuda"
        preprocess$preprocess_backend <- "cuda"
      }
    } else if (identical(backend, "cuda")) {
      preprocess$preprocess_backend_reason <- "cuda_preprocessing_unavailable"
    } else if (identical(backend, "metal") && metal_metric_available()) {
      standardized <- tryCatch(
        standardize_metal_cpp(x),
        error = function(e) {
          preprocess$preprocess_backend_reason <<- conditionMessage(e)
          NULL
        }
      )
      if (!is.null(standardized)) {
        x <- standardized$data
        used_native <- TRUE
        preprocess$standardize_backend <- "metal"
        preprocess$preprocess_backend <- "metal"
      }
    } else if (identical(backend, "metal")) {
      preprocess$preprocess_backend_reason <- "metal_preprocessing_unavailable"
    }
    if (!used_native) {
      standardized <- standardize_cpu_cpp(x)
      x <- standardized$data
      preprocess$standardize_backend <- "cpu"
      preprocess$preprocess_backend <- "cpu"
    }
  }

  if (!is.null(pca_dims)) {
    pca_dims <- as.integer(pca_dims)
    if (length(pca_dims) != 1L || is.na(pca_dims) || !is.finite(pca_dims) || pca_dims < 1L) {
      stop("`pca_dims` must be NULL or a positive integer.", call. = FALSE)
    }
    rank <- min(pca_dims, nrow(x) - 1L, ncol(x))
    if (rank >= 1L && rank < ncol(x)) {
      pca <- fastpls_rsvd_pca_scores(
        x,
        rank = rank,
        seed = seed,
        backend = backend
      )
      x <- pca$scores
      preprocess$pca_dims <- as.integer(ncol(x))
      preprocess$pca_backend <- pca$backend
      preprocess$pca_method <- pca$method
      preprocess$pca_oversample <- pca$oversample
      preprocess$pca_power <- pca$power
      preprocess$pca_backend_reason <- pca$backend_reason
      preprocess$preprocess_backend <- combine_preprocess_backends(
        preprocess$standardize_backend,
        pca$backend
      )
    }
  }
  list(data = x, preprocess = preprocess)
}

resolve_preprocess_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
    return("cpu")
  }
  if (identical(backend, "gpu")) {
    return(resolve_backend_request("gpu", need_embedding = TRUE))
  }
  if (backend %in% c("cuda", "metal")) {
    return(backend)
  }
  "cpu"
}

combine_preprocess_backends <- function(standardize_backend, pca_backend) {
  standardize_backend <- if (is.null(standardize_backend)) "none" else standardize_backend
  pca_backend <- if (is.null(pca_backend)) "none" else pca_backend
  if (identical(standardize_backend, "none")) {
    return(pca_backend)
  }
  if (identical(pca_backend, "none")) {
    return(standardize_backend)
  }
  if (identical(standardize_backend, pca_backend)) {
    return(standardize_backend)
  }
  paste(standardize_backend, pca_backend, sep = "_")
}

validate_pca_backend <- function(backend) {
  backend <- as.character(backend)[1L]
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
    return("cpu")
  }
  if (identical(backend, "gpu")) {
    return(resolve_backend_request(backend, need_embedding = TRUE))
  }
  if (!backend %in% c("cpu", "cuda", "metal")) {
    stop("`backend` must be one of 'cpu', 'cuda', or 'metal'.", call. = FALSE)
  }
  backend
}

fastembedr_metal_tsvd_pca <- function(data,
                                      ncomp,
                                      center = TRUE,
                                      scale = FALSE,
                                      seed = 4L) {
  x <- if (is_float32_matrix(data)) {
    data
  } else {
    x <- as.matrix(data)
    storage.mode(x) <- "double"
    x
  }
  fit <- pca_tsvd_metal_cpp(
    x,
    as.integer(ncomp),
    isTRUE(center),
    isTRUE(scale),
    as.integer(seed)
  )
  out <- list(
    scores = fit$scores,
    loadings = fit$loadings,
    singular_values = fit$singular_values,
    center = fit$center,
    scale = fit$scale,
    ncomp = as.integer(ncol(fit$scores)),
    method = fit$method,
    backend = fit$backend,
    backend_reason = NA_character_,
    engine = "native_metal_mps",
    precision = fit$precision,
    oversample = as.integer(fit$oversample),
    power = as.integer(fit$power),
    seed = as.integer(seed),
    timing = fit$timing
  )
  class(out) <- "fastEmbedR_pca"
  out
}

prepare_pca_matrix <- function(data, center = TRUE, scale = FALSE) {
  x <- if (is_float32_matrix(data)) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to use float32 input.", call. = FALSE)
    }
    float::dbl(data)
  } else {
    as.matrix(data)
  }
  storage.mode(x) <- "double"
  if (nrow(x) < 2L || ncol(x) < 1L) {
    stop("`data` must have at least two rows and one column.", call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }

  center_values <- rep(0, ncol(x))
  scale_values <- rep(1, ncol(x))
  if (isTRUE(center)) {
    center_values <- colMeans(x)
    x <- sweep(x, 2L, center_values, check.margin = FALSE)
  }
  if (isTRUE(scale)) {
    scale_values <- apply(x, 2L, stats::sd)
    scale_values[!is.finite(scale_values) | scale_values == 0] <- 1
    x <- sweep(x, 2L, scale_values, "/", check.margin = FALSE)
  }
  list(data = x, center = center_values, scale = scale_values)
}

attach_opentsne_pca_init <- function(fit, requested) {
  if (!is.logical(requested) || length(requested) != 1L || is.na(requested)) {
    stop("`opentsne_init` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!isTRUE(requested)) return(fit)

  init <- normalize_opentsne_pca_scores(fit$scores, fit$ncomp)
  attr(init, "fastEmbedR_init_method") <- paste0("pca_", fit$method)
  attr(init, "fastEmbedR_init_backend") <- fit$backend
  attr(init, "fastEmbedR_init_backend_reason") <- fit$backend_reason %||% NA_character_
  attr(init, "fastEmbedR_init_seed") <- fit$seed
  fit$opentsne_init <- init
  fit
}

project_pca_test_scores <- function(xtest, fit) {
  use_float32 <- is_float32_matrix(xtest) && is_float32_matrix(fit$loadings)
  x_test <- if (use_float32) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to project float32 `xtest`.", call. = FALSE)
    }
    xtest
  } else if (is_float32_matrix(xtest)) {
    float::dbl(xtest)
  } else {
    x <- as.matrix(xtest)
    storage.mode(x) <- "double"
    x
  }

  if (nrow(x_test) < 1L || ncol(x_test) != nrow(fit$loadings)) {
    stop(
      "`xtest` must have at least one row and the same number of columns as `x`.",
      call. = FALSE
    )
  }
  finite <- if (use_float32) {
    isTRUE(float32_all_finite_cpp(x_test))
  } else {
    all(is.finite(x_test))
  }
  if (!finite) {
    stop("`xtest` must contain only finite values.", call. = FALSE)
  }

  if (use_float32) {
    center <- float::fl(matrix(as.numeric(fit$center), nrow = 1L))
    scale <- float::fl(matrix(as.numeric(fit$scale), nrow = 1L))
    scaled <- float::sweep(x_test, 2L, center, "-", check.margin = FALSE)
    scaled <- float::sweep(scaled, 2L, scale, "/", check.margin = FALSE)
    scores <- scaled %*% fit$loadings
  } else {
    loadings <- if (is_float32_matrix(fit$loadings)) {
      float::dbl(fit$loadings)
    } else {
      as.matrix(fit$loadings)
    }
    scaled <- sweep(x_test, 2L, as.numeric(fit$center), "-", check.margin = FALSE)
    scaled <- sweep(scaled, 2L, as.numeric(fit$scale), "/", check.margin = FALSE)
    scores <- scaled %*% loadings
  }
  colnames(scores) <- colnames(fit$scores)
  scores
}

finalize_pca_fit <- function(fit, xtest, opentsne_init) {
  if (!is.null(xtest)) {
    fit$scores_test <- project_pca_test_scores(xtest, fit)
  }
  attach_opentsne_pca_init(fit, opentsne_init)
}

#' Backend-native truncated PCA
#'
#' `pca()` computes principal component scores with a backend-native truncated
#' decomposition. CPU uses the fastPLS-style randomized SVD family. Metal uses
#' a package-native float32 block-subspace TSVD whose large matrix products are
#' executed with Metal Performance Shaders while data and work buffers remain
#' resident in unified GPU memory. With `float::float32` input, native Metal
#' preprocessing and the returned scores/loadings also remain float32. The API
#' intentionally has no decomposition method menu and does not call Python.
#' Explicit unavailable GPU requests fail instead of being reported as GPU
#' work.
#'
#' @param x Numeric matrix/data frame or `float::float32` matrix with
#'   observations in rows.
#' @param ncomp Number of principal components.
#' @param xtest Optional test/query matrix with the same columns as `x`.
#'   Projected coordinates are returned in `scores_test`.
#' @param center If `TRUE`, mean-center columns before decomposition.
#' @param scale If `TRUE`, scale centered columns to unit sample standard
#'   deviation before decomposition.
#' @param backend PCA backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param seed Random seed for the Gaussian subspace sketch.
#' @param opentsne_init If `TRUE`, add `opentsne_init` to the returned PCA
#'   object. This matrix is centered and rescaled so its largest component
#'   standard deviation is `1e-4`, ready to pass as `Y_init` to [opentsne()]
#'   or [opentsne_knn()].
#' @return A `fastEmbedR_pca` list with `scores`, `loadings`,
#'   `singular_values`, centering/scaling vectors, backend metadata, and
#'   decomposition metadata. When `opentsne_init = TRUE`, the list also
#'   contains `opentsne_init`; when `xtest` is supplied, it also contains
#'   `scores_test`. Metal preserves `float::float32` scores, loadings,
#'   initialization, and compatible test projections when the input is
#'   float32.
#' @examples
#' fit <- pca(
#'   as.matrix(iris[, 1:4]),
#'   ncomp = 2,
#'   seed = 1,
#'   opentsne_init = TRUE
#' )
#' plot(fit$scores, pch = 21, bg = iris$Species)
#' plot(fit$opentsne_init, pch = 21, bg = iris$Species)
#' @export
pca <- function(x,
                ncomp = 2L,
                xtest = NULL,
                center = TRUE,
                scale = FALSE,
                backend = c("cpu", "cuda", "metal"),
                seed = 4L,
                opentsne_init = FALSE) {
  backend <- validate_pca_backend(match.arg(backend))
  if (!is.logical(opentsne_init) || length(opentsne_init) != 1L ||
      is.na(opentsne_init)) {
    stop("`opentsne_init` must be TRUE or FALSE.", call. = FALSE)
  }
  ncomp <- as.integer(ncomp)
  if (length(ncomp) != 1L || is.na(ncomp) || !is.finite(ncomp) || ncomp < 1L) {
    stop("`ncomp` must be a positive integer.", call. = FALSE)
  }
  if (identical(backend, "metal")) {
    fit <- fastembedr_metal_tsvd_pca(
      x,
      ncomp = ncomp,
      center = center,
      scale = scale,
      seed = seed
    )
    return(finalize_pca_fit(fit, xtest, opentsne_init))
  }
  fastpls_fit <- fastembedr_fastpls_pca(
    x,
    ncomp = ncomp,
    center = center,
    scale = scale,
    backend = backend,
    seed = seed
  )
  if (!is.null(fastpls_fit)) {
    return(finalize_pca_fit(fastpls_fit, xtest, opentsne_init))
  }

  prepared <- prepare_pca_matrix(x, center = center, scale = scale)
  rank <- min(ncomp, nrow(prepared$data) - 1L, ncol(prepared$data))
  if (rank < 1L) {
    stop("`data` has no usable PCA rank.", call. = FALSE)
  }
  fit <- fastpls_rsvd_pca_scores(
    prepared$data,
    rank = rank,
    seed = seed,
    backend = backend
  )
  if (!identical(backend, "cpu") &&
      !startsWith(as.character(fit$backend), paste0(backend, "_"))) {
    reason <- fit$backend_reason
    if (is.null(reason) || length(reason) != 1L || is.na(reason)) {
      reason <- paste0(backend, " RSVD backend is unavailable")
    }
    stop(reason, call. = FALSE)
  }
  out <- list(
    scores = fit$scores,
    loadings = fit$loadings,
    singular_values = fit$singular_values,
    center = prepared$center,
    scale = prepared$scale,
    ncomp = as.integer(ncol(fit$scores)),
    method = "rsvd",
    backend = fit$backend,
    backend_reason = fit$backend_reason,
    oversample = fit$oversample,
    power = fit$power,
    seed = as.integer(seed)
  )
  class(out) <- "fastEmbedR_pca"
  finalize_pca_fit(out, xtest, opentsne_init)
}

fastembedr_fastpls_pca <- function(data,
                                   ncomp,
                                   center,
                                   scale,
                                   backend,
                                   seed) {
  package <- "fastPLS"
  if (!fastembedr_optional_namespace_available(package)) return(NULL)
  installed_version <- utils::packageVersion(package)
  if (installed_version < numeric_version("0.99.3")) return(NULL)
  pca_fun <- tryCatch(
    fastembedr_optional_export(package, "pca"),
    error = function(e) NULL
  )
  if (is.null(pca_fun)) return(NULL)

  fit <- tryCatch(
    pca_fun(
      data,
      ncomp = as.integer(ncomp),
      center = isTRUE(center),
      scale = isTRUE(scale),
      backend = backend,
      method = "rsvd",
      seed = as.integer(seed)
    ),
    error = function(e) NULL
  )
  if (is.null(fit) || is.null(fit$scores) || is.null(fit$loadings)) return(NULL)

  singular_values <- fit$svd$d %||% NULL
  resolved_backend <- fit$svd.method %||% fit$svd$svd.method %||%
    paste0(backend, "_rsvd")
  out <- list(
    scores = fit$scores,
    loadings = fit$loadings,
    singular_values = singular_values,
    center = fit$center,
    scale = fit$scale,
    ncomp = as.integer(ncol(fit$scores)),
    method = "rsvd",
    backend = as.character(resolved_backend),
    backend_reason = NA_character_,
    engine = "fastPLS",
    engine_version = as.character(installed_version),
    precision = fit$precision %||%
      if (is_float32_matrix(fit$scores)) "float32" else "double",
    oversample = fit$svd$oversample %||% NA_integer_,
    power = fit$svd$power %||% NA_integer_,
    seed = as.integer(seed)
  )
  class(out) <- "fastEmbedR_pca"
  out
}

fastpls_rsvd_tuning <- function(n, p, rank, backend) {
  backend <- if (is.null(backend)) "cpu" else backend
  if (identical(backend, "cuda")) {
    oversample <- if (n * p >= 5e6) 16L else 10L
    power <- if (rank <= 20L || p <= 128L) 2L else 1L
  } else if (identical(backend, "metal")) {
    oversample <- if (n * p >= 5e6) 16L else 10L
    power <- if (rank <= 30L) 1L else 0L
  } else {
    oversample <- if (rank <= 10L) 10L else min(20L, max(10L, ceiling(rank / 2)))
    power <- if (rank <= 30L) 1L else 0L
  }
  list(
    oversample = as.integer(max(0L, oversample)),
    power = as.integer(max(0L, power))
  )
}

fastpls_rsvd_pca_scores <- function(x,
                                    rank,
                                    seed,
                                    backend = "cpu") {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  n <- nrow(x)
  p <- ncol(x)
  max_rank <- min(n, p)
  rank <- max(1L, min(as.integer(rank), max_rank))
  backend_requested <- if (identical(backend, "gpu")) {
    resolve_backend_request(backend, need_embedding = TRUE)
  } else if (identical(backend, "auto")) {
    "cpu"
  } else {
    backend
  }
  tuning <- fastpls_rsvd_tuning(n, p, rank, backend_requested)
  sketch_rank <- min(max_rank, rank + tuning$oversample)

  run_backend <- function(selected_backend) {
    fastpls_rsvd_pca_scores_backend(
      x,
      rank = rank,
      sketch_rank = sketch_rank,
      seed = seed,
      power = tuning$power,
      backend = selected_backend
    )
  }

  native_reason <- NA_character_
  if (backend_requested %in% c("cuda", "metal")) {
    native <- tryCatch(
      run_backend(backend_requested),
      error = function(e) {
        native_reason <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(native)) {
      return(c(native, list(backend_reason = NA_character_)))
    }
  }

  cpu <- run_backend("cpu")
  if (!is.na(native_reason)) {
    cpu$backend_reason <- paste0(backend_requested, "_rsvd_unavailable: ", native_reason)
  } else if (!identical(backend_requested, "cpu")) {
    cpu$backend_reason <- paste0(backend_requested, "_rsvd_not_requested")
  } else {
    cpu$backend_reason <- NA_character_
  }
  cpu
}

fastpls_rsvd_pca_scores_backend <- function(x,
                                            rank,
                                            sketch_rank,
                                            seed,
                                            power,
                                            backend) {
  restore_seed <- set_local_seed(seed)
  on.exit(restore_seed(), add = TRUE)
  omega <- matrix(stats::rnorm(ncol(x) * sketch_rank), nrow = ncol(x), ncol = sketch_rank)
  multiply <- function(left, right, transpose_left = FALSE) {
    rsvd_matrix_multiply(left, right, transpose_left = transpose_left, backend = backend)
  }

  y <- multiply(x, omega)
  power <- max(0L, as.integer(power))
  if (power == 1L) {
    y <- multiply(x, multiply(x, y, transpose_left = TRUE))
  } else if (power > 1L) {
    for (i in seq_len(power)) {
      z <- multiply(x, y, transpose_left = TRUE)
      qz <- qr.Q(qr(z))
      y <- multiply(x, qz)
    }
  }

  q <- qr.Q(qr(y))
  b <- multiply(q, x, transpose_left = TRUE)
  small <- svd(b, nu = rank, nv = rank)
  usable <- min(rank, length(small$d), ncol(small$u), ncol(small$v))
  if (usable < 1L) {
    stop("RSVD PCA produced no usable singular vectors.", call. = FALSE)
  }

  u <- q %*% small$u[, seq_len(usable), drop = FALSE]
  scores <- sweep(u, 2L, small$d[seq_len(usable)], "*")
  loadings <- small$v[, seq_len(usable), drop = FALSE]
  scores <- orient_pca_scores(scores, loadings)
  colnames(scores) <- paste0("PC", seq_len(ncol(scores)))

  list(
    scores = scores,
    loadings = attr(scores, "loadings"),
    singular_values = small$d[seq_len(usable)],
    backend = paste0(backend, "_rsvd"),
    method = "rsvd",
    oversample = as.integer(sketch_rank - rank),
    power = as.integer(power)
  )
}

rsvd_matrix_multiply <- function(left,
                                 right,
                                 transpose_left = FALSE,
                                 backend = "cpu") {
  left <- as.matrix(left)
  right <- as.matrix(right)
  storage.mode(left) <- "double"
  storage.mode(right) <- "double"
  if (isTRUE(transpose_left)) {
    if (nrow(left) != nrow(right)) {
      stop("Non-conformable RSVD cross-product.", call. = FALSE)
    }
  } else if (ncol(left) != nrow(right)) {
    stop("Non-conformable RSVD matrix multiply.", call. = FALSE)
  }

  if (identical(backend, "cuda")) {
    if (!cuda_metric_available()) {
      stop("CUDA RSVD matrix multiply is not available.", call. = FALSE)
    }
    return(rsvd_multiply_cuda_cpp(left, right, isTRUE(transpose_left)))
  }
  if (identical(backend, "metal")) {
    if (!metal_metric_available()) {
      stop("Metal RSVD matrix multiply is not available.", call. = FALSE)
    }
    return(rsvd_multiply_metal_cpp(left, right, isTRUE(transpose_left)))
  }

  if (isTRUE(transpose_left)) {
    crossprod(left, right)
  } else {
    left %*% right
  }
}

orient_pca_scores <- function(scores, loadings) {
  for (j in seq_len(ncol(scores))) {
    pivot <- which.max(abs(loadings[, j]))
    if (length(pivot) == 1L && is.finite(loadings[pivot, j]) && loadings[pivot, j] < 0) {
      scores[, j] <- -scores[, j]
      loadings[, j] <- -loadings[, j]
    }
  }
  attr(scores, "loadings") <- loadings
  scores
}

normalize_supplied_knn <- function(nn, n, n_neighbors = NULL, keep_self = FALSE) {
  if (!all(c("indices", "distances") %in% names(nn))) {
    stop("`nn` must contain `indices` and `distances`.", call. = FALSE)
  }
  indices <- nn$indices
  distances <- nn$distances
  if (!is.matrix(indices)) indices <- as.matrix(indices)
  distance_is_float32 <- is_float32_matrix(distances)
  if (!distance_is_float32 && !is.matrix(distances)) distances <- as.matrix(distances)
  if (!is.integer(indices)) storage.mode(indices) <- "integer"
  if (!distance_is_float32 && !identical(typeof(distances), "double")) {
    storage.mode(distances) <- "double"
  }
  if (!identical(dim(indices), dim(distances))) {
    stop("KNN `indices` and `distances` must have the same dimensions.", call. = FALSE)
  }
  if (nrow(indices) != n) {
    stop("KNN matrix row count must match `nrow(data)`.", call. = FALSE)
  }
  if (ncol(indices) < 1L) {
    stop("KNN matrices must have at least one neighbor column.", call. = FALSE)
  }
  if (!distance_is_float32 && (any(!is.finite(distances)) || any(distances < 0))) {
    stop("KNN `distances` must be finite and non-negative.", call. = FALSE)
  }

  stripped <- if (distance_is_float32) {
    strip_self_neighbors_float_cpp(indices, distances)
  } else {
    strip_self_neighbors(indices, distances)
  }
  has_self <- stripped$has_self
  knn_with_self <- if (isTRUE(keep_self) && has_self) {
    list(indices = indices, distances = distances)
  } else {
    NULL
  }
  if (has_self) {
    indices <- stripped$indices
    distances <- stripped$distances
  }
  if (ncol(indices) < 1L) {
    stop("KNN matrices must contain at least one non-self neighbor.", call. = FALSE)
  }

  if (is.null(n_neighbors)) {
    n_neighbors <- ncol(indices)
  } else {
    n_neighbors <- as.integer(n_neighbors)
    if (length(n_neighbors) != 1L || is.na(n_neighbors) || !is.finite(n_neighbors) || n_neighbors < 1L) {
      stop("`n_neighbors` must be NULL or a positive integer.", call. = FALSE)
    }
    if (n_neighbors > ncol(indices)) {
      stop("`n_neighbors` is larger than the supplied KNN width.", call. = FALSE)
    }
    indices <- indices[, seq_len(n_neighbors), drop = FALSE]
    distances <- distances[, seq_len(n_neighbors), drop = FALSE]
  }
  list(
    indices = indices,
    distances = distances,
    n_neighbors = as.integer(n_neighbors),
    has_self = isTRUE(has_self),
    knn_with_self = knn_with_self
  )
}

embedding_scores <- function(layout,
                             labels,
                             indices,
                             silhouette_sample,
                             preserve_sample,
                             preserve_k,
                             seed,
                             preserve_keep = NULL,
                             backend = "cpu") {
  n <- nrow(layout)
  labels_factor <- if (is.null(labels)) NULL else as.factor(labels)
  labels_int <- if (is.null(labels_factor)) NULL else as.integer(labels_factor)
  n_label_levels <- if (is.null(labels_factor)) 0L else length(levels(labels_factor))
  silhouette_keep <- sample_indices(n, silhouette_sample, seed)
  if (is.null(preserve_keep)) {
    preserve_keep <- sample_indices(n, preserve_sample, seed)
  }
  preserve_k <- if (is.null(preserve_k)) ncol(indices) else min(as.integer(preserve_k), ncol(indices))

  silhouette_result <- if (is.null(labels_int) || length(silhouette_keep) == 0L) {
    list(value = NA_real_, backend = if (is.null(labels_int)) "none" else "skipped", reason = NA_character_)
  } else {
    silhouette_score_with_backend(
      labels_int[silhouette_keep],
      layout[silhouette_keep, , drop = FALSE],
      n_label_levels,
      backend = backend
    )
  }
  structure_result <- structure_score_with_backend(
    layout,
    indices,
    preserve_keep,
    preserve_k,
    if (is.null(labels_int)) integer(0L) else labels_int,
    n_label_levels,
    backend = backend
  )
  structure <- structure_result$values
  out <- data.frame(
    silhouette = silhouette_result$value,
    knn_preservation = unname(structure["knn_preservation"]),
    local_trustworthiness = unname(structure["local_trustworthiness"]),
    local_continuity = unname(structure["local_continuity"]),
    structure_score = unname(structure["structure_score"]),
    embedding_knn_accuracy = unname(structure["embedding_knn_accuracy"]),
    stringsAsFactors = FALSE
  )
  attr(out, "structure_backend") <- structure_result$backend
  attr(out, "structure_backend_reason") <- structure_result$reason
  attr(out, "silhouette_backend") <- silhouette_result$backend
  attr(out, "silhouette_backend_reason") <- silhouette_result$reason
  attr(out, "backend") <- paste(
    paste0("structure:", structure_result$backend),
    paste0("silhouette:", silhouette_result$backend),
    sep = ";"
  )
  out
}

auto_landmark_count <- function(n) {
  if (n <= 3L) {
    return(n)
  }
  if (n <= 80L) {
    return(as.integer(min(n - 1L, max(2L, ceiling(n * 0.5)))))
  }
  if (n <= 1000L) {
    return(as.integer(min(n - 1L, max(80L, ceiling(sqrt(n) * 7)))))
  }
  as.integer(min(n - 1L, max(300L, ceiling(sqrt(n) * 7))))
}

resolve_landmarks <- function(landmarks, x, seed) {
  n <- nrow(x)
  if (is.null(landmarks) || identical(landmarks, FALSE)) {
    return(NULL)
  }

  if (identical(landmarks, TRUE)) {
    count <- auto_landmark_count(n)
    return(select_landmark_rows(x, count, seed))
  }

  if (length(landmarks) == 1L && is.numeric(landmarks)) {
    value <- as.numeric(landmarks)
    if (!is.finite(value) || value <= 0) {
      stop("`landmarks` must be NULL, TRUE, a positive count, a fraction in (0, 1), or row indices.", call. = FALSE)
    }
    count <- if (value > 0 && value < 1) {
      ceiling(n * value)
    } else {
      as.integer(round(value))
    }
    if (count < 2L) {
      stop("Landmark mode requires at least two landmarks.", call. = FALSE)
    }
    if (count >= n) {
      return(NULL)
    }
    return(select_landmark_rows(x, count, seed))
  }

  if (!is.numeric(landmarks)) {
    stop("`landmarks` must be NULL, TRUE, a positive count, a fraction in (0, 1), or row indices.", call. = FALSE)
  }
  idx <- sort(unique(as.integer(landmarks)))
  if (length(idx) < 2L || any(is.na(idx)) || any(idx < 1L) || any(idx > n)) {
    stop("Landmark row indices must contain at least two valid row numbers.", call. = FALSE)
  }
  if (length(idx) >= n) {
    return(NULL)
  }
  idx
}

select_landmark_rows <- function(x, count, seed) {
  n <- nrow(x)
  count <- as.integer(min(max(2L, count), n))
  if (count >= n) {
    return(seq_len(n))
  }

  z <- landmark_selection_features(x, seed)
  if (count <= 2000L) {
    candidate_count <- min(n, max(count, min(12000L, max(500L, 4L * count))))
    candidates <- projection_quantile_rows(z, candidate_count, seed)
    picked <- farthest_landmark_subset(z[candidates, , drop = FALSE], count)
    selected <- candidates[picked]
    method <- "projected_farthest"
  } else {
    selected <- projection_quantile_rows(z, count, seed)
    method <- "multi_projection_quantiles"
  }
  selected <- sort(unique(as.integer(selected)))
  if (length(selected) < count) {
    selected <- fill_landmark_rows(selected, n, count, seed)
  }
  selected <- sort(selected[seq_len(count)])
  attr(selected, "selection_method") <- method
  selected
}

landmark_selection_features <- function(x, seed) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  direct <- x[, seq_len(min(4L, p)), drop = FALSE]
  n_random <- min(4L, p)

  restore_seed <- set_local_seed(seed)
  on.exit(restore_seed(), add = TRUE)
  directions <- matrix(stats::rnorm(p * n_random), nrow = p, ncol = n_random)
  norms <- sqrt(colSums(directions * directions))
  norms[!is.finite(norms) | norms == 0] <- 1
  directions <- sweep(directions, 2L, norms, "/")
  z <- cbind(direct, x %*% directions)
  z <- as.matrix(z)
  storage.mode(z) <- "double"

  center <- colMeans(z)
  z <- sweep(z, 2L, center, "-")
  scale <- sqrt(colSums(z * z) / max(1L, n - 1L))
  keep <- is.finite(scale) & scale > 0
  if (!any(keep)) {
    return(matrix(seq_len(n), ncol = 1L))
  }
  sweep(z[, keep, drop = FALSE], 2L, scale[keep], "/")
}

projection_quantile_rows <- function(z, count, seed) {
  n <- nrow(z)
  count <- as.integer(min(max(1L, count), n))
  n_axes <- ncol(z)
  per_axis <- max(2L, ceiling(1.25 * count / max(1L, n_axes)))
  selected <- integer(0)

  center_order <- order(rowSums(z * z), seq_len(n))
  selected <- c(selected, center_order[1L])
  for (axis in seq_len(n_axes)) {
    ordered <- order(z[, axis], seq_len(n))
    positions <- unique(round(seq(1, n, length.out = per_axis)))
    selected <- c(selected, ordered[positions])
  }
  selected <- unique(as.integer(selected))

  if (length(selected) < count) {
    selected <- fill_landmark_rows(selected, n, count, seed)
  }
  if (length(selected) > count) {
    ordered <- order(z[selected, 1L], selected)
    positions <- unique(round(seq(1, length(selected), length.out = count)))
    selected <- selected[ordered[positions]]
  }
  sort(unique(as.integer(selected)))[seq_len(count)]
}

farthest_landmark_subset <- function(z, count) {
  n <- nrow(z)
  count <- as.integer(min(max(1L, count), n))
  if (count >= n) {
    return(seq_len(n))
  }

  z_norm <- rowSums(z * z)
  selected <- integer(count)
  selected[1L] <- which.min(z_norm)
  min_dist <- rep(Inf, n)

  for (i in seq_len(count)) {
    if (i > 1L) {
      selected[i] <- which.max(min_dist)
    }
    center <- z[selected[i], , drop = FALSE]
    dist <- z_norm + z_norm[selected[i]] - 2 * drop(z %*% t(center))
    min_dist <- pmin(min_dist, pmax(0, dist))
    min_dist[selected[seq_len(i)]] <- -Inf
  }
  selected
}

fill_landmark_rows <- function(selected, n, count, seed) {
  selected <- unique(as.integer(selected))
  if (length(selected) >= count) {
    return(selected[seq_len(count)])
  }
  restore_seed <- set_local_seed(seed + 1009L)
  on.exit(restore_seed(), add = TRUE)
  remaining <- setdiff(seq_len(n), selected)
  need <- count - length(selected)
  c(selected, sort(sample(remaining, need)))
}

sampled_score_indices <- function(x,
                                  keep,
                                  preserve_k,
                                  backend,
                                  n_threads = NULL) {
  n <- nrow(x)
  preserve_k <- as.integer(max(1L, min(preserve_k, n - 1L)))
  keep <- as.integer(keep)
  if (length(keep) == 0L) {
    return(matrix(integer(0L), nrow = 0L, ncol = preserve_k))
  }
  out <- matrix(1L, nrow = length(keep), ncol = preserve_k)
  query_k <- min(n, preserve_k + 1L)
  batch_size <- max(256L, min(length(keep), as.integer(floor(2e6 / max(1L, query_k)))))
  starts <- seq.int(1L, length(keep), by = batch_size)
  for (start in starts) {
    end <- min(length(keep), start + batch_size - 1L)
    rows <- start:end
    batch_keep <- keep[rows]
    raw <- fastembedr_faissr_nn(
      x,
      x[batch_keep, , drop = FALSE],
      k = query_k,
      backend = backend,
      n_threads = n_threads
    )
    for (local_i in seq_along(batch_keep)) {
      row <- unique(raw$indices[local_i, raw$indices[local_i, ] != batch_keep[local_i]])
      if (length(row) < preserve_k) {
        fill <- setdiff(seq_len(n), c(batch_keep[local_i], row))
        row <- c(row, fill)
      }
      out[rows[local_i], ] <- row[seq_len(preserve_k)]
    }
    rm(raw)
    if (length(starts) > 1L) gc(FALSE)
  }
  out
}

#' @export
print.fastEmbedR_embedding <- function(x, ...) {
  cat("fastEmbedR embedding\n")
  cat("  method: ", x$parameters$method, "\n", sep = "")
  cat("  observations: ", x$parameters$n, "\n", sep = "")
  cat("  dimensions: ", ncol(x$layout), "\n", sep = "")
  cat("  neighbors: ", x$parameters$n_neighbors, "\n", sep = "")
  if (isTRUE(x$parameters$landmark)) {
    cat("  landmarks: ", x$parameters$n_landmarks, "\n", sep = "")
  }
  cat("  embedding backend: ", x$parameters$backend, "\n", sep = "")
  cat("  KNN backend: ", x$parameters$nn_backend, "\n", sep = "")
  cat("  elapsed: ", format(round(x$metrics$elapsed, 3L), nsmall = 3L), " sec\n", sep = "")
  silhouette <- if ("silhouette" %in% names(x$metrics)) x$metrics$silhouette[[1L]] else NA_real_
  if (length(silhouette) == 1L && is.finite(silhouette)) {
    cat("  silhouette: ", format(round(silhouette, 4L), nsmall = 4L), "\n", sep = "")
  }
  knn_preservation <- if ("knn_preservation" %in% names(x$metrics)) x$metrics$knn_preservation[[1L]] else NA_real_
  if (length(knn_preservation) == 1L && is.finite(knn_preservation)) {
    cat("  KNN preservation: ", format(round(knn_preservation, 4L), nsmall = 4L), "\n", sep = "")
  }
  invisible(x)
}

#' @export
plot.fastEmbedR_embedding <- function(x,
                                      labels = x$labels,
                                      pch = 21,
                                      bg = NULL,
                                      col = "black",
                                      xlab = "Component 1",
                                      ylab = "Component 2",
                                      main = NULL,
                                      ...) {
  layout <- embedding_dense_double_matrix(x$layout)
  if (ncol(layout) < 2L) {
    stop("Plotting requires at least two embedding dimensions.", call. = FALSE)
  }
  if (is.null(main)) {
    main <- paste0("fastEmbedR ", x$parameters$method)
  }
  if (is.null(bg)) {
    bg <- if (is.null(labels)) {
      "white"
    } else {
      as.integer(as.factor(labels))
    }
  }
  graphics::plot(
    layout[, 1L],
    layout[, 2L],
    pch = pch,
    bg = bg,
    col = col,
    xlab = xlab,
    ylab = ylab,
    main = main,
    ...
  )
  invisible(x)
}
