# Backend-aware randomized-PCA initialization for openTSNE workflows.

make_opentsne_pca_init <- function(x,
                                   n_components,
                                   seed,
                                   backend = "cpu",
                                   n_threads = NULL,
                                   .thread_controlled = FALSE) {
  x <- opentsne_pca_input_matrix(x)
  n_components <- as.integer(n_components)
  if (length(n_components) != 1L || is.na(n_components) || n_components < 1L) {
    stop("`n_components` must be a positive integer.", call. = FALSE)
  }
  if (identical(backend, "cpu") && !isTRUE(.thread_controlled)) {
    requested <- if (is.null(n_threads)) default_tsne_threads() else n_threads
    return(with_pca_cpu_threads(
      requested,
      make_opentsne_pca_init(
        x,
        n_components = n_components,
        seed = seed,
        backend = backend,
        n_threads = requested,
        .thread_controlled = TRUE
      )
    )$value)
  }
  if (identical(backend, "metal")) {
    pca <- fastembedr_metal_tsvd_pca(
      x,
      ncomp = n_components,
      center = TRUE,
      scale = FALSE,
      seed = seed
    )
    init <- normalize_opentsne_pca_scores(pca$scores, n_components)
    attr(init, "fastEmbedR_init_method") <- paste0("pca_", pca$method)
    attr(init, "fastEmbedR_init_backend") <- pca$backend
    attr(init, "fastEmbedR_init_backend_reason") <- pca$backend_reason
    attr(init, "fastEmbedR_init_package") <- "fastEmbedR native Metal/MPS"
    attr(init, "fastEmbedR_init_package_version") <-
      as.character(utils::packageVersion("fastEmbedR"))
    attr(init, "fastEmbedR_init_timing") <- pca$timing
    return(init)
  }
  cuda_init_reason <- NA_character_
  if (identical(backend, "cuda")) {
    if (!exists("cuml_tsvd_init_cuda_cpp", mode = "function")) {
      stop(
        "CUDA PCA initialization requires native RAPIDS RAFT/cuML TSVD support, ",
        "but the package was not built with that backend.",
        call. = FALSE
      )
    }
    native_cuda <- tryCatch(
      cuml_tsvd_init_cuda_cpp(opentsne_dense_numeric_matrix(x), n_components),
      error = function(e) {
        cuda_init_reason <<- conditionMessage(e)
        NULL
      }
    )
    if (!is.null(native_cuda)) {
      pca <- list(
        scores = native_cuda,
        loadings = NULL,
        singular_values = NULL,
        backend = "cuda_raft_tsvd",
        method = "cuda_raft_tsvd_pca",
        oversample = NA_integer_,
        power = NA_integer_,
        backend_reason = NA_character_,
        package = "RAPIDS RAFT TSVD",
        package_version = NA_character_
      )
      init <- normalize_opentsne_pca_scores(pca$scores, n_components)
      attr(init, "fastEmbedR_init_method") <- paste0("pca_", pca$method)
      attr(init, "fastEmbedR_init_backend") <- pca$backend
      attr(init, "fastEmbedR_init_backend_reason") <- pca$backend_reason
      attr(init, "fastEmbedR_init_package") <- pca$package
      attr(init, "fastEmbedR_init_package_version") <- pca$package_version
      return(init)
    }
    stop(
      "CUDA PCA initialization failed in native RAPIDS RAFT/cuML TSVD: ",
      cuda_init_reason,
      call. = FALSE
    )
  }
  requested_threads <- if (is.null(n_threads)) {
    default_tsne_threads()
  } else {
    normalize_pca_threads(n_threads)
  }
  pca <- fastembedr_cpu_rsvd_pca(
    x,
    ncomp = n_components,
    center = TRUE,
    scale = FALSE,
    seed = seed,
    n.cores = requested_threads
  )
  pca$package <- "fastEmbedR native CPU RSVD"
  pca$package_version <- as.character(utils::packageVersion("fastEmbedR"))
  init <- normalize_opentsne_pca_scores(pca$scores, n_components)
  attr(init, "fastEmbedR_init_method") <- paste0("pca_", pca$method)
  attr(init, "fastEmbedR_init_backend") <- pca$backend
  attr(init, "fastEmbedR_init_backend_reason") <- pca$backend_reason
  if (!is.null(pca$package)) {
    attr(init, "fastEmbedR_init_package") <- pca$package
    attr(init, "fastEmbedR_init_package_version") <- pca$package_version
  }
  init
}

#' Compute or reuse a PCA initialization for openTSNE
#'
#' `opentsne_pca_init()` creates the small-scale PCA initialization used by
#' [opentsne()] and [opentsne_knn()]. Supplying `cache_file` stores the result
#' as an RDS file; later calls with the same path reuse the saved matrix instead
#' of recomputing PCA. This is useful when comparing several KNN backends with
#' exactly the same initialization. For `backend = "cuda"`, fastEmbedR requires
#' native RAPIDS RAFT/cuML TSVD support compiled into the CUDA backend and fails
#' loudly if that backend is unavailable. Metal uses fastEmbedR's native
#' float32 block-subspace TSVD with native Metal centering/scaling, Metal
#' Performance Shaders matrix products, and one resident GPU workspace. CPU
#' uses fastEmbedR's native float32 blocked RSVD implementation. The package
#' does not call Python or `reticulate` for PCA initialization.
#'
#' @param data Numeric matrix/data frame or `float::float32` matrix with
#'   observations in rows.
#' @param n_components Output dimensionality, usually `2`.
#' @param seed Random seed used by the truncated PCA subspace sketch.
#' @param backend Backend used for PCA when available: `"cpu"`, `"metal"`,
#'   or `"cuda"`.
#' @param n.cores Number of CPU cores used for CPU PCA initialization. Metal
#'   and CUDA ignore this argument.
#' @param cache_file Optional `.rds` file path. If it exists and
#'   `force_recompute = FALSE`, the saved initialization is loaded and
#'   validated.
#' @param force_recompute If `TRUE`, ignore any existing cache and recompute.
#' @return An initialization matrix suitable for `Y_init`. Float32 input is
#'   returned as `float::float32` when the selected PCA backend preserves it.
#' @examples
#' init <- opentsne_pca_init(as.matrix(iris[, 1:4]), seed = 1)
#' plot(init, pch = 21, bg = iris$Species)
#' @export
opentsne_pca_init <- function(data,
                              n_components = 2L,
                              seed = 4L,
                              backend = c("cpu", "metal", "cuda"),
                              n.cores = 1L,
                              cache_file = NULL,
                              force_recompute = FALSE) {
  backend <- match.arg(backend)
  n_components <- as.integer(n_components)
  if (length(n_components) != 1L || is.na(n_components) || n_components < 1L) {
    stop("`n_components` must be a positive integer.", call. = FALSE)
  }
  x <- opentsne_pca_input_matrix(data)
  if (nrow(x) < 2L || ncol(x) < 1L) {
    stop("`data` must have at least two rows and one column.", call. = FALSE)
  }
  if (!(identical(backend, "metal") && is_float32_matrix(x)) &&
      opentsne_pca_has_nonfinite(x)) {
    stop("`data` must contain only finite values.", call. = FALSE)
  }
  if (!is.null(cache_file)) {
    cache_file <- path.expand(as.character(cache_file)[1L])
    if (!isTRUE(force_recompute) && file.exists(cache_file)) {
      init <- readRDS(cache_file)
      init <- resolve_opentsne_y_init(
        init,
        n = nrow(x),
        n_components = n_components
      )
      attr(init, "fastEmbedR_init_cache_file") <- cache_file
      attr(init, "fastEmbedR_init_cache_hit") <- TRUE
      return(init)
    }
  }
  init <- make_opentsne_pca_init(
    x,
    n_components = n_components,
    seed = seed,
    backend = backend,
    n_threads = n.cores
  )
  if (!is.null(cache_file)) {
    dir.create(dirname(cache_file), recursive = TRUE, showWarnings = FALSE)
    saveRDS(init, cache_file, version = 2)
    attr(init, "fastEmbedR_init_cache_file") <- cache_file
    attr(init, "fastEmbedR_init_cache_hit") <- FALSE
  }
  init
}

resolve_opentsne_y_init <- function(Y_init, n, n_components) {
  if (is.null(Y_init)) return(NULL)
  if (is.character(Y_init) && length(Y_init) == 1L) {
    path <- path.expand(Y_init)
    if (!file.exists(path)) {
      stop("`Y_init` file does not exist: ", path, call. = FALSE)
    }
    Y_init <- readRDS(path)
    attr(Y_init, "fastEmbedR_init_cache_file") <- path
    attr(Y_init, "fastEmbedR_init_cache_hit") <- TRUE
  }
  float_init <- is_float32_matrix(Y_init)
  if (!float_init) {
    Y_init <- opentsne_dense_numeric_matrix(Y_init)
  }
  if (nrow(Y_init) != n || ncol(Y_init) != n_components) {
    stop(
      "`Y_init` must have ", n, " rows and ", n_components,
      " columns.",
      call. = FALSE
    )
  }
  finite_init <- if (float_init) {
    isTRUE(float32_all_finite_cpp(Y_init))
  } else {
    all(is.finite(Y_init))
  }
  if (!finite_init) {
    stop("`Y_init` must contain only finite values.", call. = FALSE)
  }
  Y_init
}

make_opentsne_pca_init_from_data <- function(init_data,
                                             n,
                                             n_components,
                                             seed,
                                             backend = "cpu",
                                             n_threads = NULL) {
  x <- opentsne_pca_input_matrix(init_data)
  if (nrow(x) != n) {
    stop("`init_data` must have one row per KNN row.", call. = FALSE)
  }
  make_opentsne_pca_init(
    x,
    n_components = n_components,
    seed = seed,
    backend = backend,
    n_threads = n_threads
  )
}
