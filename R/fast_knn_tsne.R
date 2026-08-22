# Public KNN-input and one-call openTSNE interfaces.

#' Run native openTSNE-style t-SNE from precomputed KNN
#'
#' `opentsne_knn()` is the direct KNN-input entry point for the native
#' openTSNE-style optimizer. It accepts either a list containing KNN `indices`
#' and `distances` or separate KNN index and distance matrices. The supplied
#' neighbors are used directly: this function performs no neighbor search or
#' input scaling. PCA is computed only when `init_data` is supplied and
#' `Y_init` is absent.
#' Here, "openTSNE-style" describes algorithmic lineage, not compatibility with
#' the Python package. fastEmbedR defines its own R API, defaults, objects, and
#' native optimizer kernels; it does not call or port Python `openTSNE`.
#'
#' @param indices A list containing KNN `indices` and `distances`, or an integer
#'   KNN index matrix.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave
#'   `NULL` when `indices` is a KNN list.
#' @param n_neighbors Optional number of non-self neighbor columns to use from
#'   the supplied KNN graph. This lets you compute a wide KNN once and reuse
#'   its first columns for comparable tests.
#' @param perplexity t-SNE perplexity. If `NULL`, the optimizer chooses a safe
#'   value from the supplied KNN width and sample size.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization for KNN-input runs. It is not used for neighbor
#'   search or optimization.
#' @inheritParams opentsne
#' @param backend Optimizer backend: `"cpu"`, `"cuda"`, or `"metal"`. This
#'   selects affinity construction and layout optimization only; it does not
#'   trigger a nearest-neighbor search. Host KNN matrices can be consumed by
#'   any compiled backend. A CUDA-resident KNN object can be consumed directly
#'   only by the CUDA backend. Unavailable GPU requests fail without CPU
#'   fallback.
#' @param n.cores Number of CPU cores used for CPU affinity construction and
#'   openTSNE optimization. No KNN search is performed. Metal and CUDA
#'   optimizers ignore this argument.
#' @param ... Additional low-level optimizer controls, including `theta` and
#'   `min_gain`.
#' @return An embedding matrix with settings stored in
#'   `attr(layout, "fastEmbedR_config")`. Float32 KNN distances, including a
#'   CUDA-resident KNN object, produce a `float::float32` layout; host double
#'   distances produce a standard R double matrix. Native optimization uses
#'   float32 in both cases. The exact returned representation is recorded in
#'   `attr(layout, "precision")` and in `fastEmbedR_config$output_precision`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 15L
#' idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
#'   nrow = nrow(d), byrow = TRUE)
#' layout <- opentsne_knn(idx, dst, init_data = x, perplexity = 5,
#'   early_exaggeration_iter = 5, n_iter = 10)
#' @export
opentsne_knn <- function(indices,
                         distances = NULL,
                         n_neighbors = NULL,
                         perplexity = NULL,
                         n_components = 2L,
                         init_data = NULL,
                         Y_init = NULL,
                         seed = 4L,
                         verbose = FALSE,
                         backend = NULL,
                          n.cores = NULL,
                          learning_rate = "auto",
                          early_exaggeration_iter = NULL,
                          early_exaggeration = "auto",
                          n_iter = NULL,
                          exaggeration = NULL,
                         initial_momentum = 0.8,
                         final_momentum = 0.8,
                          max_step_norm = "auto",
                          negative_gradient_method = "auto",
                          record_costs = FALSE,
                          auto_config = TRUE,
                          ...) {
  n_threads <- n.cores
  backend <- resolve_embedding_backend(backend)
  if (fastembedr_is_gpu_knn(indices) && identical(backend, "cuda")) {
    indices <- fastembedr_as_gpu_knn(indices)
    if (!is.null(distances)) {
      stop("Do not pass `distances` when `indices` is a GPU-resident KNN object.", call. = FALSE)
    }
    gpu_info <- fastembedr_gpu_knn_info(indices)
    if (isTRUE(gpu_info$has_self)) {
      stop(
        "CUDA GPU-resident openTSNE requires non-self KNN. ",
        "Use fastEmbedR's native matrix-input CUDA route or provide a native ",
        "fastEmbedR GPU KNN object with non-self neighbors.",
        call. = FALSE
      )
    }
    policy <- opentsne_neighbor_policy(
      gpu_info$n,
      perplexity = perplexity,
      available = gpu_info$k
    )
    if (is.null(n_neighbors)) {
      n_neighbors <- policy$n_neighbors
    } else {
      n_neighbors <- as.integer(n_neighbors)
      if (length(n_neighbors) != 1L || is.na(n_neighbors) ||
          !is.finite(n_neighbors) || n_neighbors < 1L ||
          n_neighbors > gpu_info$k || n_neighbors >= gpu_info$n) {
        stop(
          "`n_neighbors` must be a positive integer available in the GPU KNN object.",
          call. = FALSE
        )
      }
    }
    if (is.null(perplexity)) perplexity <- policy$perplexity
    Y_init <- resolve_opentsne_y_init(Y_init, gpu_info$n, n_components)
    cuda_init_data <- NULL
    if (is.null(Y_init) && !is.null(init_data)) {
      init_check <- opentsne_pca_input_matrix(init_data)
      if (nrow(init_check) != gpu_info$n) {
        stop("`init_data` must have one row per GPU KNN row.", call. = FALSE)
      }
      cuda_init_data <- init_check
    }
    return(fast_knn_opentsne_materialized(
      NULL,
      NULL,
      n_components = n_components,
      perplexity = perplexity,
      Y_init = Y_init,
      seed = seed,
      verbose = verbose,
      backend = backend,
      n_threads = n_threads,
      learning_rate = learning_rate,
      early_exaggeration_iter = early_exaggeration_iter,
      early_exaggeration = early_exaggeration,
      n_iter = n_iter,
      exaggeration = exaggeration,
      initial_momentum = initial_momentum,
      final_momentum = final_momentum,
      max_step_norm = max_step_norm,
      negative_gradient_method = negative_gradient_method,
      record_costs = record_costs,
      auto_config = auto_config,
      input_had_self = FALSE,
      input_backend = gpu_info$input_backend,
      gpu_knn = indices,
      gpu_n = gpu_info$n,
      gpu_k = as.integer(n_neighbors),
      cuda_init_data = cuda_init_data,
      ...
    ))
  }
  if (inherits(indices, "fastEmbedR_opentsne_prepared")) {
    if (!is.null(distances)) {
      stop("Do not pass `distances` when `indices` is a prepared openTSNE object.", call. = FALSE)
    }
    knn <- indices$knn
    if (is.null(perplexity)) perplexity <- indices$perplexity
  } else {
    knn0 <- coerce_knn_input(indices, distances)
    policy <- opentsne_neighbor_policy(
      nrow(knn0$indices),
      perplexity = perplexity,
      available = knn0$n_neighbors
    )
    if (is.null(n_neighbors)) n_neighbors <- policy$n_neighbors
    if (is.null(perplexity)) perplexity <- policy$perplexity
    knn <- normalize_opentsne_knn_input(indices, distances, n_neighbors)
  }
  Y_init <- resolve_opentsne_y_init(Y_init, knn$n, n_components)
  if (is.null(Y_init) && !is.null(init_data)) {
    init_backend <- if (backend %in% c("metal", "cuda")) backend else "cpu"
    Y_init <- make_opentsne_pca_init_from_data(
      init_data,
      n = knn$n,
      n_components = n_components,
      seed = seed,
      backend = init_backend,
      n_threads = n_threads
    )
  }
  fast_knn_opentsne_materialized(
    knn$indices,
    knn$distances,
    n_components = n_components,
    perplexity = perplexity,
    Y_init = Y_init,
    seed = seed,
    verbose = verbose,
    backend = backend,
    n_threads = n_threads,
    learning_rate = learning_rate,
    early_exaggeration_iter = early_exaggeration_iter,
    early_exaggeration = early_exaggeration,
    n_iter = n_iter,
    exaggeration = exaggeration,
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    max_step_norm = max_step_norm,
     negative_gradient_method = negative_gradient_method,
     record_costs = record_costs,
     auto_config = auto_config,
     input_had_self = knn$has_self,
    input_backend = knn$input_backend,
    ...
  )
}

#' Run native openTSNE-style t-SNE from a data matrix
#'
#' `opentsne()` computes or reuses a KNN graph, then runs the package-native
#' openTSNE-style two-phase optimizer. The default optimizer is CPU-native.
#' Explicit `backend = "metal"` and `backend = "cuda"` requests use the
#' matching package-native GPU optimizer when compiled and fail clearly if the
#' requested backend is unavailable.
#' "openTSNE-style" describes the published sparse-affinity,
#' interpolation-based t-SNE workflow, not Python API, object, default, or
#' coordinate compatibility. No Python `openTSNE` code is called.
#'
#' @param data Numeric matrix/data frame with observations in rows, or a list
#'   containing KNN `indices` and `distances`.
#' @param perplexity t-SNE perplexity. The one-call API uses
#'   `ceiling(perplexity)` non-self neighbors internally. If `NULL`, uses the
#'   largest safe value up to 30 that is available for the input.
#' @param n_components Output dimensionality, from 1 to 3.
#' @param init_data Optional original high-dimensional data matrix used only to
#'   compute PCA initialization with [opentsne_pca_init()]. It is not used for
#'   neighbor search or optimization.
#' @param Y_init Optional explicit initial layout. Use [opentsne_pca_init()] to
#'   precompute and reuse a PCA initialization.
#' @param standardize Center and scale columns before KNN. Defaults to `FALSE`
#'   so one-call results match a KNN object computed from the supplied matrix.
#' @param pca_dims Optional PCA dimension before KNN.
#' @param metric KNN distance metric for one-call matrix input: `"euclidean"`,
#'   `"cosine"`, `"correlation"`, or `"inner_product"`.
#' @param nn Optional precomputed KNN output when `data` is a data matrix.
#' @param seed Random seed.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`. CPU KNN
#'   uses package-native HNSW. Metal uses package-native exact or recall-tuned
#'   IVF-Flat search. CUDA uses direct FAISS GPU exact search below 100,000
#'   rows and direct RAPIDS cuVS IVF-Flat above that threshold, then passes the
#'   device pointers directly to native CUDA openTSNE.
#'   Unsupported GPU requests fail clearly and are not relabelled CPU runs.
#' @param keep_knn If `TRUE`, retain KNN matrices in the returned object.
#' @param verbose Print optimizer progress.
#' @param n.cores Number of CPU cores used by CPU KNN and CPU
#'   openTSNE optimization. Native GPU optimizers ignore this argument.
#' @param learning_rate Positive number or `"auto"`. With `"auto"`, the native
#'   optimizer uses `n / exaggeration` separately for each phase.
#' @param early_exaggeration_iter Number of early-exaggeration iterations.
#' @param early_exaggeration Early-exaggeration multiplier, or `"auto"` for
#'   openTSNE's default rule.
#' @param n_iter Number of normal optimization iterations after early
#'   exaggeration.
#' @param exaggeration Normal-phase exaggeration. `NULL` means 1.
#' @param initial_momentum Momentum during early exaggeration.
#' @param final_momentum Momentum during normal optimization.
#' @param max_step_norm Maximum per-point update norm. `"auto"` uses the
#'   standard CPU limit and a tighter native Metal FFT-grid limit to avoid
#'   float32 outlier steps. Use `NULL` or `NA` to disable clipping.
#' @param negative_gradient_method `"auto"`, `"exact"`, or
#'   `"fft"`. On CPU, `"auto"` resolves to the native grid-FFT
#'   FIt-SNE-style negative-gradient approximation. Native GPU FFT/exact paths
#'   are used only when the corresponding compiled symbols are available;
#'   otherwise GPU requests fail clearly rather than falling back to CPU.
#' @param record_costs If `TRUE`, compute diagnostic KL/cost traces.
#' @param auto_config If `TRUE`, choose missing t-SNE settings with a native
#'   C++ opt-SNE-inspired policy. The policy uses `n / early_exaggeration` for
#'   `"auto"` learning rate, chooses missing iteration limits, and enables
#'   KLD-based early stopping only on CPU/small exact runs where the monitor is
#'   not prohibitively expensive. Explicit user-supplied values are respected.
#' @param ... Additional low-level parameters passed to [opentsne_knn()].
#' @return A `fastEmbedR_embedding` object.
#' @examples
#' fit <- opentsne(
#'   as.matrix(iris[, 1:4]), perplexity = 5,
#'   early_exaggeration_iter = 5, n_iter = 10, seed = 1
#' )
#' plot(fit, labels = iris$Species)
#' @export
opentsne <- function(data,
                     perplexity = NULL,
                     n_components = 2L,
                     init_data = NULL,
                     Y_init = NULL,
                     standardize = FALSE,
                     pca_dims = NULL,
                     metric = c("euclidean", "cosine", "correlation", "inner_product"),
                     nn = NULL,
                     seed = 4L,
                     backend = NULL,
                     keep_knn = FALSE,
                     verbose = FALSE,
                      n.cores = NULL,
                      learning_rate = "auto",
                      early_exaggeration_iter = NULL,
                      early_exaggeration = "auto",
                      n_iter = NULL,
                     exaggeration = NULL,
                     initial_momentum = 0.8,
                     final_momentum = 0.8,
                     max_step_norm = "auto",
                      negative_gradient_method = "auto",
                      record_costs = FALSE,
                      auto_config = TRUE,
                      ...) {
  n_threads <- n.cores
  backend <- resolve_embedding_backend(backend)
  input_is_float32 <- is_float32_matrix(data)
  dots <- list(...)
  if ("init" %in% names(dots)) {
    stop(
      "`init` is not an argument of `opentsne()`; use `Y_init` or `init_data` ",
      "for PCA initialization.",
      call. = FALSE
    )
  }
  if ("n_neighbors" %in% names(dots)) {
    stop(
      "`n_neighbors` is not an argument of `opentsne()`; use `perplexity`, ",
      "which also determines the internal non-self KNN width.",
      call. = FALSE
    )
  }
  optimizer_backend <- backend
  if (fastembedr_is_gpu_knn(data)) {
    data <- fastembedr_as_gpu_knn(data)
    if (!is.null(nn)) {
      stop("When `data` is a GPU KNN object, do not also pass `nn`.", call. = FALSE)
    }
    layout_time <- system.time({
      layout <- opentsne_knn(
        data,
        n_components = n_components,
        perplexity = perplexity,
        init_data = init_data,
        Y_init = Y_init,
        seed = seed,
        verbose = verbose,
        backend = optimizer_backend,
        n.cores = n_threads,
        learning_rate = learning_rate,
        early_exaggeration_iter = early_exaggeration_iter,
        early_exaggeration = early_exaggeration,
        n_iter = n_iter,
        exaggeration = exaggeration,
        initial_momentum = initial_momentum,
        final_momentum = final_momentum,
        max_step_norm = max_step_norm,
        negative_gradient_method = negative_gradient_method,
        record_costs = record_costs,
        auto_config = auto_config,
        ...
      )
    })
    cfg <- attr(layout, "fastEmbedR_config")
    gpu_info <- fastembedr_gpu_knn_info(data)
    zero_time <- layout_time
    zero_time[] <- 0
    timings <- rbind(
      preprocess = zero_time,
      knn = zero_time,
      embedding = layout_time
    )
    metrics <- data.frame(
      method = "opentsne",
      n = nrow(layout),
      p = NA_integer_,
      n_neighbors = cfg$n_neighbors %||% gpu_info$k,
      perplexity = cfg$perplexity,
      elapsed = sum(timings[, "elapsed"]),
      preprocess_elapsed = 0,
      knn_elapsed = 0,
      embedding_elapsed = layout_time["elapsed"],
      stringsAsFactors = FALSE
    )
    init_label <- if (is.null(Y_init) && is.null(init_data)) {
      "none_precomputed_gpu_knn_cuda_random_init"
    } else {
      "none_precomputed_gpu_knn"
    }
    out <- list(
      layout = layout,
      labels = NULL,
      method = "opentsne",
      metrics = metrics,
      parameters = c(
        list(
          method = "opentsne",
          input = "gpu_resident_knn",
          n = nrow(layout),
          p = NA_integer_,
          n_neighbors = cfg$n_neighbors %||% gpu_info$k,
          k = cfg$n_neighbors %||% gpu_info$k,
          n_components = as.integer(n_components),
          seed = as.integer(seed),
          nn_backend = gpu_info$input_backend,
          keep_knn = keep_knn
        ),
        cfg,
        list(preprocess = init_label)
      ),
      timings = timings,
      knn = if (isTRUE(keep_knn)) data else NULL,
      knn_with_self = NULL,
      preprocess = list(input = init_label),
      diagnostics = list()
    )
    class(out) <- "fastEmbedR_embedding"
    return(out)
  }
  if (is_knn_input(data)) {
    if (!is.null(nn)) {
      stop("When `data` is a KNN object, do not also pass `nn`.", call. = FALSE)
    }
    full_knn <- normalize_opentsne_knn_input(data, NULL, NULL)
    n <- full_knn$n
    neighbor_policy <- opentsne_neighbor_policy(
      n,
      perplexity = perplexity,
      available = full_knn$n_neighbors
    )
    perplexity <- neighbor_policy$perplexity
    knn_result <- normalize_opentsne_knn_input(
      data,
      NULL,
      neighbor_policy$n_neighbors
    )
    Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
    if (is.null(Y_init) && !is.null(init_data)) {
      init_backend <- if (optimizer_backend %in% c("metal", "cuda")) optimizer_backend else "cpu"
      Y_init <- make_opentsne_pca_init_from_data(
        init_data,
        n = n,
        n_components = n_components,
        seed = seed,
        backend = init_backend,
        n_threads = n_threads
      )
    }

    embedding_time <- system.time({
      layout <- opentsne_knn(
        data,
        n_neighbors = knn_result$n_neighbors,
        n_components = n_components,
        perplexity = perplexity,
        init_data = init_data,
        Y_init = Y_init,
        seed = seed,
        verbose = verbose,
        backend = optimizer_backend,
        n.cores = n_threads,
        learning_rate = learning_rate,
        early_exaggeration_iter = early_exaggeration_iter,
        early_exaggeration = early_exaggeration,
        n_iter = n_iter,
        exaggeration = exaggeration,
        initial_momentum = initial_momentum,
        final_momentum = final_momentum,
        max_step_norm = max_step_norm,
         negative_gradient_method = negative_gradient_method,
         record_costs = record_costs,
         auto_config = auto_config,
        ...
      )
    })
    cfg <- attr(layout, "fastEmbedR_config")
    zero_time <- embedding_time
    zero_time[] <- 0
    timings <- rbind(
      preprocess = zero_time,
      knn = zero_time,
      embedding = embedding_time
    )
    knn_backend <- knn_result$input_backend
    if (is.na(knn_backend) || is.null(knn_backend)) knn_backend <- "supplied"
    metrics <- data.frame(
      method = "opentsne",
      n = n,
      p = NA_integer_,
      n_neighbors = knn_result$n_neighbors,
      perplexity = cfg$perplexity,
      elapsed = sum(timings[, "elapsed"]),
      preprocess_elapsed = 0,
      knn_elapsed = 0,
      embedding_elapsed = embedding_time["elapsed"],
      stringsAsFactors = FALSE
    )
    parameters <- c(
      list(
        method = "opentsne",
        input = "knn",
        n = n,
        p = NA_integer_,
        n_neighbors = knn_result$n_neighbors,
        k = knn_result$n_neighbors + 1L,
        n_components = as.integer(n_components),
        seed = as.integer(seed),
        nn_backend = knn_backend,
        keep_knn = keep_knn
      ),
      cfg,
      list(preprocess = "none_precomputed_knn")
    )
    out <- list(
      layout = layout,
      labels = NULL,
      method = "opentsne",
      metrics = metrics,
      parameters = parameters,
      timings = timings,
      knn = if (isTRUE(keep_knn)) {
        list(indices = knn_result$indices, distances = knn_result$distances)
      } else {
        NULL
      },
      knn_with_self = NULL,
      preprocess = list(input = "precomputed_knn"),
      diagnostics = list(
        metal_trace = attr(layout, "metal_trace"),
        metal_stage_timing = attr(layout, "metal_stage_timing")
      )
    )
    class(out) <- "fastEmbedR_embedding"
    return(out)
  }

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
  metric <- resolve_embedding_metric(metric, x)
  n <- nrow(x)
  neighbor_policy <- opentsne_neighbor_policy(n, perplexity = perplexity)
  perplexity <- neighbor_policy$perplexity
  n_neighbors <- neighbor_policy$n_neighbors

  knn_engine <- "supplied"
  knn_time <- system.time({
    if (is.null(nn)) {
      knn_policy <- fastembedr_embedding_nn_policy(backend, n = n)
      knn_engine <- fastembedr_nn_policy_engine(
        knn_policy,
        keep_gpu = identical(knn_policy$backend, "cuda")
      )
      raw_knn <- fastembedr_nn_without_self(
        x,
        k = n_neighbors,
        backend = knn_policy$backend,
        method = knn_policy$method,
        metric = metric,
        output = fastembedr_knn_output_type(x, knn_policy$backend),
        n_threads = n_threads,
        tuning = knn_policy$tuning,
        target_recall = knn_policy$target_recall,
        keep_gpu = identical(knn_policy$backend, "cuda")
      )
      if (fastembedr_is_gpu_knn(raw_knn)) {
        gpu_info <- fastembedr_gpu_knn_info(raw_knn)
        if (gpu_info$n != n || gpu_info$k < n_neighbors || isTRUE(gpu_info$has_self)) {
          stop("GPU-resident KNN output is incompatible with openTSNE input.", call. = FALSE)
        }
        knn_result <- list(
          indices = NULL,
          distances = NULL,
          n_neighbors = as.integer(n_neighbors),
          has_self = FALSE,
          knn_with_self = NULL,
          nn_backend = gpu_info$input_backend
        )
      } else {
        knn_result <- normalize_supplied_knn(raw_knn, n, n_neighbors)
        knn_result$nn_backend <- attr(raw_knn, "backend")
      }
      embedding_knn_input <- raw_knn
    } else {
      if (fastembedr_is_gpu_knn(nn)) {
        nn <- fastembedr_as_gpu_knn(nn)
        gpu_info <- fastembedr_gpu_knn_info(nn)
        incompatible <- gpu_info$n != n ||
          gpu_info$k < n_neighbors ||
          isTRUE(gpu_info$has_self)
        if (incompatible) {
          stop(
            "Supplied GPU-resident KNN output is incompatible with openTSNE input.",
            call. = FALSE
          )
        }
        knn_result <- list(
          indices = NULL,
          distances = NULL,
          n_neighbors = as.integer(n_neighbors),
          has_self = FALSE,
          knn_with_self = NULL,
          nn_backend = gpu_info$input_backend
        )
      } else {
        knn_result <- normalize_supplied_knn(nn, n, n_neighbors, keep_self = keep_knn)
        knn_result$nn_backend <- attr(nn, "backend")
        if (is.null(knn_result$nn_backend)) knn_result$nn_backend <- "supplied"
      }
      embedding_knn_input <- nn
    }
  })

  initialization_time <- system.time({
    Y_init <- resolve_opentsne_y_init(Y_init, n, n_components)
    init_info <- list(method = "user", backend = NA_character_)
    cuda_init_data <- NULL
    if (is.null(Y_init)) {
      init_source <- if (is.null(init_data)) x else init_data
      init_backend <- if (optimizer_backend %in% c("metal", "cuda")) optimizer_backend else "cpu"
      if (identical(optimizer_backend, "cuda") && fastembedr_is_gpu_knn(embedding_knn_input)) {
        cuda_init_candidate <- opentsne_pca_input_matrix(init_source)
        if (nrow(cuda_init_candidate) != n) {
          stop("`init_data` must have one row per input row.", call. = FALSE)
        }
        cuda_init_data <- cuda_init_candidate
        init_info$method <- "pca_cuda_raft_tsvd_pca_device"
        init_info$backend <- "cuda_raft_tsvd_device"
      } else {
        init_result <- tryCatch(
          list(
            layout = make_opentsne_pca_init_from_data(
              init_source,
              n = n,
              n_components = n_components,
              seed = seed,
              backend = init_backend,
              n_threads = n_threads
            ),
            problem = NULL
          ),
          error = function(e) {
            list(layout = NULL, problem = conditionMessage(e))
          }
        )
        Y_init <- init_result$layout
        if (!is.null(Y_init)) {
          init_info$method <- attr(Y_init, "fastEmbedR_init_method") %||% "pca"
          init_info$backend <- attr(Y_init, "fastEmbedR_init_backend") %||% init_backend
        } else {
          stop(
            "openTSNE PCA initialization failed for backend `",
            init_backend,
            "`: ",
            init_result$problem %||% "unknown problem",
            call. = FALSE
          )
        }
      }
    }
  })

  embedding_time <- system.time({
    layout <- opentsne_knn(
      embedding_knn_input,
      n_neighbors = knn_result$n_neighbors,
      n_components = n_components,
      perplexity = perplexity,
      init_data = cuda_init_data %||% init_data,
      Y_init = Y_init,
      seed = seed,
      verbose = verbose,
      backend = optimizer_backend,
      n.cores = n_threads,
      learning_rate = learning_rate,
      early_exaggeration_iter = early_exaggeration_iter,
      early_exaggeration = early_exaggeration,
      n_iter = n_iter,
      exaggeration = exaggeration,
      initial_momentum = initial_momentum,
      final_momentum = final_momentum,
      max_step_norm = max_step_norm,
            negative_gradient_method = negative_gradient_method,
            record_costs = record_costs,
            auto_config = auto_config,
      ...
    )
  })
  layout <- finalize_embedding_layout(
    layout,
    "openTSNE",
    return_float32 = input_is_float32 && is_float32_matrix(x)
  )
  cfg <- attr(layout, "fastEmbedR_config")
  timings <- rbind(
    preprocess = preprocess_time,
    knn = knn_time,
    initialization = initialization_time,
    embedding = embedding_time
  )
  metrics <- data.frame(
    method = "opentsne",
    n = n,
    p = ncol(x),
    n_neighbors = knn_result$n_neighbors,
    perplexity = cfg$perplexity,
    elapsed = sum(timings[, "elapsed"]),
    preprocess_elapsed = preprocess_time["elapsed"],
    knn_elapsed = knn_time["elapsed"],
    initialization_elapsed = initialization_time["elapsed"],
    embedding_elapsed = embedding_time["elapsed"],
    stringsAsFactors = FALSE
  )
  parameters <- c(
    list(
      method = "opentsne",
      n = n,
      p = ncol(x),
      n_neighbors = knn_result$n_neighbors,
      k = knn_result$n_neighbors + 1L,
      n_components = as.integer(n_components),
      seed = as.integer(seed),
      nn_backend = knn_result$nn_backend,
      nn_engine = knn_engine,
      metric = metric,
      keep_knn = keep_knn
    ),
    cfg,
    prepared$preprocess,
    list(init = init_info$method, init_backend = init_info$backend)
  )
  out <- list(
    layout = layout,
    labels = NULL,
    method = "opentsne",
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    knn = if (isTRUE(keep_knn)) {
      if (fastembedr_is_gpu_knn(embedding_knn_input)) {
        embedding_knn_input
      } else {
        list(indices = knn_result$indices, distances = knn_result$distances)
      }
    } else {
      NULL
    },
    knn_with_self = if (isTRUE(keep_knn)) knn_result$knn_with_self else NULL,
    preprocess = prepared$preprocess,
    diagnostics = list(
      metal_trace = attr(layout, "metal_trace"),
      metal_stage_timing = attr(layout, "metal_stage_timing")
    )
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
