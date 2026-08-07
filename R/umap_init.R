#' Precompute a reusable UMAP initialization
#'
#' `umap_init()` separates UMAP neighbor search, graph construction, and
#' initialization from stochastic optimization. The returned object can be
#' passed directly to [umap_knn()] so repeated or cross-backend runs reuse the
#' same KNN graph and initial coordinates.
#'
#' @param x Numeric data, a KNN object, or an object returned by
#'   [prepare_umap_knn()].
#' @param distances Optional KNN distance matrix when `x` is an index matrix.
#' @param n_neighbors Number of non-self neighbors when `x` is data. `NULL`
#'   uses the package's size-aware default.
#' @param n_components Output dimensionality.
#' @param metric Distance metric used only when KNN must be computed.
#' @param backend Initialization backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param seed Random seed.
#' @param n.cores Number of CPU cores.
#' @param graph_mode UMAP graph weighting. `"fuzzy"` is the standard default;
#'   `"binary"` requests the adjacency-only sensitivity graph.
#'
#' @details
#' For the fuzzy graph, CPU initialization is computed from the prepared CSR
#' graph, while native Metal or CUDA spectral kernels are used when requested
#' and available. Binary initialization is computed from the same unit-weight
#' CSR graph used by the optimizer, so its mathematical input is unchanged.
#' Explicit GPU requests fail when the corresponding native backend is
#' unavailable.
#'
#' CUDA KNN input is materialized on the host only for this standalone
#' diagnostic API. The ordinary one-call CUDA UMAP path remains device
#' resident.
#'
#' @return A `fastEmbedR_umap_initialization` object with `layout`, `prepared`,
#'   resolved `parameters`, and component timings. Pass the complete object to
#'   [umap_knn()] or use its `layout` component directly for diagnostics.
#'
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' initialization <- umap_init(
#'   x,
#'   n_neighbors = 15,
#'   backend = "cpu",
#'   graph_mode = "fuzzy",
#'   seed = 1
#' )
#' layout <- umap_knn(initialization, backend = "cpu", seed = 1)
#' plot(layout, pch = 21, bg = iris$Species)
#' @export
umap_init <- function(x,
                      distances = NULL,
                      n_neighbors = NULL,
                      n_components = 2L,
                      metric = c(
                        "euclidean", "cosine", "correlation", "inner_product"
                      ),
                      backend = c("cpu", "cuda", "metal"),
                      seed = 4L,
                      n.cores = NULL,
                      graph_mode = c("fuzzy", "binary")) {
  n_threads <- n.cores
  backend <- resolve_embedding_backend(backend)
  n_components <- validate_n_components(n_components)
  supplied_prepared <- inherits(x, "fastEmbedR_umap_prepared")
  graph_mode_missing <- missing(graph_mode)
  graph_mode <- match.arg(graph_mode)

  if (supplied_prepared && graph_mode_missing) {
    graph_mode <- x$config$graph_mode %||% graph_mode
  }
  if (supplied_prepared &&
      !identical(as.character(x$config$graph_mode), graph_mode)) {
    stop(
      "`graph_mode` does not match the supplied prepared UMAP graph.",
      call. = FALSE
    )
  }

  knn_elapsed <- 0
  graph_elapsed <- 0
  prepared <- if (supplied_prepared) {
    if (!is.null(distances)) {
      stop(
        "Do not pass `distances` with a prepared UMAP object.",
        call. = FALSE
      )
    }
    x
  } else if (!is.null(distances) || is_knn_input(x)) {
    graph_time <- system.time({
      prepared_result <- prepare_umap_knn(
        x,
        distances = distances,
        backend = backend,
        n.cores = n_threads,
        graph_mode = graph_mode
      )
    })
    graph_elapsed <- unname(graph_time[["elapsed"]])
    prepared_result
  } else {
    metric <- resolve_embedding_metric(metric, x)
    n <- nrow(x)
    if (is.null(n_neighbors)) {
      n_neighbors <- auto_embedding_k(n, method = "umap", include_self = FALSE)
    }
    n_neighbors <- as.integer(n_neighbors)
    knn_time <- system.time({
      knn <- precompute_knn(
        x,
        k = n_neighbors,
        metric = metric,
        backend = backend,
        n.cores = n_threads
      )
    })
    knn_elapsed <- unname(knn_time[["elapsed"]])
    if (fastembedr_is_gpu_knn(knn)) {
      knn <- fastembedr_gpu_knn_to_host(knn)
    }
    graph_time <- system.time({
      prepared_result <- prepare_umap_knn(
        knn,
        backend = backend,
        n.cores = n_threads,
        graph_mode = graph_mode
      )
    })
    graph_elapsed <- unname(graph_time[["elapsed"]])
    prepared_result
  }

  cfg <- prepared$config
  cfg$backend <- backend
  cfg$graph_mode <- graph_mode
  if (!is.null(n_threads)) {
    requested_threads <- suppressWarnings(as.integer(n_threads))
    if (length(requested_threads) != 1L || is.na(requested_threads) ||
        requested_threads < 1L) {
      stop("`n.cores` must be NULL or a positive integer.", call. = FALSE)
    }
    cfg$n_threads <- max(1L, min(4L, requested_threads))
  }
  if (n_components != 2L && backend %in% c("cuda", "metal")) {
    stop(
      "Native Metal and CUDA UMAP initialization currently supports two dimensions.",
      call. = FALSE
    )
  }

  initialization_time <- system.time({
    use_native_knn_spectral <- backend %in% c("cuda", "metal") &&
      identical(graph_mode, "fuzzy")
    layout <- if (use_native_knn_spectral) {
      knn <- prepared$knn
      materialized <- materialize_knn_range(
        knn$indices,
        knn$distances,
        knn$col_start,
        knn$n_neighbors
      )
      init_distances <- if (is_float32_matrix(materialized$distances)) {
        matrix(
          as.numeric(materialized$distances),
          nrow = nrow(materialized$indices),
          ncol = ncol(materialized$indices)
        )
      } else {
        materialized$distances
      }
      result <- spectral_knn_init(
        materialized$indices,
        init_distances,
        n_components = n_components,
        min_dist = cfg$min_dist,
        spectral_n_iter = cfg$spectral_n_iter,
        seed = seed,
        backend = backend,
        n_threads = cfg$n_threads
      )
      scale_embedding_sdev_r(result, cfg$init_scale)
    } else {
      umap_init_from_csr_graph(
        prepared$graph,
        n_components = n_components,
        cfg = cfg,
        seed = seed,
        verbose = FALSE
      )
    }
  })
  initialization_elapsed <- unname(initialization_time[["elapsed"]])
  if (!is.matrix(layout) || !identical(dim(layout), c(
    nrow(prepared$knn$indices), as.integer(n_components)
  )) || any(!is.finite(layout))) {
    stop("UMAP initialization produced an invalid layout.", call. = FALSE)
  }

  cfg$init_backend <- attr(layout, "backend") %||% "cpu"
  cfg$initialization_reusable <- TRUE
  cfg$initialization_seed <- as.integer(seed)
  cfg$initialization_n_components <- as.integer(n_components)
  attr(layout, "fastEmbedR_config") <- public_core_config(cfg)
  prepared$initialization <- layout
  prepared$initialization_parameters <- cfg

  out <- list(
    layout = layout,
    prepared = prepared,
    parameters = public_core_config(cfg),
    timings = data.frame(
      stage = c("knn", "graph", "initialization", "total"),
      elapsed_sec = c(
        knn_elapsed,
        graph_elapsed,
        initialization_elapsed,
        knn_elapsed + graph_elapsed + initialization_elapsed
      ),
      stringsAsFactors = FALSE
    )
  )
  class(out) <- c("fastEmbedR_umap_initialization", "list")
  out
}

#' @export
print.fastEmbedR_umap_initialization <- function(x, ...) {
  cat("<fastEmbedR_umap_initialization>\n")
  cat("  observations: ", nrow(x$layout), "\n", sep = "")
  cat("  components:   ", ncol(x$layout), "\n", sep = "")
  cat("  graph mode:   ", x$parameters$graph_mode, "\n", sep = "")
  cat("  init backend: ", x$parameters$init_backend, "\n", sep = "")
  total <- x$timings$elapsed_sec[x$timings$stage == "total"]
  if (length(total) && is.finite(total)) {
    cat("  elapsed:      ", format(round(total, 3L), nsmall = 3L), " s\n", sep = "")
  }
  invisible(x)
}
