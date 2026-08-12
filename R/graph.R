#' Build a compact nearest-neighbor graph
#'
#' `knn_graph()` converts a data matrix, a `fastEmbedR_embedding`, or a
#' precomputed KNN object into a native undirected edge graph. When neighbors
#' must be computed, the function uses fastEmbedR's existing native KNN route;
#' it does not expose a second nearest-neighbor tuning API.
#'
#' @param x Numeric data matrix, a `fastEmbedR_embedding`, a list containing
#'   `indices` and `distances`, or an existing `fastEmbedR_graph`.
#' @param k Number of non-self neighbors retained.
#' @param backend KNN-search backend used only when `x` is data: `"cpu"`,
#'   `"cuda"`, or `"metal"`.
#' @param metric Distance metric used only when KNN must be computed:
#'   `"euclidean"`, `"cosine"`, or `"correlation"`.
#' @param weight Edge weighting: shared-neighbor Jaccard (`"snn"`),
#'   inverse distance (`"distance"`), or unit weights (`"binary"`).
#' @param mutual Keep only reciprocal KNN edges.
#' @param prune Remove edges whose final weight is less than or equal to this
#'   non-negative threshold.
#' @param n.cores CPU cores used by KNN search and graph construction.
#' @return A compact `fastEmbedR_graph` list with `from`, `to`, `weight`,
#'   `n_vertices`, and construction metadata.
#' @references
#' Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
#' communities in large networks. J Stat Mech. 2008;2008:P10008.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' graph <- knn_graph(x, k = 15, weight = "snn")
#' graph
#' @export
knn_graph <- function(x,
                      k = 30L,
                      backend = NULL,
                      metric = c("euclidean", "cosine", "correlation"),
                      weight = c("snn", "distance", "binary"),
                      mutual = FALSE,
                      prune = 0,
                      n.cores = 1L) {
  n_threads <- n.cores
  if (inherits(x, "fastEmbedR_graph")) return(validate_fastembedr_graph(x))
  backend <- resolve_embedding_backend(backend)
  metric <- match.arg(metric)
  weight <- match.arg(weight)
  k <- graph_positive_integer(k, "k")
  n_threads <- graph_positive_integer(n_threads, "n.cores")
  if (length(mutual) != 1L || is.na(mutual)) {
    stop("`mutual` must be TRUE or FALSE.", call. = FALSE)
  }
  mutual <- isTRUE(mutual)
  prune <- as.numeric(prune)
  if (length(prune) != 1L || !is.finite(prune) || prune < 0) {
    stop("`prune` must be one finite non-negative number.", call. = FALSE)
  }

  started <- proc.time()[[3L]]
  source <- "precomputed_knn"
  knn_backend <- attr(x, "backend") %||% NA_character_
  if (inherits(x, "fastEmbedR_embedding")) {
    x <- x$layout
    source <- "embedding"
  }

  if (is_knn_input(x)) {
    knn <- coerce_knn_input(x)
  } else {
    if (!(is.matrix(x) || is.data.frame(x) || is_float32_matrix(x))) {
      stop(
        "`x` must be data, a fastEmbedR embedding, or a precomputed KNN object.",
        call. = FALSE
      )
    }
    n <- nrow(x)
    if (is.null(n) || n < 2L) stop("`x` must contain at least two rows.", call. = FALSE)
    k <- min(k, n - 1L)
    policy <- fastembedr_embedding_nn_policy(backend, n)
    found <- fastembedr_nn_without_self(
      x,
      k = k,
      backend = policy$backend,
      method = policy$method,
      metric = metric,
      output = "double",
      n_threads = n_threads,
      tuning = policy$tuning,
      target_recall = policy$target_recall,
      keep_gpu = FALSE
    )
    knn_backend <- attr(found, "backend") %||% policy$backend
    source <- if (identical(source, "embedding")) "embedding" else "data"
    knn <- coerce_knn_input(found)
  }

  k <- min(k, knn$n_neighbors)
  if (k < 1L) stop("The KNN input has no usable non-self neighbors.", call. = FALSE)
  selected <- materialize_knn_range(
    knn$indices,
    knn$distances,
    col_start = knn$col_start,
    n_neighbors = k
  )
  indices <- as.matrix(selected$indices)
  storage.mode(indices) <- "integer"
  distances <- embedding_dense_double_matrix(selected$distances)
  graph <- fastembedr_graph_from_knn_cpp(
    indices,
    distances,
    weight_type = weight,
    mutual = mutual,
    prune = prune,
    n_threads = n_threads
  )
  graph$parameters <- list(
    k = k,
    metric = metric,
    weight = weight,
    mutual = mutual,
    prune = prune,
    source = source,
    requested_knn_backend = if (identical(source, "precomputed_knn")) NA_character_ else backend,
    knn_backend = graph_backend_name(knn$input_backend, knn_backend),
    n.cores = n_threads,
    elapsed_sec = unname(proc.time()[[3L]] - started)
  )
  class(graph) <- c("fastEmbedR_graph", "list")
  graph
}

#' Native graph community detection
#'
#' `graph_cluster()` clusters a compact `fastEmbedR_graph` with a native
#' multilevel Louvain implementation, a Leiden local-move/refine/aggregate
#' implementation, or the Pons-Latapy Walktrap algorithm. Louvain and Leiden
#' have CPU, CUDA, and Metal implementations. Walktrap is CPU-only. No Python,
#' `igraph`, cuGraph, or external clustering function is called at run time.
#'
#' @param graph A `fastEmbedR_graph`, or an edge-list list containing `from`,
#'   `to`, `weight`, and `n_vertices`.
#' @param method `"leiden"`, `"louvain"`, or `"walktrap"`. Walktrap is the
#'   package's true random-walk community method.
#' @param backend Execution backend. `"cuda"` and `"metal"` are available for
#'   Louvain and Leiden only and fail explicitly if the requested accelerator
#'   was not compiled or is unavailable. They never fall back to CPU.
#' @param resolution Positive modularity resolution for Louvain and Leiden.
#'   Walktrap uses its reference resolution of 1.
#' @param n_iterations Maximum local-moving passes for Louvain and Leiden.
#' @param n_runs Independent seeded Louvain/Leiden runs; the highest-modularity
#'   result is returned.
#' @param steps Random-walk length for Walktrap.
#' @param seed Reproducible random seed for Louvain and Leiden.
#' @return A `fastEmbedR_graph_cluster` containing one-based `membership`,
#'   modularity, method, implementation, and run metadata.
#' @references
#' Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
#' communities in large networks. J Stat Mech. 2008;2008:P10008.
#'
#' Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
#' well-connected communities. Sci Rep. 2019;9:5233.
#'
#' Pons P, Latapy M. Computing communities in large networks using random
#' walks. J Graph Algorithms Appl. 2006;10:191-218.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' graph <- knn_graph(x, k = 15)
#' clusters <- graph_cluster(graph, method = "leiden", seed = 1)
#' table(clusters$membership)
#' @export
graph_cluster <- function(graph,
                          method = c("leiden", "louvain", "walktrap"),
                          backend = NULL,
                          resolution = 1,
                          n_iterations = 10L,
                          n_runs = 1L,
                          steps = 4L,
                          seed = 1L) {
  method <- match.arg(method)
  backend <- resolve_embedding_backend(backend)
  graph <- validate_fastembedr_graph(graph)
  resolution <- as.numeric(resolution)
  if (length(resolution) != 1L || !is.finite(resolution) || resolution <= 0) {
    stop("`resolution` must be one finite positive number.", call. = FALSE)
  }
  n_iterations <- graph_positive_integer(n_iterations, "n_iterations")
  n_runs <- graph_positive_integer(n_runs, "n_runs")
  steps <- graph_positive_integer(steps, "steps")
  seed <- as.integer(seed)
  if (length(seed) != 1L || is.na(seed)) stop("`seed` must be one integer.", call. = FALSE)
  if (identical(method, "walktrap") && !isTRUE(all.equal(resolution, 1))) {
    stop("Walktrap uses its reference modularity resolution of 1.", call. = FALSE)
  }
  if (identical(method, "walktrap") && !identical(backend, "cpu")) {
    stop(
      "Walktrap is supported only with `backend = \"cpu\"`; fastEmbedR does not silently replace a requested GPU backend.",
      call. = FALSE
    )
  }

  started <- proc.time()[[3L]]
  result <- if (identical(method, "walktrap")) {
    fastembedr_walktrap_cpp(
      graph$from, graph$to, graph$weight,
      n_vertices = graph$n_vertices,
      steps = steps
    )
  } else if (identical(backend, "cpu")) {
    fastembedr_graph_cluster_cpp(
      graph$from, graph$to, graph$weight,
      n_vertices = graph$n_vertices,
      method = method,
      resolution = resolution,
      n_iterations = n_iterations,
      n_runs = n_runs,
      seed = seed
    )
  } else if (identical(backend, "cuda")) {
    fastembedr_graph_cluster_cuda_cpp(
      graph$from, graph$to, graph$weight,
      n_vertices = graph$n_vertices,
      method = method,
      resolution = resolution,
      n_iterations = n_iterations,
      n_runs = n_runs,
      seed = seed
    )
  } else {
    fastembedr_graph_cluster_metal_cpp(
      graph$from, graph$to, graph$weight,
      n_vertices = graph$n_vertices,
      method = method,
      resolution = resolution,
      n_iterations = n_iterations,
      n_runs = n_runs,
      seed = seed
    )
  }
  result$method <- method
  result$backend_requested <- backend
  result$backend <- backend
  result$parameters <- list(
    resolution = if (identical(method, "walktrap")) 1 else resolution,
    n_iterations = if (identical(method, "walktrap")) NA_integer_ else n_iterations,
    n_runs = if (identical(method, "walktrap")) 1L else n_runs,
    steps = if (identical(method, "walktrap")) steps else NA_integer_,
    seed = if (identical(method, "walktrap")) NA_integer_ else seed,
    n_vertices = graph$n_vertices,
    n_edges = graph$n_edges,
    elapsed_sec = unname(proc.time()[[3L]] - started)
  )
  class(result) <- c("fastEmbedR_graph_cluster", "list")
  result
}

validate_fastembedr_graph <- function(graph) {
  required <- c("from", "to", "weight", "n_vertices")
  if (!is.list(graph) || !all(required %in% names(graph))) {
    stop(
      "`graph` must be a fastEmbedR graph or an edge list with from, to, weight, and n_vertices.",
      call. = FALSE
    )
  }
  graph$from <- as.integer(graph$from)
  graph$to <- as.integer(graph$to)
  graph$weight <- as.numeric(graph$weight)
  graph$n_vertices <- graph_positive_integer(graph$n_vertices, "graph$n_vertices")
  if (length(graph$from) != length(graph$to) ||
      length(graph$from) != length(graph$weight)) {
    stop("Graph edge vectors must have equal lengths.", call. = FALSE)
  }
  if (anyNA(graph$from) || anyNA(graph$to) || anyNA(graph$weight) ||
      any(!is.finite(graph$weight)) || any(graph$weight < 0)) {
    stop("Graph edges must have valid vertices and finite non-negative weights.", call. = FALSE)
  }
  if (length(graph$from) > 0L &&
      (min(graph$from) < 1L || min(graph$to) < 1L ||
       max(graph$from) > graph$n_vertices || max(graph$to) > graph$n_vertices)) {
    stop("Graph vertices must be between 1 and n_vertices.", call. = FALSE)
  }
  graph$n_edges <- length(graph$from)
  class(graph) <- c("fastEmbedR_graph", "list")
  graph
}

graph_positive_integer <- function(x, name) {
  numeric_value <- suppressWarnings(as.numeric(x))
  value <- suppressWarnings(as.integer(numeric_value))
  if (length(numeric_value) != 1L || !is.finite(numeric_value) ||
      numeric_value != value || value < 1L) {
    stop("`", name, "` must be one positive integer.", call. = FALSE)
  }
  value
}

graph_backend_name <- function(primary, fallback) {
  valid <- function(value) {
    length(value) == 1L && !is.na(value) && nzchar(value)
  }
  if (valid(primary)) as.character(primary) else as.character(fallback)
}

#' @export
print.fastEmbedR_graph <- function(x, ...) {
  cat("fastEmbedR KNN graph\n")
  cat("  vertices: ", x$n_vertices, "\n", sep = "")
  cat("  edges: ", x$n_edges, "\n", sep = "")
  if (!is.null(x$weight_type)) cat("  weights: ", x$weight_type, "\n", sep = "")
  invisible(x)
}

#' @export
print.fastEmbedR_graph_cluster <- function(x, ...) {
  cat("fastEmbedR graph clustering\n")
  cat("  method: ", x$method, "\n", sep = "")
  cat("  backend: ", x$backend, "\n", sep = "")
  cat("  communities: ", x$n_communities, "\n", sep = "")
  cat("  modularity: ", format(x$modularity, digits = 5), "\n", sep = "")
  invisible(x)
}
