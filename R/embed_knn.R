#' Embed from precomputed KNN
#'
#' @param indices A list containing KNN `indices` and `distances`, or an integer
#'   KNN index matrix. If a self-neighbor first column is present it is removed
#'   automatically.
#' @param distances Numeric KNN distance matrix matching `indices`. Leave as
#'   `NULL` when `indices` is a KNN list.
#' @param method Embedding method: `"opentsne"` or `"umap"`.
#' @param n_components Output dimensionality.
#' @param seed Random seed.
#' @param verbose Print native optimizer progress.
#' @param backend Execution backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores used by the CPU optimizer.
#'   Native GPU optimizers ignore this argument.
#' @param ... Additional openTSNE-specific parameters such as `perplexity`,
#'   iteration controls, `Y_init`, learning-rate controls, and exaggeration
#'   controls.
#' @return A numeric embedding matrix with resolved settings stored in
#'   `attr(layout, "fastEmbedR_config")`.
#' @examples
#' x <- scale(as.matrix(iris[1:30, 1:4]))
#' d <- as.matrix(stats::dist(x))
#' diag(d) <- Inf
#' k <- 5L
#' indices <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
#' distances <- matrix(
#'   d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(indices)))],
#'   nrow = nrow(d), byrow = TRUE
#' )
#' layout <- embed_knn(indices, distances, method = "umap", seed = 1)
#' @export
embed_knn <- function(indices,
                      distances = NULL,
                      method = "umap",
                      n_components = 2L,
                      seed = 4L,
                      verbose = FALSE,
                      backend = NULL,
                      n.cores = NULL,
                      ...) {
  n_threads <- n.cores
  method <- match.arg(method, c("umap", "opentsne"))
  backend <- resolve_embedding_backend(backend)
  n_components <- validate_n_components(n_components)

  if (identical(method, "umap")) {
    return(fast_knn_umap(
      indices,
      distances,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend,
      n_threads = n_threads,
      ...
    ))
  }

  fast_knn_opentsne_core(
    indices,
    distances,
    n_components = n_components,
    seed = seed,
    verbose = verbose,
    backend = backend,
    n_threads = n_threads,
    ...
  )
}
