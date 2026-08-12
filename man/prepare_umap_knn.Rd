\name{prepare_umap_knn}
\alias{prepare_umap_knn}
\title{Precompute reusable UMAP graph state from KNN}
\usage{
prepare_umap_knn(
  indices,
  distances = NULL,
  backend = NULL,
  n.cores = NULL,
  graph_mode = c("fuzzy", "binary")
)
}
\arguments{
\item{indices}{A KNN object, or an integer KNN index matrix.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave \code{NULL} when \code{indices} is a KNN object.}
\item{backend}{Execution backend used to resolve size-aware defaults: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}.}
\item{n.cores}{Number of CPU cores used during graph construction.}
\item{graph_mode}{Graph weighting mode. \code{"fuzzy"} (the default) uses
standard UMAP fuzzy graph weights. \code{"binary"} uses a symmetric
unit-weight sensitivity graph.}
}
\value{
A prepared UMAP object containing normalized KNN data, a CSR graph, graph weights, and \code{epochs_per_sample}. Pass it to \code{umap_knn()} for repeated runs without rebuilding the graph.
}
\description{
\code{prepare_umap_knn()} builds the reusable UMAP graph once from a KNN object or index/distance matrices. This is useful for repeated seeds, benchmark grids, and CPU/GPU comparisons where the neighbor graph should remain fixed.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
d <- as.matrix(stats::dist(x))
diag(d) <- Inf
k <- 10L
idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
  nrow = nrow(d), byrow = TRUE)
prep <- prepare_umap_knn(idx, dst)
y <- umap_knn(prep, seed = 1)
plot(y, pch = 21, bg = iris$Species)
}
