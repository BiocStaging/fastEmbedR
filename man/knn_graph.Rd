\name{knn_graph}
\alias{knn_graph}
\alias{print.fastEmbedR_graph}
\title{Build a compact nearest-neighbour graph}
\usage{
knn_graph(
  x,
  k = 30L,
  backend = c("cpu", "cuda", "metal"),
  metric = c("euclidean", "cosine", "correlation"),
  weight = c("snn", "distance", "binary"),
  mutual = FALSE,
  prune = 0,
  n_threads = 1L
)
}
\arguments{
\item{x}{Numeric data, a \code{fastEmbedR_embedding}, a precomputed KNN list,
or an existing \code{fastEmbedR_graph}.}
\item{k}{Number of non-self neighbours retained.}
\item{backend}{KNN-search backend used only for data input.}
\item{metric}{Euclidean, cosine, or correlation distance.}
\item{weight}{SNN/Jaccard, inverse-distance, or binary edge weights.}
\item{mutual}{Whether to retain only reciprocal KNN edges.}
\item{prune}{Non-negative edge-weight pruning threshold.}
\item{n_threads}{CPU threads for search and graph construction.}
}
\value{A compact \code{fastEmbedR_graph} edge-list object.}
\description{
Builds one undirected weighted graph from data, an embedding, or precomputed
neighbours. The function reuses fastEmbedR's existing KNN route and does not
expose another nearest-neighbour tuning surface.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
graph <- knn_graph(x, k = 15)
graph
}
