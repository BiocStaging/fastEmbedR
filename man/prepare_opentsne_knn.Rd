\name{prepare_opentsne_knn}
\alias{prepare_opentsne_knn}
\title{Precompute reusable openTSNE KNN state}
\usage{
prepare_opentsne_knn(
  indices,
  distances = NULL,
  n_neighbors = NULL,
  perplexity = NULL,
  affinity_support = c("standard", "compact")
)
}
\arguments{
\item{indices}{A KNN object, or an integer KNN index matrix.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave \code{NULL} when \code{indices} is a KNN object.}
\item{n_neighbors}{Optional number of non-self neighbor columns to retain.}
\item{perplexity}{Optional t-SNE perplexity used to choose the retained KNN width when \code{n_neighbors} is not supplied.}
\item{affinity_support}{Affinity support policy. \code{"standard"} retains
\code{ceiling(3 * perplexity)} neighbors; \code{"compact"} retains
\code{ceiling(perplexity)} and is an explicit approximation.}
}
\value{
A prepared openTSNE KNN object. Pass it to \code{opentsne_knn()} for repeated runs without repeating KNN normalization/materialization.
}
\description{
\code{prepare_opentsne_knn()} strips self-neighbors, trims to the selected
non-self affinity support, and stores the materialized KNN matrices once. The
native optimizer builds t-SNE affinities internally from this reusable KNN
state.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
d <- as.matrix(stats::dist(x))
diag(d) <- Inf
k <- 15L
idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
  nrow = nrow(d), byrow = TRUE)
prep <- prepare_opentsne_knn(idx, dst, perplexity = 5)
y <- opentsne_knn(prep, early_exaggeration_iter = 10, n_iter = 20)
plot(y, pch = 21, bg = iris$Species)
}
