\name{umap_knn}
\alias{umap_knn}
\title{Run UMAP from precomputed KNN}
\usage{
umap_knn(
    indices,
    distances = NULL,
    n_components = 2L,
    seed = 42L,
    verbose = FALSE,
    backend = NULL,
    n.cores = NULL,
    graph_mode = c("fuzzy", "binary")
)
}
\arguments{
\item{indices}{A KNN object, or an integer KNN
index matrix. A self-neighbor first column is removed automatically.}

\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave
\code{NULL} when \code{indices} is an \code{nn()} result.}

\item{n_components}{Output dimensionality. The CPU backend supports positive
dimensions, including three-dimensional embeddings. The current Metal and
CUDA optimizers support only \code{2L}.}

\item{seed}{Random seed.}

\item{verbose}{Print native optimizer progress.}

\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or
\code{"metal"}. GPU requests must resolve to a real native backend; the
package does not label CPU work as GPU.}

\item{n.cores}{Number of CPU cores for CPU UMAP.}

\item{graph_mode}{Graph weighting mode. \code{"fuzzy"} (the default) uses
standard UMAP fuzzy graph weights. \code{"binary"} uses a symmetric
unit-weight sensitivity graph.}

}
\value{
A numeric embedding matrix with resolved settings in
\code{attr(layout, "fastEmbedR_config")}.
}
\description{
\code{umap_knn()} embeds observations from a supplied nearest-neighbor graph
using the stable native UMAP path. It is the KNN-input UMAP entry
point used by \code{\link{embed_knn}} with \code{method = "umap"}.
}
\details{
This function reuses the supplied neighbors but follows the same deliberately
constrained optimizer policy as \code{\link{umap}}. Epochs, minimum distance,
spread, learning rate, repulsion strength, negative-sample rate, spectral
iterations, and backend update mode are selected internally. The resolved
configuration is stored in \code{attr(layout, "fastEmbedR_config")}.

Pass an object from \code{\link{umap_init}} to reuse both the prepared graph
and package-native initialization. This API is intended for fixed-boundary
comparisons and repeated seeds, not arbitrary UMAP hyperparameter sweeps.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
d <- as.matrix(stats::dist(x))
diag(d) <- Inf
k <- 10L
idx <- t(apply(d, 1L, order))[, seq_len(k), drop = FALSE]
dst <- matrix(d[cbind(rep(seq_len(nrow(d)), each = k), as.vector(t(idx)))],
    nrow = nrow(d), byrow = TRUE)
layout <- umap_knn(idx, dst, seed = 1)
plot(layout, pch = 21, bg = iris$Species)
}
