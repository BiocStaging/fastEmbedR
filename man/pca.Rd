\name{pca}
\alias{pca}
\title{Backend-native truncated PCA}
\usage{
pca(data, ncomp = 2L, center = TRUE, scale = FALSE,
  backend = c("cpu", "cuda", "metal"), seed = 4L)
}
\arguments{
\item{data}{Numeric matrix/data frame with observations in rows.}

\item{ncomp}{Number of principal components.}

\item{center}{If \code{TRUE}, mean-center columns before decomposition.}

\item{scale}{If \code{TRUE}, scale centered columns to unit sample standard
deviation before decomposition.}

\item{backend}{PCA backend: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}.}

\item{seed}{Random seed for the Gaussian subspace sketch.}
}
\value{
A \code{fastEmbedR_pca} list with \code{scores}, \code{loadings},
\code{singular_values}, centering/scaling vectors, backend metadata, and
decomposition metadata.
}
\description{
\code{pca()} computes principal component scores with a backend-native truncated
decomposition. CPU uses the fastPLS-style randomized SVD family. Metal uses a
package-native float32 block-subspace TSVD whose large matrix products are
executed with Metal Performance Shaders while data and work buffers remain
resident in unified GPU memory. The API intentionally has no decomposition
method menu and does not call Python. Explicit unavailable GPU requests fail
instead of being reported as GPU work.
}
\examples{
fit <- pca(as.matrix(iris[, 1:4]), ncomp = 2, seed = 1)
plot(fit$scores, pch = 21, bg = iris$Species)
}
