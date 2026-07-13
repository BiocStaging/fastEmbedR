\name{pca}
\alias{pca}
\title{Backend-native truncated PCA}
\usage{
pca(x, ncomp = 2L, xtest = NULL, center = TRUE, scale = FALSE,
  backend = c("cpu", "cuda", "metal"), seed = 4L,
  opentsne_init = FALSE)
}
\arguments{
\item{x}{Numeric matrix/data frame or \code{float::float32} matrix with
observations in rows.}

\item{ncomp}{Number of principal components.}

\item{xtest}{Optional test/query matrix with the same columns as \code{x}.
Projected coordinates are returned in \code{scores_test}.}

\item{center}{If \code{TRUE}, mean-center columns before decomposition.}

\item{scale}{If \code{TRUE}, scale centered columns to unit sample standard
deviation before decomposition.}

\item{backend}{PCA backend: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}.}

\item{seed}{Random seed for the Gaussian subspace sketch.}

\item{opentsne_init}{If \code{TRUE}, add \code{opentsne_init} to the
returned PCA object. This matrix is centered and rescaled so its largest
component standard deviation is \code{1e-4}, ready to pass as \code{Y_init}
to \code{opentsne()} or \code{opentsne_knn()}.}
}
\value{
A \code{fastEmbedR_pca} list with \code{scores}, \code{loadings},
\code{singular_values}, centering/scaling vectors, backend metadata, and
decomposition metadata. When \code{opentsne_init = TRUE}, the list also
contains \code{opentsne_init}; when \code{xtest} is supplied, it also contains
\code{scores_test}. Metal preserves \code{float::float32} scores, loadings,
initialization, and compatible test projections when the input is float32.
}
\description{
\code{pca()} computes principal component scores with a backend-native truncated
decomposition. CPU uses the fastPLS-style randomized SVD family. Metal uses a
package-native float32 block-subspace TSVD whose large matrix products are
executed with Metal Performance Shaders while data and work buffers remain
resident in unified GPU memory. With \code{float::float32} input, native Metal
preprocessing and the returned scores/loadings also remain float32. The API
intentionally has no decomposition method menu and does not call Python.
Explicit unavailable GPU requests fail instead of being reported as GPU work.
}
\examples{
fit <- pca(
  as.matrix(iris[, 1:4]),
  ncomp = 2,
  seed = 1,
  opentsne_init = TRUE
)
plot(fit$scores, pch = 21, bg = iris$Species)
plot(fit$opentsne_init, pch = 21, bg = iris$Species)
}
