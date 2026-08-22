\name{pca}
\alias{pca}
\title{Backend-native truncated PCA}
\usage{
pca(x, ncomp = 2L, xtest = NULL, center = TRUE, scale = FALSE,
  backend = NULL, n.cores = 1L, seed = 4L,
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

\item{n.cores}{Positive integer CPU core limit. It controls the linked
BLAS/OpenMP numerical kernels when \code{backend = "cpu"} and is ignored by
Metal and CUDA.}

\item{seed}{Random seed for backends that use a Gaussian subspace sketch.
The native RAFT covariance-eigensolver route records this value for
provenance but does not consume random numbers.}

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
\code{scores_test}. Metal and CUDA preserve \code{float::float32} scores,
loadings, initialization, and compatible test projections when the input is
float32. CPU preserves float32 scores and loadings for float32 input and records
\code{n.cores_requested},
\code{n.cores_effective}, and \code{core_control}.
}
\description{
\code{pca()} computes principal component scores with a backend-native truncated
decomposition. CPU uses a package-native float32 blocked randomized SVD
(RSVD). Metal uses a package-native float32 block-subspace TSVD whose large
matrix products are executed with Metal Performance Shaders while data and
work buffers remain resident in unified GPU memory. CUDA uses native RAPIDS
RAFT TSVD with float32 input, score, and loading buffers. CUDA requests fail
explicitly when RAFT TSVD support is unavailable; they never fall back to CPU
PCA.

CPU centering/scaling and large matrix products run in C++; Apple builds use
Accelerate SGEMM and other platforms use package-threaded float32 kernels.
Set \code{n.cores} to control the CPU core limit. The requested value is
applied temporarily through standard BLAS/OpenMP environment variables and,
when installed, \code{RhpcBLASctl}; the prior process settings are restored
after the call. With \code{float::float32} input, native CPU, Metal, and CUDA
preprocessing and returned scores/loadings remain float32. The API
intentionally has no decomposition method menu and does not call Python.
Explicit unavailable GPU requests fail instead of being reported as GPU work.
}
\examples{
fit <- pca(
  as.matrix(iris[, 1:4]),
  ncomp = 2,
  n.cores = 2,
  seed = 1,
  opentsne_init = TRUE
)
plot(fit$scores, pch = 21, bg = iris$Species)
plot(fit$opentsne_init, pch = 21, bg = iris$Species)
}
