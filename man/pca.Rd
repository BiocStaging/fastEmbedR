\name{pca}
\alias{pca}
\title{Randomized SVD PCA}
\usage{
pca(data, ncomp = 2L, center = TRUE, scale = FALSE,
  backend = c("cpu", "cuda", "metal"), seed = 4L)
}
\arguments{
\item{data}{Numeric matrix/data frame with observations in rows.}

\item{ncomp}{Number of principal components.}

\item{center}{If \code{TRUE}, mean-center columns before RSVD.}

\item{scale}{If \code{TRUE}, scale centered columns to unit sample standard
deviation before RSVD.}

\item{backend}{PCA backend: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}.}

\item{seed}{Random seed for the Gaussian RSVD sketch.}
}
\value{
A \code{fastEmbedR_pca} list with \code{scores}, \code{loadings},
\code{singular_values}, centering/scaling vectors, backend metadata, and RSVD
tuning metadata.
}
\description{
\code{pca()} computes principal component scores using the fastPLS-style
randomized SVD algorithm used internally by fastEmbedR. The implementation is
intentionally restricted to RSVD: it does not use \code{irlba}, ARPACK, or a
method-selection menu. CUDA and Metal backends accelerate the RSVD
matrix-multiply steps when available.
}
\examples{
fit <- pca(as.matrix(iris[, 1:4]), ncomp = 2, seed = 1)
plot(fit$scores, pch = 21, bg = iris$Species)
}
