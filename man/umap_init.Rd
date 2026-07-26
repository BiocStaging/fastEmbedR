\name{umap_init}
\alias{umap_init}
\title{Precompute a Reusable UMAP Initialization}
\usage{
umap_init(
  x,
  distances = NULL,
  n_neighbors = NULL,
  n_components = 2L,
  metric = c("euclidean", "cosine", "correlation", "inner_product"),
  backend = c("cpu", "cuda", "metal"),
  seed = 4L,
  n_threads = NULL,
  graph_mode = c("binary", "fuzzy")
)
}
\arguments{
\item{x}{Numeric data, a KNN object, or an object returned by
\code{prepare_umap_knn()}.}

\item{distances}{Optional KNN distance matrix when \code{x} is an index
matrix.}

\item{n_neighbors}{Number of non-self neighbours when KNN must be computed.}

\item{n_components}{Output dimensionality.}

\item{metric}{Distance metric used only when KNN must be computed.}

\item{backend}{Initialization backend: \code{"cpu"}, \code{"cuda"}, or
\code{"metal"}.}

\item{seed}{Random seed.}

\item{n_threads}{Number of CPU worker threads.}

\item{graph_mode}{UMAP graph weighting: \code{"binary"} or \code{"fuzzy"}.}
}
\value{
A \code{fastEmbedR_umap_initialization} object containing the initial layout,
prepared KNN/CSR graph, resolved parameters, and component timings. The object
can be passed directly to \code{umap_knn()}.
}
\description{
Separates UMAP neighbour search, graph construction, and initialization from
stochastic optimization. This permits repeated CPU, Metal, or CUDA optimizer
runs from the same graph and coordinates without recomputing either stage.
}
\details{
For a fuzzy graph, native Metal or CUDA spectral kernels are used when the
corresponding backend is requested and available. Binary initialization uses
the same unit-weight CSR graph consumed by the optimizer. Explicit unavailable
GPU backends fail rather than silently falling back to CPU.

CUDA KNN input is copied to the host only for this standalone diagnostic API.
The ordinary one-call CUDA UMAP workflow retains its device-resident path.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
initialization <- umap_init(
  x,
  n_neighbors = 15,
  backend = "cpu",
  graph_mode = "fuzzy",
  seed = 1
)
layout <- umap_knn(initialization, backend = "cpu", seed = 1)
plot(layout, pch = 21, bg = iris$Species)
}
