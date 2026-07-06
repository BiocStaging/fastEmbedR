\name{umap}
\alias{umap}
\title{Run UMAP from data or precomputed KNN}
\usage{
umap(
  data,
  n_neighbors = NULL,
  n_components = 2L,
  standardize = FALSE,
  pca_dims = NULL,
  metric = c("euclidean", "cosine", "correlation", "inner_product"),
  nn = NULL,
  seed = 4L,
  backend = c("cpu", "cuda", "metal"),
  n_threads = NULL,
  keep_knn = FALSE,
  graph_mode = c("binary", "fuzzy"),
  verbose = FALSE
)
}
\arguments{
\item{data}{Numeric matrix/data frame, or a KNN object with \code{indices} and \code{distances} fields.}

\item{n_neighbors}{Number of non-self neighbours. \code{NULL} chooses the
package default for the data size.}

\item{n_components}{Output dimensionality.}

\item{standardize}{Center and scale columns before KNN when \code{data} is a
matrix. Defaults to \code{FALSE} so one-call results match a KNN object computed
from the supplied matrix.}

\item{pca_dims}{Optional PCA dimension before KNN.}

\item{metric}{KNN distance metric for one-call matrix input: \code{"euclidean"},
\code{"cosine"}, \code{"correlation"}, or \code{"inner_product"}.}

\item{nn}{Optional precomputed KNN result when \code{data} is a matrix.}

\item{seed}{Random seed.}

\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}. KNN is delegated to faissR through an internal bridge. The current default asks for exact KNN below 100,000 samples and IVF above that threshold. CUDA requests CUDA faissR KNN, then materializes the KNN result for the validated host-prepared graph path before native CUDA UMAP optimization. This preserves the visually validated binary/fuzzy graph behaviour while the fully GPU-resident UMAP graph path remains available only for explicit \code{umap_knn()} experiments. GPU requests must resolve to a real native backend; the package does not relabel CPU work as GPU.}

\item{n_threads}{Number of CPU worker threads for KNN and CPU UMAP.}

\item{keep_knn}{If \code{TRUE}, retain KNN matrices in the returned object.}

\item{graph_mode}{Graph weighting mode. \code{"binary"} uses a symmetric
unit-weight graph. \code{"fuzzy"} uses standard UMAP fuzzy graph weights.}

\item{verbose}{Print progress.}
}
\value{
A \code{fastEmbedR_embedding} object.
}
\description{
\code{umap()} computes or reuses a nearest-neighbour graph and then calls
\code{\link{umap_knn}}. Quality metrics are computed explicitly with
\code{\link{evaluate_embedding}} rather than inside the embedding call.
}
\examples{
pkg <- "faissR"
if (requireNamespace(pkg, quietly = TRUE)) {
  x <- scale(as.matrix(iris[, 1:4]))
  fit <- umap(x, n_neighbors = 15, seed = 1)
  plot(fit)
}
}
