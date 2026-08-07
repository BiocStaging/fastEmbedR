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
  n.cores = NULL,
  keep_knn = FALSE,
  graph_mode = c("fuzzy", "binary"),
  verbose = FALSE
)
}
\arguments{
\item{data}{Numeric matrix/data frame, or a KNN object with \code{indices} and \code{distances} fields.}

\item{n_neighbors}{Number of non-self neighbors. \code{NULL} chooses the
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

\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}. CPU KNN uses package-native HNSW. Metal uses package-native exact or recall-tuned IVF-Flat search. CUDA uses direct FAISS GPU exact search below 100,000 rows and direct RAPIDS cuVS IVF-Flat above that threshold; the KNN result stays on the device through graph construction and optimization. GPU requests must resolve to a real native backend; the package does not relabel CPU work as GPU.}

\item{n.cores}{Number of CPU cores for KNN and CPU UMAP.}

\item{keep_knn}{If \code{TRUE}, retain KNN matrices in the returned object.}

\item{graph_mode}{Graph weighting mode. \code{"fuzzy"} (the default) uses the
standard UMAP fuzzy graph. \code{"binary"} uses a symmetric unit-weight graph
as an adjacency-only sensitivity approximation; it is not standard UMAP or
necessarily faster.}

\item{verbose}{Print progress.}
}
\value{
A \code{fastEmbedR_embedding} object.
}
\description{
\code{umap()} computes or reuses a nearest-neighbor graph and then calls
\code{\link{umap_knn}}. Quality metrics are computed explicitly with
\code{\link{evaluate_embedding}} rather than inside the embedding call.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- umap(x, n_neighbors = 15, seed = 1)
plot(fit)
}
