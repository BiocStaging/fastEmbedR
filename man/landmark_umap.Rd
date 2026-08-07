\name{landmark_umap}
\alias{landmark_umap}
\title{Run Landmark UMAP}
\usage{
landmark_umap(
  data,
  landmarks = 0.5,
  n_neighbors = NULL,
  n_components = 2L,
  standardize = TRUE,
  pca_dims = NULL,
  seed = 4L,
  backend = c("cpu", "cuda", "metal"),
  transform_k = NULL,
  n.cores = NULL,
  keep_knn = FALSE,
  graph_mode = c("fuzzy", "binary"),
  verbose = FALSE
)
}
\description{
Embed a landmark subset with \code{\link{umap}} and project the remaining
observations against the fixed landmark embedding using query-to-reference
KNN, local affine placement, and optional fixed-reference refinement.
}
\arguments{
\item{data}{Numeric matrix or data frame with observations in rows.}
\item{landmarks}{\code{TRUE} for an automatic subset, a fraction such as
\code{0.5}, a landmark count, or explicit row indices.}
\item{n_neighbors}{Number of non-self neighbors.}
\item{n_components}{Output dimensionality.}
\item{standardize}{Center and scale columns before KNN.}
\item{pca_dims}{Optional PCA dimension before KNN.}
\item{seed}{Random seed.}
\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or
\code{"metal"}. CPU uses native HNSW; Metal uses native exact/IVF-Flat;
CUDA requests package-native RAPIDS cuVS KNN with \code{target_recall = 0.99}. GPU requests must use a
real native backend and are not silently relabelled as CPU.}
\item{transform_k}{Number of landmark neighbors used to project
non-landmark observations. Defaults to \code{n_neighbors}.}
\item{n.cores}{Number of CPU cores.}
\item{keep_knn}{Keep projection KNN matrices in the returned object.}
\item{graph_mode}{Graph weighting mode passed unchanged to the reference
\code{\link{umap}} fit. \code{"fuzzy"} (the default) uses standard UMAP fuzzy
graph weights; \code{"binary"} uses a symmetric unit-weight sensitivity
graph.}
\item{verbose}{Print progress.}
}
\value{
A \code{fastEmbedR_embedding} object with the full layout, landmark metadata,
timings, and parameters.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- landmark_umap(x, landmarks = 0.5, n_neighbors = 15, seed = 1)
plot(fit)
}
