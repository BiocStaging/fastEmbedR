\name{graph_cluster}
\alias{graph_cluster}
\alias{print.fastEmbedR_graph_cluster}
\title{Native graph community detection}
\usage{
graph_cluster(
  graph,
  method = c("leiden", "louvain", "walktrap"),
  backend = c("cpu", "cuda", "metal"),
  resolution = 1,
  n_iterations = 10L,
  n_runs = 1L,
  steps = 4L,
  seed = 1L
)
}
\arguments{
\item{graph}{A \code{fastEmbedR_graph} or compatible weighted edge list.}
\item{method}{Native Leiden, Louvain, or exact Walktrap.}
\item{backend}{Execution backend. CUDA and Metal are supported for Louvain
and Leiden and fail explicitly when unavailable. Walktrap is CPU-only.
Accelerator requests never fall back silently to CPU.}
\item{resolution}{Positive modularity resolution for Leiden and Louvain.}
\item{n_iterations}{Maximum local-moving and Leiden iterations.}
\item{n_runs}{Independent Louvain or Leiden runs.}
\item{steps}{Random-walk length for Walktrap.}
\item{seed}{Random seed for Louvain and Leiden.}
}
\value{
A \code{fastEmbedR_graph_cluster} list with one-based membership, modularity,
implementation metadata, and timing.
}
\description{
Runs package-native graph clustering without calling Python, igraph, cuGraph,
or an external clustering routine. CPU clustering uses native C++. CUDA and
Metal use float32 CSR graph storage, accelerator-resident parallel community
moves, and package-owned C++ graph compaction and coarsening. Walktrap is the
Pons-Latapy random-walk method and is memory-guarded because its exact
transition matrix is quadratic.
}
\references{
Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of
communities in large networks. J Stat Mech. 2008;2008:P10008.

Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing
well-connected communities. Sci Rep. 2019;9:5233.

Pons P, Latapy M. Computing communities in large networks using random walks.
J Graph Algorithms Appl. 2006;10:191-218.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
graph <- knn_graph(x, k = 15)
clusters <- graph_cluster(graph, method = "leiden")
table(clusters$membership)
}
