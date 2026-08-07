#include <Rcpp.h>

#include "graph_clustering_accel_common.h"
#include "graph_clustering_cuda_api.h"

// [[Rcpp::export]]
bool graph_clustering_cuda_available_cpp() {
  return graph_clustering_cuda_available_impl();
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_cluster_cuda_cpp(
    Rcpp::IntegerVector from,
    Rcpp::IntegerVector to,
    Rcpp::NumericVector weight,
    int n_vertices,
    std::string method = "leiden",
    double resolution = 1.0,
    int n_iterations = 10,
    int n_runs = 1,
    double seed = 1) {
  if (!graph_clustering_cuda_available_impl()) {
    Rcpp::stop(
      "CUDA graph clustering is unavailable. Reinstall fastEmbedR with "
      "FASTEMBEDR_USE_CUDA=1 on a machine with a CUDA device."
    );
  }
  return fastembedr_graph_accel::cluster(
    from, to, weight, n_vertices, method, resolution, n_iterations, n_runs,
    seed, "cuda", graph_local_move_cuda_impl
  );
}
