#include "graph_clustering_cuda_api.h"

bool graph_clustering_cuda_available_impl() {
  return false;
}

fastembedr_graph_accel::MoveResult graph_local_move_cuda_impl(
    const fastembedr_graph_accel::Graph&,
    const std::vector<int>&,
    const std::vector<int>*,
    int,
    float,
    std::uint64_t,
    bool) {
  Rcpp::stop(
    "CUDA graph clustering is available only when fastEmbedR is built with "
    "FASTEMBEDR_USE_CUDA=1."
  );
}
