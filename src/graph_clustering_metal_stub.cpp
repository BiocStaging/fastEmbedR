#include "graph_clustering_metal_api.h"

bool graph_clustering_metal_available_impl() {
  return false;
}

std::string graph_clustering_metal_error_impl() {
  return "this fastEmbedR build does not contain Metal clustering support";
}

fastembedr_graph_accel::MoveResult graph_local_move_metal_impl(
    const fastembedr_graph_accel::Graph&,
    const std::vector<int>&,
    const std::vector<int>*,
    int,
    float,
    std::uint64_t,
    bool) {
  Rcpp::stop(
    "Metal graph clustering is available only in a Metal-enabled fastEmbedR "
    "build on macOS."
  );
}
