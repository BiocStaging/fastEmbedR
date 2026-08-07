#ifndef FASTEMBEDR_GRAPH_CLUSTERING_CUDA_API_H
#define FASTEMBEDR_GRAPH_CLUSTERING_CUDA_API_H

#include "graph_clustering_accel_common.h"

bool graph_clustering_cuda_available_impl();

fastembedr_graph_accel::MoveResult graph_local_move_cuda_impl(
  const fastembedr_graph_accel::Graph& graph,
  const std::vector<int>& initial,
  const std::vector<int>* parent,
  int max_passes,
  float resolution,
  std::uint64_t seed,
  bool refinement
);

#endif
