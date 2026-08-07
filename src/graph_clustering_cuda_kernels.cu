#include "graph_clustering_cuda_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kColorCount = 8;
constexpr int kLocalCandidateCount = 64;

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(status);
  Rcpp::stop(message.str());
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
        "cudaMalloc"
      );
    }
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

__device__ __forceinline__ unsigned int mix32(unsigned int value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

__device__ __forceinline__ int vertex_color(
    int vertex, unsigned int seed) {
  return static_cast<int>(mix32(
    static_cast<unsigned int>(vertex) ^ seed
  ) & (kColorCount - 1));
}

__global__ void propose_moves_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* edge_weight,
    const float* degree,
    const int* labels,
    const int* parent,
    const float* community_volume,
    const int* community_count,
    int* proposal,
    int n,
    float graph_volume,
    float resolution,
    float tolerance,
    int color,
    unsigned int seed,
    int refinement) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n || vertex_color(u, seed) != color) return;

  const int old = labels[u];
  proposal[u] = old;
  const float node_degree = degree[u];
  if (!(node_degree > 0.0f)) return;
  if (refinement && community_count[old] != 1) return;

  const int begin = row_ptr[u];
  const int end = row_ptr[u + 1];
  int candidate_labels[kLocalCandidateCount];
  float candidate_weights[kLocalCandidateCount];
  int unique_count = 0;
  bool overflow = false;
  float old_inside = 0.0f;
  for (int position = begin; position < end; ++position) {
    const int v = col_idx[position];
    if (v == u || (refinement && parent[v] != parent[u])) continue;
    const int candidate = labels[v];
    if (candidate == old) old_inside += edge_weight[position];
    int slot = -1;
    for (int index = 0; index < unique_count; ++index) {
      if (candidate_labels[index] == candidate) {
        slot = index;
        break;
      }
    }
    if (slot >= 0) {
      candidate_weights[slot] += edge_weight[position];
    } else if (unique_count < kLocalCandidateCount) {
      candidate_labels[unique_count] = candidate;
      candidate_weights[unique_count] = edge_weight[position];
      ++unique_count;
    } else {
      overflow = true;
    }
  }
  const float old_volume = fmaxf(0.0f, community_volume[old] - node_degree);
  const float old_score =
    old_inside - resolution * node_degree * old_volume / graph_volume;
  float best_score = old_score;
  int best = old;

  if (!overflow) {
    for (int index = 0; index < unique_count; ++index) {
      const int candidate = candidate_labels[index];
      if (candidate == old || community_count[candidate] <= 0) continue;
      const float score = candidate_weights[index] -
        resolution * node_degree * community_volume[candidate] / graph_volume;
      if (score > best_score + tolerance ||
          (fabsf(score - best_score) <= tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
      }
    }
  } else {
    for (int candidate_position = begin;
         candidate_position < end; ++candidate_position) {
      const int neighbor = col_idx[candidate_position];
      if (neighbor == u) continue;
      if (refinement && parent[neighbor] != parent[u]) continue;
      const int candidate = labels[neighbor];
      if (candidate == old || community_count[candidate] <= 0) continue;
      bool seen = false;
      for (int previous = begin; previous < candidate_position; ++previous) {
        const int previous_neighbor = col_idx[previous];
        if (previous_neighbor != u &&
            (!refinement || parent[previous_neighbor] == parent[u]) &&
            labels[previous_neighbor] == candidate) {
          seen = true;
          break;
        }
      }
      if (seen) continue;
      float inside = 0.0f;
      for (int position = begin; position < end; ++position) {
        const int v = col_idx[position];
        if (v != u &&
            (!refinement || parent[v] == parent[u]) &&
            labels[v] == candidate) {
          inside += edge_weight[position];
        }
      }
      const float score = inside - resolution * node_degree *
        community_volume[candidate] / graph_volume;
      if (score > best_score + tolerance ||
          (fabsf(score - best_score) <= tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
      }
    }
  }
  if (best != old && best_score > old_score + tolerance) proposal[u] = best;
}

__global__ void apply_moves_kernel(
    const float* degree,
    int* labels,
    const int* proposal,
    float* community_volume,
    int* community_count,
    unsigned int* move_count,
    int n,
    int color,
    unsigned int seed,
    int refinement) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n || vertex_color(u, seed) != color) return;
  const int old = labels[u];
  const int next = proposal[u];
  if (next == old) return;
  if (refinement && community_count[old] != 1) return;

  labels[u] = next;
  atomicAdd(&community_volume[old], -degree[u]);
  atomicAdd(&community_volume[next], degree[u]);
  atomicSub(&community_count[old], 1);
  atomicAdd(&community_count[next], 1);
  atomicAdd(move_count, 1U);
}

}  // namespace

bool graph_clustering_cuda_available_impl() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

fastembedr_graph_accel::MoveResult graph_local_move_cuda_impl(
    const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& initial,
    const std::vector<int>* parent,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    bool refinement) {
  using fastembedr_graph_accel::MoveResult;
  if (!graph_clustering_cuda_available_impl()) {
    Rcpp::stop("No CUDA device is available for graph clustering.");
  }
  if (initial.size() != static_cast<std::size_t>(graph.n)) {
    Rcpp::stop("Internal CUDA clustering membership size mismatch.");
  }
  if (refinement &&
      (parent == nullptr || parent->size() != initial.size())) {
    Rcpp::stop("Internal CUDA Leiden parent-partition size mismatch.");
  }

  MoveResult result;
  result.membership = initial;
  if (graph.n < 2 || !(graph.volume > 0.0f)) return result;

  std::vector<float> volume(static_cast<std::size_t>(graph.n), 0.0f);
  std::vector<int> count(static_cast<std::size_t>(graph.n), 0);
  for (int u = 0; u < graph.n; ++u) {
    const int community = initial[static_cast<std::size_t>(u)];
    volume[static_cast<std::size_t>(community)] +=
      graph.degree[static_cast<std::size_t>(u)];
    ++count[static_cast<std::size_t>(community)];
  }
  std::vector<int> parent_storage;
  if (parent == nullptr) {
    parent_storage.assign(static_cast<std::size_t>(graph.n), 0);
    parent = &parent_storage;
  }

  DeviceBuffer<int> d_row_ptr(graph.row_ptr.size());
  DeviceBuffer<int> d_col_idx(graph.col_idx.size());
  DeviceBuffer<float> d_weight(graph.weight.size());
  DeviceBuffer<float> d_degree(graph.degree.size());
  DeviceBuffer<int> d_labels(initial.size());
  DeviceBuffer<int> d_parent(parent->size());
  DeviceBuffer<float> d_volume(volume.size());
  DeviceBuffer<int> d_count(count.size());
  DeviceBuffer<int> d_proposal(initial.size());
  DeviceBuffer<unsigned int> d_moves(1);

  check_cuda(cudaMemcpy(
    d_row_ptr.data(), graph.row_ptr.data(),
    graph.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy CSR row offsets");
  check_cuda(cudaMemcpy(
    d_col_idx.data(), graph.col_idx.data(),
    graph.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy CSR column indices");
  check_cuda(cudaMemcpy(
    d_weight.data(), graph.weight.data(),
    graph.weight.size() * sizeof(float), cudaMemcpyHostToDevice
  ), "copy CSR weights");
  check_cuda(cudaMemcpy(
    d_degree.data(), graph.degree.data(),
    graph.degree.size() * sizeof(float), cudaMemcpyHostToDevice
  ), "copy graph degrees");
  check_cuda(cudaMemcpy(
    d_labels.data(), initial.data(),
    initial.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy graph labels");
  check_cuda(cudaMemcpy(
    d_parent.data(), parent->data(),
    parent->size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy Leiden parent labels");
  check_cuda(cudaMemcpy(
    d_volume.data(), volume.data(),
    volume.size() * sizeof(float), cudaMemcpyHostToDevice
  ), "copy community volumes");
  check_cuda(cudaMemcpy(
    d_count.data(), count.data(),
    count.size() * sizeof(int), cudaMemcpyHostToDevice
  ), "copy community counts");

  constexpr int threads = 256;
  const int blocks = (graph.n + threads - 1) / threads;
  const float tolerance = 1e-7f * std::max(
    1.0f,
    static_cast<float>(graph.total_edge_weight) /
      static_cast<float>(std::max(1, graph.n))
  );
  const unsigned int base_device_seed =
    static_cast<unsigned int>(seed ^ (seed >> 32U));

  for (int pass = 0; pass < max_passes; ++pass) {
    const unsigned int device_seed = base_device_seed ^
      (0x9e3779b9U * static_cast<unsigned int>(pass + 1));
    check_cuda(cudaMemset(d_moves.data(), 0, sizeof(unsigned int)),
               "clear CUDA graph move counter");
    for (int color = 0; color < kColorCount; ++color) {
      propose_moves_kernel<<<blocks, threads>>>(
        d_row_ptr.data(), d_col_idx.data(), d_weight.data(), d_degree.data(),
        d_labels.data(), d_parent.data(), d_volume.data(), d_count.data(),
        d_proposal.data(), graph.n, graph.volume, resolution, tolerance,
        color, device_seed, refinement ? 1 : 0
      );
      check_cuda(cudaGetLastError(), "launch CUDA graph proposal kernel");
      apply_moves_kernel<<<blocks, threads>>>(
        d_degree.data(), d_labels.data(), d_proposal.data(), d_volume.data(),
        d_count.data(), d_moves.data(), graph.n, color, device_seed,
        refinement ? 1 : 0
      );
      check_cuda(cudaGetLastError(), "launch CUDA graph apply kernel");
    }
    unsigned int moves = 0;
    check_cuda(cudaMemcpy(
      &moves, d_moves.data(), sizeof(unsigned int), cudaMemcpyDeviceToHost
    ), "read CUDA graph move counter");
    result.moves += static_cast<int>(moves);
    ++result.passes;
    if (moves == 0U) break;
  }

  check_cuda(cudaMemcpy(
    result.membership.data(), d_labels.data(),
    result.membership.size() * sizeof(int), cudaMemcpyDeviceToHost
  ), "copy CUDA graph membership");
  return result;
}
