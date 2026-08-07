#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "graph_clustering_metal_api.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kColorCount = 8;

struct MetalParams {
  std::uint32_t n;
  std::uint32_t color;
  std::uint32_t seed;
  std::uint32_t refinement;
  float graph_volume;
  float resolution;
  float tolerance;
  float padding;
};

struct MetalGraphState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> proposal_pipeline = nil;
  id<MTLComputePipelineState> apply_pipeline = nil;
  std::string error;
  bool initialized = false;
};

MetalGraphState& state() {
  static MetalGraphState value;
  return value;
}

std::mutex& state_mutex() {
  static std::mutex value;
  return value;
}

const char* metal_source() {
  return R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint color_count = 8;
constant int local_candidate_count = 64;

struct Params {
  uint n;
  uint color;
  uint seed;
  uint refinement;
  float graph_volume;
  float resolution;
  float tolerance;
  float padding;
};

inline uint mix32(uint value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

inline uint vertex_color(uint vertex_id, uint seed) {
  return mix32(vertex_id ^ seed) & (color_count - 1);
}

kernel void graph_propose(
    device const int* row_ptr [[buffer(0)]],
    device const int* col_idx [[buffer(1)]],
    device const float* edge_weight [[buffer(2)]],
    device const float* degree [[buffer(3)]],
    device const int* labels [[buffer(4)]],
    device const int* parent [[buffer(5)]],
    device const atomic_float* community_volume [[buffer(6)]],
    device const atomic_int* community_count [[buffer(7)]],
    device int* proposal [[buffer(8)]],
    constant Params& params [[buffer(9)]],
    uint u [[thread_position_in_grid]]) {
  if (u >= params.n || vertex_color(u, params.seed) != params.color) return;

  const int old = labels[u];
  proposal[u] = old;
  const float node_degree = degree[u];
  if (!(node_degree > 0.0f)) return;
  if (params.refinement &&
      atomic_load_explicit(&community_count[old], memory_order_relaxed) != 1) {
    return;
  }

  const int begin = row_ptr[u];
  const int end = row_ptr[u + 1];
  int candidate_labels[local_candidate_count];
  float candidate_weights[local_candidate_count];
  int unique_count = 0;
  bool overflow = false;
  float old_inside = 0.0f;
  for (int position = begin; position < end; ++position) {
    const int v = col_idx[position];
    if (v == int(u) ||
        (params.refinement && parent[v] != parent[u])) {
      continue;
    }
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
    } else if (unique_count < local_candidate_count) {
      candidate_labels[unique_count] = candidate;
      candidate_weights[unique_count] = edge_weight[position];
      ++unique_count;
    } else {
      overflow = true;
    }
  }
  const float old_total = atomic_load_explicit(
    &community_volume[old], memory_order_relaxed
  );
  const float old_volume = max(0.0f, old_total - node_degree);
  const float old_score = old_inside - params.resolution * node_degree *
    old_volume / params.graph_volume;
  float best_score = old_score;
  int best = old;

  if (!overflow) {
    for (int index = 0; index < unique_count; ++index) {
      const int candidate = candidate_labels[index];
      if (candidate == old ||
          atomic_load_explicit(
            &community_count[candidate], memory_order_relaxed
          ) <= 0) {
        continue;
      }
      const float candidate_volume = atomic_load_explicit(
        &community_volume[candidate], memory_order_relaxed
      );
      const float score = candidate_weights[index] -
        params.resolution * node_degree *
          candidate_volume / params.graph_volume;
      if (score > best_score + params.tolerance ||
          (fabs(score - best_score) <= params.tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
      }
    }
  } else {
    for (int candidate_position = begin;
         candidate_position < end; ++candidate_position) {
      const int neighbor = col_idx[candidate_position];
      if (neighbor == int(u)) continue;
      if (params.refinement && parent[neighbor] != parent[u]) continue;
      const int candidate = labels[neighbor];
      if (candidate == old ||
          atomic_load_explicit(
            &community_count[candidate], memory_order_relaxed
          ) <= 0) {
        continue;
      }
      bool seen = false;
      for (int previous = begin; previous < candidate_position; ++previous) {
        const int previous_neighbor = col_idx[previous];
        if (previous_neighbor != int(u) &&
            (!params.refinement || parent[previous_neighbor] == parent[u]) &&
            labels[previous_neighbor] == candidate) {
          seen = true;
          break;
        }
      }
      if (seen) continue;
      float inside = 0.0f;
      for (int position = begin; position < end; ++position) {
        const int v = col_idx[position];
        if (v != int(u) &&
            (!params.refinement || parent[v] == parent[u]) &&
            labels[v] == candidate) {
          inside += edge_weight[position];
        }
      }
      const float candidate_volume = atomic_load_explicit(
        &community_volume[candidate], memory_order_relaxed
      );
      const float score = inside - params.resolution * node_degree *
        candidate_volume / params.graph_volume;
      if (score > best_score + params.tolerance ||
          (fabs(score - best_score) <= params.tolerance && candidate < best)) {
        best = candidate;
        best_score = score;
      }
    }
  }
  if (best != old && best_score > old_score + params.tolerance) {
    proposal[u] = best;
  }
}

kernel void graph_apply(
    device const float* degree [[buffer(0)]],
    device int* labels [[buffer(1)]],
    device const int* proposal [[buffer(2)]],
    device atomic_float* community_volume [[buffer(3)]],
    device atomic_int* community_count [[buffer(4)]],
    device atomic_uint* move_count [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint u [[thread_position_in_grid]]) {
  if (u >= params.n || vertex_color(u, params.seed) != params.color) return;
  const int old = labels[u];
  const int next = proposal[u];
  if (next == old) return;
  if (params.refinement &&
      atomic_load_explicit(&community_count[old], memory_order_relaxed) != 1) {
    return;
  }
  labels[u] = next;
  atomic_fetch_add_explicit(
    &community_volume[old], -degree[u], memory_order_relaxed
  );
  atomic_fetch_add_explicit(
    &community_volume[next], degree[u], memory_order_relaxed
  );
  atomic_fetch_sub_explicit(&community_count[old], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&community_count[next], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(move_count, 1U, memory_order_relaxed);
}
)METAL";
}

void initialize_state() {
  std::lock_guard<std::mutex> lock(state_mutex());
  MetalGraphState& current = state();
  if (current.initialized) return;
  current.initialized = true;
  current.device = MTLCreateSystemDefaultDevice();
  if (current.device == nil) {
    current.error = "No Metal device is available.";
    return;
  }
  current.queue = [current.device newCommandQueue];
  if (current.queue == nil) {
    current.error = "Could not create a Metal command queue.";
    return;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  options.fastMathEnabled = YES;
  current.library = [current.device
    newLibraryWithSource:[NSString stringWithUTF8String:metal_source()]
    options:options
    error:&error];
  if (current.library == nil) {
    current.error = error == nil ? "Metal source compilation failed." :
      std::string([[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> proposal =
    [current.library newFunctionWithName:@"graph_propose"];
  id<MTLFunction> apply =
    [current.library newFunctionWithName:@"graph_apply"];
  current.proposal_pipeline = [current.device
    newComputePipelineStateWithFunction:proposal error:&error];
  if (current.proposal_pipeline == nil) {
    current.error = error == nil ? "Could not create Metal proposal pipeline." :
      std::string([[error localizedDescription] UTF8String]);
    return;
  }
  current.apply_pipeline = [current.device
    newComputePipelineStateWithFunction:apply error:&error];
  if (current.apply_pipeline == nil) {
    current.error = error == nil ? "Could not create Metal apply pipeline." :
      std::string([[error localizedDescription] UTF8String]);
  }
}

template <typename T>
id<MTLBuffer> shared_buffer(id<MTLDevice> device,
                            const std::vector<T>& values) {
  return [device
    newBufferWithBytes:values.data()
    length:values.size() * sizeof(T)
    options:MTLResourceStorageModeShared];
}

NSUInteger thread_count(id<MTLComputePipelineState> pipeline) {
  return std::max<NSUInteger>(
    1, std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup)
  );
}

void encode_pipeline(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> pipeline,
                     NSUInteger n) {
  [encoder setComputePipelineState:pipeline];
  const NSUInteger threads = thread_count(pipeline);
  [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

}  // namespace

bool graph_clustering_metal_available_impl() {
  @autoreleasepool {
    initialize_state();
    const MetalGraphState& current = state();
    return current.device != nil &&
      current.proposal_pipeline != nil &&
      current.apply_pipeline != nil;
  }
}

std::string graph_clustering_metal_error_impl() {
  @autoreleasepool {
    initialize_state();
    return state().error;
  }
}

fastembedr_graph_accel::MoveResult graph_local_move_metal_impl(
    const fastembedr_graph_accel::Graph& graph,
    const std::vector<int>& initial,
    const std::vector<int>* parent,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    bool refinement) {
  using fastembedr_graph_accel::MoveResult;
  @autoreleasepool {
    initialize_state();
    MetalGraphState& current = state();
    if (!graph_clustering_metal_available_impl()) {
      Rcpp::stop(
        current.error.empty() ? "Metal graph clustering is unavailable." :
        current.error
      );
    }
    if (initial.size() != static_cast<std::size_t>(graph.n)) {
      Rcpp::stop("Internal Metal clustering membership size mismatch.");
    }
    if (refinement &&
        (parent == nullptr || parent->size() != initial.size())) {
      Rcpp::stop("Internal Metal Leiden parent-partition size mismatch.");
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
    std::vector<int> proposal(initial.size(), 0);
    std::vector<std::uint32_t> moves(1, 0U);

    id<MTLBuffer> row_buffer = shared_buffer(current.device, graph.row_ptr);
    id<MTLBuffer> col_buffer = shared_buffer(current.device, graph.col_idx);
    id<MTLBuffer> weight_buffer = shared_buffer(current.device, graph.weight);
    id<MTLBuffer> degree_buffer = shared_buffer(current.device, graph.degree);
    id<MTLBuffer> label_buffer = shared_buffer(current.device, initial);
    id<MTLBuffer> parent_buffer = shared_buffer(current.device, *parent);
    id<MTLBuffer> volume_buffer = shared_buffer(current.device, volume);
    id<MTLBuffer> count_buffer = shared_buffer(current.device, count);
    id<MTLBuffer> proposal_buffer = shared_buffer(current.device, proposal);
    id<MTLBuffer> move_buffer = shared_buffer(current.device, moves);
    if (row_buffer == nil || col_buffer == nil || weight_buffer == nil ||
        degree_buffer == nil || label_buffer == nil || parent_buffer == nil ||
        volume_buffer == nil || count_buffer == nil ||
        proposal_buffer == nil || move_buffer == nil) {
      Rcpp::stop("Metal graph clustering buffer allocation failed.");
    }

    const float tolerance = 1e-7f * std::max(
      1.0f,
      static_cast<float>(graph.total_edge_weight) /
        static_cast<float>(std::max(1, graph.n))
    );
    MetalParams params{
      static_cast<std::uint32_t>(graph.n),
      0U,
      static_cast<std::uint32_t>(seed ^ (seed >> 32U)),
      refinement ? 1U : 0U,
      graph.volume,
      resolution,
      tolerance,
      0.0f
    };

    for (int pass = 0; pass < max_passes; ++pass) {
      params.seed = static_cast<std::uint32_t>(seed ^ (seed >> 32U)) ^
        (0x9e3779b9U * static_cast<std::uint32_t>(pass + 1));
      *static_cast<std::uint32_t*>(move_buffer.contents) = 0U;
      id<MTLCommandBuffer> command = [current.queue commandBuffer];
      if (command == nil) Rcpp::stop("Could not create Metal command buffer.");
      for (int color = 0; color < kColorCount; ++color) {
        params.color = static_cast<std::uint32_t>(color);
        id<MTLComputeCommandEncoder> proposal_encoder =
          [command computeCommandEncoder];
        [proposal_encoder setBuffer:row_buffer offset:0 atIndex:0];
        [proposal_encoder setBuffer:col_buffer offset:0 atIndex:1];
        [proposal_encoder setBuffer:weight_buffer offset:0 atIndex:2];
        [proposal_encoder setBuffer:degree_buffer offset:0 atIndex:3];
        [proposal_encoder setBuffer:label_buffer offset:0 atIndex:4];
        [proposal_encoder setBuffer:parent_buffer offset:0 atIndex:5];
        [proposal_encoder setBuffer:volume_buffer offset:0 atIndex:6];
        [proposal_encoder setBuffer:count_buffer offset:0 atIndex:7];
        [proposal_encoder setBuffer:proposal_buffer offset:0 atIndex:8];
        [proposal_encoder setBytes:&params length:sizeof(params) atIndex:9];
        encode_pipeline(
          proposal_encoder, current.proposal_pipeline,
          static_cast<NSUInteger>(graph.n)
        );
        [proposal_encoder endEncoding];

        id<MTLComputeCommandEncoder> apply_encoder =
          [command computeCommandEncoder];
        [apply_encoder setBuffer:degree_buffer offset:0 atIndex:0];
        [apply_encoder setBuffer:label_buffer offset:0 atIndex:1];
        [apply_encoder setBuffer:proposal_buffer offset:0 atIndex:2];
        [apply_encoder setBuffer:volume_buffer offset:0 atIndex:3];
        [apply_encoder setBuffer:count_buffer offset:0 atIndex:4];
        [apply_encoder setBuffer:move_buffer offset:0 atIndex:5];
        [apply_encoder setBytes:&params length:sizeof(params) atIndex:6];
        encode_pipeline(
          apply_encoder, current.apply_pipeline,
          static_cast<NSUInteger>(graph.n)
        );
        [apply_encoder endEncoding];
      }
      [command commit];
      [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError) {
        NSString* description = command.error.localizedDescription;
        Rcpp::stop(
          description == nil ? "Metal graph clustering command failed." :
          std::string([description UTF8String])
        );
      }
      const std::uint32_t pass_moves =
        *static_cast<std::uint32_t*>(move_buffer.contents);
      result.moves += static_cast<int>(pass_moves);
      ++result.passes;
      if (pass_moves == 0U) break;
    }

    const int* labels = static_cast<const int*>(label_buffer.contents);
    std::copy(labels, labels + graph.n, result.membership.begin());
    return result;
  }
}
