/*
 * Compact HNSW implementation distilled from the algorithmic organization in
 * FAISS 1.14.3, commit 0ca9df4792b173d573044ee14ca0704780176e82.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2026 Stefano Caccia.
 *
 * Licensed under the MIT License. The FAISS copyright and MIT license must be
 * retained with redistributed derivatives of this file.
 */

#include <Rcpp.h>

#include "native_knn_common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct NodeDistance {
  float distance;
  int id;
};

struct CloserFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance > b.distance;
    return a.id > b.id;
  }
};

struct FartherFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.id < b.id;
  }
};

inline bool closer(const NodeDistance& a, const NodeDistance& b) {
  return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

class CompactHNSW {
 public:
  CompactHNSW(std::vector<float> data, int n, int p, int m, int ef_construction, int ef_search)
      : data_(std::move(data)), n_(n), p_(p), m_(m), ef_construction_(ef_construction), ef_search_(ef_search) {
    generate_levels();
    allocate_graph();
  }

  void build() {
    if (n_ == 0) return;
    entry_point_ = 0;
    current_max_level_ = levels_[0];
    for (int point = 1; point < n_; ++point) add_point(point);
  }

  void search_all(int k, int n_threads, std::vector<int>& output_ids, std::vector<float>& output_distances) const {
    output_ids.assign(static_cast<std::size_t>(n_) * k, -1);
    output_distances.assign(static_cast<std::size_t>(n_) * k, std::numeric_limits<float>::infinity());
    std::atomic<int> next(0);
    n_threads = std::max(1, std::min(n_threads, n_));
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int thread_id = 0; thread_id < n_threads; ++thread_id) {
      workers.emplace_back([&, thread_id]() {
        (void)thread_id;
        VisitTable visited(n_);
        while (true) {
          int query = next.fetch_add(1, std::memory_order_relaxed);
          if (query >= n_) break;
          std::vector<NodeDistance> candidates = search_query(query, std::max(k + 1, ef_search_), visited);
          int used = 0;
          for (const NodeDistance& candidate : candidates) {
            if (candidate.id == query) continue;
            std::size_t pos = static_cast<std::size_t>(query) * k + used;
            output_ids[pos] = candidate.id;
            output_distances[pos] = candidate.distance;
            if (++used == k) break;
          }
          if (used < k) exact_fill(query, k, used, output_ids, output_distances);
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
  }

  std::size_t graph_bytes() const {
    return neighbors_.size() * sizeof(std::int32_t) + levels_.size() * sizeof(int) +
      node_offsets_.size() * sizeof(std::size_t) + level_offsets_.size() * sizeof(std::size_t) +
      counts_.size() * sizeof(std::uint16_t);
  }

 private:
  struct VisitTable {
    explicit VisitTable(int n) : marks(n, 0), generation(1) {}
    void reset() {
      if (++generation == 0) { std::fill(marks.begin(), marks.end(), 0); generation = 1; }
    }
    bool set(int id) {
      if (marks[id] == generation) return false;
      marks[id] = generation;
      return true;
    }
    std::vector<std::uint32_t> marks;
    std::uint32_t generation;
  };

  std::vector<float> data_;
  int n_;
  int p_;
  int m_;
  int ef_construction_;
  int ef_search_;
  int entry_point_ = -1;
  int current_max_level_ = -1;
  std::vector<int> levels_;
  std::vector<std::size_t> node_offsets_;
  std::vector<std::size_t> level_offsets_;
  std::vector<std::uint16_t> counts_;
  std::vector<std::int32_t> neighbors_;

  inline const float* point(int id) const { return data_.data() + static_cast<std::size_t>(id) * p_; }

  inline float distance(const float* a, const float* b) const {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int d = 0;
    for (; d + 15 < p_; d += 16) {
      float x0 = a[d] - b[d]; float x1 = a[d + 1] - b[d + 1];
      float x2 = a[d + 2] - b[d + 2]; float x3 = a[d + 3] - b[d + 3];
      float x4 = a[d + 4] - b[d + 4]; float x5 = a[d + 5] - b[d + 5];
      float x6 = a[d + 6] - b[d + 6]; float x7 = a[d + 7] - b[d + 7];
      float x8 = a[d + 8] - b[d + 8]; float x9 = a[d + 9] - b[d + 9];
      float xa = a[d + 10] - b[d + 10]; float xb = a[d + 11] - b[d + 11];
      float xc = a[d + 12] - b[d + 12]; float xd = a[d + 13] - b[d + 13];
      float xe = a[d + 14] - b[d + 14]; float xf = a[d + 15] - b[d + 15];
      sum0 += x0*x0 + x4*x4 + x8*x8 + xc*xc;
      sum1 += x1*x1 + x5*x5 + x9*x9 + xd*xd;
      sum2 += x2*x2 + x6*x6 + xa*xa + xe*xe;
      sum3 += x3*x3 + x7*x7 + xb*xb + xf*xf;
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; d < p_; ++d) { float delta = a[d] - b[d]; sum += delta * delta; }
    return sum;
  }

  inline float distance(int a, int b) const { return distance(point(a), point(b)); }

  int capacity(int level) const { return level == 0 ? 2 * m_ : m_; }

  std::size_t level_index(int node, int level) const { return level_offsets_[node] + static_cast<std::size_t>(level); }

  std::size_t neighbor_offset(int node, int level) const {
    return node_offsets_[node] + (level == 0 ? 0u : static_cast<std::size_t>(2 * m_ + (level - 1) * m_));
  }

  void generate_levels() {
    levels_.resize(n_);
    std::mt19937 generator(12345u);
    std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
    const double level_multiplier = 1.0 / std::log(static_cast<double>(m_));
    for (int i = 0; i < n_; ++i) levels_[i] = static_cast<int>(-std::log(uniform(generator)) * level_multiplier);
  }

  void allocate_graph() {
    node_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    level_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    for (int i = 0; i < n_; ++i) {
      node_offsets_[i + 1] = node_offsets_[i] + static_cast<std::size_t>(2 * m_ + levels_[i] * m_);
      level_offsets_[i + 1] = level_offsets_[i] + static_cast<std::size_t>(levels_[i] + 1);
    }
    neighbors_.assign(node_offsets_.back(), -1);
    counts_.assign(level_offsets_.back(), 0);
  }

  std::pair<const std::int32_t*, int> neighbor_range(int node, int level) const {
    std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, counts_[level_index(node, level)]};
  }

  std::pair<std::int32_t*, std::uint16_t*> mutable_neighbor_range(int node, int level) {
    std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, &counts_[level_index(node, level)]};
  }

  int greedy_search(const float* query, int entry, int level, float& entry_distance) const {
    bool changed = true;
    while (changed) {
      changed = false;
      auto range = neighbor_range(entry, level);
      for (int j = 0; j < range.second; ++j) {
        int candidate = range.first[j];
        float candidate_distance = distance(query, point(candidate));
        NodeDistance candidate_pair{candidate_distance, candidate};
        NodeDistance current_pair{entry_distance, entry};
        if (closer(candidate_pair, current_pair)) {
          entry = candidate;
          entry_distance = candidate_distance;
          changed = true;
        }
      }
    }
    return entry;
  }

  std::vector<NodeDistance> search_layer(const float* query, int entry, int ef, int level, VisitTable& visited) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, CloserFirst> candidates;
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> results;
    visited.reset();
    float initial_distance = distance(query, point(entry));
    NodeDistance initial{initial_distance, entry};
    candidates.push(initial); results.push(initial); visited.set(entry);
    while (!candidates.empty()) {
      NodeDistance current = candidates.top();
      NodeDistance worst = results.top();
      if (results.size() >= static_cast<std::size_t>(ef) && closer(worst, current)) break;
      candidates.pop();
      auto range = neighbor_range(current.id, level);
      for (int j = 0; j < range.second; ++j) {
        int candidate_id = range.first[j];
        if (!visited.set(candidate_id)) continue;
        NodeDistance candidate{distance(query, point(candidate_id)), candidate_id};
        if (results.size() < static_cast<std::size_t>(ef) || closer(candidate, results.top())) {
          candidates.push(candidate);
          results.push(candidate);
          if (results.size() > static_cast<std::size_t>(ef)) results.pop();
        }
      }
    }
    std::vector<NodeDistance> output;
    output.reserve(results.size());
    while (!results.empty()) { output.push_back(results.top()); results.pop(); }
    std::sort(output.begin(), output.end(), closer);
    return output;
  }

  std::vector<int> select_diverse(int query_id, const std::vector<NodeDistance>& candidates, int max_size) const {
    std::vector<int> selected;
    std::vector<int> rejected;
    selected.reserve(max_size); rejected.reserve(max_size);
    for (const NodeDistance& candidate : candidates) {
      if (candidate.id == query_id) continue;
      bool good = true;
      for (int other : selected) {
        if (distance(candidate.id, other) < candidate.distance) { good = false; break; }
      }
      if (good) {
        selected.push_back(candidate.id);
        if (static_cast<int>(selected.size()) == max_size) break;
      } else {
        rejected.push_back(candidate.id);
      }
    }
    for (int candidate : rejected) {
      if (static_cast<int>(selected.size()) == max_size) break;
      selected.push_back(candidate);
    }
    return selected;
  }

  void replace_neighbors(int node, int level, const std::vector<int>& selected) {
    auto range = mutable_neighbor_range(node, level);
    int cap = capacity(level);
    int size = std::min(cap, static_cast<int>(selected.size()));
    for (int i = 0; i < size; ++i) range.first[i] = selected[i];
    for (int i = size; i < cap; ++i) range.first[i] = -1;
    *range.second = static_cast<std::uint16_t>(size);
  }

  void add_reciprocal(int node, int other, int level) {
    auto range = mutable_neighbor_range(node, level);
    int count = *range.second;
    for (int i = 0; i < count; ++i) if (range.first[i] == other) return;
    int cap = capacity(level);
    if (count < cap) {
      range.first[count] = other;
      *range.second = static_cast<std::uint16_t>(count + 1);
      return;
    }
    std::vector<NodeDistance> candidates;
    candidates.reserve(static_cast<std::size_t>(count) + 1u);
    for (int i = 0; i < count; ++i) candidates.push_back({distance(node, range.first[i]), range.first[i]});
    candidates.push_back({distance(node, other), other});
    std::sort(candidates.begin(), candidates.end(), closer);
    replace_neighbors(node, level, select_diverse(node, candidates, cap));
  }

  void add_point(int point_id) {
    VisitTable visited(n_);
    int nearest = entry_point_;
    float nearest_distance = distance(point_id, nearest);
    for (int level = current_max_level_; level > levels_[point_id]; --level)
      nearest = greedy_search(point(point_id), nearest, level, nearest_distance);
    for (int level = std::min(levels_[point_id], current_max_level_); level >= 0; --level) {
      std::vector<NodeDistance> candidates = search_layer(point(point_id), nearest, ef_construction_, level, visited);
      std::vector<int> selected = select_diverse(point_id, candidates, capacity(level));
      replace_neighbors(point_id, level, selected);
      for (int other : selected) add_reciprocal(other, point_id, level);
      if (!candidates.empty()) { nearest = candidates.front().id; nearest_distance = candidates.front().distance; }
    }
    if (levels_[point_id] > current_max_level_) {
      entry_point_ = point_id;
      current_max_level_ = levels_[point_id];
    }
  }

  std::vector<NodeDistance> search_query(int query_id, int ef, VisitTable& visited) const {
    const float* query = point(query_id);
    int nearest = entry_point_;
    float nearest_distance = distance(query, point(nearest));
    for (int level = current_max_level_; level >= 1; --level)
      nearest = greedy_search(query, nearest, level, nearest_distance);
    return search_layer(query, nearest, ef, 0, visited);
  }

  void exact_fill(int query, int k, int used, std::vector<int>& output_ids, std::vector<float>& output_distances) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> heap;
    for (int candidate = 0; candidate < n_; ++candidate) {
      if (candidate == query) continue;
      NodeDistance value{distance(query, candidate), candidate};
      if (heap.size() < static_cast<std::size_t>(k) || closer(value, heap.top())) {
        heap.push(value);
        if (heap.size() > static_cast<std::size_t>(k)) heap.pop();
      }
    }
    std::vector<NodeDistance> exact;
    while (!heap.empty()) { exact.push_back(heap.top()); heap.pop(); }
    std::sort(exact.begin(), exact.end(), closer);
    (void)used;
    for (int rank = 0; rank < k; ++rank) {
      std::size_t pos = static_cast<std::size_t>(query) * k + rank;
      output_ids[pos] = exact[rank].id;
      output_distances[pos] = exact[rank].distance;
    }
  }
};

} // namespace

Rcpp::List native_hnsw_knn_impl(SEXP data_sexp,
                                int k,
                                int n_threads,
                                const std::string& metric_name,
                                double target_recall) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const fastembedr::KnnMetric metric = fastembedr::parse_knn_metric(metric_name);
  if (metric == fastembedr::KnnMetric::InnerProduct) {
    Rcpp::stop("Native CPU HNSW does not support raw inner-product distance.");
  }
  fastembedr::FloatMatrix input = fastembedr::matrix_to_row_major_float(data_sexp, metric);
  const auto converted = Clock::now();
  const int n = input.nrow;
  const int p = input.ncol;
  if (n < 2 || p < 1 || k < 1 || k >= n) Rcpp::stop("invalid HNSW input");
  const bool large_high_dim = n >= 50000 && p >= 128 && k <= 30;
  const int m = large_high_dim ? 10 : 16;
  const int ef_construction = large_high_dim ? 40 : 80;
  const int ef_search = large_high_dim ? 40 : 64;
  CompactHNSW index(std::move(input.values), n, p, m, ef_construction, ef_search);
  index.build();
  const auto built = Clock::now();
  std::vector<int> ids;
  std::vector<float> squared_distances;
  index.search_all(k, n_threads, ids, squared_distances);
  const auto searched = Clock::now();
  Rcpp::IntegerMatrix indices(n, k);
  Rcpp::NumericMatrix distances(n, k);
  for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) {
    std::size_t pos = static_cast<std::size_t>(i) * k + j;
    indices(i, j) = ids[pos] + 1;
    distances(i, j) = fastembedr::output_distance(squared_distances[pos], metric);
  }
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("method") = "native_hnsw",
    Rcpp::Named("metric") = metric_name,
    Rcpp::Named("target_recall") = target_recall,
    Rcpp::Named("M") = m,
    Rcpp::Named("efConstruction") = ef_construction,
    Rcpp::Named("efSearch") = ef_search,
    Rcpp::Named("graph_bytes") = static_cast<double>(index.graph_bytes()),
    Rcpp::Named("timing") = Rcpp::NumericVector::create(
      Rcpp::Named("convert") = std::chrono::duration<double>(converted - start).count(),
      Rcpp::Named("build") = std::chrono::duration<double>(built - converted).count(),
      Rcpp::Named("query") = std::chrono::duration<double>(searched - built).count()
    )
  );
  result.attr("backend") = "cpu";
  result.attr("method") = "native_hnsw";
  result.attr("target_recall") = target_recall;
  result.attr("metric") = metric_name;
  return result;
}
