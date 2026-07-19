adjusted_rand_index <- function(left, right) {
  contingency <- table(left, right)
  choose2 <- function(x) x * (x - 1) / 2
  n <- sum(contingency)
  same <- sum(choose2(contingency))
  row_pairs <- sum(choose2(rowSums(contingency)))
  col_pairs <- sum(choose2(colSums(contingency)))
  expected <- row_pairs * col_pairs / choose2(n)
  maximum <- (row_pairs + col_pairs) / 2
  if (maximum == expected) return(1)
  (same - expected) / (maximum - expected)
}

three_clique_graph <- function() {
  groups <- split(seq_len(18L), rep(seq_len(3L), each = 6L))
  from <- to <- integer()
  weight <- numeric()
  for (group in groups) {
    pairs <- utils::combn(group, 2L)
    from <- c(from, pairs[1L, ])
    to <- c(to, pairs[2L, ])
    weight <- c(weight, rep(1, ncol(pairs)))
  }
  from <- c(from, 6L, 12L)
  to <- c(to, 7L, 13L)
  weight <- c(weight, 0.15, 0.15)
  structure(
    list(
      from = from, to = to, weight = weight,
      n_vertices = 18L, n_edges = length(from)
    ),
    class = c("fastEmbedR_graph", "list")
  )
}

test_that("knn_graph builds a compact native graph", {
  set.seed(17)
  x <- rbind(
    matrix(rnorm(80, 0, 0.2), ncol = 4),
    matrix(rnorm(80, 3, 0.2), ncol = 4)
  )
  graph <- knn_graph(x, k = 8L, weight = "snn", n_threads = 2L)
  expect_s3_class(graph, "fastEmbedR_graph")
  expect_identical(graph$n_vertices, nrow(x))
  expect_equal(length(graph$from), graph$n_edges)
  expect_true(all(graph$from < graph$to))
  expect_true(all(graph$weight > 0 & graph$weight <= 1))
  expect_identical(graph$parameters$weight, "snn")
  expect_identical(graph$parameters$n_threads, 2L)
})

test_that("native graph methods recover canonical communities", {
  graph <- three_clique_graph()
  truth <- rep(seq_len(3L), each = 6L)
  for (method in c("louvain", "leiden", "walktrap")) {
    result <- graph_cluster(
      graph,
      method = method,
      n_iterations = 20L,
      n_runs = 2L,
      seed = 3L
    )
    expect_s3_class(result, "fastEmbedR_graph_cluster")
    expect_identical(result$backend, "cpu")
    expect_identical(result$n_communities, 3L)
    expect_equal(adjusted_rand_index(result$membership, truth), 1)
    expect_gt(result$modularity, 0.6)
    if (identical(method, "leiden")) {
      expect_true(result$connected_communities)
    }
  }
})

test_that("Walktrap agrees with the igraph oracle", {
  skip_if_not_installed("igraph")
  graph <- three_clique_graph()
  igraph_graph <- igraph::graph_from_data_frame(
    data.frame(from = graph$from, to = graph$to, weight = graph$weight),
    directed = FALSE,
    vertices = seq_len(graph$n_vertices)
  )
  reference <- igraph::cluster_walktrap(
    igraph_graph,
    weights = igraph::E(igraph_graph)$weight,
    steps = 4L
  )
  native <- graph_cluster(graph, method = "walktrap", steps = 4L)
  expect_equal(
    adjusted_rand_index(native$membership, igraph::membership(reference)),
    1
  )
  expect_equal(native$modularity, igraph::modularity(reference), tolerance = 1e-10)
})

test_that("Louvain and Leiden remain competitive with igraph", {
  skip_if_not_installed("igraph")
  set.seed(103)
  sizes <- c(45L, 55L, 65L, 75L)
  preference <- matrix(0.006, nrow = 4L, ncol = 4L)
  diag(preference) <- 0.07
  igraph_graph <- igraph::sample_sbm(
    sum(sizes),
    pref.matrix = preference,
    block.sizes = sizes,
    directed = FALSE,
    loops = FALSE
  )
  igraph::E(igraph_graph)$weight <- runif(igraph::ecount(igraph_graph), 0.5, 1.5)
  edges <- igraph::as_data_frame(igraph_graph, what = "edges")
  graph <- structure(
    list(
      from = as.integer(edges$from),
      to = as.integer(edges$to),
      weight = edges$weight,
      n_vertices = igraph::vcount(igraph_graph),
      n_edges = igraph::ecount(igraph_graph)
    ),
    class = c("fastEmbedR_graph", "list")
  )

  native_louvain <- graph_cluster(
    graph, "louvain", n_iterations = 20L, n_runs = 3L, seed = 7L
  )
  set.seed(7)
  reference_louvain <- igraph::cluster_louvain(
    igraph_graph,
    weights = igraph::E(igraph_graph)$weight,
    resolution = 1
  )
  expect_gte(native_louvain$modularity, igraph::modularity(reference_louvain) - 0.03)

  native_leiden <- graph_cluster(
    graph, "leiden", n_iterations = 10L, n_runs = 3L, seed = 7L
  )
  set.seed(7)
  reference_leiden <- igraph::cluster_leiden(
    igraph_graph,
    objective_function = "modularity",
    weights = igraph::E(igraph_graph)$weight,
    resolution = 1,
    n_iterations = 10L
  )
  reference_modularity <- igraph::modularity(
    igraph_graph,
    igraph::membership(reference_leiden),
    weights = igraph::E(igraph_graph)$weight,
    resolution = 1
  )
  expect_true(native_leiden$connected_communities)
  expect_gte(native_leiden$modularity, reference_modularity - 0.03)
})

test_that("Walktrap fails before unsafe dense allocation", {
  graph <- structure(
    list(
      from = 1L,
      to = 2L,
      weight = 1,
      n_vertices = 4001L,
      n_edges = 1L
    ),
    class = c("fastEmbedR_graph", "list")
  )
  expect_error(
    graph_cluster(graph, method = "walktrap"),
    "16-million-entry"
  )
})
