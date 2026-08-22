args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: validate-hardware.R <cpu|metal|cuda> <output-directory>", call. = FALSE)
}

backend <- match.arg(args[[1L]], c("cpu", "metal", "cuda"))
out_dir <- normalizePath(args[[2L]], mustWork = TRUE)

suppressPackageStartupMessages(library(fastEmbedR))

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
capabilities <- fastEmbedR_capabilities()
utils::write.csv(capabilities, file.path(out_dir, "capabilities.csv"), row.names = FALSE)

row <- capabilities[capabilities$backend == backend, , drop = FALSE]
if (nrow(row) != 1L || !isTRUE(row$knn_available) ||
    !isTRUE(row$embedding_available) || !isTRUE(row$clustering_available)) {
  stop("Requested backend is not fully available: ", backend, call. = FALSE)
}

seed <- 20260822L
set.seed(seed)
n <- 768L
p <- 32L
group <- rep(seq_len(6L), length.out = n)
x <- matrix(rnorm(n * p, sd = 0.35), nrow = n, ncol = p)
x[, seq_len(6L)] <- x[, seq_len(6L), drop = FALSE] +
  stats::model.matrix(~ factor(group) - 1)

elapsed <- function(label, expression) {
  gc()
  timing <- system.time(value <- force(expression))
  list(label = label, value = value, elapsed = unname(timing[["elapsed"]]))
}

pca_run <- elapsed("pca", pca(
  x, ncomp = 2L, backend = backend, n.cores = 4L, seed = seed
))
knn_run <- elapsed("knn", precompute_knn(
  x, k = 15L, metric = "euclidean", backend = backend, n.cores = 4L
))
umap_run <- elapsed("umap", umap(
  x, n_neighbors = 15L, metric = "euclidean", backend = backend,
  n.cores = 4L, seed = seed, graph_mode = "fuzzy"
))
tsne_run <- elapsed("opentsne", opentsne(
  x, perplexity = 15L, metric = "euclidean", backend = backend,
  n.cores = 4L, seed = seed, early_exaggeration_iter = 10L,
  n_iter = 20L, auto_config = FALSE, negative_gradient_method = "fft"
))

graph <- knn_graph(knn_run$value, weight = "snn", backend = backend, n.cores = 4L)
cluster_run <- elapsed("leiden", graph_cluster(
  graph, method = "leiden", backend = backend, n_iterations = 3L,
  n_runs = 1L, seed = seed
))

if (!identical(attr(knn_run$value, "backend"), backend)) {
  stop("KNN backend mismatch", call. = FALSE)
}
if (!startsWith(pca_run$value$backend, backend)) {
  stop("PCA backend mismatch: ", pca_run$value$backend, call. = FALSE)
}
umap_backend <- attr(umap_run$value$layout, "fastEmbedR_config")$backend
tsne_backend <- attr(tsne_run$value$layout, "fastEmbedR_config")$backend
if (!identical(umap_backend, backend)) stop("UMAP backend mismatch", call. = FALSE)
if (!identical(tsne_backend, backend)) stop("openTSNE backend mismatch", call. = FALSE)
if (!identical(cluster_run$value$backend_requested, backend) ||
    !identical(cluster_run$value$backend, backend)) {
  stop("Leiden backend mismatch", call. = FALSE)
}

layouts <- list(
  pca = pca_run$value$scores,
  umap = umap_run$value$layout,
  opentsne = tsne_run$value$layout,
  leiden_membership = cluster_run$value$membership
)
saveRDS(layouts, file.path(out_dir, "validation-results.rds"), version = 3L)

summary <- data.frame(
  operation = c("pca", "knn", "umap", "opentsne", "leiden"),
  backend_requested = backend,
  backend_used = c(
    pca_run$value$backend,
    attr(knn_run$value, "backend"),
    umap_backend,
    tsne_backend,
    cluster_run$value$backend
  ),
  elapsed_sec = c(
    pca_run$elapsed, knn_run$elapsed, umap_run$elapsed,
    tsne_run$elapsed, cluster_run$elapsed
  ),
  n = n,
  p = p,
  seed = seed,
  stringsAsFactors = FALSE
)
utils::write.csv(summary, file.path(out_dir, "hardware-benchmark.csv"), row.names = FALSE)

cat("All public operations used the requested", backend, "backend.\n")
print(summary)
