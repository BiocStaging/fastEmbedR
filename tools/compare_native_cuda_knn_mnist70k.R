args <- commandArgs(trailingOnly = TRUE)
data_file <- if (length(args) >= 1L) args[[1L]] else {
  path.expand("~/cuda_pca_test_20260711/input/MNIST_float32.RData")
}
output_dir <- if (length(args) >= 2L) args[[2L]] else {
  path.expand("~/cuda_pca_test_20260711/native_cuda_knn_comparison_mnist70k")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(float))
suppressPackageStartupMessages(library(fastEmbedR))

loaded <- new.env(parent = emptyenv())
load(data_file, envir = loaded)
objects <- mget(ls(loaded, all.names = TRUE), envir = loaded, inherits = FALSE)
dataset <- NULL
for (object in objects) {
  if (is.list(object) && !is.null(object$data)) {
    dataset <- object
    break
  }
}
if (is.null(dataset)) stop("No dataset$data object was found in ", data_file)
x <- dataset$data
labels <- dataset$labels
stopifnot(nrow(x) == 70000L, ncol(x) == 784L, length(labels) == nrow(x))

recall_at_k <- function(observed, reference) {
  stopifnot(identical(dim(observed$indices), dim(reference$indices)))
  mean(vapply(seq_len(nrow(reference$indices)), function(i) {
    length(intersect(observed$indices[i, ], reference$indices[i, ])) /
      ncol(reference$indices)
  }, numeric(1)))
}

time_knn <- function(name, expression, to_host) {
  gc()
  elapsed <- system.time({ result <- expression })[["elapsed"]]
  list(
    name = name,
    object = result,
    host = to_host(result),
    elapsed = as.numeric(elapsed)
  )
}

exact <- time_knn(
  "native FAISS GPU exact",
  fastEmbedR:::native_cuda_knn_cpp(
    x, 30L, "exact", "euclidean", 0.99, TRUE
  ),
  fastEmbedR:::native_cuda_knn_to_host_cpp
)

ivf <- time_knn(
  "native cuVS IVF-Flat",
  fastEmbedR:::native_cuda_knn_cpp(
    x, 30L, "ivf", "euclidean", 0.99, TRUE
  ),
  fastEmbedR:::native_cuda_knn_to_host_cpp
)
ivf_recall <- recall_at_k(ivf$host, exact$host)

faiss <- NULL
if (requireNamespace("faissR", quietly = TRUE) &&
    "nn_gpu" %in% getNamespaceExports("faissR")) {
  faiss <- tryCatch(
    time_knn(
      "faissR GPU exact",
      faissR::nn_gpu(
        x,
        k = 30L,
        exclude_self = TRUE,
        method = "exact",
        metric = "euclidean",
        tuning = "auto",
        target_recall = 0.99
      ),
      faissR::gpu_knn_to_host
    ),
    error = function(e) {
      message("faissR comparison unavailable: ", conditionMessage(e))
      NULL
    }
  )
}

run_umap <- function(route) {
  gc()
  elapsed <- system.time({
    fit <- fastEmbedR::umap(
      route$object,
      n_neighbors = 30L,
      backend = "cuda",
      graph_mode = "fuzzy",
      n_threads = 4L,
      seed = 4L
    )
  })[["elapsed"]]
  list(fit = fit, elapsed = as.numeric(elapsed))
}

set.seed(4)
exact_umap <- run_umap(exact)
set.seed(4)
ivf_umap <- run_umap(ivf)
faiss_umap <- NULL
if (!is.null(faiss)) {
  set.seed(4)
  faiss_umap <- tryCatch(run_umap(faiss), error = function(e) {
    message("faissR GPU object was not accepted by this build: ", conditionMessage(e))
    NULL
  })
}

rows <- list(
  data.frame(
    route = exact$name,
    knn_sec = exact$elapsed,
    embedding_sec = exact_umap$elapsed,
    total_sec = exact$elapsed + exact_umap$elapsed,
    recall_at_30 = 1,
    nlist = NA_integer_,
    nprobe = NA_integer_,
    pilot_recall = 1,
    stringsAsFactors = FALSE
  ),
  data.frame(
    route = ivf$name,
    knn_sec = ivf$elapsed,
    embedding_sec = ivf_umap$elapsed,
    total_sec = ivf$elapsed + ivf_umap$elapsed,
    recall_at_30 = ivf_recall,
    nlist = ivf$object$nlist,
    nprobe = ivf$object$nprobe,
    pilot_recall = ivf$object$pilot_recall,
    stringsAsFactors = FALSE
  )
)
if (!is.null(faiss) && !is.null(faiss_umap)) {
  rows[[length(rows) + 1L]] <- data.frame(
    route = faiss$name,
    knn_sec = faiss$elapsed,
    embedding_sec = faiss_umap$elapsed,
    total_sec = faiss$elapsed + faiss_umap$elapsed,
    recall_at_30 = recall_at_k(faiss$host, exact$host),
    nlist = NA_integer_,
    nprobe = NA_integer_,
    pilot_recall = NA_real_,
    stringsAsFactors = FALSE
  )
}
results <- do.call(rbind, rows)
utils::write.csv(results, file.path(output_dir, "knn_route_comparison.csv"), row.names = FALSE)

to_double <- function(layout) {
  if (inherits(layout, "float32")) float::dbl(layout) else as.matrix(layout)
}
layouts <- list(
  `native FAISS GPU exact` = exact_umap$fit$layout,
  `native cuVS IVF-Flat` = ivf_umap$fit$layout
)
if (!is.null(faiss_umap)) layouts[["faissR GPU exact"]] <- faiss_umap$fit$layout
palette <- grDevices::hcl.colors(length(unique(labels)), "Dark 3")
grDevices::png(
  file.path(output_dir, "knn_route_umap_comparison.png"),
  width = 1200L * length(layouts), height = 1200L, res = 180
)
op <- par(mfrow = c(1, length(layouts)), mar = c(4, 4, 3, 1))
for (name in names(layouts)) {
  row <- results[results$route == name, , drop = FALSE]
  plot(
    to_double(layouts[[name]]),
    col = palette[as.integer(factor(labels))],
    pch = 16,
    cex = 0.24,
    xlab = "Component 1",
    ylab = "Component 2",
    main = sprintf("%s\n%.2fs, recall %.3f", name, row$total_sec, row$recall_at_30)
  )
}
par(op)
grDevices::dev.off()

saveRDS(layouts, file.path(output_dir, "knn_route_umap_layouts.rds"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
print(results)
