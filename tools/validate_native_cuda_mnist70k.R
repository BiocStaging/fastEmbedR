args <- commandArgs(trailingOnly = TRUE)
data_file <- if (length(args) >= 1L) args[[1L]] else {
  path.expand("~/cuda_pca_test_20260711/input/MNIST_float32.RData")
}
output_dir <- if (length(args) >= 2L) args[[2L]] else {
  path.expand("~/cuda_pca_test_20260711/native_cuda_mnist70k")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_file)) {
  stop("MNIST input was not found: ", data_file)
}

if (!requireNamespace("float", quietly = TRUE)) {
  stop("The float package is required to load the MNIST float32 input.")
}
suppressPackageStartupMessages(library(float))

loaded <- new.env(parent = emptyenv())
load(data_file, envir = loaded)

# The primary benchmark files use dataset$data and dataset$labels. The fallback
# below handles older files containing separate data/label objects.
objects <- mget(ls(loaded, all.names = TRUE), envir = loaded, inherits = FALSE)
dataset_object <- NULL
for (object in objects) {
  if (is.list(object) && !is.null(object$data)) {
    dataset_object <- object
    break
  }
}
if (!is.null(dataset_object)) {
  x <- dataset_object$data
  labels <- dataset_object$labels
} else {
  matrix_names <- names(objects)[vapply(objects, function(object) {
    is.matrix(object) || inherits(object, "float32")
  }, logical(1))]
  if (!length(matrix_names)) stop("The RData file contains no matrix-like data object.")
  x <- objects[[matrix_names[[1L]]]]
  labels <- NULL
  for (name in c("labels", "label", "y", "Y")) {
    if (!is.null(objects[[name]]) && length(objects[[name]]) == nrow(x)) {
      labels <- objects[[name]]
      break
    }
  }
}

if (nrow(x) != 70000L || ncol(x) != 784L) {
  stop("Expected flattened MNIST70k (70000 x 784), found ",
       paste(dim(x), collapse = " x "), ".")
}
if (is.null(labels) || length(labels) != nrow(x)) {
  stop("MNIST labels are missing or have the wrong length.")
}

suppressPackageStartupMessages(library(fastEmbedR))
if (!isTRUE(fastEmbedR:::native_cuda_knn_available_cpp())) {
  stop("The installed fastEmbedR build does not provide native cuVS KNN.")
}

run_method <- function(name, expression) {
  gc()
  elapsed <- system.time({
    fit <- expression
  })[["elapsed"]]
  engine <- fit$parameters$nn_engine
  if (!identical(engine, "native_faiss_gpu_exact")) {
    stop(name, " used unexpected KNN engine: ", paste(engine, collapse = ", "))
  }
  metrics <- fit$metrics[1L, , drop = FALSE]
  data.frame(
    method = name,
    n = nrow(fit$layout),
    p = ncol(x),
    k = 30L,
    total_sec = as.numeric(elapsed),
    preprocess_sec = as.numeric(metrics$preprocess_elapsed),
    knn_sec = as.numeric(metrics$knn_elapsed),
    init_sec = if ("initialization_elapsed" %in% names(metrics)) {
      as.numeric(metrics$initialization_elapsed)
    } else {
      0
    },
    embedding_sec = as.numeric(metrics$embedding_elapsed),
    nn_engine = engine,
    nn_backend = fit$parameters$nn_backend,
    input_class = paste(class(x), collapse = ","),
    stringsAsFactors = FALSE
  ) -> row
  list(fit = fit, row = row)
}

set.seed(4)
tsne <- run_method(
  "fastEmbedR openTSNE CUDA",
  fastEmbedR::opentsne(
    x,
    perplexity = 30,
    backend = "cuda",
    n_threads = 4,
    seed = 4
  )
)
set.seed(4)
umap_fuzzy <- run_method(
  "fastEmbedR UMAP CUDA fuzzy",
  fastEmbedR::umap(
    x,
    n_neighbors = 30,
    backend = "cuda",
    graph_mode = "fuzzy",
    n_threads = 4,
    seed = 4
  )
)
set.seed(4)
umap_binary <- run_method(
  "fastEmbedR UMAP CUDA binary",
  fastEmbedR::umap(
    x,
    n_neighbors = 30,
    backend = "cuda",
    graph_mode = "binary",
    n_threads = 4,
    seed = 4
  )
)

results <- rbind(tsne$row, umap_fuzzy$row, umap_binary$row)
utils::write.csv(
  results,
  file.path(output_dir, "native_cuda_mnist70k.csv"),
  row.names = FALSE
)

saveRDS(tsne$fit$layout, file.path(output_dir, "opentsne_cuda_layout.rds"))
saveRDS(umap_fuzzy$fit$layout, file.path(output_dir, "umap_cuda_fuzzy_layout.rds"))
saveRDS(umap_binary$fit$layout, file.path(output_dir, "umap_cuda_binary_layout.rds"))

plot_layout <- function(layout, title) {
  if (inherits(layout, "float32")) layout <- float::dbl(layout)
  plot(
    layout,
    col = palette[as.integer(factor(labels))],
    pch = 16,
    cex = 0.24,
    xlab = "Component 1",
    ylab = "Component 2",
    main = title
  )
}

palette <- grDevices::hcl.colors(length(unique(labels)), "Dark 3")
grDevices::png(
  file.path(output_dir, "native_cuda_mnist70k.png"),
  width = 3600,
  height = 1250,
  res = 180
)
op <- par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
plot_layout(tsne$fit$layout, sprintf("openTSNE CUDA (%.2fs)", tsne$row$total_sec))
plot_layout(umap_fuzzy$fit$layout, sprintf("UMAP CUDA fuzzy (%.2fs)", umap_fuzzy$row$total_sec))
plot_layout(umap_binary$fit$layout, sprintf("UMAP CUDA binary (%.2fs)", umap_binary$row$total_sec))
grDevices::dev.off()
par(op)

writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
print(results)
