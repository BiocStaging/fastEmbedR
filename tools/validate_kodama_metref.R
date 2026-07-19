#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1L) args[[1L]] else "/mnt/sata_ssd/fastEmbedR_Data/MetRef/MetRef.RData"
output_dir <- if (length(args) >= 2L) args[[2L]] else "/mnt/sata_ssd/fastEmbedR/singularity/kodama_metref_validation"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(kodamaR))

env <- new.env(parent = emptyenv())
load(data_path, envir = env)
if (!exists("dataset", envir = env, inherits = FALSE)) {
  stop("MetRef RData must contain an object named `dataset`.")
}
dataset <- get("dataset", envir = env, inherits = FALSE)
if (!is.list(dataset) || is.null(dataset$data) || is.null(dataset$labels)) {
  stop("MetRef `dataset` must contain `data` and `labels`.")
}

x <- as.matrix(dataset$data)
storage.mode(x) <- "double"
labels <- as.factor(dataset$labels)
if (nrow(x) != length(labels)) stop("MetRef data and labels have different lengths.")
if (anyNA(x) || any(!is.finite(x))) stop("MetRef data contain non-finite values.")

set.seed(4L)
n_threads <- 4L
n_classes <- nlevels(labels)
ncomp <- min(20L, ncol(x), nrow(x) - 1L)
splitting <- max(2L, n_classes)
landmarks <- nrow(x)

rows <- list()
objects <- list()

record_stage <- function(stage, backend, fn, reported = NULL, score = NULL) {
  key <- paste(stage, backend, sep = "_")
  cat("Running ", stage, " [", backend, "]\n", sep = "")
  started <- proc.time()[["elapsed"]]
  value <- tryCatch(fn(), error = identity)
  elapsed <- proc.time()[["elapsed"]] - started
  if (inherits(value, "error")) {
    rows[[key]] <<- data.frame(
      dataset = "MetRef", stage = stage, backend = backend,
      elapsed_sec = elapsed, reported_sec = NA_real_, score = NA_real_,
      peak_memory_mb = NA_real_, status = "failed",
      error = conditionMessage(value), stringsAsFactors = FALSE
    )
    cat("  failed: ", conditionMessage(value), "\n", sep = "")
    return(NULL)
  }
  reported_value <- if (is.null(reported)) NA_real_ else as.numeric(reported(value))
  score_value <- if (is.null(score)) NA_real_ else as.numeric(score(value))
  peak_memory <- if (is.list(value) && !is.null(value$peak_memory_mb)) {
    as.numeric(value$peak_memory_mb)
  } else {
    NA_real_
  }
  rows[[key]] <<- data.frame(
    dataset = "MetRef", stage = stage, backend = backend,
    elapsed_sec = elapsed, reported_sec = reported_value, score = score_value,
    peak_memory_mb = peak_memory, status = "success", error = "",
    stringsAsFactors = FALSE
  )
  objects[[key]] <<- value
  cat("  success: ", format(elapsed, digits = 5), " sec\n", sep = "")
  value
}

runtime_seconds <- function(z) z$runtime_seconds %||% NA_real_
accuracy <- function(z) z$accuracy %||% NA_real_
objective_accuracy <- function(z) {
  if (is.null(z$acc) || length(z$acc) == 0L) return(NA_real_)
  max(as.numeric(z$acc), na.rm = TRUE)
}
`%||%` <- function(x, y) if (is.null(x)) y else x
validated_embedding <- function(value, method, backend) {
  expected <- c(nrow(x), 2L)
  if (!all(dim(value) == expected)) {
    stop(method, " ", backend, " returned ", paste(dim(value), collapse = "x"),
         "; expected ", paste(expected, collapse = "x"), ".")
  }
  bad <- sum(!is.finite(value))
  if (bad > 0L) stop(method, " ", backend, " returned ", bad, " non-finite coordinates.")
  value
}

cv <- list()
pca <- list()
km_knn <- list()
km_pls <- list()
embeddings <- list()

for (backend in c("cpu", "cuda")) {
  cv[[paste0("knn_", backend)]] <- record_stage(
    "KNNCV", backend,
    function() KNNCV(
      x, labels, folds = 5L, k = 10L, metric = "cosine",
      backend = backend, n.cores = n_threads, seed = 4L
    ),
    runtime_seconds, accuracy
  )
  cv[[paste0("pls_lda_", backend)]] <- record_stage(
    "PLSLDACV", backend,
    function() PLSLDACV(
      x, labels, folds = 5L, ncomp = ncomp, center = TRUE, scale = TRUE,
      backend = backend, n.cores = n_threads, seed = 4L
    ),
    runtime_seconds, accuracy
  )
  pca[[backend]] <- record_stage(
    "PCA", backend,
    function() KODAMA.pca(
      x, ncomp = 10L, center = TRUE, scale = FALSE,
      backend = backend, n.cores = n_threads, seed = 4L
    ),
    runtime_seconds
  )
  km_knn[[backend]] <- record_stage(
    "KODAMA_KNN", backend,
    function() KODAMA.matrix(
      x, classifier = "knn", backend = backend,
      M = 3L, Tcycle = 3L, landmarks = landmarks, splitting = splitting,
      graph.neighbors = 30L, knn.k = 10L, n.cores = n_threads,
      seed = 4L, visual.init = TRUE, progress = FALSE
    ),
    runtime_seconds, objective_accuracy
  )
  km_pls[[backend]] <- record_stage(
    "KODAMA_PLSLDA", backend,
    function() KODAMA.matrix(
      x, classifier = "pls_lda", backend = backend,
      M = 2L, Tcycle = 2L, ncomp = ncomp,
      landmarks = landmarks, splitting = splitting,
      graph.neighbors = 30L, knn.k = 10L, n.cores = n_threads,
      seed = 4L, visual.init = FALSE, progress = FALSE
    ),
    runtime_seconds, objective_accuracy
  )

  fit <- km_knn[[backend]]
  if (!is.null(fit)) {
    embeddings[[paste0("umap_", backend)]] <- record_stage(
      "UMAP", backend,
      function() validated_embedding(
        KODAMA.visualization(
          fit, "UMAP", k = 30L, backend = backend, n.cores = n_threads,
          n.epochs = 200L, graph.mode = "binary", seed = 4L
        ),
        "UMAP", backend
      )
    )
    embeddings[[paste0("opentsne_", backend)]] <- record_stage(
      "openTSNE", backend,
      function() validated_embedding(
        KODAMA.visualization(
          fit, "opentsne", k = 30L, perplexity = 10,
          backend = backend, n.cores = n_threads, n.iter = 500L, seed = 4L
        ),
        "openTSNE", backend
      )
    )
  }
}

for (name in names(cv)) {
  value <- cv[[name]]
  if (!is.null(value) && grepl("cuda$", name) && !identical(value$backend, "cuda")) {
    stop("CUDA fallback detected in ", name, ": ", value$backend)
  }
}
for (name in names(pca)) {
  value <- pca[[name]]
  if (!is.null(value)) {
    stopifnot(all(dim(value$scores) == c(nrow(x), 10L)), all(is.finite(value$scores)))
    if (name == "cuda" && !identical(value$backend, "cuda")) {
      stop("CUDA fallback detected in PCA: ", value$backend)
    }
  }
}
for (collection in list(km_knn, km_pls)) {
  for (name in names(collection)) {
    value <- collection[[name]]
    if (!is.null(value)) {
      stopifnot(length(value$best_labels) == nrow(x), all(is.finite(value$acc)))
      if (name == "cuda" && !identical(value$backend, "cuda")) {
        stop("CUDA fallback detected in KODAMA.matrix: ", value$backend)
      }
    }
  }
}
for (name in names(embeddings)) {
  value <- embeddings[[name]]
  if (!is.null(value)) {
    stopifnot(all(dim(value) == c(nrow(x), 2L)), all(is.finite(value)))
  }
}

results <- do.call(rbind, rows)
row.names(results) <- NULL
results$n <- nrow(x)
results$p <- ncol(x)
results$classes <- n_classes
results$threads <- n_threads

csv_path <- file.path(output_dir, "kodama_metref_validation.csv")
rds_path <- file.path(output_dir, "kodama_metref_validation.rds")
plot_path <- file.path(output_dir, "kodama_metref_embeddings.png")
write.csv(results, csv_path, row.names = FALSE)
saveRDS(
  list(results = results, cv = cv, pca = pca, kodama_knn = km_knn,
       kodama_plslda = km_pls, embeddings = embeddings),
  rds_path, compress = FALSE
)

required_plots <- c("opentsne_cpu", "opentsne_cuda", "umap_cpu", "umap_cuda")
if (all(required_plots %in% names(embeddings)) &&
    all(vapply(embeddings[required_plots], Negate(is.null), logical(1)))) {
  colors <- grDevices::hcl.colors(n_classes, "Dynamic")[as.integer(labels)]
  grDevices::png(plot_path, width = 2200, height = 1800, res = 220)
  old <- par(mfrow = c(2, 2), mar = c(0.5, 0.5, 2.2, 0.5))
  for (name in required_plots) {
    title <- switch(
      name,
      opentsne_cpu = "KODAMA openTSNE CPU",
      opentsne_cuda = "KODAMA openTSNE CUDA",
      umap_cpu = "KODAMA UMAP CPU",
      umap_cuda = "KODAMA UMAP CUDA"
    )
    plot(
      embeddings[[name]], pch = 20, cex = 0.7, col = colors,
      axes = FALSE, ann = FALSE, frame.plot = FALSE
    )
    title(main = title, cex.main = 1.15)
  }
  par(old)
  grDevices::dev.off()
}

print(results)
failed <- results$status != "success"
if (any(failed)) {
  stop("MetRef validation failed for: ", paste(results$stage[failed], results$backend[failed], collapse = ", "))
}
cat("MetRef KODAMA CPU/CUDA validation OK\n")
cat("CSV: ", csv_path, "\n", sep = "")
cat("RDS: ", rds_path, "\n", sep = "")
cat("Plot: ", plot_path, "\n", sep = "")
