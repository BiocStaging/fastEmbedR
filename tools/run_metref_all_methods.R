#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(fastEmbedR)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[1L]], fixed = TRUE) else default
}

data_file <- get_arg("data", "/Users/stefano/Documents/fastEmbedR/Data/MetRef/MetRef.RData")
pca_file <- get_arg("pca", "/Users/stefano/Documents/fastEmbedR/Data/MetRef/MetRef_fastPLS_pca2_init.RData")
out_dir <- get_arg("out-dir", file.path("results", paste0("metref_all_methods_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
threads <- as.integer(get_arg("threads", "4"))
seed <- as.integer(get_arg("seed", "4"))
perplexity <- as.numeric(get_arg("perplexity", "15"))
k_umap <- as.integer(get_arg("k", "30"))
point_cex <- as.numeric(get_arg("cex", "0.7"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "layouts"), recursive = TRUE, showWarnings = FALSE)

load(data_file)
dataset <- get("dataset")
x <- as.matrix(dataset$data)
storage.mode(x) <- "double"
labels <- as.factor(dataset$labels)

Y_init <- NULL
if (file.exists(pca_file)) {
  load(pca_file)
  if (exists("pca_init") && is.list(pca_init) && !is.null(pca_init$layout)) {
    Y_init <- as.matrix(pca_init$layout)
  }
}

set.seed(seed)
knn <- NULL
knn_sec <- NA_real_
if (requireNamespace("faissR", quietly = TRUE)) {
  run_faiss_knn <- function() {
    nn_args <- names(formals(faissR::nn))
    if ("exclude_self" %in% nn_args) {
      return(faissR::nn(x, k = k_umap, backend = "cpu", metric = "euclidean",
                        exclude_self = TRUE, n_threads = threads))
    }
    if ("nn_without_self" %in% getNamespaceExports("faissR")) {
      return(faissR::nn_without_self(x, k = k_umap, backend = "cpu",
                                     metric = "euclidean", n_threads = threads))
    }
    raw <- faissR::nn(x, k = k_umap + 1L, backend = "cpu",
                      metric = "euclidean", n_threads = threads)
    raw$indices <- raw$indices[, -1L, drop = FALSE]
    raw$distances <- raw$distances[, -1L, drop = FALSE]
    raw
  }
  knn_sec <- system.time({
    knn <- run_faiss_knn()
  })[["elapsed"]]
}

layout_matrix <- function(z) {
  if (is.list(z) && !is.null(z$layout)) z <- z$layout
  if (is.list(z) && !is.null(z$Y)) z <- z$Y
  z <- as.matrix(z)
  z[, 1:2, drop = FALSE]
}

safe_eval <- function(expr) {
  gc()
  t <- system.time({
    value <- tryCatch(force(expr), error = function(e) e)
  })[["elapsed"]]
  list(value = value, sec = as.numeric(t))
}

available_fft <- function() {
  # Spectre's bundled fftRtsne can abort the local R process on this Mac because
  # it initializes a second OpenMP runtime. Use the dedicated fftRtsne wrapper
  # only; unavailable FIt-SNE is reported as a failed row without stopping.
  for (package in c("fftRtsne")) {
    if (requireNamespace(package, quietly = TRUE) &&
        exists("fftRtsne", envir = asNamespace(package), inherits = FALSE)) {
      return(get("fftRtsne", envir = asNamespace(package), inherits = FALSE))
    }
  }
  NULL
}

run_specs <- list(
  list(
    method = "fastEmbedR openTSNE CPU",
    backend = "cpu",
    uses_precomputed_knn = TRUE,
    run = function() fastEmbedR::opentsne_knn(knn, perplexity = perplexity,
                                             Y_init = Y_init, backend = "cpu",
                                             n_threads = threads, seed = seed)
  ),
  list(
    method = "fastEmbedR openTSNE Metal",
    backend = "metal",
    uses_precomputed_knn = TRUE,
    run = function() fastEmbedR::opentsne_knn(knn, perplexity = perplexity,
                                             Y_init = Y_init, backend = "metal",
                                             n_threads = threads, seed = seed)
  ),
  list(
    method = "Rtsne full",
    backend = "cpu",
    uses_precomputed_knn = FALSE,
    run = function() {
      if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne is not installed.")
      Rtsne::Rtsne(x, perplexity = perplexity, check_duplicates = FALSE,
                   pca = TRUE, num_threads = threads, verbose = FALSE)$Y
    }
  ),
  list(
    method = "tsne package",
    backend = "cpu",
    uses_precomputed_knn = FALSE,
    run = function() {
      if (!requireNamespace("tsne", quietly = TRUE)) stop("tsne is not installed.")
      tsne::tsne(x, k = 2L, perplexity = perplexity, max_iter = 1000L)
    }
  ),
  list(
    method = "KlugerLab FIt-SNE",
    backend = "cpu_fft",
    uses_precomputed_knn = FALSE,
    run = function() {
      fun <- available_fft()
      if (is.null(fun)) stop("No fftRtsne/Spectre FIt-SNE wrapper installed.")
      f <- names(formals(fun))
      call_args <- list(
        X = x,
        dims = 2L,
        perplexity = perplexity,
        max_iter = 1000L,
        rand_seed = seed,
        nthreads = threads,
        verbose = FALSE
      )
      if (!is.null(Y_init)) {
        if ("Y_init" %in% f) call_args$Y_init <- Y_init
        if ("initial_config" %in% f) call_args$initial_config <- Y_init
      }
      do.call(fun, call_args[names(call_args) %in% f])
    }
  ),
  list(
    method = "fastEmbedR UMAP CPU fuzzy",
    backend = "cpu",
    uses_precomputed_knn = TRUE,
    run = function() fastEmbedR::umap_knn(knn, backend = "cpu",
                                         graph_mode = "fuzzy",
                                         n_threads = threads, seed = seed)
  ),
  list(
    method = "fastEmbedR UMAP CPU binary",
    backend = "cpu",
    uses_precomputed_knn = TRUE,
    run = function() fastEmbedR::umap_knn(knn, backend = "cpu",
                                         graph_mode = "binary",
                                         n_threads = threads, seed = seed)
  ),
  list(
    method = "fastEmbedR UMAP Metal fuzzy",
    backend = "metal",
    uses_precomputed_knn = TRUE,
    run = function() fastEmbedR::umap_knn(knn, backend = "metal",
                                         graph_mode = "fuzzy",
                                         n_threads = threads, seed = seed)
  ),
  list(
    method = "fastEmbedR UMAP Metal binary",
    backend = "metal",
    uses_precomputed_knn = TRUE,
    run = function() fastEmbedR::umap_knn(knn, backend = "metal",
                                         graph_mode = "binary",
                                         n_threads = threads, seed = seed)
  ),
  list(
    method = "umap package",
    backend = "cpu",
    uses_precomputed_knn = FALSE,
    run = function() {
      if (!requireNamespace("umap", quietly = TRUE)) stop("umap is not installed.")
      cfg <- umap::umap.defaults
      cfg$n_neighbors <- k_umap
      umap::umap(x, config = cfg)$layout
    }
  ),
  list(
    method = "uwot default",
    backend = "cpu",
    uses_precomputed_knn = FALSE,
    run = function() {
      if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.")
      uwot::umap(x, n_neighbors = k_umap, n_threads = threads,
                 n_sgd_threads = 1, fast_sgd = FALSE, verbose = FALSE)
    }
  ),
  list(
    method = "uwot fast_sgd",
    backend = "cpu",
    uses_precomputed_knn = FALSE,
    run = function() {
      if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.")
      uwot::umap(x, n_neighbors = k_umap, n_threads = threads,
                 n_sgd_threads = threads, fast_sgd = TRUE, verbose = FALSE)
    }
  )
)

rows <- list()
layouts <- list()

for (spec in run_specs) {
  message("Running ", spec$method)
  measured <- safe_eval(spec$run())
  status <- "success"
  error <- NA_character_
  layout <- NULL
  if (inherits(measured$value, "error")) {
    status <- "failed"
    error <- conditionMessage(measured$value)
  } else {
    layout <- layout_matrix(measured$value)
    layouts[[spec$method]] <- layout
    saveRDS(list(layout = layout, labels = labels, method = spec$method),
            file.path(out_dir, "layouts", paste0(gsub("[^A-Za-z0-9]+", "_", spec$method), ".rds")))
  }
  rows[[length(rows) + 1L]] <- data.frame(
    dataset = "MetRef",
    method = spec$method,
    backend = spec$backend,
    status = status,
    n = nrow(x),
    p = ncol(x),
    k = k_umap,
    perplexity = perplexity,
    uses_precomputed_knn = spec$uses_precomputed_knn,
    knn_sec = if (isTRUE(spec$uses_precomputed_knn)) knn_sec else NA_real_,
    embed_sec = measured$sec,
    error = error,
    stringsAsFactors = FALSE
  )
}

results <- do.call(rbind, rows)
utils::write.csv(results, file.path(out_dir, "metref_all_methods_results.csv"), row.names = FALSE)

plot_one <- function(layout, title, path) {
  png(path, width = 1500, height = 1300, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(2, 2, 3, 1), bg = "white")
  pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
  plot(layout[, 1], layout[, 2], pch = 16, cex = point_cex,
       col = pal[as.integer(labels)], axes = FALSE, xlab = "", ylab = "",
       main = title)
  box(col = "grey75")
}

for (nm in names(layouts)) {
  plot_one(layouts[[nm]], nm, file.path(out_dir, "plots", paste0(gsub("[^A-Za-z0-9]+", "_", nm), ".png")))
}

panel_file <- file.path(out_dir, "metref_all_methods_panel.png")
png(panel_file, width = 3000, height = 2400, res = 180)
par(mfrow = c(3, 4), mar = c(1, 1, 3, 1), bg = "white")
pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
for (nm in names(layouts)) {
  layout <- layouts[[nm]]
  sec <- results$embed_sec[match(nm, results$method)]
  plot(layout[, 1], layout[, 2], pch = 16, cex = point_cex,
       col = pal[as.integer(labels)], axes = FALSE, xlab = "", ylab = "",
       main = sprintf("%s\n%.3fs", nm, sec))
  box(col = "grey80")
}
dev.off()

bar_file <- file.path(out_dir, "metref_all_methods_time_barplot.png")
ok <- results[results$status == "success", , drop = FALSE]
png(bar_file, width = 2200, height = 1400, res = 180)
par(mar = c(10, 5, 3, 1), bg = "white")
cols <- ifelse(grepl("Metal", ok$method), "#E69F00",
               ifelse(grepl("fastEmbedR", ok$method), "#0072B2", "#999999"))
barplot(ok$embed_sec, names.arg = ok$method, las = 2, cex.names = 0.65,
        col = cols, ylab = "Embedding seconds",
        main = "MetRef runtime by method")
dev.off()

message("Wrote results to: ", normalizePath(out_dir, mustWork = FALSE))
message("Panel: ", normalizePath(panel_file, mustWork = FALSE))
message("Barplot: ", normalizePath(bar_file, mustWork = FALSE))
print(results)
