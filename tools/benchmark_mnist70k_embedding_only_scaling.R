#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit)) sub(prefix, "", hit[[length(hit)]], fixed = TRUE) else default
}

arg_flag <- function(name, default = FALSE) {
  value <- arg_value(name, NA_character_)
  if (is.na(value)) return(isTRUE(default))
  tolower(value) %in% c("1", "true", "yes", "y")
}

arg_int <- function(name, default) {
  value <- suppressWarnings(as.integer(arg_value(name, as.character(default))))
  if (length(value) == 1L && is.finite(value)) value else default
}

split_arg <- function(name, default) {
  value <- arg_value(name, default)
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

timed <- function(expr) {
  gc()
  t <- system.time(value <- force(expr))
  list(value = value, sec = unname(t[["elapsed"]]))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

load_dataset <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  if (exists("dataset", envir = env, inherits = FALSE)) {
    dataset <- get("dataset", envir = env)
  } else {
    objects <- mget(ls(env), env, inherits = FALSE)
    dataset <- objects[[1L]]
  }
  if (!is.list(dataset) || is.null(dataset$data)) {
    stop("Expected an RData object with $data and $labels: ", path, call. = FALSE)
  }
  dataset
}

load_pca_init <- function(path, data, backend, seed) {
  if (!is.na(path) && nzchar(path) && file.exists(path)) {
    env <- new.env(parent = emptyenv())
    load(path, envir = env)
    if (exists("pca_init", envir = env, inherits = FALSE)) {
      pca_init <- get("pca_init", envir = env)
      if (is.list(pca_init) && !is.null(pca_init$layout)) {
        return(as.matrix(pca_init$layout))
      }
      return(as.matrix(pca_init))
    }
  }
  fastEmbedR::opentsne_pca_init(data, seed = seed, backend = backend)
}

normalize_knn <- function(knn, k) {
  idx <- knn$indices
  dst <- knn$distances
  if (is.null(idx) || is.null(dst)) stop("KNN object must contain indices and distances.", call. = FALSE)
  idx <- as.matrix(idx)
  dst <- as.matrix(dst)
  if (ncol(idx) > k) {
    zero_based <- suppressWarnings(min(idx, na.rm = TRUE) == 0L)
    keep_idx <- matrix(NA_integer_, nrow(idx), k)
    keep_dst <- matrix(NA_real_, nrow(idx), k)
    for (i in seq_len(nrow(idx))) {
      self_id <- if (zero_based) i - 1L else i
      keep <- idx[i, ] != self_id
      kept_idx <- idx[i, keep]
      kept_dst <- dst[i, keep]
      if (length(kept_idx) < k) {
        kept_idx <- idx[i, ]
        kept_dst <- dst[i, ]
      }
      keep_idx[i, ] <- kept_idx[seq_len(k)]
      keep_dst[i, ] <- kept_dst[seq_len(k)]
    }
    idx <- keep_idx
    dst <- keep_dst
  }
  list(indices = idx[, seq_len(k), drop = FALSE], distances = dst[, seq_len(k), drop = FALSE])
}

machine_specs <- function() {
  sys <- Sys.info()
  cpu <- tryCatch({
    if (sys[["sysname"]] == "Darwin") {
      system2("sysctl", c("-n", "machdep.cpu.brand_string"), stdout = TRUE, stderr = FALSE)[1L]
    } else if (file.exists("/proc/cpuinfo")) {
      line <- grep("^model name", readLines("/proc/cpuinfo", warn = FALSE), value = TRUE)[1L]
      sub("^model name[[:space:]]*:[[:space:]]*", "", line)
    } else {
      NA_character_
    }
  }, error = function(e) NA_character_)
  gpu <- tryCatch(
    system2("nvidia-smi", c("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"), stdout = TRUE, stderr = FALSE)[1L],
    error = function(e) NA_character_
  )
  data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    hostname = unname(sys[["nodename"]]),
    system = paste(unname(sys[["sysname"]]), unname(sys[["release"]])),
    cpu = cpu,
    gpu = gpu,
    logical_cores = parallel::detectCores(logical = TRUE),
    r_version = as.character(getRversion()),
    fastEmbedR_version = as.character(utils::packageVersion("fastEmbedR")),
    faissR_version = if (requireNamespace("faissR", quietly = TRUE)) as.character(utils::packageVersion("faissR")) else NA_character_,
    stringsAsFactors = FALSE
  )
}

suppressPackageStartupMessages({
  library(fastEmbedR)
})

data_path <- arg_value("data", "/Users/stefano/Documents/fastEmbedR/Data/MNIST/MNIST.RData")
pca_path <- arg_value("pca-init", "/Users/stefano/Documents/fastEmbedR/Data/MNIST/MNIST_fastPLS_pca2_init.RData")
out_dir <- arg_value("out-dir", file.path("results", paste0("mnist70k_embedding_only_scaling_", format(Sys.time(), "%Y%m%d_%H%M%S"))))
machine_label <- arg_value("machine", Sys.info()[["nodename"]])
cache_dir <- arg_value("cache-dir", file.path(out_dir, "cache"))
k <- arg_int("k", 15L)
perplexity <- as.numeric(arg_value("perplexity", as.character(k)))
seed <- arg_int("seed", 4L)
threads <- as.integer(split_arg("threads", "1,2,4"))
backends <- split_arg("backends", "cpu,metal")
force_recompute <- arg_flag("force-recompute", FALSE)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

dataset <- load_dataset(data_path)
x <- dataset$data
if (!is.matrix(x)) x <- as.matrix(x)
labels <- if (!is.null(dataset$labels)) factor(dataset$labels) else factor(rep(1L, nrow(x)))

knn_cache <- file.path(cache_dir, sprintf("MNIST_k%d_knn.rds", k))
pca_cache <- file.path(cache_dir, "MNIST_pca_init.rds")

if (!file.exists(knn_cache) || force_recompute) {
  if (!requireNamespace("faissR", quietly = TRUE)) {
    stop("faissR is required to create the KNN cache.", call. = FALSE)
  }
  message("Computing KNN cache outside timed embedding section.")
  nn_formals <- names(formals(faissR::nn))
  if ("exclude_self" %in% nn_formals) {
    knn_run <- timed(faissR::nn(x, k = k, exclude_self = TRUE))
  } else {
    knn_run <- timed(faissR::nn(x, k = k + 1L))
  }
  knn <- normalize_knn(knn_run$value, k)
  saveRDS(list(knn = knn, knn_sec = knn_run$sec), knn_cache, version = 2)
} else {
  message("Using KNN cache: ", knn_cache)
}
knn_obj <- readRDS(knn_cache)
knn <- knn_obj$knn %||% normalize_knn(knn_obj, k)
knn_sec <- knn_obj$knn_sec %||% NA_real_

if (!file.exists(pca_cache) || force_recompute) {
  message("Creating PCA initialization outside timed embedding section.")
  pca_run <- timed(load_pca_init(pca_path, x, backend = "cpu", seed = seed))
  saveRDS(list(Y_init = as.matrix(pca_run$value), pca_sec = pca_run$sec), pca_cache, version = 2)
} else {
  message("Using PCA initialization cache: ", pca_cache)
}
pca_obj <- readRDS(pca_cache)
Y_init <- as.matrix(pca_obj$Y_init %||% pca_obj)
pca_sec <- pca_obj$pca_sec %||% NA_real_

run_embedding <- function(backend, n_threads) {
  Sys.setenv(
    OMP_NUM_THREADS = as.character(n_threads),
    OPENBLAS_NUM_THREADS = as.character(n_threads),
    MKL_NUM_THREADS = as.character(n_threads),
    RCPP_PARALLEL_NUM_THREADS = as.character(n_threads)
  )
  message("Embedding-only run: backend=", backend, " threads=", n_threads)
  ans <- tryCatch({
    run <- timed(fastEmbedR::opentsne_knn(
      knn$indices,
      knn$distances,
      perplexity = perplexity,
      Y_init = Y_init,
      seed = seed,
      backend = backend,
      n_threads = n_threads,
      verbose = FALSE
    ))
    cfg <- attr(run$value, "fastEmbedR_config")
    data.frame(
      machine = machine_label,
      method = "fastEmbedR openTSNE embedding-only",
      backend = backend,
      threads = n_threads,
      n = nrow(x),
      p = ncol(x),
      k = k,
      perplexity = perplexity,
      embedding_sec = run$sec,
      knn_sec_excluded = knn_sec,
      pca_sec_excluded = pca_sec,
      optimizer = cfg$optimizer %||% NA_character_,
      repulsion = cfg$repulsion %||% NA_character_,
      status = "success",
      error = NA_character_,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      machine = machine_label,
      method = "fastEmbedR openTSNE embedding-only",
      backend = backend,
      threads = n_threads,
      n = nrow(x),
      p = ncol(x),
      k = k,
      perplexity = perplexity,
      embedding_sec = NA_real_,
      knn_sec_excluded = knn_sec,
      pca_sec_excluded = pca_sec,
      optimizer = NA_character_,
      repulsion = NA_character_,
      status = "failed",
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  ans
}

rows <- list()
for (nt in threads) {
  rows[[length(rows) + 1L]] <- run_embedding("cpu", nt)
}
for (backend in setdiff(backends, "cpu")) {
  rows[[length(rows) + 1L]] <- run_embedding(backend, max(threads, na.rm = TRUE))
}

results <- do.call(rbind, rows)
utils::write.csv(results, file.path(out_dir, "embedding_only_scaling.csv"), row.names = FALSE)
utils::write.csv(machine_specs(), file.path(out_dir, "machine_specs.csv"), row.names = FALSE)

plot_path <- file.path(out_dir, "embedding_only_scaling.png")
ok <- results[results$status == "success" & is.finite(results$embedding_sec), , drop = FALSE]
if (nrow(ok)) {
  label <- ifelse(ok$backend == "cpu", paste0("CPU\n", ok$threads, " core", ifelse(ok$threads == 1L, "", "s")), toupper(ok$backend))
  cols <- ifelse(ok$backend == "cpu", "#4C78A8", ifelse(ok$backend == "metal", "#F58518", "#54A24B"))
  png(plot_path, width = 1300, height = 850, res = 150)
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mar = c(5.5, 5, 4, 1))
  bp <- barplot(
    ok$embedding_sec,
    names.arg = label,
    col = cols,
    border = NA,
    ylab = "Embedding time only (seconds)",
    main = sprintf("MNIST70k openTSNE embedding-only: %s", machine_label),
    ylim = c(0, max(ok$embedding_sec) * 1.2)
  )
  text(bp, ok$embedding_sec, labels = sprintf("%.2fs", ok$embedding_sec), pos = 3, cex = 0.9)
  legend("topright", fill = unique(cols), legend = unique(ifelse(ok$backend == "cpu", "CPU", toupper(ok$backend))), bty = "n")
  dev.off()
}

print(results)
message("Wrote: ", normalizePath(file.path(out_dir, "embedding_only_scaling.csv"), mustWork = FALSE))
message("Wrote: ", normalizePath(plot_path, mustWork = FALSE))
