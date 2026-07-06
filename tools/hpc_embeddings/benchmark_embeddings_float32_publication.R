#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[kv[[1L]]]] <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || (length(x) == 1L && is.na(x))) y else x
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L] %||% ""),
  mustWork = FALSE
)
if (!nzchar(script_path) || !file.exists(script_path)) {
  script_path <- normalizePath(args$script %||% "benchmark_embeddings_float32_publication.R", mustWork = FALSE)
}

bool_arg <- function(name, default = FALSE) {
  value <- args[[name]]
  if (is.null(value)) return(default)
  isTRUE(as.logical(value))
}

int_arg <- function(name, default) {
  value <- suppressWarnings(as.integer(args[[name]] %||% default))
  if (length(value) != 1L || is.na(value)) as.integer(default) else value
}

num_arg <- function(name, default) {
  value <- suppressWarnings(as.numeric(args[[name]] %||% default))
  if (length(value) != 1L || is.na(value)) as.numeric(default) else value
}

csv_arg <- function(name, default) {
  value <- args[[name]] %||% default
  x <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
  x[nzchar(x)]
}

base_dir <- normalizePath(args$base_dir %||% "/scratch/firenze/NN", mustWork = FALSE)
data_root <- normalizePath(args$data_root %||% file.path(base_dir, "Data"), mustWork = FALSE)
out_dir <- normalizePath(args$out_dir %||% file.path(base_dir, "benchmark_embeddings_float32_publication"), mustWork = FALSE)
threads <- int_arg("threads", 12L)
timeout <- int_arg("timeout", 10800L)
seed <- int_arg("seed", 4L)
k <- int_arg("k", 30L)
perplexity <- num_arg("perplexity", 15)
backend_group <- args$backend_group %||% "cpu"
force <- bool_arg("force", FALSE)
worker <- bool_arg("worker", FALSE)
thread_grid <- csv_arg("thread_grid", as.character(threads))
thread_grid <- suppressWarnings(as.integer(thread_grid))
thread_grid <- unique(thread_grid[!is.na(thread_grid) & thread_grid > 0L])
if (!length(thread_grid)) thread_grid <- threads
if (identical(backend_group, "cuda") || worker) {
  thread_grid <- threads
}

datasets <- csv_arg(
  "datasets",
  "COIL20,USPS,FashionMNIST,FlowRepository_FR-FCM-ZYRM_files,flow18,MNIST,imagenet,MetRef,mass41,TabulaMuris"
)

default_methods <- if (identical(backend_group, "cuda")) {
  paste(
    "fastEmbedR_opentsne_cuda",
    "fastEmbedR_umap_cuda_fuzzy",
    "fastEmbedR_umap_cuda_binary",
    "rapids_cuml_umap_full",
    "rapids_cuml_tsne_full",
    "rapids_cuml_umap_full_direct",
    "rapids_cuml_tsne_full_direct",
    sep = ","
  )
} else {
  paste(
    "fastEmbedR_opentsne_cpu",
    "fastEmbedR_umap_cpu_fuzzy",
    "fastEmbedR_umap_cpu_binary",
    "Rtsne_full",
    "KlugerLab_FItSNE",
    "python_opentsne_fft",
    "python_opentsne_fft_direct",
    "umap_package",
    "uwot_default",
    "uwot_fast_sgd",
    "python_umap_learn",
    "python_umap_learn_direct",
    sep = ","
  )
}
methods <- csv_arg("methods", default_methods)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "logs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "layouts"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "worker_results"), recursive = TRUE, showWarnings = FALSE)

Sys.setenv(
  OMP_NUM_THREADS = as.character(threads),
  OPENBLAS_NUM_THREADS = as.character(threads),
  MKL_NUM_THREADS = as.character(threads),
  VECLIB_MAXIMUM_THREADS = as.character(threads),
  RCPP_PARALLEL_NUM_THREADS = as.character(threads)
)

log_file <- file.path(out_dir, "benchmark.log")
log_msg <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))
  cat(line, "\n")
  cat(line, "\n", file = log_file, append = TRUE)
  flush.console()
}

find_source_rdata <- function(dataset) {
  folder <- file.path(data_root, dataset)
  hits <- list.files(folder, pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
  hits <- hits[!grepl(
    "float32|_nn|pca|manifest|summary|backup|faissR|reference|worker|benchmark",
    basename(hits),
    ignore.case = TRUE
  )]
  if (!length(hits)) return(NA_character_)
  exact <- hits[tolower(tools::file_path_sans_ext(basename(hits))) == tolower(dataset)]
  if (length(exact)) return(exact[[1L]])
  hits <- hits[order(nchar(basename(hits)), basename(hits))]
  hits[[1L]]
}

find_float_rdata <- function(dataset) {
  folder <- file.path(data_root, dataset)
  hits <- list.files(folder, pattern = "float32.*\\.[Rr][Dd]ata$", full.names = TRUE, ignore.case = TRUE)
  if (!length(hits)) stop("No float32 RData found for ", dataset, call. = FALSE)
  hits[[1L]]
}

pick_dataset_object <- function(path) {
  env <- new.env(parent = emptyenv())
  object_names <- load(path, envir = env)
  objects <- mget(object_names, env, inherits = FALSE)
  for (nm in names(objects)) {
    obj <- objects[[nm]]
    if (is.list(obj) && !is.null(obj$data)) {
      return(list(data = obj$data, labels = obj$labels %||% NULL, object_name = nm))
    }
  }
  for (nm in names(objects)) {
    obj <- objects[[nm]]
    if (is.matrix(obj) || is.data.frame(obj) || inherits(obj, "Matrix") || inherits(obj, "float32")) {
      labels <- NULL
      for (candidate in c("labels", "label", "Y", "y", "classes", "class")) {
        if (exists(candidate, envir = env, inherits = FALSE)) {
          lab <- get(candidate, envir = env, inherits = FALSE)
          if (length(lab) == nrow(obj)) labels <- lab
        }
      }
      return(list(data = obj, labels = labels, object_name = nm))
    }
  }
  stop("Could not find a data matrix/list in ", path, call. = FALSE)
}

as_double_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) stop("float package is required.", call. = FALSE)
    x <- float::dbl(x)
  }
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

as_fastembedr_float_input <- function(x) {
  if (inherits(x, "float32")) return(x)
  if (!requireNamespace("float", quietly = TRUE)) {
    stop("The float package is required for fastEmbedR float32 benchmark input.", call. = FALSE)
  }
  float::fl(as_double_matrix(x))
}

layout_matrix <- function(x) {
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) stop("float package required to coerce layout.", call. = FALSE)
    x <- float::dbl(x)
  }
  y <- as.matrix(x)
  y[, 1:2, drop = FALSE]
}

sample_for_metrics <- function(n, size, seed) {
  if (n <= size) return(seq_len(n))
  set.seed(seed)
  sort(sample.int(n, size))
}

empty_score_record <- function() {
  list(
    trustworthiness = NA_real_,
    knn_preservation = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    silhouette = NA_real_,
    label_knn_accuracy = NA_real_,
    quality_sample_n = NA_integer_
  )
}

score_layout <- function(x_standard, layout, labels) {
  out <- empty_score_record()
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) return(out)
  rows <- sample_for_metrics(nrow(layout), min(5000L, nrow(layout)), seed + 19L)
  out$quality_sample_n <- length(rows)
  score <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x_standard[rows, , drop = FALSE],
      layout[rows, , drop = FALSE],
      labels = if (is.null(labels)) NULL else labels[rows],
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(3000L, length(rows)),
      sample_size_for_local_metrics = min(3000L, length(rows)),
      seed = seed,
      n_threads = threads,
      dataset = args$dataset %||% NA_character_
    ),
    error = function(e) e
  )
  if (!inherits(score, "error")) {
    out$trustworthiness <- as.numeric(score$trustworthiness %||% NA_real_)
    out$knn_preservation_15 <- as.numeric(score$knn_preservation_15 %||% NA_real_)
    out$knn_preservation_30 <- as.numeric(score$knn_preservation_30 %||% score$knn_preservation %||% NA_real_)
    out$knn_preservation_50 <- as.numeric(score$knn_preservation_50 %||% NA_real_)
    out$knn_preservation <- out$knn_preservation_30
    out$silhouette <- as.numeric(score$silhouette %||% NA_real_)
    out$label_knn_accuracy <- as.numeric(score$nn_accuracy %||% score$label_knn_accuracy %||% NA_real_)
  }
  out
}

plot_layout <- function(layout, labels, path, title) {
  png(path, width = 1800, height = 1400, res = 180)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(2, 2, 3, 1))
  if (is.null(labels)) {
    plot(layout[, 1], layout[, 2], pch = 16, cex = 0.28, col = "#1f77b4",
         axes = FALSE, xlab = "", ylab = "", main = title)
  } else {
    labels <- as.factor(labels)
    pal <- grDevices::hcl.colors(nlevels(labels), "Dark 3")
    plot(layout[, 1], layout[, 2], pch = 16, cex = 0.28, col = pal[as.integer(labels)],
         axes = FALSE, xlab = "", ylab = "", main = title)
  }
  box(col = "grey70")
}

fast_tsne_path <- function() {
  candidates <- c(
    Sys.getenv("FASTEMBEDR_FAST_TSNE_PATH", ""),
    Sys.getenv("FAST_TSNE_PATH", ""),
    "/opt/fit-sne/bin/fast_tsne",
    "/mnt/sata_ssd/FIt-SNE/bin/fast_tsne",
    file.path(Sys.getenv("HOME"), ".local", "bin", "fast_tsne"),
    Sys.which("fast_tsne")
  )
  candidates <- candidates[nzchar(candidates)]
  candidates <- candidates[file.exists(candidates) & file.access(candidates, 1L) == 0L]
  if (length(candidates)) normalizePath(candidates[[1L]], mustWork = FALSE) else ""
}

fitsne_wrapper <- function() {
  for (package in c("fftRtsne", "Spectre")) {
    if (requireNamespace(package, quietly = TRUE) &&
        exists("fftRtsne", envir = asNamespace(package), inherits = FALSE)) {
      return(list(name = package, fun = get("fftRtsne", envir = asNamespace(package), inherits = FALSE)))
    }
  }
  wrappers <- c(
    Sys.getenv("FASTEMBEDR_FAST_TSNE_R", ""),
    Sys.getenv("FAST_TSNE_R", ""),
    "/opt/fit-sne/bin/fast_tsne.R",
    "/mnt/sata_ssd/FIt-SNE/fast_tsne.R",
    "/mnt/sata_ssd/FIt-SNE/bin/fast_tsne.R"
  )
  wrappers <- wrappers[nzchar(wrappers) & file.exists(wrappers)]
  if (!length(wrappers)) return(NULL)
  env <- new.env(parent = .GlobalEnv)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(dirname(wrappers[[1L]]))
  source(wrappers[[1L]], local = env, chdir = FALSE)
  setwd(old_wd)
  if (!exists("fftRtsne", envir = env, inherits = FALSE)) return(NULL)
  fun <- get("fftRtsne", envir = env, inherits = FALSE)
  list(name = "KlugerLab_FItSNE_source", fun = fun, exe = fast_tsne_path())
}

run_fitsne <- function(x, y_init = NULL) {
  x <- as_double_matrix(x)
  wrapper <- fitsne_wrapper()
  exe <- fast_tsne_path()
  if (!nzchar(exe)) stop("fast_tsne executable not found.", call. = FALSE)
  if (is.null(wrapper)) stop("FIt-SNE R wrapper not found.", call. = FALSE)
  formals_names <- names(formals(wrapper$fun))
  call_args <- list(
    X = x,
    dims = 2L,
    perplexity = perplexity,
    max_iter = 750L,
    rand_seed = seed,
    theta = 0.5,
    nthreads = threads,
    fast_tsne_path = exe,
    verbose = FALSE
  )
  if (!is.null(y_init)) {
    if ("Y_init" %in% formals_names) call_args$Y_init <- y_init
    if ("initial_config" %in% formals_names) call_args$initial_config <- y_init
    if ("init" %in% formals_names) call_args$init <- y_init
    if ("initialization" %in% formals_names) call_args$initialization <- y_init
  }
  call_args <- call_args[names(call_args) %in% formals_names]
  do.call(wrapper$fun, call_args)
}

ensure_reticulate <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate is not installed.", call. = FALSE)
  }
  py <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(py)) reticulate::use_python(py, required = FALSE)
  invisible(TRUE)
}

numpy_float32 <- function(x) {
  ensure_reticulate()
  np <- reticulate::import("numpy", convert = FALSE)
  np$array(as_double_matrix(x), dtype = "float32", order = "C")
}

py_array_to_matrix <- function(x) {
  ensure_reticulate()
  np <- reticulate::import("numpy", convert = FALSE)
  if (inherits(x, "python.builtin.object")) {
    # cuML commonly returns a CuPy/cuDF-backed object. Try GPU-to-host conversion
    # only at the final embedding boundary.
    as_numpy <- tryCatch(x$to_numpy(), error = function(e) NULL)
    if (!is.null(as_numpy)) x <- as_numpy
    cp <- tryCatch(reticulate::import("cupy", convert = FALSE), error = function(e) NULL)
    if (!is.null(cp)) {
      host <- tryCatch(cp$asnumpy(x), error = function(e) NULL)
      if (!is.null(host)) x <- host
    }
    x <- tryCatch(np$asarray(x), error = function(e) x)
  }
  y <- reticulate::py_to_r(x)
  y <- as.matrix(y)
  storage.mode(y) <- "double"
  y[, 1:2, drop = FALSE]
}

run_python_umap_learn <- function(x) {
  ensure_reticulate()
  umap_py <- reticulate::import("umap", convert = FALSE)
  x_np <- numpy_float32(x)
  model <- umap_py$UMAP(
    n_neighbors = as.integer(k),
    n_components = 2L,
    metric = "euclidean",
    random_state = as.integer(seed),
    n_jobs = as.integer(threads),
    verbose = FALSE
  )
  py_array_to_matrix(model$fit_transform(x_np))
}

run_python_opentsne_fft <- function(x) {
  ensure_reticulate()
  openTSNE <- reticulate::import("openTSNE", convert = FALSE)
  x_np <- numpy_float32(x)
  model <- openTSNE$TSNE(
    n_components = 2L,
    perplexity = as.numeric(perplexity),
    initialization = "pca",
    negative_gradient_method = "fft",
    n_iter = 500L,
    early_exaggeration_iter = 250L,
    random_state = as.integer(seed),
    n_jobs = as.integer(threads),
    verbose = FALSE
  )
  py_array_to_matrix(model$fit(x_np))
}

run_rapids_cuml_umap <- function(x) {
  ensure_reticulate()
  cuml_umap <- reticulate::import("cuml.manifold", convert = FALSE)$UMAP
  x_np <- numpy_float32(x)
  model <- cuml_umap(
    n_neighbors = as.integer(k),
    n_components = 2L,
    metric = "euclidean",
    random_state = as.integer(seed),
    verbose = FALSE
  )
  py_array_to_matrix(model$fit_transform(x_np))
}

run_rapids_cuml_tsne <- function(x) {
  ensure_reticulate()
  cuml_tsne <- reticulate::import("cuml.manifold", convert = FALSE)$TSNE
  x_np <- numpy_float32(x)
  base_args <- list(
    n_components = 2L,
    perplexity = as.numeric(perplexity),
    random_state = as.integer(seed),
    verbose = FALSE
  )
  # RAPIDS/cuML TSNE signatures have changed across releases. Try the modern
  # full-embedding call first, then fall back to a minimal compatible call.
  model <- tryCatch(
    do.call(cuml_tsne, c(base_args, list(method = "fft"))),
    error = function(e) do.call(cuml_tsne, base_args)
  )
  py_array_to_matrix(model$fit_transform(x_np))
}

is_direct_python_method <- function(method) {
  method %in% c(
    "python_opentsne_fft_direct",
    "python_umap_learn_direct",
    "rapids_cuml_umap_full_direct",
    "rapids_cuml_tsne_full_direct"
  )
}

python_direct_helper <- function() {
  candidates <- c(
    file.path(dirname(script_path), "benchmark_python_direct.py"),
    file.path(getwd(), "tools", "hpc_embeddings", "benchmark_python_direct.py"),
    file.path(base_dir, "benchmark_python_direct.py")
  )
  candidates <- candidates[file.exists(candidates)]
  if (!length(candidates)) {
    stop("Cannot find benchmark_python_direct.py next to the benchmark R script.", call. = FALSE)
  }
  normalizePath(candidates[[1L]], mustWork = TRUE)
}

python_executable <- function() {
  py <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(py) && file.exists(py)) return(py)
  py <- Sys.which("python")
  if (nzchar(py)) return(py)
  py <- Sys.which("python3")
  if (nzchar(py)) return(py)
  stop("Cannot find a Python executable for direct Python benchmark methods.", call. = FALSE)
}

prepare_direct_python_npz <- function(dataset, x_ref) {
  if (is.null(x_ref)) {
    stop("Standard R dataset is required for direct Python benchmark methods.", call. = FALSE)
  }
  ensure_reticulate()
  np <- reticulate::import("numpy", convert = FALSE)
  input_dir <- file.path(out_dir, "python_inputs")
  dir.create(input_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(input_dir, paste0(dataset, "_float32_for_python.npz"))
  if (!file.exists(path)) {
    x_np <- np$array(as_double_matrix(x_ref), dtype = "float32", order = "C")
    np$savez_compressed(path, x = x_np)
  }
  path
}

read_direct_python_json <- function(path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required to read direct Python benchmark metadata.", call. = FALSE)
  }
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

run_python_direct_method <- function(method, npz_path) {
  if (is.null(npz_path) || !nzchar(npz_path) || !file.exists(npz_path)) {
    stop("Direct Python benchmark requires a pre-exported NPZ input.", call. = FALSE)
  }
  helper <- python_direct_helper()
  py <- python_executable()
  direct_dir <- file.path(out_dir, "python_direct")
  dir.create(direct_dir, recursive = TRUE, showWarnings = FALSE)
  stem <- paste0(args$dataset %||% "dataset", "_", method, "_threads", threads, "_seed", seed)
  layout_csv <- file.path(direct_dir, paste0(stem, "_layout.csv"))
  meta_json <- file.path(direct_dir, paste0(stem, "_metadata.json"))
  cmd_args <- c(
    helper,
    paste0("--method=", method),
    paste0("--input=", npz_path),
    paste0("--layout=", layout_csv),
    paste0("--json=", meta_json),
    paste0("--k=", k),
    paste0("--perplexity=", perplexity),
    paste0("--threads=", threads),
    paste0("--seed=", seed)
  )
  process_time <- system.time({
    status <- system2(py, cmd_args, stdout = TRUE, stderr = TRUE)
  })
  process_elapsed <- unname(process_time[["elapsed"]])
  if (!file.exists(meta_json)) {
    stop("Direct Python benchmark did not write metadata. Output: ",
         paste(status, collapse = "\n"), call. = FALSE)
  }
  meta <- read_direct_python_json(meta_json)
  if (!identical(meta$status, "success")) {
    stop("Direct Python benchmark failed: ", meta$error %||% "unknown error",
         call. = FALSE)
  }
  if (!file.exists(layout_csv)) {
    stop("Direct Python benchmark did not write a layout CSV.", call. = FALSE)
  }
  layout <- as.matrix(utils::read.csv(layout_csv, header = FALSE))
  storage.mode(layout) <- "double"
  layout <- layout[, 1:2, drop = FALSE]
  attr(layout, "python_fit_sec") <- as.numeric(meta$python_fit_sec %||% NA_real_)
  attr(layout, "process_elapsed_sec") <- process_elapsed
  attr(layout, "python_executable") <- as.character(meta$python_executable %||% py)
  layout
}

is_tsne_method <- function(method) {
  grepl("tsne|Rtsne|FItSNE|opentsne", method, ignore.case = TRUE)
}

is_umap_method <- function(method) {
  grepl("umap", method, ignore.case = TRUE)
}

method_backend <- function(method) {
  if (grepl("_cuda$", method) || grepl("_cuda_", method) || startsWith(method, "rapids_cuml_")) {
    "cuda"
  } else {
    "cpu"
  }
}

method_parameter_row <- function(method) {
  backend <- method_backend(method)
  tsne_method <- is_tsne_method(method)
  umap_method <- is_umap_method(method)
  base <- list(
    method = method,
    backend = backend,
    n_neighbors_k = NA_character_,
    perplexity = NA_character_,
    iterations_or_epochs = NA_character_,
    early_exaggeration = NA_character_,
    learning_rate = NA_character_,
    initialization = NA_character_,
    distance_metric = "euclidean",
    thread_count = as.character(threads),
    random_seed = as.character(seed),
    knn_precomputed = "no",
    knn_source = NA_character_,
    knn_exact_or_approximate = NA_character_,
    input_matrix = if (grepl("^fastEmbedR", method)) "float32" else "standard R matrix",
    notes = NA_character_
  )
  if (method == "fastEmbedR_opentsne_cpu") {
    base$n_neighbors_k <- paste0("ceiling(perplexity) = ", ceiling(perplexity))
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "auto opt-SNE policy; default 250 early-exaggeration + 500 normal iterations unless auto stop applies"
    base$early_exaggeration <- "auto; default 12"
    base$learning_rate <- "auto; n / early_exaggeration"
    base$initialization <- "PCA initialization from fastPLS rSVD when no Y_init is supplied"
    base$knn_source <- "internal faissR CPU HNSW policy, target_recall = 0.99"
    base$knn_exact_or_approximate <- "approximate"
    base$notes <- "total runtime includes faissR KNN, PCA initialization, affinity construction, and embedding"
  } else if (method == "fastEmbedR_opentsne_cuda") {
    base$n_neighbors_k <- paste0("ceiling(perplexity) = ", ceiling(perplexity))
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "auto opt-SNE policy; default 250 early-exaggeration + 500 normal iterations unless auto stop applies"
    base$early_exaggeration <- "auto; default 12"
    base$learning_rate <- "auto; n / early_exaggeration"
    base$initialization <- "native CUDA RAFT/cuML TSVD PCA initialization"
    base$knn_source <- "internal faissR CUDA auto policy, target_recall = 0.99"
    base$knn_exact_or_approximate <- "approximate"
    base$notes <- "CUDA run is counted only when native CUDA backend and RAFT/cuML TSVD initialization are available; no host fallback is used"
  } else if (method %in% c("fastEmbedR_umap_cpu_fuzzy", "fastEmbedR_umap_cpu_binary")) {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "fastEmbedR auto UMAP policy; resolved per dataset and stored in the returned parameters"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "fastEmbedR auto UMAP policy; default starts at 1"
    base$initialization <- "spectral initialization from the KNN graph"
    base$knn_source <- "internal faissR CPU HNSW policy, target_recall = 0.99"
    base$knn_exact_or_approximate <- "approximate"
    base$notes <- if (grepl("_binary$", method)) {
      "binary symmetric graph_mode"
    } else {
      "standard fuzzy UMAP graph_mode"
    }
  } else if (method %in% c("fastEmbedR_umap_cuda_fuzzy", "fastEmbedR_umap_cuda_binary")) {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "fastEmbedR auto UMAP policy; resolved per dataset and stored in the returned parameters"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "fastEmbedR auto UMAP policy; default starts at 1"
    base$initialization <- "spectral initialization from the KNN graph"
    base$knn_source <- "internal faissR CUDA auto policy, target_recall = 0.99"
    base$knn_exact_or_approximate <- "approximate"
    base$notes <- if (grepl("_binary$", method)) {
      "binary symmetric graph_mode; CUDA run is counted only when native CUDA backend is available"
    } else {
      "standard fuzzy UMAP graph_mode; CUDA run is counted only when native CUDA backend is available"
    }
  } else if (method == "Rtsne_full") {
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "Rtsne default max_iter = 1000"
    base$early_exaggeration <- "Rtsne default exaggeration_factor = 12, stop_lying_iter = 250"
    base$learning_rate <- "Rtsne default eta = 200"
    base$initialization <- "Rtsne PCA preprocessing enabled with pca = TRUE"
    base$knn_source <- "internal Rtsne nearest-neighbour/affinity construction"
    base$knn_exact_or_approximate <- "package internal"
    base$notes <- "total runtime includes Rtsne internal preprocessing, KNN/affinity construction, and embedding"
  } else if (method == "KlugerLab_FItSNE") {
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "max_iter = 750"
    base$early_exaggeration <- "FIt-SNE wrapper default"
    base$learning_rate <- "FIt-SNE wrapper default"
    base$initialization <- "FIt-SNE wrapper default unless external wrapper supplies otherwise"
    base$knn_source <- "internal FIt-SNE nearest-neighbour/affinity construction"
    base$knn_exact_or_approximate <- "package/executable internal"
    base$notes <- "theta = 0.5; nthreads set to benchmark thread count"
  } else if (method == "python_opentsne_fft") {
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "Python openTSNE n_iter = 500 plus early_exaggeration_iter = 250"
    base$early_exaggeration <- "Python openTSNE default"
    base$learning_rate <- "Python openTSNE auto/default"
    base$initialization <- "Python openTSNE PCA initialization"
    base$knn_source <- "Python openTSNE internal affinity/neighbor construction"
    base$knn_exact_or_approximate <- "package internal"
    base$notes <- "negative_gradient_method = fft; run through reticulate for reference benchmarking"
  } else if (method == "python_opentsne_fft_direct") {
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "Python openTSNE n_iter = 500 plus early_exaggeration_iter = 250"
    base$early_exaggeration <- "Python openTSNE default"
    base$learning_rate <- "Python openTSNE auto/default"
    base$initialization <- "Python openTSNE PCA initialization"
    base$knn_source <- "Python openTSNE internal affinity/neighbor construction"
    base$knn_exact_or_approximate <- "package internal"
    base$notes <- "negative_gradient_method = fft; run in a native Python subprocess; elapsed_sec is Python-side fit time, process_elapsed_sec includes process startup and data load"
  } else if (method == "umap_package") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "umap::umap.defaults except n_neighbors = k"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "umap package default"
    base$initialization <- "umap package default"
    base$knn_source <- "internal umap package KNN"
    base$knn_exact_or_approximate <- "package internal"
    base$notes <- "total runtime includes umap package internal KNN and embedding"
  } else if (method == "uwot_default") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "uwot defaults"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "uwot default"
    base$initialization <- "uwot default"
    base$knn_source <- "internal uwot KNN with package default nn_method"
    base$knn_exact_or_approximate <- "package internal"
    base$notes <- "fast_sgd = FALSE; n_sgd_threads = 1"
  } else if (method == "uwot_fast_sgd") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "uwot defaults"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "uwot default"
    base$initialization <- "uwot default"
    base$knn_source <- "internal uwot KNN with package default nn_method"
    base$knn_exact_or_approximate <- "package internal"
    base$notes <- "fast_sgd = TRUE; n_sgd_threads set to benchmark thread count"
  } else if (method == "python_umap_learn") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "Python umap-learn defaults"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "Python umap-learn default"
    base$initialization <- "Python umap-learn default"
    base$knn_source <- "Python umap-learn internal nearest-neighbor graph construction"
    base$knn_exact_or_approximate <- "package internal approximate"
    base$notes <- "run through reticulate for reference benchmarking"
  } else if (method == "python_umap_learn_direct") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "Python umap-learn defaults"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "Python umap-learn default"
    base$initialization <- "Python umap-learn default"
    base$knn_source <- "Python umap-learn internal nearest-neighbor graph construction"
    base$knn_exact_or_approximate <- "package internal approximate"
    base$notes <- "run in a native Python subprocess; elapsed_sec is Python-side fit time, process_elapsed_sec includes process startup and data load"
  } else if (method == "rapids_cuml_umap_full") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "RAPIDS cuML UMAP defaults"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "RAPIDS cuML default"
    base$initialization <- "RAPIDS cuML default"
    base$knn_source <- "RAPIDS cuML internal GPU nearest-neighbor graph construction"
    base$knn_exact_or_approximate <- "RAPIDS internal"
    base$notes <- "full RAPIDS cuML UMAP through reticulate; total runtime includes Python/R boundary and GPU transfer"
  } else if (method == "rapids_cuml_umap_full_direct") {
    base$n_neighbors_k <- as.character(k)
    base$iterations_or_epochs <- "RAPIDS cuML UMAP defaults"
    base$early_exaggeration <- "not used"
    base$learning_rate <- "RAPIDS cuML default"
    base$initialization <- "RAPIDS cuML default"
    base$knn_source <- "RAPIDS cuML internal GPU nearest-neighbor graph construction"
    base$knn_exact_or_approximate <- "RAPIDS internal"
    base$notes <- "full RAPIDS cuML UMAP in a native Python subprocess; elapsed_sec is Python-side fit time, process_elapsed_sec includes process startup and data load"
  } else if (method == "rapids_cuml_tsne_full") {
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "RAPIDS cuML TSNE defaults"
    base$early_exaggeration <- "RAPIDS cuML default"
    base$learning_rate <- "RAPIDS cuML default"
    base$initialization <- "RAPIDS cuML default"
    base$knn_source <- "RAPIDS cuML internal GPU affinity construction"
    base$knn_exact_or_approximate <- "RAPIDS internal"
    base$notes <- "full RAPIDS cuML TSNE through reticulate; total runtime includes Python/R boundary and GPU transfer"
  } else if (method == "rapids_cuml_tsne_full_direct") {
    base$perplexity <- as.character(perplexity)
    base$iterations_or_epochs <- "RAPIDS cuML TSNE defaults"
    base$early_exaggeration <- "RAPIDS cuML default"
    base$learning_rate <- "RAPIDS cuML default"
    base$initialization <- "RAPIDS cuML default"
    base$knn_source <- "RAPIDS cuML internal GPU affinity construction"
    base$knn_exact_or_approximate <- "RAPIDS internal"
    base$notes <- "full RAPIDS cuML TSNE in a native Python subprocess; elapsed_sec is Python-side fit time, process_elapsed_sec includes process startup and data load"
  } else {
    base$n_neighbors_k <- if (umap_method) as.character(k) else NA_character_
    base$perplexity <- if (tsne_method) as.character(perplexity) else NA_character_
    base$notes <- "unknown method; inspect worker log"
  }
  as.data.frame(base, stringsAsFactors = FALSE)
}

method_parameter_summary <- function(method) {
  row <- method_parameter_row(method)
  pairs <- paste(names(row), as.character(row[1, , drop = TRUE]), sep = "=")
  paste(pairs[!is.na(row[1, , drop = TRUE])], collapse = "; ")
}

cuda_status_message <- function() {
  parts <- c("CUDA backend unavailable.")
  if (requireNamespace("fastEmbedR", quietly = TRUE)) {
    parts <- c(
      parts,
      paste0(
        " fastEmbedR version=",
        paste(capture.output(print(try(utils::packageVersion("fastEmbedR"), silent = TRUE))), collapse = " ")
      ),
      paste0(
        " fastEmbedR exports cuda_available=",
        "cuda_available" %in% getNamespaceExports("fastEmbedR")
      )
    )
  } else {
    parts <- c(parts, " fastEmbedR is not installed.")
  }
  if (requireNamespace("faissR", quietly = TRUE)) {
    parts <- c(
      parts,
      paste0(
        " faissR::backend_info()=",
        paste(capture.output(print(try(faissR::backend_info(), silent = TRUE))), collapse = " ")
      ),
      paste0(
        " faissR::cuda_available()=",
        paste(capture.output(print(try(faissR::cuda_available(), silent = TRUE))), collapse = " ")
      ),
      paste0(
        " faissR::cuvs_available()=",
        paste(capture.output(print(try(faissR::cuvs_available(), silent = TRUE))), collapse = " ")
      )
    )
  } else {
    parts <- c(parts, " faissR is not installed.")
  }
  paste(parts, collapse = "")
}

cuda_ready_for_benchmark <- function() {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) return(FALSE)
  if (!requireNamespace("faissR", quietly = TRUE)) return(FALSE)
  isTRUE(tryCatch(faissR::cuda_available(), error = function(e) FALSE))
}

run_embedding_method <- function(method, dataset, x_fast, x_ref, labels, python_npz = NULL) {
  set.seed(seed)
  if (method == "fastEmbedR_opentsne_cpu") {
    return(fastEmbedR::opentsne(x_fast, perplexity = perplexity, backend = "cpu",
                                n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_opentsne_cuda") {
    return(fastEmbedR::opentsne(x_fast, perplexity = perplexity, backend = "cuda",
                                n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cpu_fuzzy") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cpu",
                            graph_mode = "fuzzy", n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cpu_binary") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cpu",
                            graph_mode = "binary", n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cuda_fuzzy") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cuda",
                            graph_mode = "fuzzy", n_threads = threads, seed = seed))
  }
  if (method == "fastEmbedR_umap_cuda_binary") {
    return(fastEmbedR::umap(x_fast, n_neighbors = k, backend = "cuda",
                            graph_mode = "binary", n_threads = threads, seed = seed))
  }
  if (method == "Rtsne_full") {
    if (is.null(x_ref)) stop("Standard R dataset is required for Rtsne_full.", call. = FALSE)
    if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne is not installed.", call. = FALSE)
    return(Rtsne::Rtsne(x_ref, perplexity = perplexity, check_duplicates = FALSE,
                        pca = TRUE, num_threads = threads)$Y)
  }
  if (method == "KlugerLab_FItSNE") {
    if (is.null(x_ref)) stop("Standard R dataset is required for KlugerLab_FItSNE.", call. = FALSE)
    return(run_fitsne(x_ref))
  }
  if (method == "python_opentsne_fft") {
    if (is.null(x_ref)) stop("Standard R dataset is required for python_opentsne_fft.", call. = FALSE)
    return(run_python_opentsne_fft(x_ref))
  }
  if (method == "python_opentsne_fft_direct") {
    return(run_python_direct_method(method, python_npz))
  }
  if (method == "rapids_cuml_tsne_full") {
    if (is.null(x_ref)) stop("Standard R dataset is required for rapids_cuml_tsne_full.", call. = FALSE)
    return(run_rapids_cuml_tsne(x_ref))
  }
  if (method == "rapids_cuml_tsne_full_direct") {
    return(run_python_direct_method(method, python_npz))
  }
  if (method == "umap_package") {
    if (is.null(x_ref)) stop("Standard R dataset is required for umap_package.", call. = FALSE)
    if (!requireNamespace("umap", quietly = TRUE)) stop("umap is not installed.", call. = FALSE)
    cfg <- umap::umap.defaults
    cfg$n_neighbors <- k
    return(umap::umap(x_ref, config = cfg)$layout)
  }
  if (method == "uwot_default") {
    if (is.null(x_ref)) stop("Standard R dataset is required for uwot_default.", call. = FALSE)
    if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
    return(uwot::umap(x_ref, n_neighbors = k, n_threads = threads,
                      n_sgd_threads = 1, fast_sgd = FALSE, verbose = FALSE))
  }
  if (method == "uwot_fast_sgd") {
    if (is.null(x_ref)) stop("Standard R dataset is required for uwot_fast_sgd.", call. = FALSE)
    if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
    return(uwot::umap(x_ref, n_neighbors = k, n_threads = threads,
                      n_sgd_threads = threads, fast_sgd = TRUE, verbose = FALSE))
  }
  if (method == "python_umap_learn") {
    if (is.null(x_ref)) stop("Standard R dataset is required for python_umap_learn.", call. = FALSE)
    return(run_python_umap_learn(x_ref))
  }
  if (method == "python_umap_learn_direct") {
    return(run_python_direct_method(method, python_npz))
  }
  if (method == "rapids_cuml_umap_full") {
    if (is.null(x_ref)) stop("Standard R dataset is required for rapids_cuml_umap_full.", call. = FALSE)
    return(run_rapids_cuml_umap(x_ref))
  }
  if (method == "rapids_cuml_umap_full_direct") {
    return(run_python_direct_method(method, python_npz))
  }
  stop("Unknown method: ", method, call. = FALSE)
}

worker_main <- function() {
  dataset <- args$dataset
  method <- args$method
  worker_out <- args$worker_out
  if (is.null(dataset) || is.null(method) || is.null(worker_out)) {
    stop("--dataset, --method, and --worker_out are required in worker mode.", call. = FALSE)
  }
  if (method_backend(method) == "cuda" && !cuda_ready_for_benchmark()) {
    stop(cuda_status_message(), call. = FALSE)
  }

  float_obj <- pick_dataset_object(find_float_rdata(dataset))
  standard_path <- find_source_rdata(dataset)
  standard <- if (is.na(standard_path)) NULL else pick_dataset_object(standard_path)
  labels <- if (is.null(standard)) float_obj$labels else (standard$labels %||% float_obj$labels)
  if (!is.null(labels)) labels <- as.factor(labels)
  # Reference packages receive the standard R data, not the float32 object.
  # Some wrappers, including KlugerLab/FIt-SNE, require a strict base matrix.
  x_ref <- if (is.null(standard)) NULL else as_double_matrix(standard$data)
  x_fast <- as_fastembedr_float_input(float_obj$data)
  if (!is.null(x_ref) && nrow(x_ref) != nrow(x_fast)) {
    stop("Standard and float32 datasets have different number of rows for ", dataset, call. = FALSE)
  }
  x_score <- if (is.null(x_ref)) as_double_matrix(float_obj$data) else as_double_matrix(x_ref)
  python_npz <- if (is_direct_python_method(method)) {
    # Export is intentionally outside the timed embedding block. Direct Python
    # rows report Python-side package fit time; process_elapsed_sec below records
    # subprocess startup and NPZ load overhead for transparency.
    prepare_direct_python_npz(dataset, x_ref)
  } else {
    NULL
  }

  gc()
  t <- system.time({
    fit <- run_embedding_method(method, dataset, x_fast, x_ref, labels, python_npz = python_npz)
  })
  process_elapsed <- unname(t[["elapsed"]])
  python_fit_sec <- as.numeric(attr(fit, "python_fit_sec", exact = TRUE) %||% NA_real_)
  direct_process_elapsed <- as.numeric(attr(fit, "process_elapsed_sec", exact = TRUE) %||% NA_real_)
  elapsed <- if (is_direct_python_method(method) && is.finite(python_fit_sec)) {
    python_fit_sec
  } else {
    process_elapsed
  }
  layout <- layout_matrix(fit)
  layout_file <- file.path(out_dir, "layouts", paste0(dataset, "_", method, "_threads", threads, "_seed", seed, ".rds"))
  plot_file <- file.path(out_dir, "plots", paste0(dataset, "_", method, "_threads", threads, "_seed", seed, ".png"))
  postprocess_warnings <- character()
  add_postprocess_warning <- function(stage, condition) {
    postprocess_warnings <<- c(
      postprocess_warnings,
      sprintf("%s: %s", stage, conditionMessage(condition))
    )
  }
  save_ok <- tryCatch({
    saveRDS(list(layout = layout, labels = labels, method = method, dataset = dataset), layout_file)
    TRUE
  }, error = function(e) {
    add_postprocess_warning("save_layout", e)
    FALSE
  })
  if (!isTRUE(save_ok)) layout_file <- NA_character_
  plot_ok <- tryCatch({
    plot_layout(layout, labels, plot_file, sprintf("%s %s %d threads %.2fs", dataset, method, threads, elapsed))
    TRUE
  }, error = function(e) {
    add_postprocess_warning("plot_layout", e)
    FALSE
  })
  if (!isTRUE(plot_ok)) plot_file <- NA_character_
  # Metrics need numeric matrix arithmetic; this conversion is only for scoring,
  # never for the reference package calls above. Metric failures should not
  # invalidate a completed embedding in long HPC runs.
  scores <- tryCatch(
    score_layout(x_score, layout, labels),
    error = function(e) {
      add_postprocess_warning("score_layout", e)
      empty_score_record()
    }
  )

  result <- data.frame(
    dataset = dataset,
    method = method,
    backend = method_backend(method),
    cpu_threads = threads,
    status = "success",
    n = nrow(x_fast),
    p = ncol(x_fast),
    k = if (is_umap_method(method)) k else NA_integer_,
    perplexity = if (is_tsne_method(method)) perplexity else NA_real_,
    parameters = method_parameter_summary(method),
    input_fastEmbedR = if (grepl("^fastEmbedR", method)) {
      "float32"
    } else {
      "standard_R_matrix"
    },
    timing_mode = if (is_direct_python_method(method)) "native_python_process" else if (grepl("python|rapids_cuml", method)) "reticulate" else "R",
    elapsed_sec = elapsed,
    process_elapsed_sec = if (is_direct_python_method(method)) direct_process_elapsed else process_elapsed,
    python_fit_sec = if (is_direct_python_method(method)) python_fit_sec else NA_real_,
    trust = scores$trustworthiness,
    trustworthiness = scores$trustworthiness,
    knn_preservation = scores$knn_preservation,
    nn_preservation = scores$knn_preservation,
    knn_preservation_15 = scores$knn_preservation_15,
    knn_preservation_30 = scores$knn_preservation_30,
    knn_preservation_50 = scores$knn_preservation_50,
    silhouette = scores$silhouette,
    label_acc = scores$label_knn_accuracy,
    knn_label_accuracy = scores$label_knn_accuracy,
    quality_sample_n = scores$quality_sample_n,
    max_rss_kb = NA_real_,
    max_rss_gb = NA_real_,
    layout_file = layout_file,
    plot_file = plot_file,
    error = if (length(postprocess_warnings)) paste(postprocess_warnings, collapse = " | ") else NA_character_,
    stringsAsFactors = FALSE
  )
  write.csv(result, worker_out, row.names = FALSE)
}

parse_time_v <- function(path) {
  if (!file.exists(path)) return(list(max_rss_kb = NA_real_, exit_status = NA_integer_))
  txt <- readLines(path, warn = FALSE)
  rss_line <- grep("Maximum resident set size", txt, value = TRUE)
  rss <- NA_real_
  if (length(rss_line)) {
    rss <- suppressWarnings(as.numeric(sub(".*: *", "", rss_line[[length(rss_line)]])))
  }
  status_line <- grep("Exit status", txt, value = TRUE)
  exit_status <- NA_integer_
  if (length(status_line)) {
    exit_status <- suppressWarnings(as.integer(sub(".*: *", "", status_line[[length(status_line)]])))
  }
  list(max_rss_kb = rss, exit_status = exit_status)
}

ensure_quality_columns <- function(tab) {
  defaults <- list(
    trust = NA_real_,
    trustworthiness = NA_real_,
    cpu_threads = NA_real_,
    knn_preservation = NA_real_,
    nn_preservation = NA_real_,
    knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_,
    knn_preservation_50 = NA_real_,
    silhouette = NA_real_,
    label_acc = NA_real_,
    knn_label_accuracy = NA_real_,
    quality_sample_n = NA_integer_,
    max_rss_kb = NA_real_,
    max_rss_gb = NA_real_,
    timing_mode = NA_character_,
    process_elapsed_sec = NA_real_,
    python_fit_sec = NA_real_
  )
  for (nm in names(defaults)) {
    if (!nm %in% names(tab)) tab[[nm]] <- defaults[[nm]]
  }
  if (all(is.na(tab$trustworthiness)) && any(!is.na(tab$trust))) {
    tab$trustworthiness <- tab$trust
  }
  if (all(is.na(tab$trust)) && any(!is.na(tab$trustworthiness))) {
    tab$trust <- tab$trustworthiness
  }
  if (all(is.na(tab$knn_preservation_30)) && any(!is.na(tab$knn_preservation))) {
    tab$knn_preservation_30 <- tab$knn_preservation
  }
  if (all(is.na(tab$knn_preservation)) && any(!is.na(tab$knn_preservation_30))) {
    tab$knn_preservation <- tab$knn_preservation_30
  }
  if (all(is.na(tab$nn_preservation)) && any(!is.na(tab$knn_preservation_30))) {
    tab$nn_preservation <- tab$knn_preservation_30
  }
  if (all(is.na(tab$knn_label_accuracy)) && any(!is.na(tab$label_acc))) {
    tab$knn_label_accuracy <- tab$label_acc
  }
  if (all(is.na(tab$label_acc)) && any(!is.na(tab$knn_label_accuracy))) {
    tab$label_acc <- tab$knn_label_accuracy
  }
  numeric_cols <- c(
    "n", "p", "k", "perplexity", "cpu_threads", "elapsed_sec", "trust",
    "trustworthiness", "knn_preservation", "nn_preservation",
    "knn_preservation_15", "knn_preservation_30", "knn_preservation_50",
    "silhouette", "label_acc", "knn_label_accuracy", "quality_sample_n",
    "max_rss_kb", "max_rss_gb", "process_elapsed_sec", "python_fit_sec"
  )
  for (nm in intersect(numeric_cols, names(tab))) {
    tab[[nm]] <- suppressWarnings(as.numeric(tab[[nm]]))
  }
  tab
}

write_markdown_table <- function(tab, path) {
  if (!nrow(tab)) return(invisible(NULL))
  numeric_cols <- vapply(tab, is.numeric, logical(1))
  display <- tab
  display[numeric_cols] <- lapply(display[numeric_cols], function(x) {
    ifelse(is.na(x), "", formatC(x, digits = 4L, format = "fg"))
  })
  display[] <- lapply(display, function(x) {
    x <- as.character(x)
    x[is.na(x) | x == "NA"] <- ""
    x
  })
  con <- file(path, open = "wt")
  on.exit(close(con), add = TRUE)
  cat("| ", paste(names(display), collapse = " | "), " |\n", sep = "", file = con)
  cat("| ", paste(rep("---", ncol(display)), collapse = " | "), " |\n", sep = "", file = con)
  for (i in seq_len(nrow(display))) {
    cat("| ", paste(as.character(display[i, , drop = TRUE]), collapse = " | "), " |\n", sep = "", file = con)
  }
  invisible(NULL)
}

write_parameter_outputs <- function(methods, thread_values = threads) {
  old_threads <- threads
  on.exit({
    threads <<- old_threads
    Sys.setenv(
      OMP_NUM_THREADS = as.character(threads),
      OPENBLAS_NUM_THREADS = as.character(threads),
      MKL_NUM_THREADS = as.character(threads),
      VECLIB_MAXIMUM_THREADS = as.character(threads),
      RCPP_PARALLEL_NUM_THREADS = as.character(threads)
    )
  }, add = TRUE)
  tabs <- lapply(thread_values, function(th) {
    threads <<- as.integer(th)
    Sys.setenv(
      OMP_NUM_THREADS = as.character(threads),
      OPENBLAS_NUM_THREADS = as.character(threads),
      MKL_NUM_THREADS = as.character(threads),
      VECLIB_MAXIMUM_THREADS = as.character(threads),
      RCPP_PARALLEL_NUM_THREADS = as.character(threads)
    )
    tab <- do.call(rbind, lapply(methods, method_parameter_row))
    tab$cpu_threads <- threads
    tab
  })
  param_tab <- do.call(rbind, tabs)
  param_tab <- param_tab[order(param_tab$cpu_threads, param_tab$method, param_tab$backend), , drop = FALSE]
  write.csv(param_tab, file.path(out_dir, "embedding_parameter_table.csv"), row.names = FALSE)
  write_markdown_table(param_tab, file.path(out_dir, "embedding_parameter_table.md"))
  invisible(param_tab)
}

run_cmd_capture <- function(command, args = character()) {
  path <- Sys.which(command)
  if (!nzchar(path)) return(NA_character_)
  out <- tryCatch(
    system2(path, args, stdout = TRUE, stderr = TRUE),
    warning = function(w) conditionMessage(w),
    error = function(e) conditionMessage(e)
  )
  paste(out, collapse = "\n")
}

git_capture <- function(args) {
  out <- run_cmd_capture("git", args)
  if (length(out) != 1L || is.na(out)) NA_character_ else trimws(out)
}

pkg_version_or_na <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(pkg))
}

write_text_manifest <- function(x, path) {
  con <- file(path, open = "wt")
  on.exit(close(con), add = TRUE)
  recurse <- function(obj, prefix = "") {
    if (is.list(obj)) {
      for (nm in names(obj)) {
        cat(prefix, nm, ":\n", sep = "", file = con)
        recurse(obj[[nm]], paste0(prefix, "  "))
      }
    } else {
      val <- paste(as.character(obj), collapse = "\n")
      if (!nzchar(val) || is.na(val)) val <- "NA"
      for (line in strsplit(val, "\n", fixed = TRUE)[[1L]]) {
        cat(prefix, line, "\n", sep = "", file = con)
      }
    }
  }
  recurse(x)
  invisible(path)
}

write_reproducibility_bundle <- function() {
  session_txt <- paste(capture.output(utils::sessionInfo()), collapse = "\n")
  writeLines(session_txt, file.path(out_dir, "sessionInfo.txt"))

  command_lines <- c(
    current_invocation = paste(commandArgs(FALSE), collapse = " "),
    hpc_cpu1_12 = paste(
      paste0("DATASETS=", paste(datasets, collapse = ",")),
      paste0("K=", k),
      paste0("PERPLEXITY=", perplexity),
      paste0("CPU_THREADS_GRID=", paste(thread_grid, collapse = ",")),
      paste0("TIMEOUT=", timeout),
      paste0("FORCE=", force),
      "sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh"
    ),
    hpc_cuda = paste(
      paste0("DATASETS=", paste(datasets, collapse = ",")),
      paste0("K=", k),
      paste0("PERPLEXITY=", perplexity),
      paste0("TIMEOUT=", timeout),
      "sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh"
    ),
    small_reference_validation = paste(
      "Rscript tools/validate_reference_implementations.R",
      "--out-dir=results/reference_validation_current",
      "--threads=2 --seed=4 --perplexity=10 --k=31"
    )
  )
  writeLines(paste(names(command_lines), command_lines, sep = ": "), file.path(out_dir, "benchmark_command_lines.txt"))

  backend_info <- if (requireNamespace("faissR", quietly = TRUE)) {
    paste(capture.output(print(try(faissR::backend_info(), silent = TRUE))), collapse = "\n")
  } else {
    "faissR not installed"
  }

  manifest <- list(
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    repository = "https://github.com/tkcaccia/fastEmbedR",
    git_commit = git_capture(c("rev-parse", "HEAD")),
    git_describe = git_capture(c("describe", "--tags", "--always", "--dirty")),
    git_status_short = git_capture(c("status", "--short")),
    manuscript_release_tag = Sys.getenv("FASTEMBEDR_MANUSCRIPT_TAG", unset = "v0.1.0-manuscript"),
    archival_snapshot = Sys.getenv("FASTEMBEDR_ZENODO_DOI", unset = "Zenodo DOI to be minted from the manuscript release tag before submission"),
    base_dir = base_dir,
    data_root = data_root,
    out_dir = out_dir,
    datasets = paste(datasets, collapse = ","),
    methods = paste(methods, collapse = ","),
    seed = seed,
    k = k,
    perplexity = perplexity,
    threads = threads,
    thread_grid = paste(thread_grid, collapse = ","),
    timeout_seconds = timeout,
    backend_group = backend_group,
    command_lines = as.list(command_lines),
    package_versions = list(
      fastEmbedR = pkg_version_or_na("fastEmbedR"),
      faissR = pkg_version_or_na("faissR"),
      Rcpp = pkg_version_or_na("Rcpp"),
      float = pkg_version_or_na("float"),
      Rtsne = pkg_version_or_na("Rtsne"),
      uwot = pkg_version_or_na("uwot"),
      umap = pkg_version_or_na("umap"),
      jsonlite = pkg_version_or_na("jsonlite")
    ),
    hardware = list(
      sys_info = Sys.info(),
      lscpu = run_cmd_capture("lscpu"),
      free_h = run_cmd_capture("free", "-h"),
      nvidia_smi_query = run_cmd_capture(
        "nvidia-smi",
        c("--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader")
      ),
      nvidia_smi = run_cmd_capture("nvidia-smi")
    ),
    cuda_stack = list(
      nvcc_version = run_cmd_capture("nvcc", "--version"),
      cuda_home = Sys.getenv("CUDA_HOME", unset = NA_character_),
      ld_library_path = Sys.getenv("LD_LIBRARY_PATH", unset = NA_character_),
      faissR_cuda_available = if (requireNamespace("faissR", quietly = TRUE)) {
        paste(capture.output(print(try(faissR::cuda_available(), silent = TRUE))), collapse = "\n")
      } else {
        "faissR not installed"
      },
      faissR_cuvs_available = if (requireNamespace("faissR", quietly = TRUE)) {
        paste(capture.output(print(try(faissR::cuvs_available(), silent = TRUE))), collapse = "\n")
      } else {
        "faissR not installed"
      },
      faissR_backend_info = backend_info
    ),
    r = list(
      version = R.version.string,
      platform = R.version$platform,
      library_paths = paste(.libPaths(), collapse = "; "),
      sessionInfo_file = "sessionInfo.txt"
    ),
    environment_files = list(
      r_environment = "tools/reproducibility/benchmark_environment.yml",
      installation = "docs/installation.md",
      backend_capabilities = "docs/backend-capabilities.md",
      hpc_driver = "tools/hpc_embeddings/benchmark_embeddings_float32_publication.R",
      direct_python_helper = "tools/hpc_embeddings/benchmark_python_direct.py",
      cpu_wrapper = "tools/hpc_embeddings/benchmark_embeddings_float32_cpu12.sh",
      cuda_wrapper = "tools/hpc_embeddings/benchmark_embeddings_float32_cuda.sh"
    )
  )
  write_text_manifest(manifest, file.path(out_dir, "reproducibility_manifest.txt"))
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::write_json(manifest, file.path(out_dir, "reproducibility_manifest.json"), auto_unbox = TRUE, pretty = TRUE)
  }
  invisible(manifest)
}

write_quality_outputs <- function(tab) {
  tab <- ensure_quality_columns(tab)
  quality_cols <- c(
    "dataset", "method", "backend", "status", "n", "p",
    "cpu_threads", "timing_mode", "elapsed_sec", "process_elapsed_sec",
    "python_fit_sec", "max_rss_gb", "trustworthiness",
    "knn_preservation_30", "silhouette", "knn_label_accuracy",
    "knn_preservation_15", "knn_preservation_50", "quality_sample_n",
    "plot_file", "error"
  )
  quality_cols <- quality_cols[quality_cols %in% names(tab)]
  quality <- tab[, quality_cols, drop = FALSE]
  names(quality) <- sub("^elapsed_sec$", "runtime_sec", names(quality))
  names(quality) <- sub("^knn_preservation_30$", "nn_preservation", names(quality))
  names(quality) <- sub("^knn_label_accuracy$", "knn_label_accuracy", names(quality))

  key_datasets <- c(
    "MNIST", "FashionMNIST", "flow18", "mass41", "imagenet",
    "FlowRepository_FR-FCM-ZYRM_files", "MetRef"
  )
  quality$key_manuscript_dataset <- quality$dataset %in% key_datasets
  quality <- quality[order(!quality$key_manuscript_dataset, quality$dataset, quality$method, quality$backend), , drop = FALSE]
  write.csv(quality, file.path(out_dir, "embedding_quality_table.csv"), row.names = FALSE)

  md_cols <- c(
    "dataset", "method", "backend", "cpu_threads", "runtime_sec", "trustworthiness",
    "nn_preservation", "silhouette", "knn_label_accuracy", "max_rss_gb",
    "timing_mode", "status"
  )
  md_cols <- md_cols[md_cols %in% names(quality)]
  md_quality <- quality[quality$key_manuscript_dataset, md_cols, drop = FALSE]
  if (!nrow(md_quality)) md_quality <- quality[, md_cols, drop = FALSE]
  write_markdown_table(
    md_quality,
    file.path(out_dir, "embedding_quality_table.md")
  )

  ok <- quality$status == "success" &
    is.finite(quality$runtime_sec) &
    is.finite(quality$trustworthiness) &
    quality$runtime_sec > 0
  if (any(ok)) {
    pareto <- quality[ok, , drop = FALSE]
    write.csv(pareto, file.path(out_dir, "embedding_runtime_quality_pareto.csv"), row.names = FALSE)
    plot_datasets <- intersect(key_datasets, unique(pareto$dataset))
    if (!length(plot_datasets)) plot_datasets <- unique(pareto$dataset)
    plot_datasets <- head(plot_datasets, 6L)
    methods <- unique(pareto$method)
    palette <- grDevices::hcl.colors(max(3L, length(methods)), "Dark 3")
    method_cols <- setNames(palette[seq_along(methods)], methods)
    png(file.path(out_dir, "embedding_runtime_quality_pareto.png"),
        width = 2100, height = 1500, res = 170)
    old_par <- par(no.readonly = TRUE)
    on.exit({
      par(old_par)
      dev.off()
    }, add = TRUE)
    par(mfrow = c(2, 3), mar = c(4.2, 4.3, 3.2, 1.1), oma = c(0, 0, 2, 0))
    for (ds in plot_datasets) {
      z <- pareto[pareto$dataset == ds, , drop = FALSE]
      xlim <- range(z$runtime_sec, finite = TRUE)
      if (length(unique(z$runtime_sec)) == 1L) xlim <- xlim * c(0.8, 1.2)
      plot(
        z$runtime_sec, z$trustworthiness,
        log = "x",
        pch = 19,
        cex = 1.15,
        col = method_cols[z$method],
        xlab = "Runtime (seconds, log scale)",
        ylab = "Trustworthiness",
        main = ds,
        xlim = xlim
      )
      point_labels <- if ("cpu_threads" %in% names(z) && any(!is.na(z$cpu_threads))) {
        paste0(z$backend, "/", z$cpu_threads, "t")
      } else {
        z$backend
      }
      text(z$runtime_sec, z$trustworthiness, labels = point_labels, pos = 3, cex = 0.7)
      grid(col = "grey88")
    }
    legend(
      "bottom",
      inset = -0.02,
      legend = methods,
      col = method_cols[methods],
      pch = 19,
      horiz = TRUE,
      cex = 0.75,
      bty = "n",
      xpd = NA
    )
    mtext("Runtime-quality Pareto comparison for manuscript datasets", outer = TRUE, cex = 1.0)
  }
  invisible(quality)
}

write_combined_outputs <- function(results) {
  if (!length(results)) return(invisible(NULL))
  tab <- do.call(rbind, results)
  tab <- ensure_quality_columns(tab)
  write.csv(tab, file.path(out_dir, "embedding_benchmark_results.csv"), row.names = FALSE)
  write_quality_outputs(tab)
  ok <- tab$status == "success" & is.finite(tab$elapsed_sec)
  if (any(ok)) {
    png(file.path(out_dir, "embedding_time_barplot.png"), width = 1800, height = 1100, res = 150)
    par(mar = c(11, 5, 3, 1))
    thread_label <- if ("cpu_threads" %in% names(tab)) paste0(tab$cpu_threads[ok], "t") else tab$backend[ok]
    labs <- paste(tab$dataset[ok], tab$method[ok], thread_label, sep = "\n")
    barplot(tab$elapsed_sec[ok], names.arg = labs, las = 2, cex.names = 0.55,
            ylab = "Seconds", main = "Embedding runtime")
    dev.off()
    mem_ok <- ok & is.finite(tab$max_rss_gb)
    if (any(mem_ok)) {
      png(file.path(out_dir, "embedding_memory_barplot.png"), width = 1800, height = 1100, res = 150)
      par(mar = c(11, 5, 3, 1))
      mem_thread_label <- if ("cpu_threads" %in% names(tab)) paste0(tab$cpu_threads[mem_ok], "t") else tab$backend[mem_ok]
      mem_labs <- paste(tab$dataset[mem_ok], tab$method[mem_ok], mem_thread_label, sep = "\n")
      barplot(tab$max_rss_gb[mem_ok], names.arg = mem_labs, las = 2, cex.names = 0.55,
              ylab = "Peak RSS (GB)", main = "Embedding peak memory")
      dev.off()
    }
  }
  invisible(tab)
}

if (worker) {
  tryCatch(worker_main(), error = function(e) {
    dataset <- args$dataset %||% NA_character_
    method <- args$method %||% NA_character_
    worker_out <- args$worker_out %||% file.path(out_dir, "worker_failed.csv")
    row <- data.frame(
      dataset = dataset,
      method = method,
      backend = method_backend(method),
      cpu_threads = threads,
      status = "failed",
      n = NA_integer_,
      p = NA_integer_,
      k = if (is_umap_method(method)) k else NA_integer_,
      perplexity = if (is_tsne_method(method)) perplexity else NA_real_,
      parameters = method_parameter_summary(method),
      input_fastEmbedR = if (grepl("^fastEmbedR", method)) {
        "float32"
      } else {
        "standard_R_matrix"
      },
      timing_mode = if (is_direct_python_method(method)) "native_python_process" else if (grepl("python|rapids_cuml", method)) "reticulate" else "R",
      elapsed_sec = NA_real_,
      process_elapsed_sec = NA_real_,
      python_fit_sec = NA_real_,
      trust = NA_real_,
      trustworthiness = NA_real_,
      knn_preservation = NA_real_,
      nn_preservation = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      knn_preservation_50 = NA_real_,
      silhouette = NA_real_,
      label_acc = NA_real_,
      knn_label_accuracy = NA_real_,
      quality_sample_n = NA_integer_,
      max_rss_kb = NA_real_,
      max_rss_gb = NA_real_,
      layout_file = NA_character_,
      plot_file = NA_character_,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
    write.csv(row, worker_out, row.names = FALSE)
    message(conditionMessage(e))
    quit(status = 1L)
  })
  quit(status = 0L)
}

main_results <- list()
timeout_bin <- Sys.getenv("TIMEOUT_BIN", Sys.which("timeout"))
time_bin <- Sys.getenv("TIME_V_BIN", "/usr/bin/time")
if (!file.exists(time_bin)) time_bin <- Sys.which("time")
if (nzchar(time_bin)) {
  has_time_v <- suppressWarnings(
    system2(time_bin, c("-v", "true"), stdout = FALSE, stderr = FALSE) == 0L
  )
  if (!isTRUE(has_time_v)) time_bin <- ""
}

log_msg("Starting benchmark: backend_group=%s thread_grid=%s timeout=%d", backend_group, paste(thread_grid, collapse = ","), timeout)
log_msg("Datasets: %s", paste(datasets, collapse = ","))
log_msg("Methods: %s", paste(methods, collapse = ","))
parameter_tab <- write_parameter_outputs(methods, thread_grid)
log_msg("Parameter table: %s", file.path(out_dir, "embedding_parameter_table.csv"))
write_reproducibility_bundle()
log_msg("Reproducibility manifest: %s", file.path(out_dir, "reproducibility_manifest.txt"))

input_audit <- do.call(rbind, lapply(datasets, function(dataset) {
  float_path <- tryCatch(find_float_rdata(dataset), error = function(e) NA_character_)
  standard_path <- find_source_rdata(dataset)
  data.frame(
    dataset = dataset,
    float32_file = if (is.na(float_path)) NA_character_ else float_path,
    standard_rdata_file = if (is.na(standard_path)) NA_character_ else standard_path,
    has_float32 = !is.na(float_path),
    has_standard_rdata = !is.na(standard_path),
    stringsAsFactors = FALSE
  )
}))
write.csv(input_audit, file.path(out_dir, "dataset_input_audit.csv"), row.names = FALSE)
missing_standard <- input_audit$dataset[!input_audit$has_standard_rdata]
if (length(missing_standard)) {
  log_msg("Datasets missing standard .RData for reference methods: %s", paste(missing_standard, collapse = ","))
}

for (current_threads in thread_grid) {
  threads <- as.integer(current_threads)
  Sys.setenv(
    OMP_NUM_THREADS = as.character(threads),
    OPENBLAS_NUM_THREADS = as.character(threads),
    MKL_NUM_THREADS = as.character(threads),
    VECLIB_MAXIMUM_THREADS = as.character(threads),
    RCPP_PARALLEL_NUM_THREADS = as.character(threads)
  )
  log_msg("Thread setting: %d", threads)
  for (dataset in datasets) {
    for (method in methods) {
      worker_csv <- file.path(out_dir, "worker_results", paste0(dataset, "_", method, "_threads", threads, ".csv"))
      worker_log <- file.path(out_dir, "logs", paste0(dataset, "_", method, "_threads", threads, ".log"))
      time_log <- file.path(out_dir, "logs", paste0(dataset, "_", method, "_threads", threads, "_time.txt"))
      if (!force && file.exists(worker_csv)) {
        log_msg("%s/%s/%dt: existing worker result, reusing", dataset, method, threads)
        row <- read.csv(worker_csv, stringsAsFactors = FALSE)
        if (!"cpu_threads" %in% names(row)) row$cpu_threads <- threads
        main_results[[length(main_results) + 1L]] <- row
        write_combined_outputs(main_results)
        next
      }
      cmd <- c()
      if (nzchar(timeout_bin)) cmd <- c(cmd, timeout_bin, as.character(timeout))
      if (nzchar(time_bin)) cmd <- c(cmd, time_bin, "-v", "-o", time_log)
      cmd <- c(
        cmd,
        file.path(R.home("bin"), "Rscript"),
        script_path,
        "--worker=TRUE",
        paste0("--base_dir=", base_dir),
        paste0("--data_root=", data_root),
        paste0("--out_dir=", out_dir),
        paste0("--dataset=", dataset),
        paste0("--method=", method),
        paste0("--worker_out=", worker_csv),
        paste0("--threads=", threads),
        paste0("--timeout=", timeout),
        paste0("--seed=", seed),
        paste0("--k=", k),
        paste0("--perplexity=", perplexity)
      )
      log_msg("%s/%s/%dt: running", dataset, method, threads)
      status <- system2(cmd[[1L]], args = cmd[-1L], stdout = worker_log, stderr = worker_log)
      time_info <- parse_time_v(time_log)
      if (file.exists(worker_csv)) {
        row <- read.csv(worker_csv, stringsAsFactors = FALSE)
        if (!"cpu_threads" %in% names(row)) row$cpu_threads <- threads
      } else {
        row <- data.frame(
          dataset = dataset,
          method = method,
          backend = method_backend(method),
          cpu_threads = threads,
          status = if (identical(status, 124L)) "timeout" else "failed",
          n = NA_integer_, p = NA_integer_,
          k = if (is_umap_method(method)) k else NA_integer_,
          perplexity = if (is_tsne_method(method)) perplexity else NA_real_,
          parameters = method_parameter_summary(method),
          input_fastEmbedR = if (grepl("^fastEmbedR", method)) "float32" else "standard_R_matrix",
          timing_mode = if (is_direct_python_method(method)) "native_python_process" else if (grepl("python|rapids_cuml", method)) "reticulate" else "R",
          elapsed_sec = NA_real_,
          process_elapsed_sec = NA_real_,
          python_fit_sec = NA_real_,
          trust = NA_real_,
          trustworthiness = NA_real_,
          knn_preservation = NA_real_,
          nn_preservation = NA_real_,
          knn_preservation_15 = NA_real_,
          knn_preservation_30 = NA_real_,
          knn_preservation_50 = NA_real_,
          silhouette = NA_real_,
          label_acc = NA_real_,
          knn_label_accuracy = NA_real_,
          quality_sample_n = NA_integer_,
          max_rss_kb = NA_real_, max_rss_gb = NA_real_,
          layout_file = NA_character_, plot_file = NA_character_,
          error = paste("worker exited with status", status),
          stringsAsFactors = FALSE
        )
        write.csv(row, worker_csv, row.names = FALSE)
      }
      row$cpu_threads <- threads
      row$max_rss_kb <- time_info$max_rss_kb
      row$max_rss_gb <- time_info$max_rss_kb / 1024^2
      write.csv(row, worker_csv, row.names = FALSE)
      main_results[[length(main_results) + 1L]] <- row
      log_msg("%s/%s/%dt: %s sec=%s rss_gb=%s", dataset, method, threads, row$status[1],
              format(row$elapsed_sec[1], digits = 4), format(row$max_rss_gb[1], digits = 4))
      write_combined_outputs(main_results)
    }
  }
}

tab <- write_combined_outputs(main_results)
print(tab)
log_msg("DONE: %s", out_dir)
