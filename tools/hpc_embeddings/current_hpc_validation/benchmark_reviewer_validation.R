#!/usr/bin/env Rscript

# Publication benchmark requested during manuscript review. The primary timing
# comparison is total user-level runtime. Component timings and precomputed-KNN
# runs are reported separately because reference packages expose different
# computational boundaries.

parse_args <- function(x) {
  out <- list()
  for (arg in x) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", kv[[1L]])]] <- if (length(kv) > 1L) {
      paste(kv[-1L], collapse = "=")
    } else {
      "TRUE"
    }
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L ||
      (length(x) == 1L && is.na(x))) y else x
}
as_bool <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)) %in% c("1", "true", "yes", "y")
}
as_int <- function(x, default) {
  ans <- suppressWarnings(as.integer(x %||% default))
  if (length(ans) != 1L || is.na(ans)) as.integer(default) else ans
}
as_num <- function(x, default) {
  ans <- suppressWarnings(as.numeric(x %||% default))
  if (length(ans) != 1L || is.na(ans) || !is.finite(ans)) as.numeric(default) else ans
}
as_csv <- function(x, default) {
  ans <- trimws(strsplit(as.character(x %||% default), ",", fixed = TRUE)[[1L]])
  ans[nzchar(ans)]
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
} else {
  normalizePath(args$script %||% "benchmark_reviewer_validation.R", mustWork = TRUE)
}
script_dir <- dirname(script_path)
source(file.path(script_dir, "publication_metrics.R"), local = TRUE)

base_dir <- normalizePath(args$base_dir %||% "/scratch/firenze/NN", mustWork = FALSE)
data_root <- normalizePath(
  args$data_root %||% file.path(base_dir, "Data"), mustWork = FALSE
)
out_dir <- normalizePath(
  args$out_dir %||% file.path(
    base_dir, paste0("benchmark_reviewer_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  ),
  mustWork = FALSE
)
layout_dir <- normalizePath(
  args$layout_dir %||% file.path(
    base_dir, "fastEmbedR-rlayout", basename(out_dir)
  ),
  mustWork = FALSE
)
cache_dir <- normalizePath(
  args$cache_dir %||% file.path(data_root, "_fastEmbedR_precomputed"),
  mustWork = FALSE
)
backend_group <- match.arg(
  args$backend_group %||% "cpu", c("cpu", "cuda", "metal", "local")
)
worker <- as_bool(args$worker, FALSE)
precompute_worker <- as_bool(args$precompute_worker, FALSE)
kodama_core_worker <- as_bool(args$kodama_core_worker, FALSE)
force <- as_bool(args$force, FALSE)
reference_validations <- as_bool(args$reference_validations, TRUE)
threads_grid <- unique(as.integer(as_csv(args$threads_grid, "1,4")))
threads_grid <- threads_grid[is.finite(threads_grid) & threads_grid > 0L]
if (!length(threads_grid)) threads_grid <- c(1L, 4L)
threads <- as_int(args$threads, max(threads_grid))
seeds <- unique(as.integer(as_csv(args$seeds, "4,17,42")))
seeds <- seeds[is.finite(seeds)]
if (!length(seeds)) seeds <- c(4L, 17L, 42L)
seed <- as_int(args$seed, seeds[[1L]])
k <- as_int(args$k, 30L)
perplexity <- as_num(args$perplexity, 30)
timeout <- as_int(args$timeout, 43200L)
quality_sample_n <- as_int(args$quality_sample_n, 3000L)
quality_max_distance_ops <- as_num(args$quality_max_distance_ops, 2e8)
validation_sample_n <- as_int(args$validation_sample_n, 2000L)
pca_ncomp <- as_int(args$pca_ncomp, 2L)
landmark_fraction <- as_num(args$landmark_fraction, 0.2)
if (landmark_fraction <= 0 || landmark_fraction >= 1) {
  stop("--landmark-fraction must be strictly between 0 and 1.", call. = FALSE)
}
kodama_m <- as_int(args$kodama_m, 100L)
kodama_tcycle <- as_int(args$kodama_tcycle, 20L)
kodama_ncomp <- as_int(args$kodama_ncomp, 50L)
kodama_landmarks <- as_int(args$kodama_landmarks, 10000000L)
kodama_graph_neighbors <- as_int(args$kodama_graph_neighbors, 100L)
kodama_n_epochs <- as_int(args$kodama_n_epochs, 200L)
kodama_n_iter <- as_int(args$kodama_n_iter, 500L)
local_cpu_max_n <- as_int(args$local_cpu_max_n, .Machine$integer.max)
local_cpu_exceptions <- as_csv(args$local_cpu_exceptions, "")
datasets <- as_csv(
  args$datasets,
  paste(
    "COIL20", "USPS", "FashionMNIST",
    "FlowRepository_FR-FCM-ZYRM_files", "flow18", "MNIST",
    "MetRef", "mass41", "TabulaMuris", "Macosko2015_retina", "imagenet",
    sep = ","
  )
)

dirs <- c(
  out_dir,
  file.path(out_dir, "logs"),
  layout_dir,
  file.path(out_dir, "plots"),
  file.path(out_dir, "worker_results"),
  file.path(out_dir, "memory"),
  file.path(out_dir, "validation_knn"),
  cache_dir
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

set_threads <- function(value) {
  value <- as.integer(value)
  Sys.setenv(
    OMP_NUM_THREADS = value,
    OPENBLAS_NUM_THREADS = value,
    MKL_NUM_THREADS = value,
    VECLIB_MAXIMUM_THREADS = value,
    RCPP_PARALLEL_NUM_THREADS = value
  )
  invisible(value)
}
set_threads(threads)

log_file <- file.path(out_dir, "benchmark.log")
log_msg <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", sprintf(...))
  cat(line, "\n")
  cat(line, "\n", file = log_file, append = TRUE)
  flush.console()
}

dataset_alias <- function(dataset) {
  if (tolower(dataset) %in% c("retina", "macosko", "macosko2015")) {
    "Macosko2015_retina"
  } else {
    dataset
  }
}

local_cpu_allowed_for <- function(dataset, n) {
  if (!is.finite(n)) return(TRUE)
  n <= local_cpu_max_n || dataset_alias(dataset) %in% local_cpu_exceptions
}

dataset_folder <- function(dataset) {
  file.path(data_root, dataset_alias(dataset))
}

find_standard_rdata <- function(dataset) {
  dataset <- dataset_alias(dataset)
  if (identical(tolower(dataset), "iris")) return("__iris__")
  folder <- dataset_folder(dataset)
  hits <- list.files(folder, pattern = "\\.[Rr][Dd]ata$", full.names = TRUE)
  hits <- hits[!grepl(
    "float32|_nn|knn_|pca|manifest|summary|backup|reference|benchmark|worker",
    basename(hits), ignore.case = TRUE
  )]
  if (!length(hits)) return(NA_character_)
  exact <- hits[
    tolower(tools::file_path_sans_ext(basename(hits))) == tolower(dataset)
  ]
  if (length(exact)) return(exact[[1L]])
  hits[order(nchar(basename(hits)), basename(hits))][[1L]]
}

find_float_rdata <- function(dataset) {
  dataset <- dataset_alias(dataset)
  if (identical(tolower(dataset), "iris")) return(NA_character_)
  hits <- list.files(
    dataset_folder(dataset), pattern = "float32.*\\.[Rr][Dd]ata$",
    full.names = TRUE, ignore.case = TRUE
  )
  if (!length(hits)) return(NA_character_)
  hits[order(nchar(basename(hits)), basename(hits))][[1L]]
}

pick_dataset_object <- function(path) {
  if (identical(path, "__iris__")) {
    return(list(
      data = as.matrix(datasets::iris[, 1:4]),
      labels = datasets::iris$Species,
      object_name = "iris"
    ))
  }
  env <- new.env(parent = emptyenv())
  loaded <- load(path, envir = env)
  objects <- mget(loaded, envir = env, inherits = FALSE)
  for (name in names(objects)) {
    value <- objects[[name]]
    if (is.list(value) && !is.null(value$data)) {
      labels <- value$labels %||% value$label %||% value$tissue %||% NULL
      return(list(data = value$data, labels = labels, object_name = name))
    }
  }
  for (name in names(objects)) {
    value <- objects[[name]]
    if (is.matrix(value) || is.data.frame(value) || inherits(value, "Matrix") ||
        inherits(value, "float32")) {
      labels <- NULL
      for (candidate in c("labels", "label", "tissue", "Y", "y", "class")) {
        if (exists(candidate, envir = env, inherits = FALSE)) {
          candidate_value <- get(candidate, envir = env, inherits = FALSE)
          if (length(candidate_value) == nrow(value)) labels <- candidate_value
        }
      }
      return(list(data = value, labels = labels, object_name = name))
    }
  }
  stop("No dataset matrix was found in ", path, call. = FALSE)
}

as_double_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required.", call. = FALSE)
    }
    x <- float::dbl(x)
  }
  if (inherits(x, "Matrix")) x <- as.matrix(x)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

as_float32_matrix <- function(x) {
  if (inherits(x, "float32")) return(x)
  if (!requireNamespace("float", quietly = TRUE)) {
    stop("The float package is required for fastEmbedR benchmark rows.", call. = FALSE)
  }
  float::fl(as_double_matrix(x))
}

load_dataset <- function(dataset, need_standard = TRUE, need_float = TRUE) {
  standard_path <- find_standard_rdata(dataset)
  float_path <- find_float_rdata(dataset)
  # A missing float32 file is generated from the standard object, so the
  # standard object must also be loaded in that case even for fastEmbedR-only
  # workers (notably the built-in iris smoke test).
  load_standard <- need_standard || (need_float && is.na(float_path))
  standard <- if (load_standard && !is.na(standard_path)) {
    pick_dataset_object(standard_path)
  } else NULL
  float_object <- if (need_float && !is.na(float_path)) {
    pick_dataset_object(float_path)
  } else NULL
  if (is.null(standard) && is.null(float_object)) {
    stop("No standard or float32 data found for ", dataset_alias(dataset), call. = FALSE)
  }
  if (is.null(float_object) && need_float) {
    float_object <- list(
      data = as_float32_matrix(standard$data),
      labels = standard$labels,
      object_name = paste0(standard$object_name, "_generated_float32")
    )
  }
  labels <- standard$labels %||% float_object$labels %||% NULL
  if (!is.null(labels)) labels <- as.factor(labels)
  list(
    standard = standard,
    float = float_object,
    labels = labels,
    standard_path = standard_path,
    float_path = float_path
  )
}

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
kodama_cache_tag <- safe_name(args$kodama_cache_tag %||% "unknown_image")
cache_backend_for_dataset <- function(dataset, backend = NULL) {
  if (!is.null(backend) && length(backend) == 1L && nzchar(backend) &&
      !identical(backend, "auto")) {
    return(backend)
  }
  requested <- args$shared_cache_backend %||% ""
  if (nzchar(requested) && !identical(requested, "auto")) return(requested)
  if (exists("precompute_table", inherits = TRUE)) {
    table <- get("precompute_table", inherits = TRUE)
    if (is.data.frame(table) && "shared_cache_backend" %in% names(table)) {
      hit <- table[table$dataset == dataset_alias(dataset), , drop = FALSE]
      if (nrow(hit) && nzchar(hit$shared_cache_backend[[1L]])) {
        return(hit$shared_cache_backend[[1L]])
      }
    }
  }
  "cpu"
}

cache_paths <- function(dataset, backend = NULL) {
  stem <- safe_name(dataset_alias(dataset))
  backend <- cache_backend_for_dataset(dataset, backend)
  pca_name <- if (identical(backend, "cpu")) {
    sprintf("%s_pca_init_2d_seed4.rds", stem)
  } else {
    sprintf("%s_pca_init_%s_2d_seed4.rds", stem, backend)
  }
  manifest_name <- if (identical(backend, "cpu")) {
    sprintf("%s_precompute_manifest.rds", stem)
  } else {
    sprintf("%s_precompute_manifest_%s.rds", stem, backend)
  }
  list(
    knn = file.path(cache_dir, sprintf("%s_knn_%s_k%d.rds", stem, backend, k)),
    rtsne_knn = file.path(
      cache_dir,
      sprintf("%s_knn_cpu_rtsne_k%d.rds", stem, max(k, ceiling(3 * perplexity) + 1L))
    ),
    pca_init = file.path(cache_dir, pca_name),
    validation = file.path(
      cache_dir, sprintf("%s_validation_n%d_seed4.rds", stem, validation_sample_n)
    ),
    manifest = file.path(cache_dir, manifest_name)
  )
}

kodama_core_paths <- function(dataset, classifier, backend, worker_threads, worker_seed) {
  version <- tryCatch(
    as.character(utils::packageVersion("kodamaR")),
    error = function(...) "unavailable"
  )
  stem <- paste(
    safe_name(dataset_alias(dataset)), classifier, backend,
    paste0("t", as.integer(worker_threads)), paste0("seed", as.integer(worker_seed)),
    paste0("M", kodama_m), paste0("C", kodama_tcycle),
    paste0("P", kodama_ncomp), paste0("L", kodama_landmarks),
    paste0("G", kodama_graph_neighbors), paste0("K", k),
    paste0("v", safe_name(version)), paste0("build", kodama_cache_tag), sep = "_"
  )
  directory <- file.path(cache_dir, "kodama_core")
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  list(
    fit = file.path(directory, paste0(stem, ".rds")),
    metrics = file.path(directory, paste0(stem, "_metrics.rds"))
  )
}

save_rds_atomic <- function(object, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(pattern = paste0(basename(path), "."), tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(object, temporary, compress = compress)
  if (!file.rename(temporary, path)) {
    if (!file.copy(temporary, path, overwrite = TRUE)) {
      stop("Could not publish cache file: ", path, call. = FALSE)
    }
    unlink(temporary)
  }
  invisible(path)
}

call_supported <- function(fun, arguments) {
  formal_names <- names(formals(fun))
  if (!"..." %in% formal_names) {
    arguments <- arguments[names(arguments) %in% formal_names]
  }
  do.call(fun, arguments)
}

host_knn_subset <- function(knn, width) {
  knn <- publication_knn_host(knn)
  width <- min(as.integer(width), ncol(knn$indices))
  list(
    indices = knn$indices[, seq_len(width), drop = FALSE],
    distances = knn$distances[, seq_len(width), drop = FALSE]
  )
}

method_family <- function(method) {
  if (grepl("pca|irlba", method, ignore.case = TRUE)) return("PCA")
  if (grepl("tsne", method, ignore.case = TRUE)) return("t-SNE")
  if (grepl("umap|uwot", method, ignore.case = TRUE)) return("UMAP")
  "unknown"
}

method_is_kodama <- function(method) startsWith(method, "KODAMA_")
method_is_landmark <- function(method) {
  startsWith(method, "fastEmbedR_") && grepl("_landmark$", method)
}

landmark_baseline_method <- function(method) {
  if (!method_is_landmark(method)) return(NA_character_)
  backend <- method_backend(method)
  if (method_is_tsne(method)) {
    sprintf("fastEmbedR_opentsne_%s_full", backend)
  } else {
    sprintf("fastEmbedR_umap_%s_binary_full", backend)
  }
}

kodama_classifier_for_method <- function(method) {
  if (!method_is_kodama(method)) return(NA_character_)
  if (grepl("_plslda_", method, fixed = TRUE)) "pls_lda" else "knn"
}

kodama_visualization_for_method <- function(method) {
  if (!method_is_kodama(method)) return(NA_character_)
  if (grepl("opentsne", method, ignore.case = TRUE)) "opentsne" else "UMAP"
}

method_backend <- function(method) {
  if (grepl("_cuda|rapids", method, ignore.case = TRUE)) return("cuda")
  if (grepl("_metal", method, ignore.case = TRUE)) return("metal")
  "cpu"
}

method_scope <- function(method) {
  if (method_is_kodama(method)) return("full_pipeline")
  if (method_is_landmark(method)) return("landmark_pipeline")
  if (grepl("_knn$|_knn_|Rtsne_neighbors", method)) {
    return("embedding_from_precomputed_knn")
  }
  if (grepl("pca|irlba", method, ignore.case = TRUE)) return("pca_only")
  "full_pipeline"
}

default_methods <- function(group) {
  cpu <- c(
    "fastEmbedR_pca_cpu", "irlba_pca",
    "fastEmbedR_opentsne_cpu_full", "fastEmbedR_opentsne_cpu_knn",
    "Rtsne_full", "Rtsne_neighbors", "KlugerLab_FItSNE",
    "fastEmbedR_umap_cpu_fuzzy_full", "fastEmbedR_umap_cpu_fuzzy_knn",
    "fastEmbedR_umap_cpu_binary_full", "fastEmbedR_umap_cpu_binary_knn",
    "uwot_default", "uwot_fast_sgd", "uwot_knn",
    "umap_package", "umap_package_knn"
  )
  metal <- c(
    "fastEmbedR_pca_metal",
    "fastEmbedR_opentsne_metal_full", "fastEmbedR_opentsne_metal_knn",
    "fastEmbedR_umap_metal_fuzzy_full", "fastEmbedR_umap_metal_fuzzy_knn",
    "fastEmbedR_umap_metal_binary_full", "fastEmbedR_umap_metal_binary_knn"
  )
  cuda <- c(
    "fastEmbedR_pca_cuda",
    "fastEmbedR_opentsne_cuda_full", "fastEmbedR_opentsne_cuda_knn",
    "fastEmbedR_umap_cuda_fuzzy_full", "fastEmbedR_umap_cuda_fuzzy_knn",
    "fastEmbedR_umap_cuda_binary_full", "fastEmbedR_umap_cuda_binary_knn",
    "rapids_cuml_tsne_full", "rapids_cuml_umap_full"
  )
  switch(group, cpu = cpu, metal = metal, cuda = cuda, local = c(cpu, metal))
}
methods <- as_csv(args$methods, paste(default_methods(backend_group), collapse = ","))

method_uses_standard <- function(method) {
  grepl("Rtsne|FItSNE|uwot|umap_package|irlba|rapids|KODAMA", method)
}

method_input_type <- function(method) {
  if (startsWith(method, "fastEmbedR")) return("float32")
  if (method_is_kodama(method)) return("standard R double; KODAMA C++ float32")
  "standard R double"
}

method_is_tsne <- function(method) identical(method_family(method), "t-SNE")
method_is_umap <- function(method) identical(method_family(method), "UMAP")

method_threads <- function(method, requested) {
  if (method_backend(method) %in% c("cuda", "metal")) return(NA_integer_)
  as.integer(requested)
}

parameter_record <- function(method, requested_threads) {
  family <- method_family(method)
  scope <- method_scope(method)
  data.frame(
    method = method,
    family = family,
    backend = method_backend(method),
    timing_scope = scope,
    landmark_fraction = if (method_is_landmark(method)) landmark_fraction else NA_real_,
    n_neighbors = if (family == "UMAP") k else NA_integer_,
    perplexity = if (family == "t-SNE") perplexity else NA_real_,
    pca_components = if (family == "PCA") pca_ncomp else if (family == "t-SNE") 2L else NA_integer_,
    initialization = if (family == "t-SNE") {
      if (method_is_kodama(method)) {
        "KODAMA stored PCA initialization when exposed by kodamaR; otherwise native default"
      } else if (scope == "embedding_from_precomputed_knn") {
        "shared fastEmbedR rSVD PCA initialization"
      } else "method internal PCA initialization"
    } else if (family == "UMAP") {
      if (method_is_kodama(method)) {
        "KODAMA stored PCA initialization when exposed by kodamaR; otherwise native default"
      } else "method internal spectral initialization"
    } else "randomized truncated PCA",
    iterations_or_epochs = if (family == "t-SNE") {
      if (method_is_kodama(method)) {
        sprintf("KODAMA openTSNE: 250 early-exaggeration + %d optimization iterations", kodama_n_iter)
      } else if (grepl("Rtsne|FItSNE", method)) "750 iterations" else "fastEmbedR/openTSNE auto policy (250 early + 500 normal default)"
    } else if (family == "UMAP") {
      if (method_is_kodama(method)) {
        sprintf("KODAMA UMAP: %d epochs", kodama_n_epochs)
      } else if (grepl("fastEmbedR", method)) "fastEmbedR size-aware epoch policy" else "reference-package default"
    } else "randomized subspace iterations",
    early_exaggeration = if (family == "t-SNE") {
      if (method_is_kodama(method)) {
        "12 for first 250 iterations"
      } else if (grepl("Rtsne", method)) "12 for first 250 iterations" else "openTSNE/fastEmbedR auto policy"
    } else "not applicable",
    learning_rate = if (family == "t-SNE") {
      if (grepl("Rtsne", method)) "200" else "automatic"
    } else if (family == "UMAP") "method default/automatic" else "not applicable",
    metric = "euclidean",
    kodama_classifier = if (method_is_kodama(method)) {
      kodama_classifier_for_method(method)
    } else NA_character_,
    kodama_M = if (method_is_kodama(method)) kodama_m else NA_integer_,
    kodama_Tcycle = if (method_is_kodama(method)) kodama_tcycle else NA_integer_,
    kodama_ncomp = if (method_is_kodama(method)) kodama_ncomp else NA_integer_,
    kodama_landmarks = if (method_is_kodama(method)) kodama_landmarks else NA_integer_,
    requested_threads = requested_threads,
    effective_threads = method_threads(method, requested_threads),
    random_seeds = paste(seeds, collapse = ","),
    input_type = method_input_type(method),
    knn_boundary = if (method_is_kodama(method)) {
      "KODAMA graph is built once per classifier and reused by both visualizations"
    } else if (method_is_landmark(method)) {
      "landmark reference and projection KNN are internal and included in elapsed time"
    } else if (scope == "embedding_from_precomputed_knn") {
      "precomputed and excluded from elapsed time"
    } else if (family == "PCA") "not applicable" else "internal and included in elapsed time",
    knn_mode = if (scope == "embedding_from_precomputed_knn") {
      "shared fastEmbedR CPU HNSW cache, target recall 0.99"
    } else if (startsWith(method, "fastEmbedR")) {
      if (method_backend(method) == "cuda") "native CUDA exact/IVF auto route, target recall 0.99" else "native CPU HNSW or Metal exact/IVF auto route, target recall 0.99"
    } else if (method_is_kodama(method)) {
      "KODAMA-corrected graph retained by the shared classifier fit"
    } else "reference-package internal",
    graph_mode = if (method_is_kodama(method) && family == "UMAP") {
      "KODAMA visualization default"
    } else if (grepl("binary", method)) "binary" else if (family == "UMAP") "fuzzy/reference default" else "not applicable",
    notes = if (method_is_kodama(method)) {
      sprintf(
        "classifier=%s; M=%d; Tcycle=%d; ncomp<=%d; landmarks<=%d; shared core charged once per workflow",
        kodama_classifier_for_method(method), kodama_m, kodama_tcycle,
        kodama_ncomp, kodama_landmarks
      )
    } else if (method_is_landmark(method)) {
      sprintf(
        "explicit %.0f%% landmark approximation; matched full run is %s",
        100 * landmark_fraction, landmark_baseline_method(method)
      )
    } else if (grepl("binary", method)) "binary symmetric graph" else if (grepl("fuzzy", method)) "fuzzy simplicial graph" else "",
    stringsAsFactors = FALSE
  )
}

write_markdown <- function(x, path) {
  x <- as.data.frame(x, stringsAsFactors = FALSE)
  display <- x
  display[] <- lapply(display, function(value) {
    if (is.numeric(value)) value <- signif(value, 5)
    value <- as.character(value)
    value[is.na(value) | value == "NA"] <- ""
    gsub("\\|", "\\\\|", value)
  })
  con <- file(path, "wt")
  on.exit(close(con), add = TRUE)
  writeLines(paste0("| ", paste(names(display), collapse = " | "), " |"), con)
  writeLines(paste0("| ", paste(rep("---", ncol(display)), collapse = " | "), " |"), con)
  for (i in seq_len(nrow(display))) {
    writeLines(paste0("| ", paste(display[i, ], collapse = " | "), " |"), con)
  }
  invisible(path)
}

precompute_dataset <- function(dataset, validation_backends) {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required for precomputation.", call. = FALSE)
  }
  # Precomputation uses the float32 object alone when present. Loading the
  # standard double object at the same time would nearly double resident RAM on
  # FlowRepository, Tabula Muris, retina, and ImageNet.
  data <- load_dataset(dataset, need_standard = FALSE, need_float = TRUE)
  x <- as_float32_matrix(data$float$data)
  n <- nrow(x)
  if (k >= n) stop("k must be smaller than the dataset size.", call. = FALSE)
  cpu_allowed <- !identical(backend_group, "local") ||
    local_cpu_allowed_for(dataset, n)
  shared_cache_backend <- args$shared_cache_backend %||% "auto"
  if (identical(shared_cache_backend, "auto")) {
    shared_cache_backend <- if (cpu_allowed) "cpu" else "metal"
  }
  shared_cache_backend <- match.arg(shared_cache_backend, c("cpu", "metal", "cuda"))
  if (!cpu_allowed && identical(shared_cache_backend, "cpu")) {
    stop(
      "Local CPU work is disabled above ", local_cpu_max_n,
      " samples; choose a Metal shared-cache backend.", call. = FALSE
    )
  }
  if (!cpu_allowed) validation_backends <- setdiff(validation_backends, "cpu")
  paths <- cache_paths(dataset, shared_cache_backend)
  cache_threads <- max(threads_grid)
  manifest <- list(
    dataset = dataset_alias(dataset), n = n, p = ncol(x),
    k = k, perplexity = perplexity, cache_threads = cache_threads,
    shared_cache_backend = shared_cache_backend,
    local_cpu_allowed = cpu_allowed,
    local_cpu_max_n = local_cpu_max_n,
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    fastEmbedR_version = as.character(utils::packageVersion("fastEmbedR")),
    standard_path = data$standard_path,
    float_path = data$float_path
  )
  running_row <- data.frame(
    dataset = dataset_alias(dataset), status = "running", error = "",
    n = n, p = ncol(x), shared_cache_backend = shared_cache_backend,
    local_cpu_allowed = cpu_allowed, knn_file = paths$knn,
    rtsne_knn_file = if (file.exists(paths$rtsne_knn)) paths$rtsne_knn else NA_character_,
    pca_init_file = paths$pca_init, validation_file = paths$validation,
    stringsAsFactors = FALSE
  )
  if (precompute_worker && !is.null(args$worker_out)) {
    write.csv(running_row, args$worker_out, row.names = FALSE)
  }

  if (force || !file.exists(paths$knn)) {
    gc()
    elapsed <- system.time({
      knn <- fastEmbedR::precompute_knn(
        x, k = k, metric = "euclidean", backend = shared_cache_backend,
        n_threads = cache_threads
      )
    })[["elapsed"]]
    knn <- publication_knn_host(knn)
    saveRDS(knn, paths$knn, compress = FALSE)
    manifest$knn_elapsed_sec <- unname(elapsed)
  }

  rtsne_width <- min(n - 1L, max(k, ceiling(3 * perplexity) + 1L))
  needs_rtsne_knn <- "Rtsne_neighbors" %in% methods
  if (needs_rtsne_knn && (force || !file.exists(paths$rtsne_knn))) {
    estimated_gb <- n * rtsne_width * (4 + 8) / 1024^3
    max_cache_gb <- as_num(args$max_knn_cache_gb, 16)
    if (estimated_gb <= max_cache_gb) {
      gc()
      elapsed <- system.time({
        wide_knn <- fastEmbedR::precompute_knn(
          x, k = rtsne_width, metric = "euclidean", backend = shared_cache_backend,
          n_threads = cache_threads
        )
      })[["elapsed"]]
      wide_knn <- publication_knn_host(wide_knn)
      saveRDS(wide_knn, paths$rtsne_knn, compress = FALSE)
      manifest$rtsne_knn_elapsed_sec <- unname(elapsed)
      manifest$rtsne_knn_width <- rtsne_width
    } else {
      manifest$rtsne_knn_skipped <- sprintf(
        "estimated %.2f GB exceeds max_knn_cache_gb %.2f",
        estimated_gb, max_cache_gb
      )
    }
  }

  if (force || !file.exists(paths$pca_init)) {
    gc()
    elapsed <- system.time({
      pca_fit <- fastEmbedR::pca(
        x, ncomp = 2L, center = TRUE, scale = FALSE,
        backend = shared_cache_backend, seed = seeds[[1L]], opentsne_init = TRUE
      )
    })[["elapsed"]]
    pca_init <- publication_layout_matrix(pca_fit$opentsne_init)
    saveRDS(pca_init, paths$pca_init, compress = FALSE)
    manifest$pca_init_elapsed_sec <- unname(elapsed)
  }

  if (force || !file.exists(paths$validation)) {
    rows <- publication_sample_rows(n, validation_sample_n, seeds[[1L]] + 1009L)
    x_sample <- as_double_matrix(x[rows, , drop = FALSE])
    exact <- publication_exact_knn(x_sample, min(k, nrow(x_sample) - 1L))
    validation <- list(
      rows = rows,
      data = x_sample,
      labels = if (is.null(data$labels)) NULL else data$labels[rows],
      exact_knn = exact,
      dataset = dataset_alias(dataset)
    )
    saveRDS(validation, paths$validation, compress = FALSE)
  }

  validation <- readRDS(paths$validation)
  for (backend in validation_backends) {
    validation_path <- file.path(
      out_dir, "validation_knn",
      sprintf("%s_%s_knn.rds", safe_name(dataset_alias(dataset)), backend)
    )
    if (!force && file.exists(validation_path)) next
    validation_error <- tryCatch({
      validation_input <- as_float32_matrix(validation$data)
      backend_knn <- fastEmbedR::precompute_knn(
        validation_input,
        k = min(k, nrow(validation$data) - 1L),
        metric = "euclidean",
        backend = backend,
        n_threads = cache_threads
      )
      backend_knn <- publication_knn_host(backend_knn)
      saveRDS(backend_knn, validation_path, compress = FALSE)
      NULL
    }, error = function(e) conditionMessage(e))
    manifest[[paste0("validation_", backend, "_status")]] <-
      if (is.null(validation_error)) "success" else "failed"
    if (!is.null(validation_error)) {
      manifest[[paste0("validation_", backend, "_error")]] <- validation_error
    }
  }
  saveRDS(manifest, paths$manifest)
  data.frame(
    dataset = dataset_alias(dataset), status = "success", error = "",
    n = n, p = ncol(x),
    shared_cache_backend = shared_cache_backend,
    local_cpu_allowed = cpu_allowed,
    knn_file = paths$knn,
    rtsne_knn_file = if (file.exists(paths$rtsne_knn)) paths$rtsne_knn else NA_character_,
    pca_init_file = paths$pca_init,
    validation_file = paths$validation,
    stringsAsFactors = FALSE
  )
}

find_fitsne <- function() {
  candidates <- c(
    Sys.getenv("FASTEMBEDR_FAST_TSNE_PATH", ""),
    Sys.getenv("FAST_TSNE_PATH", ""),
    "/opt/fit-sne/bin/fast_tsne",
    file.path(Sys.getenv("HOME"), ".local", "bin", "fast_tsne"),
    Sys.which("fast_tsne")
  )
  candidates <- candidates[nzchar(candidates)]
  candidates <- candidates[file.exists(candidates) & file.access(candidates, 1L) == 0L]
  if (length(candidates)) candidates[[1L]] else ""
}

run_fitsne <- function(x, y_init = NULL) {
  exe <- find_fitsne()
  if (!nzchar(exe)) stop("KlugerLab FIt-SNE executable was not found.", call. = FALSE)
  wrapper <- NULL
  for (package in c("fftRtsne", "Spectre")) {
    if (requireNamespace(package, quietly = TRUE) &&
        exists("fftRtsne", envir = asNamespace(package), inherits = FALSE)) {
      wrapper <- get("fftRtsne", envir = asNamespace(package), inherits = FALSE)
      break
    }
  }
  if (is.null(wrapper)) {
    candidates <- c(
      Sys.getenv("FASTEMBEDR_FAST_TSNE_R", ""),
      Sys.getenv("FAST_TSNE_R", ""),
      "/opt/fit-sne/bin/fast_tsne.R",
      "/mnt/sata_ssd/FIt-SNE/fast_tsne.R",
      "/mnt/sata_ssd/FIt-SNE/bin/fast_tsne.R"
    )
    candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
    if (length(candidates)) {
      wrapper_env <- new.env(parent = .GlobalEnv)
      source(candidates[[1L]], local = wrapper_env, chdir = TRUE)
      if (exists("fftRtsne", envir = wrapper_env, inherits = FALSE)) {
        wrapper <- get("fftRtsne", envir = wrapper_env, inherits = FALSE)
      }
    }
  }
  if (is.null(wrapper)) stop("No FIt-SNE R wrapper was found.", call. = FALSE)
  available <- names(formals(wrapper))
  call <- list(
    X = as_double_matrix(x), dims = 2L, perplexity = perplexity,
    max_iter = 750L, rand_seed = seed, theta = 0.5,
    nthreads = threads, fast_tsne_path = exe, verbose = FALSE
  )
  if (!is.null(y_init)) {
    for (name in c("Y_init", "initial_config", "init", "initialization")) {
      if (name %in% available) call[[name]] <- y_init
    }
  }
  do.call(wrapper, call[names(call) %in% available])
}

python_executable_for_validation <- function() {
  candidates <- c(
    Sys.getenv("RETICULATE_PYTHON", ""),
    Sys.which("python3"), Sys.which("python")
  )
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (length(candidates)) candidates[[1L]] else ""
}

read_opentsne_affinity_binary <- function(path) {
  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  nnz <- readBin(con, integer(), n = 1L, size = 4L, endian = "little", signed = TRUE)
  if (!length(nnz) || nnz < 1L) stop("Invalid openTSNE affinity output.", call. = FALSE)
  i <- readBin(con, integer(), n = nnz, size = 4L, endian = "little", signed = TRUE)
  j <- readBin(con, integer(), n = nnz, size = 4L, endian = "little", signed = TRUE)
  weight <- readBin(con, numeric(), n = nnz, size = 4L, endian = "little")
  publication_sparse_triplet_edges(i, j, weight, one_based = FALSE)
}

reference_affinity_validation_table <- function() {
  helper <- file.path(script_dir, "reference_opentsne_affinity.py")
  python <- python_executable_for_validation()
  rows <- list()
  for (dataset in datasets) {
    path <- cache_paths(dataset)$validation
    base <- data.frame(
      dataset = dataset_alias(dataset), reference = "Python openTSNE exact affinity",
      status = "not_available", edge_jaccard = NA_real_, weight_pearson = NA_real_,
      weight_spearman = NA_real_, weight_l1_similarity = NA_real_,
      validation_sample_n = NA_integer_, error = "", stringsAsFactors = FALSE
    )
    if (!file.exists(path)) {
      base$error <- "validation cache is missing"
      rows[[length(rows) + 1L]] <- base
      next
    }
    validation <- readRDS(path)
    base$validation_sample_n <- nrow(validation$data)
    if (!nzchar(python) || !file.exists(helper)) {
      base$error <- "Python or reference_opentsne_affinity.py is unavailable"
      rows[[length(rows) + 1L]] <- base
      next
    }
    work <- file.path(out_dir, "reference_affinity", safe_name(dataset_alias(dataset)))
    dir.create(work, recursive = TRUE, showWarnings = FALSE)
    input <- file.path(work, "validation_float32.bin")
    output <- file.path(work, "opentsne_affinity.bin")
    log <- file.path(work, "opentsne_affinity.log")
    con <- file(input, open = "wb")
    writeBin(as.vector(t(as.matrix(validation$data))), con, size = 4L, endian = "little")
    close(con)
    command <- c(
      helper, paste0("--input=", input), paste0("--n=", nrow(validation$data)),
      paste0("--p=", ncol(validation$data)), paste0("--perplexity=", perplexity),
      paste0("--k=", min(k, nrow(validation$data) - 1L)),
      paste0("--threads=", max(threads_grid)), paste0("--seed=", seeds[[1L]]),
      paste0("--output=", output)
    )
    status <- tryCatch(
      system2(python, command, stdout = log, stderr = log,
              env = paste0("NUMBA_CACHE_DIR=", file.path(work, "numba_cache"))),
      error = function(e) 1L
    )
    if (!identical(as.integer(status), 0L) || !file.exists(output)) {
      base$status <- "failed"
      base$error <- if (file.exists(log)) paste(readLines(log, warn = FALSE), collapse = " | ") else "reference process failed"
      rows[[length(rows) + 1L]] <- base
      next
    }
    reference <- tryCatch(read_opentsne_affinity_binary(output), error = identity)
    if (inherits(reference, "error")) {
      base$status <- "failed"
      base$error <- conditionMessage(reference)
      rows[[length(rows) + 1L]] <- base
      next
    }
    candidate <- publication_sparse_affinities(validation$exact_knn, perplexity)
    agreement <- publication_edge_agreement(reference, candidate)
    base$status <- "success"
    base$edge_jaccard <- agreement$edge_jaccard
    base$weight_pearson <- agreement$weight_pearson
    base$weight_spearman <- agreement$weight_spearman
    base$weight_l1_similarity <- agreement$weight_l1_similarity
    rows[[length(rows) + 1L]] <- base
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

python_float32 <- function(x) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate is not installed.", call. = FALSE)
  }
  python <- Sys.getenv("RETICULATE_PYTHON", "")
  if (nzchar(python)) reticulate::use_python(python, required = FALSE)
  np <- reticulate::import("numpy", convert = FALSE)
  np$array(as_double_matrix(x), dtype = "float32", order = "C")
}

python_layout <- function(x) {
  np <- reticulate::import("numpy", convert = FALSE)
  cp <- tryCatch(reticulate::import("cupy", convert = FALSE), error = function(e) NULL)
  if (!is.null(cp)) x <- tryCatch(cp$asnumpy(x), error = function(e) x)
  x <- tryCatch(x$to_numpy(), error = function(e) x)
  x <- reticulate::py_to_r(np$asarray(x))
  publication_layout_matrix(x)
}

run_rapids <- function(method, x) {
  manifold <- reticulate::import("cuml.manifold", convert = FALSE)
  x <- python_float32(x)
  if (grepl("tsne", method, ignore.case = TRUE)) {
    model <- manifold$TSNE(
      n_components = 2L,
      perplexity = as.numeric(perplexity),
      n_neighbors = as.integer(max(ceiling(3 * perplexity) + 1L, 4L)),
      method = "fft", random_state = as.integer(seed), verbose = FALSE
    )
  } else {
    model <- manifold$UMAP(
      n_neighbors = as.integer(k), n_components = 2L,
      metric = "euclidean", random_state = as.integer(seed), verbose = FALSE
    )
  }
  python_layout(model$fit_transform(x))
}

build_kodama_core <- function(dataset_data, classifier, backend) {
  if (!requireNamespace("kodamaR", quietly = TRUE)) {
    stop("kodamaR is not installed.", call. = FALSE)
  }
  if (!backend %in% c("cpu", "cuda")) {
    stop("KODAMA benchmark backends are limited to cpu and cuda.", call. = FALSE)
  }
  if (is.null(dataset_data$standard)) {
    stop("Standard R data are required by the current kodamaR wrapper.", call. = FALSE)
  }
  x <- as_double_matrix(dataset_data$standard$data)
  n <- nrow(x)
  p <- ncol(x)
  effective_ncomp <- max(1L, min(kodama_ncomp, p, n - 1L))
  effective_landmarks <- max(1L, min(kodama_landmarks, n))
  effective_graph_neighbors <- max(1L, min(kodama_graph_neighbors, n - 1L))
  effective_knn_k <- max(1L, min(k, n - 1L))
  fun <- getExportedValue("kodamaR", "KODAMA.matrix")
  arguments <- list(
    data = x,
    M = as.integer(kodama_m),
    Tcycle = as.integer(kodama_tcycle),
    ncomp = as.integer(effective_ncomp),
    landmarks = as.integer(effective_landmarks),
    n.cores = as.integer(threads),
    graph.neighbors = as.integer(effective_graph_neighbors),
    knn.k = as.integer(effective_knn_k),
    metric = "euclidean",
    classifier = classifier,
    backend = backend,
    seed = as.integer(seed),
    visual.init = TRUE,
    progress = FALSE,
    apply.kodama.dissimilarity = TRUE
  )
  elapsed <- system.time({
    fit <- call_supported(fun, arguments)
  })[["elapsed"]]
  list(
    fit = fit,
    elapsed_sec = unname(elapsed),
    n = n,
    p = p,
    classifier = classifier,
    backend = backend,
    threads = as.integer(threads),
    seed = as.integer(seed),
    effective_ncomp = effective_ncomp,
    effective_landmarks = effective_landmarks,
    effective_graph_neighbors = effective_graph_neighbors,
    effective_knn_k = effective_knn_k
  )
}

load_kodama_core <- function(method) {
  classifier <- kodama_classifier_for_method(method)
  backend <- method_backend(method)
  paths <- kodama_core_paths(args$dataset, classifier, backend, threads, seed)
  if (!file.exists(paths$fit) || !file.exists(paths$metrics)) {
    stop(
      "Shared KODAMA core cache is unavailable for ", classifier, "/", backend,
      ". The core worker must complete before visualization.", call. = FALSE
    )
  }
  cached <- readRDS(paths$fit)
  metrics <- readRDS(paths$metrics)
  list(
    fit = if (is.list(cached) && !is.null(cached$fit)) cached$fit else cached,
    metrics = metrics,
    paths = paths,
    classifier = classifier,
    backend = backend
  )
}

run_kodama_visualization <- function(method, core) {
  fun <- getExportedValue("kodamaR", "KODAMA.visualization")
  visualization <- kodama_visualization_for_method(method)
  init_key <- if (identical(visualization, "UMAP")) "umap" else "opentsne"
  stored_init <- if (is.list(core$fit) && is.list(core$fit$visual_init)) {
    core$fit$visual_init[[init_key]] %||% NULL
  } else if (is.list(core$fit)) {
    core$fit$visual_init %||% NULL
  } else NULL
  arguments <- list(
    x = core$fit,
    method = visualization,
    init = stored_init,
    k = as.integer(k),
    metric = "euclidean",
    backend = core$backend,
    n.cores = as.integer(threads),
    gpu.device = 0L,
    n.epochs = as.integer(kodama_n_epochs),
    n.iter = as.integer(kodama_n_iter),
    perplexity = as.numeric(perplexity),
    seed = as.integer(seed)
  )
  call_supported(fun, arguments)
}

fit_layout <- function(fit) {
  publication_layout_matrix(fit)
}

extract_fastembedr_timings <- function(fit) {
  out <- c(preprocess_sec = NA_real_, knn_sec = NA_real_, init_sec = NA_real_,
           graph_or_affinity_sec = NA_real_, embedding_sec = NA_real_)
  if (!is.list(fit)) return(out)
  timings <- fit$timings %||% NULL
  if (!is.null(timings) && is.matrix(timings) && "elapsed" %in% colnames(timings)) {
    if ("preprocess" %in% rownames(timings)) out[["preprocess_sec"]] <- timings["preprocess", "elapsed"]
    if ("knn" %in% rownames(timings)) out[["knn_sec"]] <- timings["knn", "elapsed"]
    if ("embedding" %in% rownames(timings)) out[["embedding_sec"]] <- timings["embedding", "elapsed"]
  }
  out
}

run_method <- function(method, dataset_data) {
  backend <- method_backend(method)
  paths <- cache_paths(args$dataset)
  x_fast <- if (!startsWith(method, "fastEmbedR") || is.null(dataset_data$float)) NULL else {
    as_float32_matrix(dataset_data$float$data)
  }
  x_standard <- if (!method_uses_standard(method) || is.null(dataset_data$standard)) NULL else {
    as_double_matrix(dataset_data$standard$data)
  }
  shared_knn <- if (method_scope(method) == "embedding_from_precomputed_knn") {
    readRDS(paths$knn)
  } else NULL
  pca_init <- if (method_scope(method) == "embedding_from_precomputed_knn" &&
                  method_is_tsne(method)) readRDS(paths$pca_init) else NULL
  shared_knn_fast <- if (is.null(shared_knn)) NULL else list(
    indices = shared_knn$indices,
    distances = as_float32_matrix(shared_knn$distances)
  )
  pca_init_fast <- if (is.null(pca_init)) NULL else as_float32_matrix(pca_init)
  kodama_core <- if (method_is_kodama(method)) load_kodama_core(method) else NULL
  component <- c(preprocess_sec = NA_real_, knn_sec = NA_real_, init_sec = NA_real_,
                 graph_or_affinity_sec = NA_real_, embedding_sec = NA_real_)

  elapsed <- system.time({
    fit <- switch(
      method,
      fastEmbedR_pca_cpu = fastEmbedR::pca(
        x_fast, ncomp = pca_ncomp, center = TRUE, scale = FALSE,
        backend = "cpu", seed = seed, opentsne_init = TRUE
      ),
      fastEmbedR_pca_metal = fastEmbedR::pca(
        x_fast, ncomp = pca_ncomp, center = TRUE, scale = FALSE,
        backend = "metal", seed = seed, opentsne_init = TRUE
      ),
      fastEmbedR_pca_cuda = fastEmbedR::pca(
        x_fast, ncomp = pca_ncomp, center = TRUE, scale = FALSE,
        backend = "cuda", seed = seed, opentsne_init = TRUE
      ),
      irlba_pca = {
        if (is.null(x_standard)) stop("Standard R data are required for irlba.", call. = FALSE)
        if (!requireNamespace("irlba", quietly = TRUE)) stop("irlba is not installed.", call. = FALSE)
        irlba::prcomp_irlba(x_standard, n = pca_ncomp, center = TRUE, scale. = FALSE)
      },
      fastEmbedR_opentsne_cpu_full = fastEmbedR::opentsne(
        x_fast, perplexity = perplexity, backend = "cpu", n_threads = threads,
        seed = seed, record_costs = FALSE
      ),
      fastEmbedR_opentsne_metal_full = fastEmbedR::opentsne(
        x_fast, perplexity = perplexity, backend = "metal", n_threads = threads,
        seed = seed, record_costs = FALSE
      ),
      fastEmbedR_opentsne_cuda_full = fastEmbedR::opentsne(
        x_fast, perplexity = perplexity, backend = "cuda", n_threads = threads,
        seed = seed, record_costs = FALSE
      ),
      fastEmbedR_opentsne_cpu_landmark = fastEmbedR::landmark_tsne(
        x_fast, landmarks = landmark_fraction, n_neighbors = k,
        perplexity = perplexity, standardize = FALSE, backend = "cpu",
        n_threads = threads, seed = seed, keep_knn = FALSE,
        verbose = FALSE
      ),
      fastEmbedR_opentsne_metal_landmark = fastEmbedR::landmark_tsne(
        x_fast, landmarks = landmark_fraction, n_neighbors = k,
        perplexity = perplexity, standardize = FALSE, backend = "metal",
        n_threads = threads, seed = seed, keep_knn = FALSE,
        verbose = FALSE
      ),
      fastEmbedR_opentsne_cuda_landmark = fastEmbedR::landmark_tsne(
        x_fast, landmarks = landmark_fraction, n_neighbors = k,
        perplexity = perplexity, standardize = FALSE, backend = "cuda",
        n_threads = threads, seed = seed, keep_knn = FALSE,
        verbose = FALSE
      ),
      fastEmbedR_opentsne_cpu_knn = fastEmbedR::opentsne_knn(
        shared_knn_fast, n_neighbors = ceiling(perplexity), perplexity = perplexity,
        Y_init = pca_init_fast, backend = "cpu", n_threads = threads,
        seed = seed, record_costs = FALSE
      ),
      fastEmbedR_opentsne_metal_knn = fastEmbedR::opentsne_knn(
        shared_knn_fast, n_neighbors = ceiling(perplexity), perplexity = perplexity,
        Y_init = pca_init_fast, backend = "metal", n_threads = threads,
        seed = seed, record_costs = FALSE
      ),
      fastEmbedR_opentsne_cuda_knn = fastEmbedR::opentsne_knn(
        shared_knn_fast, n_neighbors = ceiling(perplexity), perplexity = perplexity,
        Y_init = pca_init_fast, backend = "cuda", n_threads = threads,
        seed = seed, record_costs = FALSE
      ),
      KODAMA_plslda_opentsne_cpu = run_kodama_visualization(method, kodama_core),
      KODAMA_knn_opentsne_cpu = run_kodama_visualization(method, kodama_core),
      KODAMA_plslda_opentsne_cuda = run_kodama_visualization(method, kodama_core),
      KODAMA_knn_opentsne_cuda = run_kodama_visualization(method, kodama_core),
      Rtsne_full = {
        if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne is not installed.", call. = FALSE)
        Rtsne::Rtsne(
          x_standard, dims = 2L, perplexity = perplexity, pca = TRUE,
          theta = 0.5, max_iter = 750L, num_threads = threads,
          check_duplicates = FALSE, verbose = FALSE
        )$Y
      },
      Rtsne_neighbors = {
        if (!requireNamespace("Rtsne", quietly = TRUE)) stop("Rtsne is not installed.", call. = FALSE)
        if (!file.exists(paths$rtsne_knn)) stop("Wide Rtsne KNN cache is unavailable.", call. = FALSE)
        wide <- readRDS(paths$rtsne_knn)
        Rtsne::Rtsne_neighbors(
          index = wide$indices, distance = wide$distances,
          dims = 2L, perplexity = perplexity, theta = 0.5,
          max_iter = 750L, Y_init = pca_init, num_threads = threads,
          verbose = FALSE
        )$Y
      },
      KlugerLab_FItSNE = run_fitsne(x_standard),
      fastEmbedR_umap_cpu_fuzzy_full = fastEmbedR::umap(
        x_fast, n_neighbors = k, backend = "cpu", n_threads = threads,
        graph_mode = "fuzzy", seed = seed
      ),
      fastEmbedR_umap_cpu_binary_full = fastEmbedR::umap(
        x_fast, n_neighbors = k, backend = "cpu", n_threads = threads,
        graph_mode = "binary", seed = seed
      ),
      fastEmbedR_umap_metal_fuzzy_full = fastEmbedR::umap(
        x_fast, n_neighbors = k, backend = "metal", n_threads = threads,
        graph_mode = "fuzzy", seed = seed
      ),
      fastEmbedR_umap_metal_binary_full = fastEmbedR::umap(
        x_fast, n_neighbors = k, backend = "metal", n_threads = threads,
        graph_mode = "binary", seed = seed
      ),
      fastEmbedR_umap_cuda_fuzzy_full = fastEmbedR::umap(
        x_fast, n_neighbors = k, backend = "cuda", n_threads = threads,
        graph_mode = "fuzzy", seed = seed
      ),
      fastEmbedR_umap_cuda_binary_full = fastEmbedR::umap(
        x_fast, n_neighbors = k, backend = "cuda", n_threads = threads,
        graph_mode = "binary", seed = seed
      ),
      fastEmbedR_umap_cpu_binary_landmark = fastEmbedR::landmark_umap(
        x_fast, landmarks = landmark_fraction, n_neighbors = k,
        standardize = FALSE, backend = "cpu", n_threads = threads,
        seed = seed, keep_knn = FALSE, verbose = FALSE
      ),
      fastEmbedR_umap_metal_binary_landmark = fastEmbedR::landmark_umap(
        x_fast, landmarks = landmark_fraction, n_neighbors = k,
        standardize = FALSE, backend = "metal", n_threads = threads,
        seed = seed, keep_knn = FALSE, verbose = FALSE
      ),
      fastEmbedR_umap_cuda_binary_landmark = fastEmbedR::landmark_umap(
        x_fast, landmarks = landmark_fraction, n_neighbors = k,
        standardize = FALSE, backend = "cuda", n_threads = threads,
        seed = seed, keep_knn = FALSE, verbose = FALSE
      ),
      fastEmbedR_umap_cpu_fuzzy_knn = fastEmbedR::umap_knn(
        shared_knn_fast, backend = "cpu", n_threads = threads,
        graph_mode = "fuzzy", seed = seed
      ),
      fastEmbedR_umap_cpu_binary_knn = fastEmbedR::umap_knn(
        shared_knn_fast, backend = "cpu", n_threads = threads,
        graph_mode = "binary", seed = seed
      ),
      fastEmbedR_umap_metal_fuzzy_knn = fastEmbedR::umap_knn(
        shared_knn_fast, backend = "metal", n_threads = threads,
        graph_mode = "fuzzy", seed = seed
      ),
      fastEmbedR_umap_metal_binary_knn = fastEmbedR::umap_knn(
        shared_knn_fast, backend = "metal", n_threads = threads,
        graph_mode = "binary", seed = seed
      ),
      fastEmbedR_umap_cuda_fuzzy_knn = fastEmbedR::umap_knn(
        shared_knn_fast, backend = "cuda", n_threads = threads,
        graph_mode = "fuzzy", seed = seed
      ),
      fastEmbedR_umap_cuda_binary_knn = fastEmbedR::umap_knn(
        shared_knn_fast, backend = "cuda", n_threads = threads,
        graph_mode = "binary", seed = seed
      ),
      KODAMA_plslda_umap_cpu = run_kodama_visualization(method, kodama_core),
      KODAMA_knn_umap_cpu = run_kodama_visualization(method, kodama_core),
      KODAMA_plslda_umap_cuda = run_kodama_visualization(method, kodama_core),
      KODAMA_knn_umap_cuda = run_kodama_visualization(method, kodama_core),
      uwot_default = {
        if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
        uwot::umap(
          x_standard, n_neighbors = k, n_threads = threads,
          n_sgd_threads = 1L, fast_sgd = FALSE, init = "spectral",
          seed = seed, verbose = FALSE
        )
      },
      uwot_fast_sgd = {
        if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
        uwot::umap(
          x_standard, n_neighbors = k, n_threads = threads,
          n_sgd_threads = threads, fast_sgd = TRUE, init = "spectral",
          seed = seed, verbose = FALSE
        )
      },
      uwot_knn = {
        if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot is not installed.", call. = FALSE)
        idx <- cbind(seq_len(nrow(shared_knn$indices)), shared_knn$indices)
        dst <- cbind(0, shared_knn$distances)
        uwot::umap(
          X = NULL, n_neighbors = k, nn_method = list(idx = idx, dist = dst),
          n_threads = threads, n_sgd_threads = threads, fast_sgd = TRUE,
          init = "spectral", seed = seed, verbose = FALSE
        )
      },
      umap_package = {
        if (!requireNamespace("umap", quietly = TRUE)) stop("umap is not installed.", call. = FALSE)
        config <- umap::umap.defaults
        config$n_neighbors <- k
        config$random_state <- seed
        umap::umap(x_standard, config = config)$layout
      },
      umap_package_knn = {
        if (!requireNamespace("umap", quietly = TRUE)) stop("umap is not installed.", call. = FALSE)
        uknn <- umap::umap.knn(shared_knn$indices, shared_knn$distances)
        config <- umap::umap.defaults
        config$n_neighbors <- k
        config$knn <- uknn
        config$random_state <- seed
        umap::umap(x_standard, knn = uknn, config = config)$layout
      },
      rapids_cuml_tsne_full = run_rapids(method, x_standard),
      rapids_cuml_umap_full = run_rapids(method, x_standard),
      stop("Unknown method: ", method, call. = FALSE)
    )
  })[["elapsed"]]

  if (startsWith(method, "fastEmbedR") && method_scope(method) == "full_pipeline") {
    component <- extract_fastembedr_timings(fit)
  }
  landmark_extra <- list()
  if (method_is_landmark(method)) {
    fit_metrics <- fit$metrics[1L, , drop = FALSE]
    value_or_na <- function(name) {
      if (name %in% names(fit_metrics)) as.numeric(fit_metrics[[name]][[1L]]) else NA_real_
    }
    component[["preprocess_sec"]] <- value_or_na("preprocess_elapsed")
    component[["embedding_sec"]] <- unname(elapsed) -
      ifelse(is.finite(component[["preprocess_sec"]]), component[["preprocess_sec"]], 0)
    landmark_extra <- list(
      landmark_fraction = value_or_na("landmark_fraction"),
      n_landmarks = as.integer(value_or_na("n_landmarks")),
      reference_embedding_sec = value_or_na("reference_embedding_elapsed"),
      landmark_projection_knn_sec = value_or_na("landmark_projection_knn_elapsed"),
      landmark_refinement_sec = value_or_na("landmark_refinement_elapsed"),
      landmark_transform_sec = value_or_na("transform_elapsed")
    )
  }
  if (method_scope(method) == "embedding_from_precomputed_knn") {
    component[["embedding_sec"]] <- unname(elapsed)
  }
  if (method_is_kodama(method)) {
    component[["embedding_sec"]] <- unname(elapsed)
  }
  if (method_family(method) == "PCA") {
    component[["init_sec"]] <- unname(elapsed)
  }
  layout <- if (method_family(method) == "PCA") {
    if (identical(method, "irlba_pca")) fit$x[, seq_len(min(2L, ncol(fit$x))), drop = FALSE] else fit$scores[, seq_len(min(2L, ncol(fit$scores))), drop = FALSE]
  } else {
    fit_layout(fit)
  }
  workflow_elapsed <- if (method_is_kodama(method)) {
    core_elapsed <- as.numeric(
      kodama_core$metrics$core_runtime_sec %||%
        kodama_core$metrics$elapsed_sec %||% NA_real_
    )
    if (!is.finite(core_elapsed)) {
      stop("KODAMA core cache has no valid runtime.", call. = FALSE)
    }
    core_elapsed + unname(elapsed)
  } else {
    unname(elapsed)
  }
  extra <- if (method_is_kodama(method)) {
    list(
      kodama_classifier = kodama_core$classifier,
      kodama_core_sec = workflow_elapsed - unname(elapsed),
      kodama_visualization_sec = unname(elapsed),
      kodama_core_cache_file = kodama_core$paths$fit,
      kodama_core_reused = TRUE
    )
  } else landmark_extra
  list(
    fit = fit, layout = publication_layout_matrix(layout),
    elapsed_sec = workflow_elapsed, component = component, extra = extra
  )
}

score_embedding <- function(dataset_data, layout, family, dataset) {
  empty <- list(
    trustworthiness = NA_real_, knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_, knn_preservation_50 = NA_real_,
    silhouette = NA_real_, label_knn_accuracy = NA_real_,
    tsne_kl = NA_real_, quality_sample_n = NA_integer_
  )
  if (identical(family, "PCA")) return(empty)
  n <- nrow(layout)
  source_data <- dataset_data$standard$data %||% dataset_data$float$data
  p <- ncol(source_data)
  budget_n <- floor(sqrt(2 * quality_max_distance_ops / max(1, p)))
  min_metric_n <- min(n, 52L)
  effective_quality_n <- max(
    min_metric_n,
    min(n, quality_sample_n, max(1L, as.integer(budget_n)))
  )
  rows <- publication_sample_rows(n, effective_quality_n, seed + 101L)
  x_sample <- as_double_matrix(source_data[rows, , drop = FALSE])
  labels <- dataset_data$labels
  label_sample <- if (is.null(labels)) NULL else labels[rows]
  score <- tryCatch(
    fastEmbedR::evaluate_embedding(
      x_sample, layout[rows, , drop = FALSE], labels = label_sample,
      k = c(15L, 30L, 50L),
      sample_size_for_global_metrics = min(2000L, length(rows)),
      sample_size_for_local_metrics = min(2000L, length(rows)),
      seed = seed, n_threads = threads, dataset = dataset
    ),
    error = function(e) NULL
  )
  if (!is.null(score)) {
    empty$trustworthiness <- as.numeric(score$trustworthiness %||% NA_real_)
    empty$knn_preservation_15 <- as.numeric(score$knn_preservation_15 %||% NA_real_)
    empty$knn_preservation_30 <- as.numeric(
      score$knn_preservation_30 %||% score$knn_preservation %||% NA_real_
    )
    empty$knn_preservation_50 <- as.numeric(score$knn_preservation_50 %||% NA_real_)
    empty$silhouette <- as.numeric(score$silhouette %||% NA_real_)
    empty$label_knn_accuracy <- as.numeric(
      score$label_knn_accuracy %||% score$nn_accuracy %||% NA_real_
    )
  }
  empty$quality_sample_n <- length(rows)
  if (identical(family, "t-SNE")) {
    validation <- readRDS(cache_paths(dataset)$validation)
    affinity <- publication_sparse_affinities(
      validation$exact_knn, min(perplexity, ncol(validation$exact_knn$indices))
    )
    empty$tsne_kl <- publication_tsne_kl(layout[validation$rows, , drop = FALSE], affinity)
  }
  empty
}

worker_result_template <- function(dataset, method, status = "failed", error = NA_character_) {
  data.frame(
    dataset = dataset_alias(dataset), method = method,
    family = method_family(method), backend = method_backend(method),
    timing_scope = method_scope(method), seed = seed,
    requested_threads = threads, effective_threads = method_threads(method, threads),
    status = status, error = error,
    n = NA_integer_, p = NA_integer_, k = if (method_is_umap(method)) k else NA_integer_,
    perplexity = if (method_is_tsne(method)) perplexity else NA_real_,
    input_type = gsub(" ", "_", method_input_type(method), fixed = TRUE),
    total_runtime_sec = NA_real_, preprocess_sec = NA_real_, knn_sec = NA_real_,
    init_sec = NA_real_, graph_or_affinity_sec = NA_real_, embedding_sec = NA_real_,
    peak_ram_kb = NA_real_, peak_ram_gb = NA_real_,
    gpu_memory_scope = NA_character_, gpu_baseline_mb = NA_real_,
    peak_gpu_mb = NA_real_, peak_gpu_delta_mb = NA_real_,
    trustworthiness = NA_real_, knn_preservation_15 = NA_real_,
    knn_preservation_30 = NA_real_, knn_preservation_50 = NA_real_,
    silhouette = NA_real_, label_knn_accuracy = NA_real_, tsne_kl = NA_real_,
    landmark_fraction = if (method_is_landmark(method)) landmark_fraction else NA_real_,
    n_landmarks = NA_integer_, reference_embedding_sec = NA_real_,
    landmark_projection_knn_sec = NA_real_, landmark_refinement_sec = NA_real_,
    landmark_transform_sec = NA_real_,
    kodama_classifier = if (method_is_kodama(method)) {
      kodama_classifier_for_method(method)
    } else NA_character_,
    kodama_core_sec = NA_real_, kodama_visualization_sec = NA_real_,
    kodama_core_peak_ram_gb = NA_real_, kodama_visualization_peak_ram_gb = NA_real_,
    kodama_core_peak_gpu_delta_mb = NA_real_,
    kodama_visualization_peak_gpu_delta_mb = NA_real_,
    kodama_core_cache_file = NA_character_, kodama_core_reused = NA,
    quality_sample_n = NA_integer_, layout_file = NA_character_,
    plot_file = NA_character_, stringsAsFactors = FALSE
  )
}

worker_main <- function() {
  dataset <- args$dataset %||% stop("--dataset is required.", call. = FALSE)
  method <- args$method %||% stop("--method is required.", call. = FALSE)
  worker_out <- args$worker_out %||% stop("--worker-out is required.", call. = FALSE)
  set_threads(threads)
  data <- load_dataset(
    dataset,
    need_standard = method_uses_standard(method) || !startsWith(method, "fastEmbedR"),
    need_float = startsWith(method, "fastEmbedR")
  )
  gc()
  result <- run_method(method, data)
  layout <- result$layout
  layout_file <- file.path(
    layout_dir,
    sprintf(
      "%s_%s_threads%s_seed%d.rds", safe_name(dataset_alias(dataset)),
      safe_name(method), threads, seed
    )
  )
  plot_file <- file.path(
    out_dir, "plots",
    sprintf(
      "%s_%s_threads%s_seed%d.png", safe_name(dataset_alias(dataset)),
      safe_name(method), threads, seed
    )
  )
  landmark_indices <- if (method_is_landmark(method)) {
    as.integer(result$fit$landmarks$indices %||% integer())
  } else {
    integer()
  }
  saveRDS(
    list(
      layout = layout, labels = data$labels, dataset = dataset_alias(dataset),
      method = method, backend = method_backend(method), seed = seed,
      requested_threads = threads, timing_scope = method_scope(method),
      landmark_indices = landmark_indices
    ),
    layout_file, compress = FALSE
  )
  publication_clean_plot(
    layout,
    data$labels,
    plot_file,
    landmark_indices = landmark_indices
  )
  scores <- score_embedding(data, layout, method_family(method), dataset)
  row <- worker_result_template(dataset, method, status = "success", error = NA_character_)
  row$n <- nrow(layout)
  row$p <- ncol(data$float$data %||% data$standard$data)
  row$total_runtime_sec <- result$elapsed_sec
  for (name in names(result$component)) row[[name]] <- result$component[[name]]
  for (name in names(result$extra %||% list())) row[[name]] <- result$extra[[name]]
  for (name in names(scores)) row[[name]] <- scores[[name]]
  row$layout_file <- layout_file
  row$plot_file <- plot_file
  write.csv(row, worker_out, row.names = FALSE)
  invisible(row)
}

read_key_values <- function(path) {
  if (!file.exists(path)) return(list())
  lines <- readLines(path, warn = FALSE)
  values <- strsplit(lines, "=", fixed = TRUE)
  values <- values[vapply(values, length, integer(1)) >= 2L]
  out <- lapply(values, function(x) paste(x[-1L], collapse = "="))
  names(out) <- vapply(values, `[[`, character(1), 1L)
  out
}

parse_memory_files <- function(time_file, gpu_file) {
  time_lines <- if (file.exists(time_file)) readLines(time_file, warn = FALSE) else character()
  rss_line <- grep("Maximum resident set size", time_lines, value = TRUE)
  rss <- if (length(rss_line)) {
    suppressWarnings(as.numeric(sub(".*: *", "", tail(rss_line, 1L))))
  } else NA_real_
  gpu <- read_key_values(gpu_file)
  list(
    peak_ram_kb = rss,
    peak_ram_gb = rss / 1024^2,
    gpu_memory_scope = gpu$gpu_memory_scope %||% NA_character_,
    gpu_baseline_mb = suppressWarnings(as.numeric(gpu$gpu_baseline_mb %||% NA_real_)),
    peak_gpu_mb = suppressWarnings(as.numeric(gpu$gpu_peak_mb %||% NA_real_)),
    peak_gpu_delta_mb = suppressWarnings(as.numeric(gpu$gpu_peak_delta_mb %||% NA_real_))
  )
}

kodama_core_result_template <- function(dataset, classifier, backend,
                                        status = "failed", error = NA_character_) {
  data.frame(
    dataset = dataset_alias(dataset), classifier = classifier, backend = backend,
    seed = seed, requested_threads = threads,
    status = status, error = error, n = NA_integer_, p = NA_integer_,
    M = kodama_m, Tcycle = kodama_tcycle, requested_ncomp = kodama_ncomp,
    requested_landmarks = kodama_landmarks,
    graph_neighbors = kodama_graph_neighbors, knn_k = k,
    kodama_cache_tag = kodama_cache_tag,
    effective_ncomp = NA_integer_, effective_landmarks = NA_integer_,
    core_runtime_sec = NA_real_, peak_ram_kb = NA_real_, peak_ram_gb = NA_real_,
    gpu_memory_scope = NA_character_, gpu_baseline_mb = NA_real_,
    peak_gpu_mb = NA_real_, peak_gpu_delta_mb = NA_real_,
    cache_file = NA_character_, core_reused = FALSE,
    stringsAsFactors = FALSE
  )
}

kodama_core_worker_main <- function() {
  dataset <- args$dataset %||% stop("--dataset is required.", call. = FALSE)
  classifier <- args$kodama_classifier %||%
    stop("--kodama-classifier is required.", call. = FALSE)
  backend <- args$kodama_backend %||% backend_group
  worker_out <- args$worker_out %||% stop("--worker-out is required.", call. = FALSE)
  if (!classifier %in% c("knn", "pls_lda")) {
    stop("KODAMA classifier must be knn or pls_lda.", call. = FALSE)
  }
  set_threads(threads)
  data <- load_dataset(dataset, need_standard = TRUE, need_float = FALSE)
  gc()
  result <- build_kodama_core(data, classifier, backend)
  paths <- kodama_core_paths(dataset, classifier, backend, threads, seed)
  save_rds_atomic(result, paths$fit, compress = FALSE)
  row <- kodama_core_result_template(
    dataset, classifier, backend, status = "success", error = NA_character_
  )
  row$n <- result$n
  row$p <- result$p
  row$effective_ncomp <- result$effective_ncomp
  row$effective_landmarks <- result$effective_landmarks
  row$core_runtime_sec <- result$elapsed_sec
  row$cache_file <- paths$fit
  write.csv(row, worker_out, row.names = FALSE)
  invisible(row)
}

run_kodama_core_isolated <- function(dataset, classifier, backend,
                                     worker_threads, worker_seed) {
  paths <- kodama_core_paths(dataset, classifier, backend, worker_threads, worker_seed)
  if (!force && file.exists(paths$fit) && file.exists(paths$metrics)) {
    cached <- tryCatch(readRDS(paths$metrics), error = function(...) NULL)
    if (is.data.frame(cached) && nrow(cached) &&
        identical(cached$status[[1L]], "success")) {
      cached$core_reused <- TRUE
      return(cached)
    }
  }
  stem <- sprintf(
    "%s_KODAMA_core_%s_%s_threads%d_seed%d",
    safe_name(dataset_alias(dataset)), classifier, backend,
    worker_threads, worker_seed
  )
  csv <- file.path(out_dir, "worker_results", paste0(stem, ".csv"))
  log <- file.path(out_dir, "logs", paste0(stem, ".log"))
  time_file <- file.path(out_dir, "memory", paste0(stem, "_ram.txt"))
  gpu_file <- file.path(out_dir, "memory", paste0(stem, "_gpu.txt"))
  monitor <- file.path(script_dir, "benchmark_worker_monitor.sh")
  command <- c(
    monitor, time_file, gpu_file, as.character(timeout),
    file.path(R.home("bin"), "Rscript"), script_path,
    "--kodama-core-worker=TRUE",
    paste0("--backend-group=", backend_group),
    paste0("--kodama-backend=", backend),
    paste0("--kodama-classifier=", classifier),
    paste0("--base-dir=", base_dir), paste0("--data-root=", data_root),
    paste0("--out-dir=", out_dir), paste0("--layout-dir=", layout_dir),
    paste0("--cache-dir=", cache_dir),
    paste0("--dataset=", dataset), paste0("--worker-out=", csv),
    paste0("--threads=", worker_threads), paste0("--seed=", worker_seed),
    paste0("--k=", k), paste0("--perplexity=", perplexity),
    paste0("--kodama-m=", kodama_m),
    paste0("--kodama-tcycle=", kodama_tcycle),
    paste0("--kodama-ncomp=", kodama_ncomp),
    paste0("--kodama-landmarks=", kodama_landmarks),
    paste0("--kodama-graph-neighbors=", kodama_graph_neighbors),
    paste0("--kodama-n-epochs=", kodama_n_epochs),
    paste0("--kodama-n-iter=", kodama_n_iter),
    paste0("--kodama-cache-tag=", kodama_cache_tag)
  )
  status <- system2("bash", command, stdout = log, stderr = log)
  row <- if (file.exists(csv)) {
    read.csv(csv, stringsAsFactors = FALSE)
  } else {
    kodama_core_result_template(
      dataset, classifier, backend,
      status = if (identical(as.integer(status), 124L)) "timeout" else "failed",
      error = describe_worker_exit(status, log, "KODAMA core worker")
    )
  }
  if (!identical(as.integer(status), 0L) && identical(row$status[[1L]], "success")) {
    row$status <- if (identical(as.integer(status), 124L)) "timeout" else "failed"
    row$error <- describe_worker_exit(status, log, "KODAMA core worker")
  }
  memory <- parse_memory_files(time_file, gpu_file)
  for (name in names(memory)) row[[name]] <- memory[[name]]
  row$core_reused <- FALSE
  write.csv(row, csv, row.names = FALSE)
  if (identical(row$status[[1L]], "success") && file.exists(paths$fit)) {
    save_rds_atomic(row, paths$metrics, compress = FALSE)
  }
  row
}

is_resource_failure <- function(status, error) {
  identical(as.character(status), "failed") &&
    grepl(
      paste(
        "failed to allocate|cannot allocate|out of memory|std::bad_alloc|\\bOOM\\b",
        "isolated worker exited with status (9|137)|SIGKILL|oom[_ -]?kill|killed",
        sep = "|"
      ),
      as.character(error %||% ""), ignore.case = TRUE, perl = TRUE
    )
}

is_backend_health_failure <- function(status, error) {
  identical(as.character(status), "failed") &&
    grepl(
      "uncorrectable ECC|cudaErrorECCUncorrectable|GPU has fallen off the bus",
      as.character(error %||% ""), ignore.case = TRUE, perl = TRUE
    )
}

describe_worker_exit <- function(status, log_file, worker_kind = "isolated worker") {
  code <- suppressWarnings(as.integer(status))
  detail <- if (file.exists(log_file)) {
    lines <- tail(readLines(log_file, warn = FALSE), 30L)
    lines <- trimws(lines[nzchar(trimws(lines))])
    if (length(lines)) tail(lines, 1L) else ""
  } else {
    ""
  }
  message <- if (identical(code, 137L) || identical(code, 9L)) {
    sprintf(
      "%s received SIGKILL (exit %d), normally because the Slurm cgroup memory limit was exceeded",
      worker_kind, code
    )
  } else if (identical(code, 124L)) {
    sprintf("%s exceeded the configured timeout", worker_kind)
  } else {
    sprintf("%s exited with status %s", worker_kind, as.character(status))
  }
  if (nzchar(detail) && !identical(detail, message)) {
    paste0(message, "; last log line: ", detail)
  } else {
    message
  }
}

run_isolated_worker <- function(
  dataset, method, worker_threads, worker_seed, shared_cache_backend = "cpu"
) {
  stem <- sprintf(
    "%s_%s_threads%d_seed%d", safe_name(dataset_alias(dataset)),
    safe_name(method), worker_threads, worker_seed
  )
  csv <- file.path(out_dir, "worker_results", paste0(stem, ".csv"))
  log <- file.path(out_dir, "logs", paste0(stem, ".log"))
  time_file <- file.path(out_dir, "memory", paste0(stem, "_ram.txt"))
  gpu_file <- file.path(out_dir, "memory", paste0(stem, "_gpu.txt"))
  if (!force && file.exists(csv)) {
    existing <- tryCatch(
      read.csv(csv, stringsAsFactors = FALSE),
      error = function(...) NULL
    )
    if (!is.null(existing) && nrow(existing) &&
        existing$status[[1L]] %in% c("success", "skipped")) {
      return(existing)
    }
  }
  monitor <- file.path(script_dir, "benchmark_worker_monitor.sh")
  command <- c(
    monitor, time_file, gpu_file, as.character(timeout),
    file.path(R.home("bin"), "Rscript"), script_path,
    "--worker=TRUE",
    paste0("--backend-group=", backend_group),
    paste0("--base-dir=", base_dir), paste0("--data-root=", data_root),
    paste0("--out-dir=", out_dir), paste0("--layout-dir=", layout_dir),
    paste0("--cache-dir=", cache_dir),
    paste0("--dataset=", dataset), paste0("--method=", method),
    paste0("--worker-out=", csv), paste0("--threads=", worker_threads),
    paste0("--seed=", worker_seed), paste0("--seeds=", worker_seed),
    paste0("--k=", k), paste0("--perplexity=", perplexity),
    paste0("--quality-sample-n=", quality_sample_n),
    paste0("--quality-max-distance-ops=", quality_max_distance_ops),
    paste0("--validation-sample-n=", validation_sample_n),
    paste0("--pca-ncomp=", pca_ncomp),
    paste0("--landmark-fraction=", landmark_fraction),
    paste0("--kodama-m=", kodama_m),
    paste0("--kodama-tcycle=", kodama_tcycle),
    paste0("--kodama-ncomp=", kodama_ncomp),
    paste0("--kodama-landmarks=", kodama_landmarks),
    paste0("--kodama-graph-neighbors=", kodama_graph_neighbors),
    paste0("--kodama-n-epochs=", kodama_n_epochs),
    paste0("--kodama-n-iter=", kodama_n_iter),
    paste0("--kodama-cache-tag=", kodama_cache_tag),
    paste0("--shared-cache-backend=", shared_cache_backend),
    paste0("--local-cpu-max-n=", local_cpu_max_n),
    paste0("--local-cpu-exceptions=", paste(local_cpu_exceptions, collapse = ","))
  )
  status <- system2("bash", command, stdout = log, stderr = log)
  row <- if (file.exists(csv)) {
    read.csv(csv, stringsAsFactors = FALSE)
  } else {
    worker_result_template(
      dataset, method,
      status = if (identical(status, 124L)) "timeout" else "failed",
      error = describe_worker_exit(status, log, "isolated method worker")
    )
  }
  memory <- parse_memory_files(time_file, gpu_file)
  for (name in names(memory)) row[[name]] <- memory[[name]]
  if (method_is_kodama(method)) {
    classifier <- kodama_classifier_for_method(method)
    core_paths <- kodama_core_paths(
      dataset, classifier, method_backend(method), worker_threads, worker_seed
    )
    core <- if (file.exists(core_paths$metrics)) {
      tryCatch(readRDS(core_paths$metrics), error = function(...) NULL)
    } else NULL
    row$kodama_visualization_peak_ram_gb <- memory$peak_ram_gb
    row$kodama_visualization_peak_gpu_delta_mb <- memory$peak_gpu_delta_mb
    if (is.data.frame(core) && nrow(core)) {
      row$kodama_core_peak_ram_gb <- core$peak_ram_gb[[1L]]
      row$kodama_core_peak_gpu_delta_mb <- core$peak_gpu_delta_mb[[1L]]
      finite_max <- function(...) {
        values <- suppressWarnings(as.numeric(c(...)))
        values <- values[is.finite(values)]
        if (length(values)) max(values) else NA_real_
      }
      row$peak_ram_gb <- finite_max(
        row$kodama_core_peak_ram_gb, row$kodama_visualization_peak_ram_gb
      )
      row$peak_ram_kb <- row$peak_ram_gb * 1024^2
      row$peak_gpu_delta_mb <- finite_max(
        row$kodama_core_peak_gpu_delta_mb,
        row$kodama_visualization_peak_gpu_delta_mb
      )
      row$peak_gpu_mb <- finite_max(core$peak_gpu_mb[[1L]], memory$peak_gpu_mb)
      row$gpu_baseline_mb <- finite_max(
        core$gpu_baseline_mb[[1L]], memory$gpu_baseline_mb
      )
      row$gpu_memory_scope <- paste(
        "maximum of isolated KODAMA core and visualization",
        "device-wide measurements"
      )
    }
  }
  write.csv(row, csv, row.names = FALSE)
  row
}

aggregate_runs <- function(runs) {
  if (!nrow(runs)) return(data.frame())
  group_columns <- c(
    "dataset", "method", "family", "backend", "timing_scope",
    "requested_threads", "effective_threads", "n", "p", "k", "perplexity",
    "input_type", "landmark_fraction"
  )
  metric_columns <- c(
    "total_runtime_sec", "preprocess_sec", "knn_sec", "init_sec",
    "graph_or_affinity_sec", "embedding_sec", "peak_ram_gb",
    "peak_gpu_mb", "peak_gpu_delta_mb", "trustworthiness",
    "knn_preservation_15", "knn_preservation_30", "knn_preservation_50",
    "silhouette", "label_knn_accuracy", "tsne_kl",
    "n_landmarks", "reference_embedding_sec", "landmark_projection_knn_sec",
    "landmark_refinement_sec", "landmark_transform_sec",
    "kodama_core_sec", "kodama_visualization_sec",
    "kodama_core_peak_ram_gb", "kodama_visualization_peak_ram_gb",
    "kodama_core_peak_gpu_delta_mb",
    "kodama_visualization_peak_gpu_delta_mb"
  )
  key_parts <- lapply(runs[group_columns], function(value) {
    value <- as.character(value)
    value[is.na(value)] <- "<NA>"
    value
  })
  key <- do.call(paste, c(key_parts, sep = "\r"))
  pieces <- split(runs, key)
  rows <- lapply(pieces, function(piece) {
    base <- piece[1L, group_columns, drop = FALSE]
    base$n_runs <- nrow(piece)
    base$n_success <- sum(piece$status == "success")
    base$status <- if (base$n_success == base$n_runs) "success" else if (base$n_success > 0L) "partial" else "failed"
    for (metric in metric_columns) {
      values <- suppressWarnings(as.numeric(piece[[metric]]))
      stats <- publication_median_iqr(values)
      for (stat in names(stats)) base[[paste0(metric, "_", stat)]] <- stats[[stat]]
    }
    base
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$dataset, out$family, out$method, out$requested_threads), , drop = FALSE]
}

load_layout_record <- function(path) {
  if (is.na(path) || !file.exists(path)) return(NULL)
  tryCatch(readRDS(path), error = function(e) NULL)
}

landmark_row_overlap <- function(reference, candidate, selected, k) {
  reference <- publication_knn_host(reference)$indices
  candidate <- publication_knn_host(candidate)$indices
  k <- min(as.integer(k), ncol(reference), ncol(candidate))
  selected <- as.integer(selected)
  selected <- selected[selected >= 1L & selected <= nrow(reference)]
  if (!length(selected) || k < 1L || nrow(reference) != nrow(candidate)) {
    return(NA_real_)
  }
  mean(vapply(selected, function(i) {
    length(intersect(reference[i, seq_len(k)], candidate[i, seq_len(k)])) / k
  }, numeric(1)))
}

landmark_validation_table <- function(runs) {
  if (!nrow(runs) || !"landmark_fraction" %in% names(runs)) return(data.frame())
  candidates <- runs[
    runs$status == "success" & vapply(runs$method, method_is_landmark, logical(1)),
    , drop = FALSE
  ]
  if (!nrow(candidates)) return(data.frame())
  scalar <- function(row, name) {
    if (name %in% names(row)) suppressWarnings(as.numeric(row[[name]][[1L]])) else NA_real_
  }
  safe_ratio <- function(numerator, denominator) {
    if (!is.finite(numerator) || !is.finite(denominator) || denominator <= 0) {
      return(NA_real_)
    }
    numerator / denominator
  }
  rows <- list()
  for (i in seq_len(nrow(candidates))) {
    landmark_run <- candidates[i, , drop = FALSE]
    baseline_method <- landmark_baseline_method(landmark_run$method[[1L]])
    baseline <- runs[
      runs$status == "success" & runs$dataset == landmark_run$dataset[[1L]] &
        runs$method == baseline_method &
        runs$requested_threads == landmark_run$requested_threads[[1L]] &
        runs$seed == landmark_run$seed[[1L]],
      , drop = FALSE
    ]
    if (!nrow(baseline)) next
    baseline <- baseline[1L, , drop = FALSE]
    full_record <- load_layout_record(baseline$layout_file[[1L]])
    landmark_record <- load_layout_record(landmark_run$layout_file[[1L]])
    if (is.null(full_record) || is.null(landmark_record)) next
    full_layout <- publication_layout_matrix(full_record$layout)
    landmark_layout <- publication_layout_matrix(landmark_record$layout)
    if (!identical(dim(full_layout), dim(landmark_layout))) next

    validation <- readRDS(cache_paths(landmark_run$dataset[[1L]])$validation)
    validation_rows <- validation$rows[validation$rows <= nrow(full_layout)]
    if (length(validation_rows) < 3L) next
    full_sample <- full_layout[validation_rows, , drop = FALSE]
    landmark_sample <- landmark_layout[validation_rows, , drop = FALSE]
    landmark_indices <- as.integer(landmark_record$landmark_indices %||% integer())
    projected_positions <- which(!validation_rows %in% landmark_indices)
    landmark_positions <- which(validation_rows %in% landmark_indices)
    neighbor_k <- min(30L, nrow(full_sample) - 1L)
    full_knn <- publication_exact_knn(full_sample, neighbor_k)
    landmark_knn <- publication_exact_knn(landmark_sample, neighbor_k)
    all_proc <- publication_procrustes(full_sample, landmark_sample)
    projected_proc <- if (length(projected_positions) >= 3L) {
      publication_procrustes(
        full_sample[projected_positions, , drop = FALSE],
        landmark_sample[projected_positions, , drop = FALSE]
      )
    } else data.frame(rmsd = NA_real_, correlation = NA_real_)
    landmark_proc <- if (length(landmark_positions) >= 3L) {
      publication_procrustes(
        full_sample[landmark_positions, , drop = FALSE],
        landmark_sample[landmark_positions, , drop = FALSE]
      )
    } else data.frame(rmsd = NA_real_, correlation = NA_real_)

    rows[[length(rows) + 1L]] <- data.frame(
      dataset = landmark_run$dataset[[1L]], family = landmark_run$family[[1L]],
      backend = landmark_run$backend[[1L]],
      requested_threads = landmark_run$requested_threads[[1L]],
      seed = landmark_run$seed[[1L]], baseline_method = baseline_method,
      landmark_method = landmark_run$method[[1L]],
      landmark_fraction = scalar(landmark_run, "landmark_fraction"),
      n_landmarks = scalar(landmark_run, "n_landmarks"),
      n_projected = nrow(landmark_layout) - scalar(landmark_run, "n_landmarks"),
      full_runtime_sec = scalar(baseline, "total_runtime_sec"),
      landmark_runtime_sec = scalar(landmark_run, "total_runtime_sec"),
      speedup_vs_full = safe_ratio(
        scalar(baseline, "total_runtime_sec"),
        scalar(landmark_run, "total_runtime_sec")
      ),
      full_peak_ram_gb = scalar(baseline, "peak_ram_gb"),
      landmark_peak_ram_gb = scalar(landmark_run, "peak_ram_gb"),
      ram_ratio_vs_full = safe_ratio(
        scalar(landmark_run, "peak_ram_gb"),
        scalar(baseline, "peak_ram_gb")
      ),
      full_peak_gpu_delta_mb = scalar(baseline, "peak_gpu_delta_mb"),
      landmark_peak_gpu_delta_mb = scalar(landmark_run, "peak_gpu_delta_mb"),
      gpu_memory_ratio_vs_full = safe_ratio(
        scalar(landmark_run, "peak_gpu_delta_mb"),
        scalar(baseline, "peak_gpu_delta_mb")
      ),
      trustworthiness_full = scalar(baseline, "trustworthiness"),
      trustworthiness_landmark = scalar(landmark_run, "trustworthiness"),
      trustworthiness_delta = scalar(landmark_run, "trustworthiness") -
        scalar(baseline, "trustworthiness"),
      knn_preservation_15_full = scalar(baseline, "knn_preservation_15"),
      knn_preservation_15_landmark = scalar(landmark_run, "knn_preservation_15"),
      knn_preservation_15_delta = scalar(landmark_run, "knn_preservation_15") -
        scalar(baseline, "knn_preservation_15"),
      label_knn_accuracy_full = scalar(baseline, "label_knn_accuracy"),
      label_knn_accuracy_landmark = scalar(landmark_run, "label_knn_accuracy"),
      label_knn_accuracy_delta = scalar(landmark_run, "label_knn_accuracy") -
        scalar(baseline, "label_knn_accuracy"),
      procrustes_rmsd_all = all_proc$rmsd[[1L]],
      procrustes_correlation_all = all_proc$correlation[[1L]],
      procrustes_rmsd_projected = projected_proc$rmsd[[1L]],
      procrustes_correlation_projected = projected_proc$correlation[[1L]],
      procrustes_rmsd_landmarks = landmark_proc$rmsd[[1L]],
      procrustes_correlation_landmarks = landmark_proc$correlation[[1L]],
      embedding_knn_overlap_15_all = landmark_row_overlap(
        full_knn, landmark_knn, seq_len(nrow(full_sample)), min(15L, neighbor_k)
      ),
      embedding_knn_overlap_15_projected = landmark_row_overlap(
        full_knn, landmark_knn, projected_positions, min(15L, neighbor_k)
      ),
      reference_embedding_sec = scalar(landmark_run, "reference_embedding_sec"),
      projection_knn_sec = scalar(landmark_run, "landmark_projection_knn_sec"),
      refinement_sec = scalar(landmark_run, "landmark_refinement_sec"),
      transform_sec = scalar(landmark_run, "landmark_transform_sec"),
      validation_sample_n = length(validation_rows),
      projected_validation_n = length(projected_positions),
      stringsAsFactors = FALSE
    )
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

stability_table <- function(runs) {
  layout_paths <- as.character(runs$layout_file)
  layout_exists <- !is.na(layout_paths) & nzchar(layout_paths)
  layout_exists[layout_exists] <- file.exists(layout_paths[layout_exists])
  ok <- runs$status == "success" & runs$family %in% c("t-SNE", "UMAP") &
    layout_exists
  x <- runs[ok, , drop = FALSE]
  if (!nrow(x)) return(data.frame())
  group_columns <- c("dataset", "method", "backend", "requested_threads")
  groups <- split(x, interaction(x[group_columns], drop = TRUE, lex.order = TRUE))
  rows <- list()
  for (piece in groups) {
    if (nrow(piece) < 2L) next
    validation <- readRDS(cache_paths(piece$dataset[[1L]])$validation)
    pairs <- utils::combn(seq_len(nrow(piece)), 2L)
    for (column in seq_len(ncol(pairs))) {
      a <- piece[pairs[1L, column], , drop = FALSE]
      b <- piece[pairs[2L, column], , drop = FALSE]
      la <- load_layout_record(a$layout_file)$layout
      lb <- load_layout_record(b$layout_file)$layout
      rows[[length(rows) + 1L]] <- cbind(
        a[, group_columns, drop = FALSE],
        seed_a = a$seed, seed_b = b$seed,
        publication_procrustes(
          la[validation$rows, , drop = FALSE],
          lb[validation$rows, , drop = FALSE]
        ),
        neighbor_stability_15 = publication_knn_overlap(
          publication_exact_knn(la[validation$rows, , drop = FALSE], 15L),
          publication_exact_knn(lb[validation$rows, , drop = FALSE], 15L),
          15L
        ),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

backend_validation_table <- function(validation_backends) {
  rows <- list()
  for (dataset in datasets) {
    validation_path <- cache_paths(dataset)$validation
    if (!file.exists(validation_path)) next
    validation <- readRDS(validation_path)
    reference_knn <- validation$exact_knn
    reference_affinity <- publication_sparse_affinities(
      reference_knn, min(perplexity, ncol(reference_knn$indices))
    )
    for (backend in validation_backends) {
      path <- file.path(
        out_dir, "validation_knn",
        sprintf("%s_%s_knn.rds", safe_name(dataset_alias(dataset)), backend)
      )
      if (!file.exists(path)) next
      candidate <- readRDS(path)
      candidate_affinity <- publication_sparse_affinities(
        candidate, min(perplexity, ncol(candidate$indices))
      )
      affinity <- publication_edge_agreement(reference_affinity, candidate_affinity)
      for (graph_mode in c("fuzzy", "binary")) {
        reference_graph <- publication_umap_edges(
          reference_knn, graph_mode = graph_mode, n_threads = max(threads_grid)
        )
        candidate_graph <- publication_umap_edges(
          candidate, graph_mode = graph_mode, n_threads = max(threads_grid)
        )
        graph <- publication_edge_agreement(reference_graph, candidate_graph)
        rows[[length(rows) + 1L]] <- data.frame(
          dataset = dataset_alias(dataset), backend = backend,
          graph_mode = graph_mode,
          knn_recall_at_k = publication_knn_overlap(reference_knn, candidate, k),
          affinity_edge_jaccard = affinity$edge_jaccard,
          affinity_weight_pearson = affinity$weight_pearson,
          affinity_weight_spearman = affinity$weight_spearman,
          affinity_weight_l1_similarity = affinity$weight_l1_similarity,
          umap_graph_edge_jaccard = graph$edge_jaccard,
          umap_graph_weight_pearson = graph$weight_pearson,
          umap_graph_weight_spearman = graph$weight_spearman,
          umap_graph_weight_l1_similarity = graph$weight_l1_similarity,
          validation_sample_n = nrow(validation$data),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

reference_graph_validation_table <- function() {
  if (!"cpu" %in% validation_backends ||
      !requireNamespace("uwot", quietly = TRUE) ||
      !requireNamespace("Matrix", quietly = TRUE)) return(data.frame())
  rows <- list()
  for (dataset in datasets) {
    path <- cache_paths(dataset)$validation
    if (!file.exists(path)) next
    validation <- readRDS(path)
    knn <- publication_knn_host(validation$exact_knn)
    idx <- cbind(seq_len(nrow(knn$indices)), knn$indices)
    dst <- cbind(0, knn$distances)
    for (graph_mode in c("fuzzy", "binary")) {
      fast_graph <- publication_umap_edges(
        knn, graph_mode = graph_mode, n_threads = max(threads_grid)
      )
      uwot_graph <- tryCatch(
        uwot::similarity_graph(
          X = NULL, n_neighbors = ncol(knn$indices),
          nn_method = list(idx = idx, dist = dst),
          n_threads = max(threads_grid),
          binary_edge_weights = identical(graph_mode, "binary"),
          verbose = FALSE
        ),
        error = function(e) NULL
      )
      if (is.null(uwot_graph)) next
      agreement <- publication_edge_agreement(
        publication_sparse_matrix_edges(uwot_graph), fast_graph
      )
      rows[[length(rows) + 1L]] <- data.frame(
        dataset = dataset_alias(dataset), graph_mode = graph_mode,
        reference = "uwot::similarity_graph",
        edge_jaccard = agreement$edge_jaccard,
        weight_pearson = agreement$weight_pearson,
        weight_spearman = agreement$weight_spearman,
        weight_l1_similarity = agreement$weight_l1_similarity,
        validation_sample_n = nrow(validation$data),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

pca_agreement_table <- function(runs) {
  successful <- runs[runs$status == "success" & runs$family == "PCA", , drop = FALSE]
  if (!nrow(successful)) return(data.frame())
  rows <- list()
  for (dataset in unique(successful$dataset)) {
    for (run_seed in unique(successful$seed[successful$dataset == dataset])) {
      reference <- successful[
        successful$dataset == dataset & successful$seed == run_seed &
          successful$method == "irlba_pca",
        , drop = FALSE
      ]
      if (!nrow(reference)) next
      reference <- reference[which.max(reference$requested_threads), , drop = FALSE]
      ref_layout <- load_layout_record(reference$layout_file)$layout
      candidates <- successful[
        successful$dataset == dataset & successful$seed == run_seed &
          startsWith(successful$method, "fastEmbedR_pca_"),
        , drop = FALSE
      ]
      for (i in seq_len(nrow(candidates))) {
        candidate <- candidates[i, , drop = FALSE]
        candidate_layout <- load_layout_record(candidate$layout_file)$layout
        agreement <- publication_procrustes(ref_layout, candidate_layout)
        rows[[length(rows) + 1L]] <- data.frame(
          dataset = dataset, seed = run_seed,
          reference = "irlba::prcomp_irlba",
          candidate = candidate$method,
          backend = candidate$backend,
          procrustes_rmsd = agreement$rmsd,
          procrustes_correlation = agreement$correlation,
          reference_runtime_sec = reference$total_runtime_sec,
          candidate_runtime_sec = candidate$total_runtime_sec,
          reference_peak_ram_gb = reference$peak_ram_gb,
          candidate_peak_ram_gb = candidate$peak_ram_gb,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

write_reproducibility <- function() {
  writeLines(capture.output(utils::sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
  git_commit <- tryCatch(
    trimws(suppressWarnings(system2(
      "git", c("-C", dirname(script_dir), "rev-parse", "HEAD"),
      stdout = TRUE, stderr = FALSE
    ))),
    error = function(e) NA_character_
  )
  if (!length(git_commit) || !nzchar(git_commit[[1L]])) git_commit <- NA_character_
  nvidia <- tryCatch(
    system2("nvidia-smi", c("--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"), stdout = TRUE, stderr = TRUE),
    error = function(e) NA_character_
  )
  package_names <- c(
    "fastEmbedR", "float", "Rcpp", "Rtsne", "uwot", "umap", "irlba",
    "reticulate", "kodamaR"
  )
  versions <- vapply(package_names, function(package) {
    if (requireNamespace(package, quietly = TRUE)) {
      as.character(utils::packageVersion(package))
    } else NA_character_
  }, character(1))
  fastembedr_dll <- tryCatch(
    system.file("libs", paste0("fastEmbedR", .Platform$dynlib.ext),
                package = "fastEmbedR"),
    error = function(e) ""
  )
  fastembedr_dll_md5 <- if (nzchar(fastembedr_dll) && file.exists(fastembedr_dll)) {
    unname(tools::md5sum(fastembedr_dll))
  } else {
    NA_character_
  }
  lines <- c(
    paste0("generated_at=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    paste0("git_commit=", git_commit),
    paste0("container_image_path=", Sys.getenv(
      "FASTEMBEDR_IMAGE_PATH", unset = NA_character_
    )),
    paste0("container_image_resolved=", Sys.getenv(
      "FASTEMBEDR_IMAGE_RESOLVED", unset = NA_character_
    )),
    paste0("container_image_identity=", Sys.getenv(
      "FASTEMBEDR_IMAGE_IDENTITY", unset = NA_character_
    )),
    paste0("fastEmbedR_dll=", fastembedr_dll),
    paste0("fastEmbedR_dll_md5=", fastembedr_dll_md5),
    paste0("slurm_job_id=", Sys.getenv("SLURM_JOB_ID", unset = NA_character_)),
    paste0("slurm_job_nodelist=", Sys.getenv(
      "SLURM_JOB_NODELIST", unset = NA_character_
    )),
    paste0("slurm_ntasks=", Sys.getenv("SLURM_NTASKS", unset = NA_character_)),
    paste0("slurm_cpus_per_task=", Sys.getenv(
      "SLURM_CPUS_PER_TASK", unset = NA_character_
    )),
    paste0("slurm_cpus_on_node=", Sys.getenv(
      "SLURM_CPUS_ON_NODE", unset = NA_character_
    )),
    paste0("results_dir=", out_dir),
    paste0("layout_dir=", layout_dir),
    paste0("backend_group=", backend_group),
    paste0("datasets=", paste(datasets, collapse = ",")),
    paste0("methods=", paste(methods, collapse = ",")),
    paste0("seeds=", paste(seeds, collapse = ",")),
    paste0("threads_grid=", paste(threads_grid, collapse = ",")),
    paste0("k=", k), paste0("perplexity=", perplexity),
    paste0("quality_sample_n=", quality_sample_n),
    paste0("quality_max_distance_ops=", quality_max_distance_ops),
    paste0("validation_sample_n=", validation_sample_n),
    paste0("landmark_fraction=", landmark_fraction),
    paste0("reference_validations=", reference_validations),
    paste0("kodama_M=", kodama_m),
    paste0("kodama_Tcycle=", kodama_tcycle),
    paste0("kodama_ncomp=", kodama_ncomp),
    paste0("kodama_landmarks=", kodama_landmarks),
    paste0("kodama_graph_neighbors=", kodama_graph_neighbors),
    paste0("kodama_n_epochs=", kodama_n_epochs),
    paste0("kodama_n_iter=", kodama_n_iter),
    paste0("kodama_cache_tag=", kodama_cache_tag),
    paste0("local_cpu_max_n=", local_cpu_max_n),
    paste0("local_cpu_exceptions=", paste(local_cpu_exceptions, collapse = ",")),
    paste0("R=", R.version.string),
    paste0("platform=", R.version$platform),
    paste0("nvidia=", paste(nvidia, collapse = " | ")),
    paste0("package_", names(versions), "=", versions)
  )
  writeLines(lines, file.path(out_dir, "reproducibility_manifest.txt"))
  writeLines(paste(commandArgs(FALSE), collapse = " "), file.path(out_dir, "benchmark_command.txt"))
}

make_summary_figures <- function(summary) {
  successful <- summary$status %in% c("success", "partial") &
    is.finite(summary$total_runtime_sec_median)
  if (!any(successful)) return(invisible(NULL))
  plot_data <- summary[
    successful & summary$timing_scope %in% c("full_pipeline", "landmark_pipeline"),
    , drop = FALSE
  ]
  if (nrow(plot_data)) {
    png(file.path(out_dir, "runtime_median_iqr.png"), width = 2200, height = 1500, res = 180)
    old <- par(no.readonly = TRUE)
    on.exit({par(old); dev.off()}, add = TRUE)
    datasets_to_plot <- head(unique(plot_data$dataset), 9L)
    par(mfrow = c(3, 3), mar = c(8, 4, 2, 1))
    for (dataset in datasets_to_plot) {
      z <- plot_data[plot_data$dataset == dataset, , drop = FALSE]
      centers <- barplot(
        z$total_runtime_sec_median, names.arg = z$method, las = 2,
        cex.names = 0.55, ylab = "Total runtime (s)", main = dataset,
        col = ifelse(z$backend == "cpu", "#4C78A8", ifelse(z$backend == "metal", "#59A14F", "#E15759"))
      )
      variable <- is.finite(z$total_runtime_sec_q1) &
        is.finite(z$total_runtime_sec_q3) &
        z$total_runtime_sec_q3 > z$total_runtime_sec_q1
      if (any(variable)) {
        arrows(
          centers[variable], z$total_runtime_sec_q1[variable],
          centers[variable], z$total_runtime_sec_q3[variable],
          angle = 90, code = 3, length = 0.04
        )
      }
    }
  }
  invisible(NULL)
}

if (kodama_core_worker) {
  tryCatch({
    kodama_core_worker_main()
  }, error = function(e) {
    row <- kodama_core_result_template(
      args$dataset %||% NA_character_,
      args$kodama_classifier %||% NA_character_,
      args$kodama_backend %||% backend_group,
      status = "failed", error = conditionMessage(e)
    )
    write.csv(
      row,
      args$worker_out %||% file.path(out_dir, "kodama_core_worker_failed.csv"),
      row.names = FALSE
    )
    message(conditionMessage(e))
    quit(status = 1L)
  })
  quit(status = 0L)
}

if (precompute_worker) {
  tryCatch({
    validation_backends <- as_csv(args$validation_backends, backend_group)
    result <- precompute_dataset(args$dataset, validation_backends)
    write.csv(result, args$worker_out, row.names = FALSE)
  }, error = function(e) {
    existing <- if (!is.null(args$worker_out) && file.exists(args$worker_out)) {
      tryCatch(read.csv(args$worker_out, stringsAsFactors = FALSE), error = function(...) NULL)
    } else NULL
    if (is.null(existing) || !nrow(existing)) {
      existing <- data.frame(
        dataset = dataset_alias(args$dataset), status = "failed", error = "",
        n = NA_real_, p = NA_real_, shared_cache_backend = NA_character_,
        local_cpu_allowed = NA, knn_file = NA_character_,
        rtsne_knn_file = NA_character_, pca_init_file = NA_character_,
        validation_file = NA_character_, stringsAsFactors = FALSE
      )
    }
    existing$status <- "failed"
    existing$error <- conditionMessage(e)
    write.csv(
      existing,
      args$worker_out, row.names = FALSE
    )
    message(conditionMessage(e))
    quit(status = 1L)
  })
  quit(status = 0L)
}

if (worker) {
  tryCatch(worker_main(), error = function(e) {
    row <- worker_result_template(
      args$dataset %||% NA_character_, args$method %||% NA_character_,
      status = "failed", error = conditionMessage(e)
    )
    write.csv(row, args$worker_out %||% file.path(out_dir, "worker_failed.csv"), row.names = FALSE)
    message(conditionMessage(e))
    quit(status = 1L)
  })
  quit(status = 0L)
}

validation_backends <- switch(
  backend_group,
  cpu = "cpu", metal = "metal", cuda = "cuda", local = c("cpu", "metal")
)
write_reproducibility()
parameter_rows <- do.call(rbind, lapply(threads_grid, function(value) {
  do.call(rbind, lapply(methods, parameter_record, requested_threads = value))
}))
write.csv(parameter_rows, file.path(out_dir, "parameter_table.csv"), row.names = FALSE)
write_markdown(parameter_rows, file.path(out_dir, "parameter_table.md"))

precompute_rows <- list()
for (dataset in datasets) {
  precompute_csv <- file.path(
    out_dir, "worker_results",
    paste0(safe_name(dataset_alias(dataset)), "_precompute.csv")
  )
  precompute_log <- file.path(
    out_dir, "logs", paste0(safe_name(dataset_alias(dataset)), "_precompute.log")
  )
  precompute_time <- file.path(
    out_dir, "memory", paste0(safe_name(dataset_alias(dataset)), "_precompute_ram.txt")
  )
  precompute_gpu <- file.path(
    out_dir, "memory", paste0(safe_name(dataset_alias(dataset)), "_precompute_gpu.txt")
  )
  if (!force && file.exists(precompute_csv)) {
    existing <- tryCatch(read.csv(precompute_csv, stringsAsFactors = FALSE), error = function(...) NULL)
    if (!is.null(existing) && nrow(existing) &&
        existing$status[[1L]] %in% "success") {
      log_msg("%s precompute: reusing terminal status %s", dataset, existing$status[[1L]])
      precompute_rows[[length(precompute_rows) + 1L]] <- existing
      next
    }
  }
  worker_command <- c(
    file.path(R.home("bin"), "Rscript"), script_path,
    "--precompute-worker=TRUE", paste0("--script=", script_path),
    paste0("--backend-group=", backend_group),
    paste0("--base-dir=", base_dir), paste0("--data-root=", data_root),
    paste0("--out-dir=", out_dir), paste0("--layout-dir=", layout_dir),
    paste0("--cache-dir=", cache_dir),
    paste0("--dataset=", dataset), paste0("--worker-out=", precompute_csv),
    paste0("--threads=", max(threads_grid)),
    paste0("--threads-grid=", paste(threads_grid, collapse = ",")),
    paste0("--seeds=", paste(seeds, collapse = ",")),
    paste0("--k=", k), paste0("--perplexity=", perplexity),
    paste0("--validation-sample-n=", validation_sample_n),
    paste0("--validation-backends=", paste(validation_backends, collapse = ",")),
    paste0("--methods=", paste(methods, collapse = ",")),
    paste0("--shared-cache-backend=auto"),
    paste0("--kodama-cache-tag=", kodama_cache_tag),
    paste0("--local-cpu-max-n=", local_cpu_max_n),
    paste0("--local-cpu-exceptions=", paste(local_cpu_exceptions, collapse = ",")),
    paste0("--force=", force)
  )
  log_msg("%s precompute: running", dataset)
  monitor <- file.path(script_dir, "benchmark_worker_monitor.sh")
  status <- system2(
    "bash",
    c(monitor, precompute_time, precompute_gpu, as.character(timeout), worker_command),
    stdout = precompute_log, stderr = precompute_log
  )
  if (file.exists(precompute_csv)) {
    row <- read.csv(precompute_csv, stringsAsFactors = FALSE)
    if (nrow(row) && !identical(as.integer(status), 0L) &&
        identical(row$status[[1L]], "running")) {
      row$status <- if (identical(as.integer(status), 124L)) "timeout" else "failed"
      row$error <- describe_worker_exit(status, precompute_log, "precompute worker")
      write.csv(row, precompute_csv, row.names = FALSE)
    }
    precompute_rows[[length(precompute_rows) + 1L]] <- row
  } else {
    precompute_rows[[length(precompute_rows) + 1L]] <- data.frame(
      dataset = dataset_alias(dataset), status = "failed",
      error = describe_worker_exit(status, precompute_log, "precompute worker"),
      n = NA_real_, p = NA_real_,
      shared_cache_backend = NA_character_, local_cpu_allowed = NA,
      knn_file = NA_character_, rtsne_knn_file = NA_character_,
      pca_init_file = NA_character_, validation_file = NA_character_,
      stringsAsFactors = FALSE
    )
  }
  write.csv(
    do.call(rbind, precompute_rows),
    file.path(out_dir, "precompute_manifest_checkpoint.csv"),
    row.names = FALSE
  )
}
precompute_table <- do.call(rbind, precompute_rows)
write.csv(precompute_table, file.path(out_dir, "precompute_manifest.csv"), row.names = FALSE)

kodama_core_rows <- list()
kodama_methods <- methods[vapply(methods, method_is_kodama, logical(1))]
if (length(kodama_methods)) {
  core_specs <- unique(data.frame(
    classifier = vapply(kodama_methods, kodama_classifier_for_method, character(1)),
    backend = vapply(kodama_methods, method_backend, character(1)),
    stringsAsFactors = FALSE
  ))
  for (dataset in datasets) {
    pre_status <- precompute_table[
      precompute_table$dataset == dataset_alias(dataset), , drop = FALSE
    ]
    dataset_n <- if (nrow(pre_status) && "n" %in% names(pre_status)) {
      suppressWarnings(as.numeric(pre_status$n[[1L]]))
    } else NA_real_
    for (spec_index in seq_len(nrow(core_specs))) {
      classifier <- core_specs$classifier[[spec_index]]
      core_backend <- core_specs$backend[[spec_index]]
      if (identical(backend_group, "local") && identical(core_backend, "cpu") &&
          !local_cpu_allowed_for(dataset, dataset_n)) {
        next
      }
      core_threads <- if (identical(core_backend, "cpu")) {
        threads_grid
      } else {
        max(threads_grid)
      }
      for (worker_threads in core_threads) {
        for (worker_seed in seeds) {
          log_msg(
            "%s/KODAMA core %s/%s/%dt/seed%d: running",
            dataset, classifier, core_backend, worker_threads, worker_seed
          )
          row <- run_kodama_core_isolated(
            dataset, classifier, core_backend, worker_threads, worker_seed
          )
          kodama_core_rows[[length(kodama_core_rows) + 1L]] <- row
          write.csv(
            do.call(rbind, kodama_core_rows),
            file.path(out_dir, "kodama_core_runs_checkpoint.csv"),
            row.names = FALSE
          )
          log_msg(
            "%s/KODAMA core %s/%s/%dt/seed%d: %s total=%s RAM=%s GPUdelta=%s reused=%s",
            dataset, classifier, core_backend, worker_threads, worker_seed,
            row$status[[1L]], format(row$core_runtime_sec[[1L]], digits = 5),
            format(row$peak_ram_gb[[1L]], digits = 4),
            format(row$peak_gpu_delta_mb[[1L]], digits = 4),
            row$core_reused[[1L]]
          )
        }
      }
    }
  }
}
kodama_core_table <- if (length(kodama_core_rows)) {
  do.call(rbind, kodama_core_rows)
} else data.frame()
write.csv(kodama_core_table, file.path(out_dir, "kodama_core_runs.csv"), row.names = FALSE)

runs <- list()
skipped_rows <- list()
for (dataset in datasets) {
  pre_status <- precompute_table[precompute_table$dataset == dataset_alias(dataset), , drop = FALSE]
  if (nrow(pre_status) && "status" %in% names(pre_status) && any(pre_status$status == "failed")) {
    log_msg("%s: precompute failed; method workers will report the concrete error", dataset)
  }
  dataset_n <- if (nrow(pre_status) && "n" %in% names(pre_status)) {
    suppressWarnings(as.numeric(pre_status$n[[1L]]))
  } else NA_real_
  shared_cache_backend <- if (
    nrow(pre_status) && "shared_cache_backend" %in% names(pre_status)
  ) pre_status$shared_cache_backend[[1L]] else "cpu"
  dataset_backend_unhealthy <- FALSE
  for (method in methods) {
    if (dataset_backend_unhealthy) {
      log_msg(
        "%s/%s: skipped because the CUDA device failed during an earlier method",
        dataset, method
      )
      skipped_rows[[length(skipped_rows) + 1L]] <- data.frame(
        dataset = dataset_alias(dataset), method = method, n = dataset_n,
        reason = "CUDA device health failure (uncorrectable ECC or device loss)",
        stringsAsFactors = FALSE
      )
      next
    }
    if (identical(backend_group, "local") &&
        !local_cpu_allowed_for(dataset, dataset_n) &&
        identical(method_backend(method), "cpu")) {
      log_msg(
        "%s/%s: skipped by local CPU size policy (n=%s > %s)",
        dataset, method, format(dataset_n, scientific = FALSE), local_cpu_max_n
      )
      skipped_rows[[length(skipped_rows) + 1L]] <- data.frame(
        dataset = dataset_alias(dataset), method = method, n = dataset_n,
        reason = sprintf("local CPU backend disabled above %d samples", local_cpu_max_n),
        stringsAsFactors = FALSE
      )
      next
    }
    if (nrow(pre_status) && identical(pre_status$status[[1L]], "timeout") &&
        !identical(method_family(method), "PCA") && !method_is_kodama(method)) {
      log_msg(
        "%s/%s: skipped after shared precompute timeout", dataset, method
      )
      skipped_rows[[length(skipped_rows) + 1L]] <- data.frame(
        dataset = dataset_alias(dataset), method = method, n = dataset_n,
        reason = "shared Metal precompute exceeded the per-stage timeout",
        stringsAsFactors = FALSE
      )
      next
    }
    if (identical(method_scope(method), "embedding_from_precomputed_knn") &&
        (is.na(pre_status$knn_file[[1L]]) || !file.exists(pre_status$knn_file[[1L]]))) {
      log_msg("%s/%s: skipped because shared KNN cache is unavailable", dataset, method)
      skipped_rows[[length(skipped_rows) + 1L]] <- data.frame(
        dataset = dataset_alias(dataset), method = method, n = dataset_n,
        reason = "shared KNN cache is unavailable", stringsAsFactors = FALSE
      )
      next
    }
    thread_values <- if (method_backend(method) == "cpu") threads_grid else max(threads_grid)
    for (worker_threads in thread_values) {
      for (worker_seed in seeds) {
        log_msg("%s/%s/%dt/seed%d: running", dataset, method, worker_threads, worker_seed)
        row <- run_isolated_worker(
          dataset, method, worker_threads, worker_seed, shared_cache_backend
        )
        runs[[length(runs) + 1L]] <- row
        write.csv(
          do.call(rbind, runs),
          file.path(out_dir, "benchmark_runs_checkpoint.csv"),
          row.names = FALSE
        )
        log_msg(
          "%s/%s/%dt/seed%d: %s total=%s RAM=%s GPUdelta=%s",
          dataset, method, worker_threads, worker_seed, row$status[[1L]],
          format(row$total_runtime_sec[[1L]], digits = 5),
          format(row$peak_ram_gb[[1L]], digits = 4),
          format(row$peak_gpu_delta_mb[[1L]], digits = 4)
        )
        if (is_resource_failure(row$status[[1L]], row$error[[1L]])) {
          log_msg(
            "%s/%s: deterministic resource failure; remaining seeds skipped",
            dataset, method
          )
          break
        }
        if (is_backend_health_failure(row$status[[1L]], row$error[[1L]])) {
          dataset_backend_unhealthy <- TRUE
          log_msg(
            "%s/%s: CUDA device health failure; remaining methods and seeds skipped",
            dataset, method
          )
          break
        }
      }
    }
  }
}
write.csv(
  if (length(skipped_rows)) do.call(rbind, skipped_rows) else data.frame(),
  file.path(out_dir, "skipped_by_local_size_policy.csv"), row.names = FALSE
)

run_table <- if (length(runs)) do.call(rbind, runs) else data.frame()
write.csv(run_table, file.path(out_dir, "benchmark_runs.csv"), row.names = FALSE)
summary_table <- aggregate_runs(run_table)
write.csv(summary_table, file.path(out_dir, "benchmark_summary_median_variability.csv"), row.names = FALSE)
write_markdown(summary_table, file.path(out_dir, "benchmark_summary_median_variability.md"))
stability <- stability_table(run_table)
write.csv(stability, file.path(out_dir, "stability_pairwise.csv"), row.names = FALSE)
landmark_validation <- landmark_validation_table(run_table)
write.csv(
  landmark_validation,
  file.path(out_dir, "landmark_validation_vs_full.csv"),
  row.names = FALSE
)
landmark_markdown <- file.path(out_dir, "landmark_validation_vs_full.md")
if (ncol(landmark_validation)) {
  write_markdown(landmark_validation, landmark_markdown)
} else {
  writeLines("No landmark methods were requested for this benchmark run.", landmark_markdown)
}
backend_validation <- backend_validation_table(validation_backends)
write.csv(
  backend_validation,
  file.path(out_dir, "knn_affinity_umap_graph_agreement.csv"),
  row.names = FALSE
)
pca_agreement <- pca_agreement_table(run_table)
write.csv(pca_agreement, file.path(out_dir, "pca_vs_irlba_agreement.csv"), row.names = FALSE)
reference_graph <- if (reference_validations) {
  reference_graph_validation_table()
} else data.frame()
write.csv(
  reference_graph,
  file.path(out_dir, "umap_graph_agreement_vs_uwot.csv"),
  row.names = FALSE
)
reference_affinity <- if (reference_validations) {
  reference_affinity_validation_table()
} else data.frame()
write.csv(
  reference_affinity,
  file.path(out_dir, "tsne_affinity_agreement_vs_python_opentsne.csv"),
  row.names = FALSE
)
make_summary_figures(summary_table)
log_msg("DONE: %s", out_dir)
print(summary_table)
