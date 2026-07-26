#!/usr/bin/env Rscript

parse_args <- function(x) {
  out <- list()
  for (arg in x) {
    if (!startsWith(arg, "--")) next
    pieces <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    key <- gsub("-", "_", pieces[[1L]])
    out[[key]] <- if (length(pieces) > 1L) {
      paste(pieces[-1L], collapse = "=")
    } else {
      "TRUE"
    }
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x)) y else x
}

as_integer_vector <- function(x) {
  as.integer(trimws(strsplit(x, ",", fixed = TRUE)[[1L]]))
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
script_dir <- dirname(script_path)
source(file.path(script_dir, "publication_metrics.R"), local = TRUE)

args <- parse_args(commandArgs(trailingOnly = TRUE))
mode <- tolower(args$mode %||% "run")
bundle_path <- normalizePath(
  args$bundle %||% file.path(getwd(), "matched_mnist2000_bundle.rds"),
  mustWork = FALSE
)
out_dir <- normalizePath(
  args$out_dir %||% file.path(getwd(), "matched_embedding_correctness"),
  mustWork = FALSE
)
dataset_path <- args$dataset %||%
  "/Users/stefano/Documents/fastEmbedR/Data/MNIST/MNIST.RData"
backend <- tolower(args$backend %||% "cpu")
threads <- as.integer(args$threads %||% "4")
seeds <- as_integer_vector(args$seeds %||% "4,17,42")
sample_n <- as.integer(args$sample_n %||% "2000")
k <- as.integer(args$k %||% "30")
perplexity <- as.numeric(args$perplexity %||% "30")
negative_gradient_method <- tolower(args$negative_gradient_method %||% "fft")
if (!negative_gradient_method %in% c("exact", "fft")) {
  stop("`--negative-gradient-method` must be exact or fft.", call. = FALSE)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

layout_matrix <- function(x) {
  publication_layout_matrix(x)
}

float_matrix <- function(x) {
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required.", call. = FALSE)
    }
    return(float::dbl(x))
  }
  as.matrix(x)
}

extract_dataset <- function(path) {
  environment <- new.env(parent = emptyenv())
  loaded <- load(path, envir = environment)
  candidates <- mget(loaded, envir = environment, inherits = FALSE)
  value <- NULL
  for (candidate in candidates) {
    if (is.list(candidate) && !is.null(candidate$data)) {
      value <- candidate
      break
    }
  }
  if (is.null(value)) stop("No list with a `data` element was found.", call. = FALSE)
  value
}

prepare_bundle <- function() {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is not installed.", call. = FALSE)
  }
  dataset <- extract_dataset(dataset_path)
  x <- float_matrix(dataset$data)
  if (nrow(x) < sample_n) {
    stop("The dataset has fewer than sample_n rows.", call. = FALSE)
  }
  set.seed(20260726L)
  rows <- sort(sample.int(nrow(x), sample_n))
  x <- x[rows, , drop = FALSE]
  labels <- if (!is.null(dataset$labels)) dataset$labels[rows] else NULL

  knn <- publication_exact_knn(x, k)
  affinity <- publication_sparse_affinities(knn, perplexity)
  initializations <- list()
  for (seed in seeds) {
    tsne_init <- fastEmbedR::opentsne_pca_init(
      x, n_components = 2L, seed = seed, backend = "cpu"
    )
    umap_prepared <- fastEmbedR::prepare_umap_knn(
      knn, backend = "cpu", n_threads = threads, graph_mode = "fuzzy"
    )
    umap_initialization <- fastEmbedR::umap_init(
      umap_prepared,
      backend = "cpu",
      seed = seed,
      n_threads = threads,
      graph_mode = "fuzzy"
    )
    initializations[[as.character(seed)]] <- list(
      tsne = layout_matrix(tsne_init),
      umap = umap_initialization
    )
  }
  bundle <- list(
    dataset = "MNIST",
    rows = rows,
    data = x,
    labels = labels,
    knn = knn,
    affinity = affinity,
    initializations = initializations,
    parameters = list(
      n = nrow(x), p = ncol(x), k = k, perplexity = perplexity,
      seeds = seeds, threads = threads,
      tsne_early_iterations = 250L,
      tsne_normal_iterations = 750L,
      tsne_early_exaggeration = 12,
      tsne_learning_rate = nrow(x) / 12,
      tsne_momentum = 0.8,
      tsne_max_step_norm = 5,
      umap_graph_mode = "fuzzy",
      umap_n_epochs = 500L,
      umap_learning_rate = 1,
      umap_negative_sample_rate = 5L,
      umap_min_dist = 0.01
    ),
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
  saveRDS(bundle, bundle_path, compress = "xz")
  message("Prepared bundle: ", bundle_path)
}

write_run_row <- function(rows, path) {
  write.csv(do.call(rbind, rows), path, row.names = FALSE)
}

run_matched_umap_optimizer <- function(initialization, backend, seed, threads) {
  namespace <- asNamespace("fastEmbedR")
  graph <- initialization$prepared$graph
  config <- initialization$parameters
  layout <- initialization$layout
  if (identical(backend, "cuda")) {
    optimizer <- get("umap_cuda_optimize_csr_cpp", envir = namespace)
    return(optimizer(
      graph$offsets,
      graph$neighbors,
      graph$weights,
      graph$epochs_per_sample,
      layout,
      as.integer(config$n_epochs),
      as.integer(config$negative_sample_rate),
      as.numeric(config$learning_rate),
      as.numeric(config$min_dist),
      as.numeric(config$repulsion_strength),
      as.integer(seed),
      0L
    ))
  }
  if (identical(backend, "metal")) {
    optimizer <- get("knn_embed_metal_csr_cpp", envir = namespace)
    return(optimizer(
      graph$offsets,
      graph$neighbors,
      graph$weights,
      layout,
      as.integer(config$n_epochs),
      as.integer(config$negative_sample_rate),
      as.numeric(config$learning_rate),
      as.numeric(config$min_dist),
      as.numeric(graph$max_weight),
      as.numeric(config$repulsion_strength),
      as.integer(seed),
      1L
    ))
  }
  optimizer <- get("fast_knn_umap_csr_init_cpp", envir = namespace)
  optimizer(
    graph$offsets,
    graph$neighbors,
    graph$weights,
    layout,
    as.integer(config$n_epochs),
    as.numeric(config$min_dist),
    as.integer(config$negative_sample_rate),
    as.numeric(config$learning_rate),
    as.numeric(config$repulsion_strength),
    as.integer(threads),
    as.integer(seed),
    FALSE
  )
}

run_fastembedr <- function() {
  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is not installed.", call. = FALSE)
  }
  bundle <- readRDS(bundle_path)
  parameters <- bundle$parameters
  if (!requireNamespace("float", quietly = TRUE)) {
    stop("The float package is required for matched backend validation.",
         call. = FALSE)
  }
  run_knn <- bundle$knn
  run_knn$distances <- float::fl(as.matrix(run_knn$distances))
  class(run_knn) <- c("fastEmbedR_knn", "list")
  run_dir <- file.path(out_dir, paste0("fastEmbedR_", backend))
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  rows <- list()
  for (seed in seeds) {
    initialization <- bundle$initializations[[as.character(seed)]]

    tsne_start <- proc.time()[["elapsed"]]
    tsne <- fastEmbedR::opentsne_knn(
      run_knn$indices,
      run_knn$distances,
      perplexity = parameters$perplexity,
      Y_init = initialization$tsne,
      seed = seed,
      backend = backend,
      n_threads = threads,
      learning_rate = parameters$tsne_learning_rate,
      early_exaggeration_iter = parameters$tsne_early_iterations,
      early_exaggeration = parameters$tsne_early_exaggeration,
      n_iter = parameters$tsne_normal_iterations,
      exaggeration = 1,
      initial_momentum = parameters$tsne_momentum,
      final_momentum = parameters$tsne_momentum,
      max_step_norm = parameters$tsne_max_step_norm,
      negative_gradient_method = negative_gradient_method,
      record_costs = TRUE,
      auto_config = FALSE
    )
    tsne_sec <- proc.time()[["elapsed"]] - tsne_start
    tsne_layout <- layout_matrix(tsne)
    tsne_path <- file.path(run_dir, sprintf("tsne_seed%d.rds", seed))
    saveRDS(tsne_layout, tsne_path, compress = FALSE)
    rows[[length(rows) + 1L]] <- data.frame(
      algorithm = "t-SNE", implementation = "fastEmbedR",
      backend = backend, seed = seed, n = nrow(tsne_layout),
      k = parameters$k, perplexity = parameters$perplexity,
      negative_gradient_method = negative_gradient_method,
      elapsed_sec = tsne_sec,
      common_affinity_kl = publication_tsne_kl(tsne_layout, bundle$affinity),
      reported_kl = NA_real_, layout_file = tsne_path,
      status = "success", error = "", stringsAsFactors = FALSE
    )

    umap_start <- proc.time()[["elapsed"]]
    umap <- run_matched_umap_optimizer(
      initialization$umap, backend = backend, seed = seed, threads = threads
    )
    umap_sec <- proc.time()[["elapsed"]] - umap_start
    umap_layout <- layout_matrix(umap)
    umap_path <- file.path(run_dir, sprintf("umap_seed%d.rds", seed))
    saveRDS(umap_layout, umap_path, compress = FALSE)
    rows[[length(rows) + 1L]] <- data.frame(
      algorithm = "UMAP", implementation = "fastEmbedR",
      backend = backend, seed = seed, n = nrow(umap_layout),
      k = parameters$k, perplexity = NA_real_,
      negative_gradient_method = NA_character_,
      elapsed_sec = umap_sec, common_affinity_kl = NA_real_,
      reported_kl = NA_real_, layout_file = umap_path,
      status = "success", error = "", stringsAsFactors = FALSE
    )
  }
  write_run_row(rows, file.path(run_dir, "runs.csv"))
  message("Completed fastEmbedR backend: ", backend)
}

write_raw_matrix <- function(x, path, size, integer = FALSE) {
  if (inherits(x, "float32")) x <- float::dbl(x)
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(
    as.vector(t(x)), connection, size = size, endian = "little",
    useBytes = TRUE
  )
}

run_references <- function() {
  bundle <- readRDS(bundle_path)
  parameters <- bundle$parameters
  reference_dir <- file.path(out_dir, "references")
  dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
  python <- args$python %||% Sys.which("python3")
  helper <- file.path(script_dir, "reference_opentsne_matched_layout.py")
  rows <- list()

  raw_indices <- file.path(reference_dir, "knn_indices_int32.bin")
  raw_distances <- file.path(reference_dir, "knn_distances_float32.bin")
  write_raw_matrix(bundle$knn$indices - 1L, raw_indices, 4L, integer = TRUE)
  write_raw_matrix(bundle$knn$distances, raw_distances, 4L)

  for (seed in seeds) {
    initialization <- bundle$initializations[[as.character(seed)]]
    raw_initialization <- file.path(
      reference_dir, sprintf("tsne_initialization_seed%d_float32.bin", seed)
    )
    raw_layout <- file.path(
      reference_dir, sprintf("python_opentsne_seed%d_float32.bin", seed)
    )
    metrics_path <- file.path(
      reference_dir, sprintf("python_opentsne_seed%d_metrics.csv", seed)
    )
    write_raw_matrix(initialization$tsne, raw_initialization, 4L)
    command <- c(
      helper,
      paste0("--indices=", raw_indices),
      paste0("--distances=", raw_distances),
      paste0("--initialization=", raw_initialization),
      paste0("--output-layout=", raw_layout),
      paste0("--output-metrics=", metrics_path),
      paste0("--n=", parameters$n),
      paste0("--k=", parameters$k),
      paste0("--perplexity=", parameters$perplexity),
      paste0("--seed=", seed),
      paste0("--threads=", threads),
      paste0("--early-iterations=", parameters$tsne_early_iterations),
      paste0("--normal-iterations=", parameters$tsne_normal_iterations),
      paste0("--negative-gradient-method=", negative_gradient_method)
    )
    log_path <- file.path(
      reference_dir, sprintf("python_opentsne_seed%d.log", seed)
    )
    status <- system2(python, command, stdout = log_path, stderr = log_path)
    if (status != 0L || !file.exists(raw_layout)) {
      stop("Python openTSNE reference failed; inspect ", log_path, call. = FALSE)
    }
    connection <- file(raw_layout, open = "rb")
    tsne_layout <- matrix(
      readBin(
        connection, numeric(), n = parameters$n * 2L, size = 4L,
        endian = "little"
      ),
      nrow = parameters$n, ncol = 2L, byrow = TRUE
    )
    close(connection)
    metrics <- read.csv(metrics_path, stringsAsFactors = FALSE)
    tsne_path <- file.path(reference_dir, sprintf("tsne_seed%d.rds", seed))
    saveRDS(tsne_layout, tsne_path, compress = FALSE)
    rows[[length(rows) + 1L]] <- data.frame(
      algorithm = "t-SNE", implementation = "Python openTSNE",
      backend = "python_cpu", seed = seed, n = parameters$n,
      k = parameters$k, perplexity = parameters$perplexity,
      negative_gradient_method = negative_gradient_method,
      elapsed_sec = metrics$total_sec[[1L]],
      common_affinity_kl = publication_tsne_kl(tsne_layout, bundle$affinity),
      reported_kl = metrics$reported_kl[[1L]], layout_file = tsne_path,
      status = "success", error = "", stringsAsFactors = FALSE
    )

    if (requireNamespace("uwot", quietly = TRUE)) {
      host_knn <- publication_knn_host(bundle$knn)
      nn_method <- list(
        idx = cbind(seq_len(parameters$n), host_knn$indices),
        dist = cbind(0, host_knn$distances)
      )
      umap_start <- proc.time()[["elapsed"]]
      umap_layout <- uwot::umap(
        X = bundle$data,
        n_neighbors = parameters$k,
        n_components = 2L,
        metric = "euclidean",
        n_epochs = parameters$umap_n_epochs,
        learning_rate = parameters$umap_learning_rate,
        init = initialization$umap$layout,
        min_dist = parameters$umap_min_dist,
        repulsion_strength = 1,
        negative_sample_rate = parameters$umap_negative_sample_rate,
        nn_method = nn_method,
        fast_sgd = TRUE,
        n_threads = threads,
        n_sgd_threads = threads,
        verbose = FALSE,
        seed = seed
      )
      umap_sec <- proc.time()[["elapsed"]] - umap_start
      umap_path <- file.path(reference_dir, sprintf("umap_seed%d.rds", seed))
      saveRDS(umap_layout, umap_path, compress = FALSE)
      rows[[length(rows) + 1L]] <- data.frame(
        algorithm = "UMAP", implementation = "uwot fast_sgd",
        backend = "reference_cpu", seed = seed, n = parameters$n,
        k = parameters$k, perplexity = NA_real_, elapsed_sec = umap_sec,
        negative_gradient_method = NA_character_,
        common_affinity_kl = NA_real_, reported_kl = NA_real_,
        layout_file = umap_path, status = "success", error = "",
        stringsAsFactors = FALSE
      )
    }
  }
  write_run_row(rows, file.path(reference_dir, "runs.csv"))
  message("Completed Python openTSNE and uwot references.")
}

read_runs <- function(paths) {
  paths <- paths[file.exists(paths)]
  if (!length(paths)) stop("No run files were found.", call. = FALSE)
  do.call(rbind, lapply(paths, function(path) {
    value <- read.csv(path, stringsAsFactors = FALSE)
    local_layout <- file.path(dirname(path), basename(value$layout_file))
    missing_layout <- !file.exists(value$layout_file) & file.exists(local_layout)
    value$layout_file[missing_layout] <- local_layout[missing_layout]
    value$source_file <- path
    value
  }))
}

pairwise_agreement <- function(runs) {
  rows <- list()
  for (algorithm in unique(runs$algorithm)) {
    for (seed in unique(runs$seed[runs$algorithm == algorithm])) {
      piece <- runs[
        runs$algorithm == algorithm & runs$seed == seed &
          runs$status == "success",
        , drop = FALSE
      ]
      if (nrow(piece) < 2L) next
      pairs <- utils::combn(seq_len(nrow(piece)), 2L)
      for (column in seq_len(ncol(pairs))) {
        a <- piece[pairs[1L, column], , drop = FALSE]
        b <- piece[pairs[2L, column], , drop = FALSE]
        layout_a <- readRDS(a$layout_file)
        layout_b <- readRDS(b$layout_file)
        agreement <- publication_procrustes(layout_a, layout_b)
        neighbor_agreement <- publication_knn_overlap(
          publication_exact_knn(layout_a, 15L),
          publication_exact_knn(layout_b, 15L),
          15L
        )
        rows[[length(rows) + 1L]] <- data.frame(
          algorithm = algorithm, seed = seed,
          implementation_a = a$implementation,
          backend_a = a$backend,
          implementation_b = b$implementation,
          backend_b = b$backend,
          procrustes_rmsd = agreement$rmsd,
          procrustes_correlation = agreement$correlation,
          embedding_neighbor_agreement_15 = neighbor_agreement,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

summarize_agreement <- function(x) {
  if (!nrow(x)) return(data.frame())
  keys <- interaction(
    x$algorithm, x$implementation_a, x$backend_a,
    x$implementation_b, x$backend_b,
    drop = TRUE, lex.order = TRUE
  )
  pieces <- split(x, keys)
  do.call(rbind, lapply(pieces, function(piece) {
    data.frame(
      algorithm = piece$algorithm[[1L]],
      implementation_a = piece$implementation_a[[1L]],
      backend_a = piece$backend_a[[1L]],
      implementation_b = piece$implementation_b[[1L]],
      backend_b = piece$backend_b[[1L]],
      seeds = nrow(piece),
      procrustes_rmsd_median = median(piece$procrustes_rmsd, na.rm = TRUE),
      procrustes_correlation_median =
        median(piece$procrustes_correlation, na.rm = TRUE),
      embedding_neighbor_agreement_15_median =
        median(piece$embedding_neighbor_agreement_15, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
}

combine_results <- function() {
  input_dirs <- trimws(strsplit(args$inputs %||% out_dir, ",", fixed = TRUE)[[1L]])
  run_files <- unlist(lapply(input_dirs, function(directory) {
    list.files(directory, pattern = "^runs[.]csv$", recursive = TRUE, full.names = TRUE)
  }), use.names = FALSE)
  runs <- read_runs(unique(run_files))
  write.csv(runs, file.path(out_dir, "matched_embedding_runs.csv"), row.names = FALSE)
  agreement <- pairwise_agreement(runs)
  write.csv(
    agreement,
    file.path(out_dir, "matched_embedding_agreement_per_seed.csv"),
    row.names = FALSE
  )
  summary <- summarize_agreement(agreement)
  write.csv(
    summary,
    file.path(out_dir, "matched_embedding_agreement_summary.csv"),
    row.names = FALSE
  )
  tsne_kl <- runs[
    runs$algorithm == "t-SNE" & runs$status == "success",
    c(
      "implementation", "backend", "seed", "common_affinity_kl",
      "reported_kl", "elapsed_sec"
    ),
    drop = FALSE
  ]
  write.csv(
    tsne_kl, file.path(out_dir, "matched_tsne_kl.csv"), row.names = FALSE
  )
  manifest <- c(
    paste0("generated_at=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    paste0("bundle=", bundle_path),
    paste0("inputs=", paste(input_dirs, collapse = ";")),
    "initialization=identical within each seed and algorithm",
    "knn=identical exact kNN matrix",
    paste0("tsne_negative_gradient_method=", negative_gradient_method),
    "tsne_objective=KL evaluated against one common sparse affinity matrix",
    "layout_agreement=orthogonal Procrustes after centering and Frobenius scaling",
    "neighborhood_agreement=mean overlap at k=15 in the final 2-D layouts"
  )
  writeLines(manifest, file.path(out_dir, "matched_validation_manifest.txt"))
  message("Combined validation written to: ", out_dir)
}

if (identical(mode, "prepare")) {
  prepare_bundle()
} else if (identical(mode, "run")) {
  run_fastembedr()
} else if (identical(mode, "reference")) {
  run_references()
} else if (identical(mode, "combine")) {
  combine_results()
} else {
  stop("Unknown --mode. Use prepare, run, reference, or combine.", call. = FALSE)
}
