#!/usr/bin/env Rscript

parse_args <- function(arguments) {
  out <- list()
  for (argument in arguments) {
    if (!startsWith(argument, "--")) next
    split <- strsplit(sub("^--", "", argument), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", split[[1L]])]] <- if (length(split) > 1L) {
      paste(split[-1L], collapse = "=")
    } else {
      "TRUE"
    }
  }
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x) || (length(x) == 1L && is.na(x))) y else x
}
csv_arg <- function(x, default) {
  value <- strsplit(as.character(x %||% default), ",", fixed = TRUE)[[1L]]
  value <- trimws(value)
  value[nzchar(value)]
}
int_arg <- function(x, default) {
  value <- suppressWarnings(as.integer(x %||% default))
  if (length(value) != 1L || is.na(value) || value < 1L) as.integer(default) else value
}
bool_arg <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  tolower(as.character(x)) %in% c("1", "true", "yes", "y")
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
script_flag <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(
  if (length(script_flag)) sub("^--file=", "", script_flag[[1L]]) else
    args$script %||% "benchmark_local_graph_clustering.R",
  mustWork = TRUE
)
script_dir <- dirname(script_path)
source(file.path(script_dir, "publication_metrics.R"), local = TRUE)

data_root <- normalizePath(
  args$data_root %||% "/Users/stefano/Documents/fastEmbedR/Data",
  mustWork = TRUE
)
out_dir <- normalizePath(
  args$out_dir %||% file.path(
    dirname(dirname(script_dir)),
    "results",
    paste0("local_graph_clustering_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  ),
  mustWork = FALSE
)
datasets <- csv_arg(
  args$datasets,
  "MetRef,COIL20,USPS,Macosko2015_retina,FashionMNIST,MNIST,TabulaMuris"
)
backends <- csv_arg(args$backends, "cpu,metal")
threads_grid <- unique(as.integer(csv_arg(args$threads_grid, "1,4")))
threads_grid <- threads_grid[is.finite(threads_grid) & threads_grid > 0L]
seeds <- unique(as.integer(csv_arg(args$seeds, "4,17,42")))
seeds <- seeds[is.finite(seeds)]
k <- int_arg(args$k, 30L)
timeout <- int_arg(args$timeout, 43200L)
igraph_max_n <- int_arg(args$igraph_max_n, 50000L)
walktrap_max_n <- min(int_arg(args$walktrap_max_n, 4000L), 4000L)
force <- bool_arg(args$force, FALSE)
worker <- bool_arg(args$worker, FALSE)
stage <- args$stage %||% ""

dirs <- c(
  out_dir,
  file.path(out_dir, "cache"),
  file.path(out_dir, "logs"),
  file.path(out_dir, "memory"),
  file.path(out_dir, "plots"),
  file.path(out_dir, "workers")
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
set_threads <- function(value) {
  value <- as.integer(value)
  Sys.setenv(
    OMP_NUM_THREADS = value,
    OPENBLAS_NUM_THREADS = value,
    VECLIB_MAXIMUM_THREADS = value,
    RCPP_PARALLEL_NUM_THREADS = value
  )
}

matrix_dimensions <- function(x) {
  if (inherits(x, "float32")) {
    dimensions <- dim(methods::slot(x, "Data"))
  } else {
    dimensions <- dim(x)
  }
  if (length(dimensions) != 2L || anyNA(dimensions)) {
    stop("Dataset input must have two finite matrix dimensions.", call. = FALSE)
  }
  as.integer(dimensions)
}

dataset_file <- function(dataset, float32 = TRUE) {
  folder <- file.path(data_root, dataset)
  pattern <- if (float32) "float32.*\\.[Rr][Dd]ata$" else "\\.[Rr][Dd]ata$"
  paths <- list.files(folder, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
  if (!float32) {
    paths <- paths[!grepl(
      "float32|pca|backup|knn|manifest|benchmark",
      basename(paths), ignore.case = TRUE
    )]
  }
  if (!length(paths)) return(NA_character_)
  paths[order(nchar(basename(paths)), basename(paths))][[1L]]
}

load_dataset_object <- function(dataset) {
  path <- dataset_file(dataset, TRUE)
  if (is.na(path)) path <- dataset_file(dataset, FALSE)
  if (is.na(path)) stop("No dataset file found for ", dataset, call. = FALSE)
  environment <- new.env(parent = emptyenv())
  loaded <- load(path, envir = environment)
  objects <- mget(loaded, envir = environment, inherits = FALSE)
  object <- NULL
  for (candidate in objects) {
    if (is.list(candidate) && !is.null(candidate$data)) {
      object <- candidate
      break
    }
  }
  if (is.null(object)) stop("No list with a `data` element in ", path, call. = FALSE)
  labels <- object$labels %||% object$label %||% object$tissue %||% NULL
  dimensions <- matrix_dimensions(object$data)
  if (!is.null(labels) && length(labels) != dimensions[[1L]]) labels <- NULL
  list(
    data = object$data,
    labels = if (is.null(labels)) NULL else as.factor(labels),
    path = path
  )
}

adjusted_rand <- function(left, right) {
  if (length(left) != length(right) || !length(left)) return(NA_real_)
  contingency <- table(left, right)
  choose_two <- function(x) x * (x - 1) / 2
  n <- sum(contingency)
  denominator <- choose_two(n)
  if (denominator <= 0) return(NA_real_)
  observed <- sum(choose_two(contingency))
  row_pairs <- sum(choose_two(rowSums(contingency)))
  column_pairs <- sum(choose_two(colSums(contingency)))
  expected <- row_pairs * column_pairs / denominator
  maximum <- (row_pairs + column_pairs) / 2
  if (maximum == expected) return(1)
  as.numeric((observed - expected) / (maximum - expected))
}

normalized_mutual_information <- function(left, right) {
  if (length(left) != length(right) || !length(left)) return(NA_real_)
  tab <- table(left, right)
  joint <- tab / sum(tab)
  p_left <- rowSums(joint)
  p_right <- colSums(joint)
  nonzero <- which(joint > 0, arr.ind = TRUE)
  mutual <- sum(vapply(seq_len(nrow(nonzero)), function(i) {
    row <- nonzero[i, 1L]
    column <- nonzero[i, 2L]
    joint[row, column] * log(joint[row, column] / (p_left[row] * p_right[column]))
  }, numeric(1)))
  entropy_left <- -sum(p_left[p_left > 0] * log(p_left[p_left > 0]))
  entropy_right <- -sum(p_right[p_right > 0] * log(p_right[p_right > 0]))
  denominator <- sqrt(entropy_left * entropy_right)
  if (!is.finite(denominator) || denominator <= 0) return(NA_real_)
  as.numeric(mutual / denominator)
}

graph_edges <- function(graph) {
  key <- paste0(pmin.int(graph$from, graph$to), ":", pmax.int(graph$from, graph$to))
  data.frame(
    key = key,
    weight = as.numeric(graph$weight) / max(sum(graph$weight), .Machine$double.xmin),
    stringsAsFactors = FALSE
  )
}

igraph_cluster <- function(graph, method, seed) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("igraph is not installed.", call. = FALSE)
  }
  igraph_graph <- igraph::graph_from_data_frame(
    data.frame(
      from = graph$from,
      to = graph$to,
      weight = graph$weight
    ),
    directed = FALSE,
    vertices = seq_len(graph$n_vertices)
  )
  set.seed(seed)
  result <- switch(
    method,
    louvain = igraph::cluster_louvain(
      igraph_graph,
      weights = igraph::E(igraph_graph)$weight,
      resolution = 1
    ),
    leiden = igraph::cluster_leiden(
      igraph_graph,
      objective_function = "modularity",
      weights = igraph::E(igraph_graph)$weight,
      resolution = 1,
      n_iterations = 10L
    ),
    walktrap = igraph::cluster_walktrap(
      igraph_graph,
      weights = igraph::E(igraph_graph)$weight,
      steps = 4L
    )
  )
  membership <- as.integer(igraph::membership(result))
  list(
    membership = membership,
    modularity = as.numeric(igraph::modularity(
      igraph_graph,
      membership,
      weights = igraph::E(igraph_graph)$weight,
      resolution = 1
    )),
    n_communities = length(unique(membership))
  )
}

cache_stem <- function(dataset, backend, threads) {
  file.path(
    out_dir,
    "cache",
    sprintf("%s_%s_t%d", safe_name(dataset), backend, as.integer(threads))
  )
}

worker_graph <- function() {
  library(fastEmbedR)
  dataset <- args$dataset
  backend <- match.arg(args$backend, c("cpu", "metal", "cuda"))
  threads <- int_arg(args$threads, 4L)
  set_threads(threads)
  loaded <- load_dataset_object(dataset)
  x <- loaded$data
  labels <- loaded$labels
  dimensions <- matrix_dimensions(x)
  n <- dimensions[[1L]]
  p <- dimensions[[2L]]
  width <- min(k, n - 1L)
  stem <- cache_stem(dataset, backend, threads)

  knn_elapsed <- system.time({
    knn <- precompute_knn(
      x,
      k = width,
      metric = "euclidean",
      backend = backend,
      n_threads = threads
    )
  })[["elapsed"]]
  graph_elapsed <- system.time({
    graph <- knn_graph(
      knn,
      k = width,
      weight = "snn",
      n_threads = threads
    )
  })[["elapsed"]]
  init_elapsed <- system.time({
    initialization <- umap_init(
      knn,
      backend = backend,
      graph_mode = "fuzzy",
      seed = 4L,
      n_threads = threads
    )
  })[["elapsed"]]
  optimizer_elapsed <- system.time({
    layout <- umap_knn(
      initialization,
      backend = backend,
      seed = 4L,
      n_threads = threads
    )
  })[["elapsed"]]

  common_init_sec <- NA_real_
  common_init_layout_file <- NA_character_
  reference_init <- args$reference_init %||% ""
  if (backend %in% c("metal", "cuda") && nzchar(reference_init) &&
      file.exists(reference_init)) {
    cpu_initialization <- readRDS(reference_init)
    common_init_sec <- system.time({
      common_layout <- umap_knn(
        cpu_initialization,
        backend = "metal",
        seed = 4L,
        n_threads = threads
      )
    })[["elapsed"]]
    common_init_layout_file <- paste0(stem, "_common_init_layout.rds")
    saveRDS(common_layout, common_init_layout_file, compress = FALSE)
  }

  graph_file <- paste0(stem, "_graph.rds")
  init_file <- paste0(stem, "_umap_init.rds")
  layout_file <- paste0(stem, "_layout.rds")
  labels_file <- paste0(stem, "_labels.rds")
  saveRDS(graph, graph_file, compress = FALSE)
  saveRDS(initialization, init_file, compress = FALSE)
  saveRDS(layout, layout_file, compress = FALSE)
  saveRDS(labels, labels_file, compress = FALSE)

  row <- data.frame(
    dataset = dataset,
    backend = backend,
    threads = threads,
    n = n,
    p = p,
    k = width,
    input_file = loaded$path,
    knn_sec = unname(knn_elapsed),
    snn_graph_sec = unname(graph_elapsed),
    umap_graph_sec = initialization$timings$elapsed_sec[
      initialization$timings$stage == "graph"
    ],
    umap_init_sec = initialization$timings$elapsed_sec[
      initialization$timings$stage == "initialization"
    ],
    umap_init_total_sec = unname(init_elapsed),
    umap_optimizer_sec = unname(optimizer_elapsed),
    common_cpu_init_metal_optimizer_sec = if (backend == "metal") {
      unname(common_init_sec)
    } else {
      NA_real_
    },
    common_cpu_init_accelerator_optimizer_sec = unname(common_init_sec),
    n_edges = graph$n_edges,
    graph_object_mb = as.numeric(object.size(graph)) / 1024^2,
    graph_file = graph_file,
    initialization_file = init_file,
    layout_file = layout_file,
    common_init_layout_file = common_init_layout_file,
    labels_file = labels_file,
    status = "success",
    error = NA_character_,
    stringsAsFactors = FALSE
  )
  write.csv(row, args$worker_out, row.names = FALSE)
}

worker_cluster <- function() {
  library(fastEmbedR)
  dataset <- args$dataset
  backend <- args$backend
  threads <- int_arg(args$threads, 4L)
  method <- match.arg(args$method, c("louvain", "leiden", "walktrap"))
  seed <- int_arg(args$seed, 4L)
  graph <- readRDS(args$graph_file)
  labels <- readRDS(args$labels_file)
  layout <- readRDS(args$layout_file)

  native_elapsed <- system.time({
    native <- graph_cluster(
      graph,
      method = method,
      resolution = 1,
      n_iterations = 10L,
      n_runs = if (method == "walktrap") 1L else 3L,
      steps = 4L,
      seed = seed
    )
  })[["elapsed"]]

  reference_status <- "success"
  reference_error <- NA_character_
  reference_elapsed <- NA_real_
  reference <- NULL
  if (graph$n_vertices > igraph_max_n) {
    reference_status <- "skipped_size_policy"
  } else {
    reference_elapsed <- system.time({
      reference <- tryCatch(
        igraph_cluster(graph, method, seed),
        error = function(error) {
          reference_error <<- conditionMessage(error)
          NULL
        }
      )
    })[["elapsed"]]
    if (is.null(reference)) reference_status <- "failed"
  }

  membership_file <- file.path(
    out_dir,
    "cache",
    sprintf(
      "%s_%s_t%d_%s_seed%d_membership.rds",
      safe_name(dataset), backend, threads, method, seed
    )
  )
  saveRDS(native$membership, membership_file, compress = FALSE)
  plot_file <- file.path(
    out_dir,
    "plots",
    sprintf(
      "%s_%s_threads%d_%s_seed%d.png",
      safe_name(dataset), backend, threads, method, seed
    )
  )
  publication_clean_plot(layout, native$membership, plot_file)

  row <- data.frame(
    dataset = dataset,
    graph_backend = backend,
    threads = threads,
    method = method,
    seed = seed,
    n = graph$n_vertices,
    n_edges = graph$n_edges,
    native_runtime_sec = unname(native_elapsed),
    native_modularity = native$modularity,
    native_communities = native$n_communities,
    label_ari = if (is.null(labels)) NA_real_ else
      adjusted_rand(native$membership, labels),
    label_nmi = if (is.null(labels)) NA_real_ else
      normalized_mutual_information(native$membership, labels),
    igraph_status = reference_status,
    igraph_runtime_sec = unname(reference_elapsed),
    igraph_modularity = if (is.null(reference)) NA_real_ else reference$modularity,
    igraph_communities = if (is.null(reference)) NA_integer_ else
      reference$n_communities,
    native_vs_igraph_ari = if (is.null(reference)) NA_real_ else
      adjusted_rand(native$membership, reference$membership),
    native_vs_igraph_nmi = if (is.null(reference)) NA_real_ else
      normalized_mutual_information(native$membership, reference$membership),
    membership_file = membership_file,
    plot_file = plot_file,
    status = "success",
    error = reference_error,
    stringsAsFactors = FALSE
  )
  write.csv(row, args$worker_out, row.names = FALSE)
}

if (worker) {
  tryCatch(
    {
      if (identical(stage, "graph")) worker_graph() else if (identical(stage, "cluster")) {
        worker_cluster()
      } else {
        stop("Unknown worker stage: ", stage, call. = FALSE)
      }
    },
    error = function(error) {
      write.csv(
        data.frame(status = "failed", error = conditionMessage(error)),
        args$worker_out,
        row.names = FALSE
      )
      message(conditionMessage(error))
      quit(status = 1L)
    }
  )
  quit(status = 0L)
}

parse_memory <- function(path) {
  if (!file.exists(path)) return(NA_real_)
  line <- grep("Maximum resident set size", readLines(path, warn = FALSE), value = TRUE)
  if (!length(line)) return(NA_real_)
  as.numeric(sub(".*: ", "", tail(line, 1L))) / 1024^2
}

bind_rows_fill <- function(rows) {
  if (!length(rows)) return(data.frame())
  columns <- unique(unlist(lapply(rows, names), use.names = FALSE))
  normalized <- lapply(rows, function(row) {
    missing <- setdiff(columns, names(row))
    for (name in missing) row[[name]] <- NA
    row[, columns, drop = FALSE]
  })
  do.call(rbind, normalized)
}

run_worker <- function(arguments, stem) {
  csv <- file.path(out_dir, "workers", paste0(stem, ".csv"))
  log <- file.path(out_dir, "logs", paste0(stem, ".log"))
  ram <- file.path(out_dir, "memory", paste0(stem, "_ram.txt"))
  gpu <- file.path(out_dir, "memory", paste0(stem, "_gpu.txt"))
  if (!force && file.exists(csv)) {
    existing <- tryCatch(read.csv(csv, stringsAsFactors = FALSE), error = function(...) NULL)
    if (!is.null(existing) && nrow(existing) &&
        identical(existing$status[[1L]], "success")) return(existing)
  }
  monitor <- file.path(script_dir, "benchmark_worker_monitor.sh")
  command <- c(
    monitor, ram, gpu, as.character(timeout),
    file.path(R.home("bin"), "Rscript"), script_path,
    "--worker=TRUE",
    paste0("--script=", script_path),
    paste0("--data-root=", data_root),
    paste0("--out-dir=", out_dir),
    paste0("--k=", k),
    paste0("--igraph-max-n=", igraph_max_n),
    paste0("--walktrap-max-n=", walktrap_max_n),
    paste0("--worker-out=", csv),
    arguments
  )
  status <- system2("bash", command, stdout = log, stderr = log)
  result <- if (file.exists(csv)) {
    read.csv(csv, stringsAsFactors = FALSE)
  } else {
    data.frame(status = "failed", error = paste("worker exit", status))
  }
  result$peak_ram_gb <- parse_memory(ram)
  write.csv(result, csv, row.names = FALSE)
  result
}

writeLines(
  c(
    capture.output(sessionInfo()),
    "",
    paste("generated_at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    paste("data_root:", data_root),
    paste("datasets:", paste(datasets, collapse = ",")),
    paste("backends:", paste(backends, collapse = ",")),
    paste("threads_grid:", paste(threads_grid, collapse = ",")),
    paste("seeds:", paste(seeds, collapse = ",")),
    paste("k:", k),
    paste("git_commit:", tryCatch(
      system2("git", c("-C", dirname(dirname(script_dir)), "rev-parse", "HEAD"),
              stdout = TRUE),
      error = function(...) NA_character_
    ))
  ),
  file.path(out_dir, "reproducibility.txt")
)

graph_rows <- list()
cluster_rows <- list()
for (dataset in datasets) {
  cpu_reference_init <- ""
  cpu_reference_layout <- ""
  for (backend in backends) {
    thread_values <- if (backend == "cpu") threads_grid else max(threads_grid)
    for (threads in thread_values) {
      stem <- sprintf("%s_%s_t%d_graph", safe_name(dataset), backend, threads)
      graph_result <- run_worker(
        c(
          "--stage=graph",
          paste0("--dataset=", dataset),
          paste0("--backend=", backend),
          paste0("--threads=", threads),
          paste0("--reference-init=", cpu_reference_init)
        ),
        stem
      )
      graph_rows[[length(graph_rows) + 1L]] <- graph_result
      if (identical(backend, "cpu") && threads == max(threads_grid) &&
          identical(graph_result$status[[1L]], "success")) {
        cpu_reference_init <- graph_result$initialization_file[[1L]]
        cpu_reference_layout <- graph_result$layout_file[[1L]]
      }
      if (!identical(graph_result$status[[1L]], "success")) next

      methods <- c("louvain", "leiden")
      if (graph_result$n[[1L]] <= walktrap_max_n) methods <- c(methods, "walktrap")
      for (method in methods) {
        method_seeds <- if (method == "walktrap") seeds[[1L]] else seeds
        for (seed in method_seeds) {
          cluster_stem <- sprintf(
            "%s_%s_t%d_%s_seed%d",
            safe_name(dataset), backend, threads, method, seed
          )
          cluster_result <- run_worker(
            c(
              "--stage=cluster",
              paste0("--dataset=", dataset),
              paste0("--backend=", backend),
              paste0("--threads=", threads),
              paste0("--method=", method),
              paste0("--seed=", seed),
              paste0("--graph-file=", graph_result$graph_file[[1L]]),
              paste0("--layout-file=", graph_result$layout_file[[1L]]),
              paste0("--labels-file=", graph_result$labels_file[[1L]])
            ),
            cluster_stem
          )
          cluster_rows[[length(cluster_rows) + 1L]] <- cluster_result
        }
      }
    }
  }
}

graphs <- bind_rows_fill(graph_rows)
clusters <- bind_rows_fill(cluster_rows)
write.csv(graphs, file.path(out_dir, "graph_initialization_runs.csv"), row.names = FALSE)
write.csv(clusters, file.path(out_dir, "clustering_runs.csv"), row.names = FALSE)

graph_agreement <- list()
for (dataset in unique(graphs$dataset[graphs$status == "success"])) {
  cpu <- graphs[
    graphs$dataset == dataset & graphs$backend == "cpu" &
      graphs$threads == max(threads_grid) & graphs$status == "success",
    , drop = FALSE
  ]
  if (!nrow(cpu)) next
  for (accelerator in intersect(c("metal", "cuda"), backends)) {
    accelerated <- graphs[
      graphs$dataset == dataset & graphs$backend == accelerator &
        graphs$status == "success",
      , drop = FALSE
    ]
    if (!nrow(accelerated)) next
    agreement <- publication_edge_agreement(
      graph_edges(readRDS(cpu$graph_file[[1L]])),
      graph_edges(readRDS(accelerated$graph_file[[1L]]))
    )
    common_layout <- accelerated$common_init_layout_file[[1L]]
    procrustes <- if (!is.na(common_layout) && nzchar(common_layout) &&
                       file.exists(common_layout)) {
      publication_procrustes(
        readRDS(cpu$layout_file[[1L]]),
        readRDS(common_layout)
      )
    } else {
      data.frame(rmsd = NA_real_, correlation = NA_real_)
    }
    graph_agreement[[length(graph_agreement) + 1L]] <- data.frame(
      dataset = dataset,
      accelerator_backend = accelerator,
      cpu_graph_threads = max(threads_grid),
      edge_jaccard = agreement$edge_jaccard,
      weight_pearson = agreement$weight_pearson,
      weight_spearman = agreement$weight_spearman,
      weight_l1_similarity = agreement$weight_l1_similarity,
      common_initialization_procrustes_rmsd = procrustes$rmsd,
      common_initialization_procrustes_correlation = procrustes$correlation,
      stringsAsFactors = FALSE
    )
  }
}
agreement_table <- if (length(graph_agreement)) {
  do.call(rbind, graph_agreement)
} else {
  data.frame()
}
write.csv(
  agreement_table,
  file.path(out_dir, "cpu_accelerator_graph_and_initialization_agreement.csv"),
  row.names = FALSE
)
if (nrow(agreement_table) &&
    any(agreement_table$accelerator_backend == "metal")) {
  write.csv(
    agreement_table[
      agreement_table$accelerator_backend == "metal",
      setdiff(names(agreement_table), "accelerator_backend"),
      drop = FALSE
    ],
    file.path(out_dir, "cpu_metal_graph_and_initialization_agreement.csv"),
    row.names = FALSE
  )
}

if (nrow(clusters)) {
  key <- interaction(
    clusters$dataset,
    clusters$graph_backend,
    clusters$threads,
    clusters$method,
    drop = TRUE
  )
  summary_rows <- lapply(split(clusters, key), function(piece) {
    runtime <- publication_median_iqr(piece$native_runtime_sec)
    data.frame(
      dataset = piece$dataset[[1L]],
      graph_backend = piece$graph_backend[[1L]],
      threads = piece$threads[[1L]],
      method = piece$method[[1L]],
      runs = nrow(piece),
      runtime_median_sec = runtime[["median"]],
      runtime_iqr_sec = runtime[["iqr"]],
      modularity_median = median(piece$native_modularity, na.rm = TRUE),
      communities_median = median(piece$native_communities, na.rm = TRUE),
      label_ari_median = median(piece$label_ari, na.rm = TRUE),
      label_nmi_median = median(piece$label_nmi, na.rm = TRUE),
      native_vs_igraph_ari_median = median(
        piece$native_vs_igraph_ari,
        na.rm = TRUE
      ),
      native_vs_igraph_nmi_median = median(
        piece$native_vs_igraph_nmi,
        na.rm = TRUE
      ),
      peak_ram_gb = max(piece$peak_ram_gb, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  cluster_summary <- do.call(rbind, summary_rows)
  numeric_columns <- names(cluster_summary)[vapply(
    cluster_summary, is.numeric, logical(1)
  )]
  for (name in numeric_columns) {
    cluster_summary[[name]][!is.finite(cluster_summary[[name]])] <- NA_real_
  }
  write.csv(
    cluster_summary,
    file.path(out_dir, "clustering_summary_median_variability.csv"),
    row.names = FALSE
  )
}

message("DONE: ", out_dir)
