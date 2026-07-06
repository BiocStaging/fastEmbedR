#!/usr/bin/env Rscript

# Small deterministic validation against reference implementations.
# This is not a speed benchmark. It checks that the native fastEmbedR
# implementations behave like established reference paths on a small dataset.

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x)) y else x

parse_args <- function() {
  raw <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in raw) {
    if (!grepl("^--", arg)) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    key <- kv[[1L]]
    val <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else "TRUE"
    out[[gsub("-", "_", key)]] <- val
  }
  out
}

as_int <- function(x, default) {
  val <- suppressWarnings(as.integer(x %||% default))
  if (length(val) != 1L || is.na(val)) default else val
}

as_num <- function(x, default) {
  val <- suppressWarnings(as.numeric(x %||% default))
  if (length(val) != 1L || is.na(val) || !is.finite(val)) default else val
}

write_markdown_table <- function(x, path) {
  x <- as.data.frame(x, stringsAsFactors = FALSE)
  if (nrow(x) == 0L) {
    writeLines("_No rows._", path)
    return(invisible(path))
  }
  display <- x
  display[] <- lapply(display, function(col) {
    if (is.numeric(col)) col <- signif(col, 5)
    col <- as.character(col)
    col[is.na(col) | col == "NA"] <- ""
    col
  })
  con <- file(path, open = "wt")
  on.exit(close(con), add = TRUE)
  writeLines(paste0("| ", paste(names(display), collapse = " | "), " |"), con)
  writeLines(paste0("| ", paste(rep("---", ncol(display)), collapse = " | "), " |"), con)
  for (i in seq_len(nrow(display))) {
    writeLines(paste0("| ", paste(display[i, ], collapse = " | "), " |"), con)
  }
  invisible(path)
}

exact_knn <- function(x, k) {
  n <- nrow(x)
  k <- min(as.integer(k), n - 1L)
  d <- as.matrix(stats::dist(x, method = "euclidean", upper = TRUE, diag = TRUE))
  indices <- matrix(NA_integer_, n, k)
  distances <- matrix(NA_real_, n, k)
  for (i in seq_len(n)) {
    ord <- order(d[i, ], decreasing = FALSE)
    ord <- ord[ord != i]
    ord <- ord[seq_len(k)]
    indices[i, ] <- ord
    distances[i, ] <- d[i, ord]
  }
  list(indices = indices, distances = distances)
}

procrustes_summary <- function(x, y) {
  if (is.null(x) || is.null(y) || any(dim(x) != dim(y))) {
    return(list(rmsd = NA_real_, correlation = NA_real_))
  }
  x <- as.matrix(x)
  y <- as.matrix(y)
  x <- scale(x, center = TRUE, scale = FALSE)
  y <- scale(y, center = TRUE, scale = FALSE)
  s <- sqrt(sum(x * x))
  if (is.finite(s) && s > 0) x <- x / s
  sv <- tryCatch(svd(crossprod(x, y)), error = function(e) NULL)
  if (is.null(sv)) return(list(rmsd = NA_real_, correlation = NA_real_))
  rot <- sv$u %*% t(sv$v)
  xa <- x %*% rot
  denom <- sqrt(mean(rowSums(y^2)))
  if (!is.finite(denom) || denom <= 0) denom <- 1
  list(
    rmsd = sqrt(mean(rowSums((xa - y)^2))) / denom,
    correlation = suppressWarnings(stats::cor(as.vector(xa), as.vector(y)))
  )
}

metric_row <- function(x, layout, labels, dataset, method_family, implementation,
                       status = "success", error = "", k_eval = c(15, 30)) {
  if (!identical(status, "success")) {
    return(data.frame(
      dataset = dataset,
      method_family = method_family,
      implementation = implementation,
      status = status,
      error = error,
      trustworthiness = NA_real_,
      knn_preservation_15 = NA_real_,
      knn_preservation_30 = NA_real_,
      label_knn_accuracy = NA_real_,
      stringsAsFactors = FALSE
    ))
  }
  met <- tryCatch(
    fastEmbedR::evaluate_embedding(x, layout, labels = labels, k = k_eval),
    error = function(e) e
  )
  if (inherits(met, "error")) {
    return(metric_row(
      x, layout, labels, dataset, method_family, implementation,
      status = "metric_failed", error = conditionMessage(met), k_eval = k_eval
    ))
  }
  data.frame(
    dataset = dataset,
    method_family = method_family,
    implementation = implementation,
    status = status,
    error = "",
    trustworthiness = met$trustworthiness[[1L]],
    knn_preservation_15 = met$knn_preservation_15[[1L]],
    knn_preservation_30 = met$knn_preservation_30[[1L]],
    label_knn_accuracy = met$label_knn_accuracy[[1L]],
    stringsAsFactors = FALSE
  )
}

csr_to_edges <- function(prepared_umap) {
  g <- prepared_umap$graph
  offsets <- as.integer(g$offsets)
  neighbors <- as.integer(g$neighbors) + 1L
  weights <- as.numeric(g$weights)
  if (length(offsets) < 2L || length(neighbors) == 0L) {
    return(data.frame(i = integer(), j = integer(), weight = numeric()))
  }
  counts <- diff(offsets)
  data.frame(
    i = rep(seq_along(counts), counts),
    j = neighbors,
    weight = weights,
    stringsAsFactors = FALSE
  )
}

matrix_to_edges <- function(mat) {
  sm <- Matrix::summary(mat)
  data.frame(
    i = as.integer(sm$i),
    j = as.integer(sm$j),
    weight = as.numeric(sm$x),
    stringsAsFactors = FALSE
  )
}

graph_agreement <- function(fast_edges, ref_edges) {
  fast_edges <- fast_edges[fast_edges$i != fast_edges$j, , drop = FALSE]
  ref_edges <- ref_edges[ref_edges$i != ref_edges$j, , drop = FALSE]
  fast_key <- paste(fast_edges$i, fast_edges$j, sep = ":")
  ref_key <- paste(ref_edges$i, ref_edges$j, sep = ":")
  common <- intersect(fast_key, ref_key)
  union_n <- length(union(fast_key, ref_key))
  jaccard <- if (union_n > 0L) length(common) / union_n else NA_real_
  fc <- match(common, fast_key)
  rc <- match(common, ref_key)
  weight_spearman <- if (length(common) > 2L) {
    suppressWarnings(stats::cor(
      fast_edges$weight[fc],
      ref_edges$weight[rc],
      method = "spearman"
    ))
  } else {
    NA_real_
  }
  list(
    graph_edge_jaccard = jaccard,
    graph_common_edges = length(common),
    graph_union_edges = union_n,
    graph_weight_spearman = weight_spearman
  )
}

safe_run <- function(expr) {
  tryCatch(
    list(status = "success", value = force(expr), error = ""),
    error = function(e) list(status = "failed", value = NULL, error = conditionMessage(e))
  )
}

plot_validation <- function(layouts, labels, path) {
  png(path, width = 1800, height = 1100, res = 150)
  on.exit(dev.off(), add = TRUE)
  old <- par(no.readonly = TRUE)
  on.exit(par(old), add = TRUE)
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  cols <- as.integer(as.factor(labels))
  for (nm in names(layouts)) {
    y <- layouts[[nm]]
    if (is.null(y)) {
      plot.new()
      title(main = paste(nm, "failed"))
    } else if (!all(is.finite(y))) {
      plot.new()
      title(main = paste(nm, "non-finite"))
    } else {
      plot(y, pch = 20, cex = 1.1, col = cols, main = nm,
           xlab = "Component 1", ylab = "Component 2")
    }
  }
  invisible(path)
}

main <- function() {
  args <- parse_args()
  out_dir <- args$out_dir %||% file.path(
    "results",
    paste0("reference_validation_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  )
  threads <- as_int(args$threads, 2L)
  seed <- as_int(args$seed, 4L)
  perplexity <- as_num(args$perplexity, 10)
  k_umap <- as_int(args$k_umap, 15L)
  k_knn <- as_int(args$k, max(ceiling(3 * perplexity), k_umap))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  if (!requireNamespace("fastEmbedR", quietly = TRUE)) {
    stop("fastEmbedR is required.", call. = FALSE)
  }
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Matrix is required for graph validation.", call. = FALSE)
  }

  set.seed(seed)
  dataset_name <- "iris"
  x <- scale(as.matrix(datasets::iris[, 1:4]))
  labels <- datasets::iris$Species
  knn <- exact_knn(x, k_knn)
  y_init_tsne <- fastEmbedR::opentsne_pca_init(x, seed = seed)

  fast_tsne <- safe_run(fastEmbedR::opentsne_knn(
    knn,
    perplexity = perplexity,
    Y_init = y_init_tsne,
    seed = seed,
    backend = "cpu",
    n_threads = threads,
    early_exaggeration_iter = 250L,
    n_iter = 500L,
    negative_gradient_method = "exact",
    record_costs = TRUE,
    auto_config = TRUE
  ))

  rtsne <- if (requireNamespace("Rtsne", quietly = TRUE)) {
    safe_run(Rtsne::Rtsne_neighbors(
      index = knn$indices,
      distance = knn$distances,
      perplexity = perplexity,
      Y_init = y_init_tsne,
      max_iter = 750L,
      stop_lying_iter = 250L,
      mom_switch_iter = 250L,
      theta = 0.5,
      num_threads = threads,
      verbose = FALSE
    ))
  } else {
    list(status = "failed", value = NULL, error = "Rtsne is not installed")
  }

  prep_umap <- safe_run(fastEmbedR::prepare_umap_knn(
    knn,
    backend = "cpu",
    n_threads = threads,
    graph_mode = "fuzzy"
  ))
  fast_umap <- if (identical(prep_umap$status, "success")) {
    safe_run(fastEmbedR::umap_knn(
      prep_umap$value,
      backend = "cpu",
      n_threads = threads,
      seed = seed,
      graph_mode = "fuzzy"
    ))
  } else {
    list(status = "failed", value = NULL, error = prep_umap$error)
  }

  uwot_graph <- if (requireNamespace("uwot", quietly = TRUE)) {
    safe_run(uwot::similarity_graph(
      x,
      n_neighbors = k_umap,
      metric = "euclidean",
      nn_method = "fnn",
      n_threads = threads,
      verbose = FALSE
    ))
  } else {
    list(status = "failed", value = NULL, error = "uwot is not installed")
  }
  uwot_umap <- if (requireNamespace("uwot", quietly = TRUE)) {
    safe_run(uwot::umap(
      x,
      n_neighbors = k_umap,
      metric = "euclidean",
      init = "spectral",
      n_threads = threads,
      n_sgd_threads = 1L,
      fast_sgd = FALSE,
      verbose = FALSE
    ))
  } else {
    list(status = "failed", value = NULL, error = "uwot is not installed")
  }

  rows <- list()
  rows[[length(rows) + 1L]] <- metric_row(
    x, fast_tsne$value, labels, dataset_name, "t-SNE",
    "fastEmbedR openTSNE-style native CPU", fast_tsne$status, fast_tsne$error
  )
  rows[[length(rows) + 1L]] <- metric_row(
    x, if (is.null(rtsne$value)) NULL else rtsne$value$Y, labels,
    dataset_name, "t-SNE", "Rtsne::Rtsne_neighbors reference",
    rtsne$status, rtsne$error
  )
  rows[[length(rows) + 1L]] <- metric_row(
    x, fast_umap$value, labels, dataset_name, "UMAP",
    "fastEmbedR fuzzy native CPU", fast_umap$status, fast_umap$error
  )
  rows[[length(rows) + 1L]] <- metric_row(
    x, uwot_umap$value, labels, dataset_name, "UMAP",
    "uwot::umap reference", uwot_umap$status, uwot_umap$error
  )

  results <- do.call(rbind, rows)
  results$n <- nrow(x)
  results$p <- ncol(x)
  results$k <- k_knn
  results$perplexity <- ifelse(results$method_family == "t-SNE", perplexity, NA_real_)
  results$seed <- seed
  results$kl_final <- NA_real_
  results$procrustes_rmsd_vs_reference <- NA_real_
  results$procrustes_correlation_vs_reference <- NA_real_
  results$graph_edge_jaccard_vs_reference <- NA_real_
  results$graph_common_edges <- NA_integer_
  results$graph_union_edges <- NA_integer_
  results$graph_weight_spearman_vs_reference <- NA_real_
  results$affinity_or_graph_agreement <- ""
  results$notes <- ""

  if (identical(fast_tsne$status, "success")) {
    idx <- results$implementation == "fastEmbedR openTSNE-style native CPU"
    costs <- attr(fast_tsne$value, "itercosts")
    if (!is.null(costs) && length(costs) > 0L) {
      results$kl_final[idx] <- tail(as.numeric(costs), 1L)
    }
    if (identical(rtsne$status, "success")) {
      pc <- procrustes_summary(fast_tsne$value, rtsne$value$Y)
      results$procrustes_rmsd_vs_reference[idx] <- pc$rmsd
      results$procrustes_correlation_vs_reference[idx] <- pc$correlation
    }
    prep_tsne <- fastEmbedR::prepare_opentsne_knn(knn, perplexity = perplexity)
    results$affinity_or_graph_agreement[idx] <- paste0(
      "fastEmbedR sparse t-SNE affinities are built internally from the exact KNN; ",
      "prepared affinity state: ", prep_tsne$affinity_state
    )
  }
  if (identical(rtsne$status, "success")) {
    idx <- results$implementation == "Rtsne::Rtsne_neighbors reference"
    if (!is.null(rtsne$value$itercosts) && length(rtsne$value$itercosts) > 0L) {
      results$kl_final[idx] <- tail(as.numeric(rtsne$value$itercosts), 1L)
    }
  }

  if (identical(fast_umap$status, "success") && identical(uwot_umap$status, "success")) {
    idx <- results$implementation == "fastEmbedR fuzzy native CPU"
    pc <- procrustes_summary(fast_umap$value, uwot_umap$value)
    results$procrustes_rmsd_vs_reference[idx] <- pc$rmsd
    results$procrustes_correlation_vs_reference[idx] <- pc$correlation
    results$notes[idx] <- "UMAP optimizers are stochastic and do not share fixed spectral initialization in this validation."
  }
  if (identical(prep_umap$status, "success") && identical(uwot_graph$status, "success")) {
    ga <- graph_agreement(csr_to_edges(prep_umap$value), matrix_to_edges(uwot_graph$value))
    idx <- results$implementation == "fastEmbedR fuzzy native CPU"
    results$graph_edge_jaccard_vs_reference[idx] <- ga$graph_edge_jaccard
    results$graph_common_edges[idx] <- ga$graph_common_edges
    results$graph_union_edges[idx] <- ga$graph_union_edges
    results$graph_weight_spearman_vs_reference[idx] <- ga$graph_weight_spearman
    results$affinity_or_graph_agreement[idx] <- "UMAP graph compared with uwot::similarity_graph on exact-FNN iris input."
  }

  utils::write.csv(
    results,
    file.path(out_dir, "reference_validation_results.csv"),
    row.names = FALSE
  )
  write_markdown_table(
    results[, c(
      "dataset", "method_family", "implementation", "status",
      "trustworthiness", "knn_preservation_15", "knn_preservation_30",
      "label_knn_accuracy", "kl_final", "procrustes_rmsd_vs_reference",
      "procrustes_correlation_vs_reference", "graph_edge_jaccard_vs_reference",
      "graph_weight_spearman_vs_reference"
    )],
    file.path(out_dir, "reference_validation_results.md")
  )

  layouts <- list(
    "fastEmbedR openTSNE" = fast_tsne$value,
    "Rtsne_neighbors" = if (is.null(rtsne$value)) NULL else rtsne$value$Y,
    "fastEmbedR UMAP fuzzy" = fast_umap$value,
    "uwot UMAP" = uwot_umap$value
  )
  plot_validation(layouts, labels, file.path(out_dir, "reference_validation_plots.png"))

  manifest <- c(
    paste0("date: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    paste0("dataset: ", dataset_name),
    paste0("n: ", nrow(x)),
    paste0("p: ", ncol(x)),
    paste0("seed: ", seed),
    paste0("threads: ", threads),
    paste0("k: ", k_knn),
    paste0("perplexity: ", perplexity),
    "purpose: small reference validation, not runtime benchmarking"
  )
  writeLines(manifest, file.path(out_dir, "reference_validation_manifest.txt"))
  message("Wrote reference validation outputs to: ", normalizePath(out_dir))
}

main()
