#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
embedding_dir <- if (length(args) >= 1L) args[[1L]] else stop(
  "Usage: build_local_cpu_metal_outputs.R EMBEDDING_DIR GRAPH_DIR [OUTPUT_DIR]",
  call. = FALSE
)
graph_dir <- if (length(args) >= 2L) args[[2L]] else stop(
  "A graph/clustering benchmark directory is required.", call. = FALSE
)
output_dir <- if (length(args) >= 3L) args[[3L]] else
  file.path("manuscript", "mloss", "generated")
figure_dir <- file.path(dirname(output_dir), "figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

read_required <- function(root, filename) {
  path <- file.path(root, filename)
  if (!file.exists(path) || file.info(path)$size == 0L) {
    stop("Missing benchmark output: ", path, call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

finite_number <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x)] <- NA_real_
  x
}

embedding <- read_required(embedding_dir, "benchmark_summary_median_variability.csv")
graph <- read_required(graph_dir, "graph_initialization_runs.csv")
clustering <- read_required(graph_dir, "clustering_runs.csv")
graph_agreement <- read_required(
  graph_dir, "cpu_metal_graph_and_initialization_agreement.csv"
)
pca_agreement_path <- file.path(embedding_dir, "pca_vs_irlba_agreement.csv")
pca_agreement <- if (file.exists(pca_agreement_path) &&
                     file.info(pca_agreement_path)$size > 0L) {
  read.csv(pca_agreement_path, stringsAsFactors = FALSE, check.names = FALSE)
} else data.frame()

successful <- embedding[
  embedding$status == "success" &
    grepl("^fastEmbedR|^irlba", embedding$method),
  ,
  drop = FALSE
]
write.csv(
  successful,
  file.path(output_dir, "local_cpu_metal_summary.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  graph,
  file.path(output_dir, "local_graph_initialization_runs.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  graph_agreement,
  file.path(output_dir, "local_cpu_metal_graph_agreement.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  clustering,
  file.path(output_dir, "local_clustering_oracle_runs.csv"),
  row.names = FALSE,
  na = ""
)
if (nrow(pca_agreement)) {
  write.csv(
    pca_agreement,
    file.path(output_dir, "local_pca_vs_irlba_agreement.csv"),
    row.names = FALSE,
    na = ""
  )
}

latex_escape <- function(x) {
  x <- gsub("\\", "\\textbackslash{}", as.character(x), fixed = TRUE)
  x <- gsub("_", "\\_", x, fixed = TRUE)
  x <- gsub("%", "\\%", x, fixed = TRUE)
  x <- gsub("&", "\\&", x, fixed = TRUE)
  x
}

format_metric <- function(x, digits = 3L) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.finite(x), formatC(x, digits = digits, format = "f"), "--")
}

graph_table_lines <- c(
  "\\begin{tabular}{lrrrr}",
  "\\toprule",
  "Dataset & Edge Jaccard & Weight Pearson & Weight Spearman & Init. Procrustes \\\\",
  "\\midrule"
)
if (nrow(graph_agreement)) {
  graph_table_lines <- c(
    graph_table_lines,
    paste0(
      latex_escape(graph_agreement$dataset), " & ",
      format_metric(graph_agreement$edge_jaccard), " & ",
      format_metric(graph_agreement$weight_pearson), " & ",
      format_metric(graph_agreement$weight_spearman), " & ",
      format_metric(
        graph_agreement$common_initialization_procrustes_correlation
      ),
      " \\\\"
    )
  )
}
graph_table_lines <- c(graph_table_lines, "\\bottomrule", "\\end{tabular}")
writeLines(
  graph_table_lines,
  file.path(output_dir, "local_cpu_metal_graph_agreement.tex")
)

oracle <- clustering[
  clustering$status == "success" &
    clustering$igraph_status == "success" &
    is.finite(finite_number(clustering$native_vs_igraph_ari)),
  ,
  drop = FALSE
]
if (nrow(oracle)) {
  keys <- interaction(
    oracle$dataset, oracle$method, drop = TRUE, lex.order = TRUE
  )
  oracle_summary <- do.call(rbind, lapply(split(oracle, keys), function(rows) {
    data.frame(
      dataset = rows$dataset[[1L]],
      method = rows$method[[1L]],
      native_vs_igraph_ari = stats::median(
        finite_number(rows$native_vs_igraph_ari), na.rm = TRUE
      ),
      native_vs_igraph_nmi = stats::median(
        finite_number(rows$native_vs_igraph_nmi), na.rm = TRUE
      ),
      native_runtime_sec = stats::median(
        finite_number(rows$native_runtime_sec), na.rm = TRUE
      ),
      igraph_runtime_sec = stats::median(
        finite_number(rows$igraph_runtime_sec), na.rm = TRUE
      ),
      stringsAsFactors = FALSE
    )
  }))
} else {
  oracle_summary <- data.frame()
}
write.csv(
  oracle_summary,
  file.path(output_dir, "local_clustering_oracle_summary.csv"),
  row.names = FALSE,
  na = ""
)
oracle_table_lines <- c(
  "\\begin{tabular}{llrrrr}",
  "\\toprule",
  "Dataset & Method & ARI & NMI & Native sec. & igraph sec. \\\\",
  "\\midrule"
)
if (nrow(oracle_summary)) {
  oracle_table_lines <- c(
    oracle_table_lines,
    paste0(
      latex_escape(oracle_summary$dataset), " & ",
      latex_escape(tools::toTitleCase(oracle_summary$method)), " & ",
      format_metric(oracle_summary$native_vs_igraph_ari), " & ",
      format_metric(oracle_summary$native_vs_igraph_nmi), " & ",
      format_metric(oracle_summary$native_runtime_sec), " & ",
      format_metric(oracle_summary$igraph_runtime_sec),
      " \\\\"
    )
  )
}
oracle_table_lines <- c(
  oracle_table_lines, "\\bottomrule", "\\end{tabular}"
)
writeLines(
  oracle_table_lines,
  file.path(output_dir, "local_clustering_oracle_summary.tex")
)

backend_label <- function(method, backend, threads) {
  if (identical(backend, "metal")) return("Metal")
  if (grepl("^irlba", method)) return(sprintf("irlba (%dt)", threads))
  sprintf("CPU (%dt)", threads)
}

method_mode <- function(method) {
  if (grepl("opentsne", method)) return("openTSNE")
  if (grepl("umap.*fuzzy", method)) return("UMAP fuzzy")
  if (grepl("umap.*binary", method)) return("UMAP binary")
  if (grepl("pca", method)) return("PCA")
  method
}

full <- successful[
  successful$timing_scope == "full_pipeline" &
    grepl("opentsne|umap", successful$method) &
    grepl("_full$", successful$method),
  ,
  drop = FALSE
]
full$implementation <- mapply(
  backend_label,
  full$method,
  full$backend,
  as.integer(full$requested_threads),
  USE.NAMES = FALSE
)
full$algorithm <- vapply(full$method, method_mode, character(1))
full$total_runtime_sec_median <- finite_number(full$total_runtime_sec_median)
full$trustworthiness_median <- finite_number(full$trustworthiness_median)
full$knn_preservation_30_median <- finite_number(full$knn_preservation_30_median)
full$peak_ram_gb_median <- finite_number(full$peak_ram_gb_median)
write.csv(
  full,
  file.path(output_dir, "local_full_pipeline_runtime_quality.csv"),
  row.names = FALSE,
  na = ""
)

pca_rows <- successful[
  successful$timing_scope == "pca_only",
  ,
  drop = FALSE
]
pca_rows$implementation <- mapply(
  backend_label,
  pca_rows$method,
  pca_rows$backend,
  as.integer(pca_rows$requested_threads),
  USE.NAMES = FALSE
)
pca_rows$total_runtime_sec_median <- finite_number(
  pca_rows$total_runtime_sec_median
)
for (column in c("total_runtime_sec_q1", "total_runtime_sec_q3")) {
  if (column %in% names(pca_rows)) {
    pca_rows[[column]] <- finite_number(pca_rows[[column]])
  }
}
write.csv(
  pca_rows,
  file.path(output_dir, "local_pca_runtime.csv"),
  row.names = FALSE,
  na = ""
)

method_colors <- c(
  "CPU (1t)" = "#0072B2",
  "CPU (4t)" = "#009E73",
  "Metal" = "#D55E00",
  "irlba (1t)" = "#6A3D9A",
  "irlba (4t)" = "#CC79A7"
)
dataset_levels <- unique(embedding$dataset)

draw_metric_panel <- function(data, algorithm, value, ylab, log_y = FALSE) {
  piece <- data[data$algorithm == algorithm & is.finite(data[[value]]), , drop = FALSE]
  if (!nrow(piece)) {
    plot.new()
    title(main = algorithm)
    text(0.5, 0.5, "No successful rows")
    return(invisible(NULL))
  }
  piece$dataset_index <- match(piece$dataset, dataset_levels)
  implementations <- intersect(names(method_colors), unique(piece$implementation))
  values <- piece[[value]]
  q1_name <- sub("_median$", "_q1", value)
  q3_name <- sub("_median$", "_q3", value)
  if (all(c(q1_name, q3_name) %in% names(piece))) {
    values <- c(values, finite_number(piece[[q1_name]]), finite_number(piece[[q3_name]]))
  }
  ylim <- range(values, finite = TRUE)
  if (log_y) ylim <- range(log10(pmax(values, .Machine$double.xmin)), finite = TRUE)
  plot(
    NA,
    xlim = c(0.6, length(dataset_levels) + 0.4),
    ylim = ylim,
    xaxt = "n",
    xlab = "",
    ylab = ylab,
    main = algorithm,
    bty = "l"
  )
  axis(1, at = seq_along(dataset_levels), labels = dataset_levels, las = 2, cex.axis = 0.72)
  for (implementation in implementations) {
    row <- piece[piece$implementation == implementation, , drop = FALSE]
    y <- row[[value]]
    has_interval <- q1_name %in% names(row) && q3_name %in% names(row)
    if (has_interval) {
      lower <- finite_number(row[[q1_name]])
      upper <- finite_number(row[[q3_name]])
    }
    if (log_y) {
      y <- log10(pmax(y, .Machine$double.xmin))
      if (has_interval) {
        lower <- log10(pmax(lower, .Machine$double.xmin))
        upper <- log10(pmax(upper, .Machine$double.xmin))
      }
    }
    if (has_interval) {
      valid <- is.finite(lower) & is.finite(upper)
      segments(
        row$dataset_index[valid],
        lower[valid],
        row$dataset_index[valid],
        upper[valid],
        col = grDevices::adjustcolor(method_colors[[implementation]], 0.65),
        lwd = 1.2
      )
    }
    lines(
      row$dataset_index,
      y,
      type = "b",
      pch = 16,
      lwd = 1.8,
      col = method_colors[[implementation]]
    )
  }
  legend(
    "topright",
    legend = implementations,
    col = method_colors[implementations],
    pch = 16,
    lwd = 1.8,
    bty = "n",
    cex = 0.78
  )
}

runtime_quality_png <- file.path(figure_dir, "current_local_runtime_quality.png")
runtime_quality_pdf <- file.path(figure_dir, "current_local_runtime_quality.pdf")
draw_runtime_quality <- function(device) {
  device()
  old <- par(mfrow = c(2, 2), mar = c(7, 4.2, 2.5, 1), oma = c(0, 0, 1.5, 0))
  on.exit({
    par(old)
    dev.off()
  }, add = TRUE)
  draw_metric_panel(full, "openTSNE", "total_runtime_sec_median", "log10 total seconds", TRUE)
  draw_metric_panel(full, "UMAP fuzzy", "total_runtime_sec_median", "log10 total seconds", TRUE)
  draw_metric_panel(full, "openTSNE", "trustworthiness_median", "Trustworthiness")
  draw_metric_panel(full, "UMAP fuzzy", "knn_preservation_30_median", "Preserve@30")
  mtext("Current-code Mac CPU/Metal benchmark", outer = TRUE, cex = 1.15, font = 2)
}
draw_runtime_quality(function() {
  png(runtime_quality_png, width = 2600, height = 1900, res = 240, bg = "white")
})
draw_runtime_quality(function() {
  pdf(runtime_quality_pdf, width = 10.8, height = 8.0, useDingbats = FALSE)
})

draw_pca_graph_cluster <- function(device) {
  device()
  old <- par(mfrow = c(2, 2), mar = c(7, 4.2, 2.5, 1), oma = c(0, 0, 1.5, 0))
  on.exit({
    par(old)
    dev.off()
  }, add = TRUE)

  pca_plot <- pca_rows[is.finite(pca_rows$total_runtime_sec_median), , drop = FALSE]
  pca_plot$dataset_index <- match(pca_plot$dataset, dataset_levels)
  pca_limits <- pca_plot$total_runtime_sec_median
  if (all(c("total_runtime_sec_q1", "total_runtime_sec_q3") %in% names(pca_plot))) {
    pca_limits <- c(
      pca_limits,
      pca_plot$total_runtime_sec_q1,
      pca_plot$total_runtime_sec_q3
    )
  }
  plot(
    NA,
    xlim = c(0.6, length(dataset_levels) + 0.4),
    ylim = range(
      log10(pmax(pca_limits, .Machine$double.xmin)),
      finite = TRUE
    ),
    xaxt = "n",
    xlab = "",
    ylab = "log10 seconds",
    main = "Randomized PCA runtime",
    bty = "l"
  )
  axis(1, at = seq_along(dataset_levels), labels = dataset_levels, las = 2, cex.axis = 0.72)
  for (implementation in intersect(names(method_colors), unique(pca_plot$implementation))) {
    row <- pca_plot[pca_plot$implementation == implementation, , drop = FALSE]
    if (all(c("total_runtime_sec_q1", "total_runtime_sec_q3") %in% names(row))) {
      valid <- is.finite(row$total_runtime_sec_q1) &
        is.finite(row$total_runtime_sec_q3)
      segments(
        row$dataset_index[valid],
        log10(pmax(row$total_runtime_sec_q1[valid], .Machine$double.xmin)),
        row$dataset_index[valid],
        log10(pmax(row$total_runtime_sec_q3[valid], .Machine$double.xmin)),
        col = grDevices::adjustcolor(method_colors[[implementation]], 0.65),
        lwd = 1.2
      )
    }
    lines(
      row$dataset_index,
      log10(pmax(row$total_runtime_sec_median, .Machine$double.xmin)),
      type = "b",
      pch = 16,
      lwd = 1.8,
      col = method_colors[[implementation]]
    )
  }
  legend(
    "topright",
    legend = intersect(names(method_colors), unique(pca_plot$implementation)),
    col = method_colors[intersect(names(method_colors), unique(pca_plot$implementation))],
    pch = 16,
    lwd = 1.8,
    bty = "n",
    cex = 0.72
  )

  if (nrow(pca_agreement)) {
    agreement <- pca_agreement[is.finite(pca_agreement$procrustes_correlation), , drop = FALSE]
    agreement$label <- ifelse(agreement$backend == "metal", "Metal", "CPU")
    boxplot(
      procrustes_correlation ~ label,
      data = agreement,
      col = c("#009E73", "#D55E00"),
      ylab = "Procrustes correlation",
      xlab = "",
      main = "PCA agreement with irlba",
      outline = FALSE
    )
  } else {
    plot.new()
    title(main = "PCA agreement with irlba")
  }

  graph$graph_total_sec <- rowSums(cbind(
    finite_number(graph$knn_sec),
    finite_number(graph$snn_graph_sec),
    finite_number(graph$umap_init_total_sec)
  ), na.rm = TRUE)
  graph$implementation <- ifelse(
    graph$backend == "metal", "Metal",
    sprintf("CPU (%dt)", as.integer(graph$threads))
  )
  graph$dataset_index <- match(graph$dataset, dataset_levels)
  plot(
    NA,
    xlim = c(0.6, length(dataset_levels) + 0.4),
    ylim = range(log10(pmax(graph$graph_total_sec, .Machine$double.xmin))),
    xaxt = "n",
    xlab = "",
    ylab = "log10 seconds",
    main = "KNN + graph + UMAP initialization",
    bty = "l"
  )
  axis(1, at = seq_along(dataset_levels), labels = dataset_levels, las = 2, cex.axis = 0.72)
  for (implementation in intersect(names(method_colors), unique(graph$implementation))) {
    row <- graph[graph$implementation == implementation, , drop = FALSE]
    lines(
      row$dataset_index,
      log10(pmax(row$graph_total_sec, .Machine$double.xmin)),
      type = "b",
      pch = 16,
      lwd = 1.8,
      col = method_colors[[implementation]]
    )
  }

  oracle <- clustering[
    clustering$igraph_status == "success" &
      is.finite(clustering$native_vs_igraph_ari),
    ,
    drop = FALSE
  ]
  if (nrow(oracle)) {
    boxplot(
      native_vs_igraph_ari ~ method,
      data = oracle,
      col = c("#56B4E9", "#E69F00", "#CC79A7"),
      ylim = c(0, 1),
      ylab = "ARI",
      xlab = "",
      main = "Native versus igraph membership",
      outline = FALSE
    )
    abline(h = 1, lty = 3, col = "#555555")
  } else {
    plot.new()
    title(main = "Native versus igraph membership")
  }
  mtext("PCA, graph, and downstream clustering validation", outer = TRUE, cex = 1.15, font = 2)
}

pca_graph_png <- file.path(figure_dir, "current_local_pca_graph_clustering.png")
pca_graph_pdf <- file.path(figure_dir, "current_local_pca_graph_clustering.pdf")
draw_pca_graph_cluster(function() {
  png(pca_graph_png, width = 2600, height = 1900, res = 240, bg = "white")
})
draw_pca_graph_cluster(function() {
  pdf(pca_graph_pdf, width = 10.8, height = 8.0, useDingbats = FALSE)
})

cat("Wrote current local manuscript outputs to ", normalizePath(output_dir), "\n", sep = "")
