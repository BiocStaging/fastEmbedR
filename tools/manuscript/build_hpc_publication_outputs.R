#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_root <- if (length(args) >= 1L) args[[1L]] else
  "/Users/stefano/Documents/fastEmbedR-results/fastEmbedR-results"
output_dir <- if (length(args) >= 2L) args[[2L]] else
  "/Users/stefano/Documents/umap/manuscript/mloss/generated"
figure_dir <- file.path(dirname(output_dir), "figures")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

read_csv_safe <- function(path) {
  tryCatch(
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) NULL
  )
}

clean_number <- function(x) {
  suppressWarnings(as.numeric(x))
}

path_parts <- function(path) {
  relative <- sub(paste0("^", gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1",
                                   normalizePath(results_root)), "/?"), "", normalizePath(path))
  strsplit(relative, "/", fixed = TRUE)[[1L]]
}

summary_files <- list.files(
  results_root,
  pattern = "^benchmark_summary_median_variability\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)
summary_files <- summary_files[!grepl("/kodama/", summary_files, fixed = TRUE)]

summary_rows <- lapply(summary_files, function(path) {
  x <- read_csv_safe(path)
  if (is.null(x) || !nrow(x)) return(NULL)
  parts <- path_parts(path)
  x$suite <- if (length(parts) >= 2L) parts[[2L]] else NA_character_
  x$profile <- if (length(parts) >= 3L) parts[[3L]] else NA_character_
  x$source_file <- path
  x
})
summary_rows <- Filter(Negate(is.null), summary_rows)
all_summary <- do.call(rbind, summary_rows)
all_summary <- all_summary[
  !grepl("KODAMA", all_summary$method, ignore.case = TRUE) &
    !grepl("kodama", all_summary$source_file, ignore.case = TRUE),
  ,
  drop = FALSE
]
write.csv(all_summary, file.path(output_dir, "all_non_kodama_summary.csv"),
          row.names = FALSE, na = "")

standard <- all_summary[all_summary$suite == "standard", , drop = FALSE]
landmark <- all_summary[all_summary$suite == "landmark", , drop = FALSE]
write.csv(standard, file.path(output_dir, "standard_summary.csv"),
          row.names = FALSE, na = "")
write.csv(landmark, file.path(output_dir, "landmark_summary.csv"),
          row.names = FALSE, na = "")

python_quality_files <- list.files(
  results_root,
  pattern = "^embedding_quality_table\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)
python_quality_files <- python_quality_files[
  grepl("/python/", python_quality_files, fixed = TRUE) &
    !grepl("/kodama/", python_quality_files, fixed = TRUE)
]
python_rows <- lapply(python_quality_files, function(path) {
  x <- read_csv_safe(path)
  if (is.null(x) || !nrow(x)) return(NULL)
  parts <- path_parts(path)
  x$suite <- if (length(parts) >= 2L) parts[[2L]] else NA_character_
  x$profile <- if (length(parts) >= 3L) parts[[3L]] else NA_character_
  seed_part <- parts[grepl("^seed_", parts)]
  x$seed <- if (length(seed_part)) sub("^seed_", "", seed_part[[1L]]) else NA_character_
  x$source_file <- path
  x
})
python_rows <- Filter(Negate(is.null), python_rows)
python_runs <- if (length(python_rows)) do.call(rbind, python_rows) else data.frame()
if (nrow(python_runs)) {
  direct_python <- python_runs$timing_mode == "native_python_process"
  python_runs$timing_scope <- ifelse(
    direct_python,
    "direct_python_fit",
    "r_mediated_total_call"
  )
  python_runs$runtime_measure <- ifelse(
    direct_python,
    "direct_python_fit_sec",
    "r_mediated_total_call_sec"
  )
  process_elapsed <- clean_number(python_runs$process_elapsed_sec)
  selected_runtime <- clean_number(python_runs$runtime_sec)
  python_fit <- clean_number(python_runs$python_fit_sec)
  python_runs$r_mediated_total_call_sec <- ifelse(
    !direct_python,
    ifelse(is.finite(process_elapsed), process_elapsed, selected_runtime),
    NA_real_
  )
  python_runs$direct_python_fit_sec <- ifelse(
    direct_python,
    ifelse(is.finite(python_fit), python_fit, selected_runtime),
    NA_real_
  )
  python_runs$direct_python_process_total_sec <- ifelse(
    direct_python,
    process_elapsed,
    NA_real_
  )
  write.csv(python_runs, file.path(output_dir, "python_non_kodama_runs.csv"),
            row.names = FALSE, na = "")
  python_success <- python_runs[python_runs$status == "success", , drop = FALSE]
  if (nrow(python_success)) {
    python_success$runtime_sec <- clean_number(python_success$runtime_sec)
    python_success$trustworthiness <- clean_number(python_success$trustworthiness)
    python_success$nn_preservation <- clean_number(python_success$nn_preservation)
    python_success$silhouette <- clean_number(python_success$silhouette)
    python_success$knn_label_accuracy <- clean_number(python_success$knn_label_accuracy)
    grouping <- interaction(
      python_success$dataset, python_success$method, python_success$backend,
      python_success$profile, drop = TRUE
    )
    summarize_time <- function(values) {
      values <- clean_number(values)
      values <- values[is.finite(values)]
      if (!length(values)) {
        return(c(median = NA_real_, q1 = NA_real_, q3 = NA_real_))
      }
      c(
        median = stats::median(values),
        q1 = as.numeric(stats::quantile(values, 0.25, names = FALSE)),
        q3 = as.numeric(stats::quantile(values, 0.75, names = FALSE))
      )
    }
    python_summary <- do.call(rbind, lapply(split(python_success, grouping), function(x) {
      selected <- summarize_time(x$runtime_sec)
      r_total <- summarize_time(x$r_mediated_total_call_sec)
      direct_fit <- summarize_time(x$direct_python_fit_sec)
      direct_process <- summarize_time(x$direct_python_process_total_sec)
      data.frame(
        dataset = x$dataset[[1L]],
        method = x$method[[1L]],
        backend = x$backend[[1L]],
        profile = x$profile[[1L]],
        timing_scope = x$timing_scope[[1L]],
        runtime_measure = x$runtime_measure[[1L]],
        n_runs = nrow(x),
        runtime_sec_median = selected[["median"]],
        runtime_sec_q1 = selected[["q1"]],
        runtime_sec_q3 = selected[["q3"]],
        r_mediated_total_call_sec_median = r_total[["median"]],
        r_mediated_total_call_sec_q1 = r_total[["q1"]],
        r_mediated_total_call_sec_q3 = r_total[["q3"]],
        direct_python_fit_sec_median = direct_fit[["median"]],
        direct_python_fit_sec_q1 = direct_fit[["q1"]],
        direct_python_fit_sec_q3 = direct_fit[["q3"]],
        direct_python_process_total_sec_median = direct_process[["median"]],
        direct_python_process_total_sec_q1 = direct_process[["q1"]],
        direct_python_process_total_sec_q3 = direct_process[["q3"]],
        trustworthiness_median = median(x$trustworthiness, na.rm = TRUE),
        nn_preservation_median = median(x$nn_preservation, na.rm = TRUE),
        silhouette_median = median(x$silhouette, na.rm = TRUE),
        label_accuracy_median = median(x$knn_label_accuracy, na.rm = TRUE)
      )
    }))
    write.csv(python_summary, file.path(output_dir, "python_summary_median.csv"),
              row.names = FALSE, na = "")
  }
}

collect_diagnostic <- function(filename) {
  paths <- list.files(results_root, pattern = paste0("^", filename, "$"),
                      recursive = TRUE, full.names = TRUE)
  paths <- paths[!grepl("/kodama/", paths, fixed = TRUE)]
  rows <- lapply(paths, function(path) {
    x <- read_csv_safe(path)
    if (is.null(x) || !nrow(x)) return(NULL)
    parts <- path_parts(path)
    x$suite <- if (length(parts) >= 2L) parts[[2L]] else NA_character_
    x$profile <- if (length(parts) >= 3L) parts[[3L]] else NA_character_
    x$source_file <- path
    x
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(data.frame())
  do.call(rbind, rows)
}

diagnostic_files <- c(
  "knn_affinity_umap_graph_agreement.csv",
  "tsne_affinity_agreement_vs_python_opentsne.csv",
  "umap_graph_agreement_vs_uwot.csv",
  "pca_vs_irlba_agreement.csv",
  "stability_pairwise.csv",
  "landmark_validation_vs_full.csv"
)
for (filename in diagnostic_files) {
  diagnostic <- collect_diagnostic(filename)
  if (nrow(diagnostic)) {
    write.csv(
      diagnostic,
      file.path(output_dir, sub("\\.csv$", "_all.csv", filename)),
      row.names = FALSE,
      na = ""
    )
  }
}

success <- standard[
  standard$status == "success" &
    !is.na(standard$timing_scope) &
    standard$timing_scope %in% c("full_pipeline", "embedding_from_precomputed_knn", "pca_only"),
  ,
  drop = FALSE
]
write.csv(success, file.path(output_dir, "successful_standard_results.csv"),
          row.names = FALSE, na = "")

method_labels <- c(
  fastEmbedR_opentsne_cpu_full = "fastEmbedR openTSNE CPU",
  fastEmbedR_opentsne_cuda_full = "fastEmbedR openTSNE CUDA",
  Rtsne_full = "Rtsne",
  KlugerLab_FItSNE = "FIt-SNE",
  fastEmbedR_umap_cpu_fuzzy_full = "fastEmbedR UMAP CPU",
  fastEmbedR_umap_cuda_fuzzy_full = "fastEmbedR UMAP CUDA",
  uwot_fast_sgd = "uwot fast SGD",
  uwot_default = "uwot default",
  umap_package = "umap package"
)

full <- success[
  success$timing_scope == "full_pipeline" &
    success$method %in% names(method_labels),
  ,
  drop = FALSE
]
full$method_label <- unname(method_labels[full$method])
full$runtime <- clean_number(full$total_runtime_sec_median)
full$runtime_q1 <- clean_number(full$total_runtime_sec_q1)
full$runtime_q3 <- clean_number(full$total_runtime_sec_q3)
full$trust <- clean_number(full$trustworthiness_median)
full$preserve30 <- clean_number(full$knn_preservation_30_median)
full$silhouette <- clean_number(full$silhouette_median)
full$label_accuracy <- clean_number(full$label_knn_accuracy_median)
full$peak_ram_gb <- clean_number(full$peak_ram_gb_median)
full$threads <- clean_number(full$requested_threads)
full <- full[
  (full$profile == "cpu4" & full$backend == "cpu") |
    (full$profile == "cuda" & full$backend == "cuda"),
  ,
  drop = FALSE
]
write.csv(full, file.path(output_dir, "full_pipeline_cpu4_cuda.csv"),
          row.names = FALSE, na = "")

matched_speedup <- function(candidate, reference, family) {
  a <- full[full$method == candidate & full$family == family,
            c("dataset", "runtime", "trust", "preserve30", "label_accuracy", "peak_ram_gb")]
  b <- full[full$method == reference & full$family == family,
            c("dataset", "runtime", "trust", "preserve30", "label_accuracy", "peak_ram_gb")]
  names(a)[-1L] <- paste0(names(a)[-1L], "_candidate")
  names(b)[-1L] <- paste0(names(b)[-1L], "_reference")
  z <- merge(a, b, by = "dataset")
  z$speedup <- z$runtime_reference / z$runtime_candidate
  z$trust_delta <- z$trust_candidate - z$trust_reference
  z$preserve30_delta <- z$preserve30_candidate - z$preserve30_reference
  z$label_accuracy_delta <- z$label_accuracy_candidate - z$label_accuracy_reference
  z$ram_ratio <- z$peak_ram_gb_candidate / z$peak_ram_gb_reference
  z$candidate <- candidate
  z$reference <- reference
  z$family <- family
  z
}

comparisons <- rbind(
  matched_speedup("fastEmbedR_opentsne_cpu_full", "Rtsne_full", "t-SNE"),
  matched_speedup("fastEmbedR_opentsne_cuda_full", "Rtsne_full", "t-SNE"),
  matched_speedup("fastEmbedR_umap_cpu_fuzzy_full", "uwot_fast_sgd", "UMAP"),
  matched_speedup("fastEmbedR_umap_cuda_fuzzy_full", "uwot_fast_sgd", "UMAP")
)
write.csv(comparisons, file.path(output_dir, "matched_reference_comparisons.csv"),
          row.names = FALSE, na = "")

claim_summary <- do.call(rbind, lapply(
  split(comparisons, interaction(comparisons$candidate, comparisons$reference, drop = TRUE)),
  function(x) {
    data.frame(
      candidate = x$candidate[[1L]],
      reference = x$reference[[1L]],
      family = x$family[[1L]],
      matched_datasets = nrow(x),
      speedup_median = median(x$speedup, na.rm = TRUE),
      speedup_q1 = unname(quantile(x$speedup, 0.25, na.rm = TRUE)),
      speedup_q3 = unname(quantile(x$speedup, 0.75, na.rm = TRUE)),
      speedup_min = min(x$speedup, na.rm = TRUE),
      speedup_max = max(x$speedup, na.rm = TRUE),
      trust_delta_median = median(x$trust_delta, na.rm = TRUE),
      trust_delta_max_abs = max(abs(x$trust_delta), na.rm = TRUE),
      preserve30_delta_median = median(x$preserve30_delta, na.rm = TRUE),
      label_accuracy_delta_median = median(x$label_accuracy_delta, na.rm = TRUE),
      ram_ratio_median = median(x$ram_ratio, na.rm = TRUE)
    )
  }
))
write.csv(claim_summary, file.path(output_dir, "claim_summary.csv"),
          row.names = FALSE, na = "")

key_datasets <- c(
  "MNIST", "FashionMNIST", "TabulaMuris", "Macosko2015_retina",
  "flow18", "mass41", "FlowRepository_FR-FCM-ZYRM_files", "imagenet"
)
key_methods <- c(
  "fastEmbedR_opentsne_cpu_full", "fastEmbedR_opentsne_cuda_full",
  "Rtsne_full", "KlugerLab_FItSNE",
  "fastEmbedR_umap_cpu_fuzzy_full", "fastEmbedR_umap_cuda_fuzzy_full",
  "uwot_fast_sgd"
)
key_table <- full[full$dataset %in% key_datasets & full$method %in% key_methods,
                  c("dataset", "family", "method_label", "backend", "threads",
                    "runtime", "runtime_q1", "runtime_q3", "peak_ram_gb",
                    "trust", "preserve30", "silhouette", "label_accuracy")]
key_table <- key_table[order(match(key_table$dataset, key_datasets),
                             key_table$family, key_table$runtime), ]
write.csv(key_table, file.path(output_dir, "key_dataset_runtime_quality.csv"),
          row.names = FALSE, na = "")

latex_escape <- function(x) {
  slash <- intToUtf8(92L)
  x <- gsub("_", paste0(slash, "_"), x, fixed = TRUE)
  x <- gsub("&", paste0(slash, "&"), x, fixed = TRUE)
  x <- gsub("%", paste0(slash, "%"), x, fixed = TRUE)
  x
}

table_rows <- key_table[
  key_table$dataset %in% c(
    "MNIST", "FashionMNIST", "TabulaMuris", "Macosko2015_retina",
    "flow18", "mass41"
  ),
  ,
  drop = FALSE
]
display_dataset <- c(
  MNIST = "MNIST", FashionMNIST = "Fashion-MNIST",
  TabulaMuris = "Tabula Muris", Macosko2015_retina = "Retina",
  flow18 = "flow18", mass41 = "mass41"
)
method_short <- c(
  "fastEmbedR openTSNE CPU" = "fastEmbedR openTSNE",
  "fastEmbedR openTSNE CUDA" = "fastEmbedR openTSNE",
  "Rtsne" = "Rtsne",
  "FIt-SNE" = "FIt-SNE",
  "fastEmbedR UMAP CPU" = "fastEmbedR fuzzy UMAP",
  "fastEmbedR UMAP CUDA" = "fastEmbedR fuzzy UMAP",
  "uwot fast SGD" = "uwot fast SGD"
)
tex <- c(
  "{\\footnotesize",
  "\\setlength{\\tabcolsep}{3pt}",
  "\\begin{longtable}{p{0.13\\textwidth}p{0.24\\textwidth}lrrrr}",
  "\\caption{Median total runtime and quality over three seeds for representative benchmark datasets.}\\label{tab:hpcquality}\\\\",
  "\\toprule",
  "Dataset & Method & Backend & Seconds & Trust. & Preserve@30 & Label acc.\\\\",
  "\\midrule",
  "\\endfirsthead",
  "\\toprule",
  "Dataset & Method & Backend & Seconds & Trust. & Preserve@30 & Label acc.\\\\",
  "\\midrule",
  "\\endhead"
)
for (i in seq_len(nrow(table_rows))) {
  row <- table_rows[i, ]
  tex <- c(tex, sprintf(
    "%s & %s & %s & %.2f & %.3f & %.3f & %.3f\\\\",
    latex_escape(unname(display_dataset[row$dataset])),
    latex_escape(unname(method_short[row$method_label])),
    toupper(latex_escape(row$backend)),
    row$runtime, row$trust, row$preserve30, row$label_accuracy
  ))
}
tex <- c(tex, "\\bottomrule", "\\end{longtable}", "}")
writeLines(tex, file.path(output_dir, "key_dataset_runtime_quality.tex"))

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("patchwork", quietly = TRUE)) {
  stop("ggplot2 and patchwork are required to build publication figures")
}

library(ggplot2)
library(patchwork)

plot_datasets <- c(
  "MNIST", "FashionMNIST", "TabulaMuris",
  "Macosko2015_retina", "flow18", "mass41"
)
plot_names <- c(
  MNIST = "MNIST", FashionMNIST = "Fashion-MNIST",
  TabulaMuris = "Tabula Muris", Macosko2015_retina = "Retina",
  flow18 = "flow18", mass41 = "mass41"
)
tsne_methods <- c(
  "fastEmbedR openTSNE CPU", "fastEmbedR openTSNE CUDA", "Rtsne", "FIt-SNE"
)
umap_methods <- c(
  "fastEmbedR UMAP CPU", "fastEmbedR UMAP CUDA", "uwot fast SGD"
)
palette <- c(
  "fastEmbedR openTSNE CPU" = "#0072B2",
  "fastEmbedR openTSNE CUDA" = "#009E73",
  "Rtsne" = "#D55E00",
  "FIt-SNE" = "#CC79A7",
  "fastEmbedR UMAP CPU" = "#0072B2",
  "fastEmbedR UMAP CUDA" = "#009E73",
  "uwot fast SGD" = "#D55E00"
)

plot_data <- full[full$dataset %in% plot_datasets, , drop = FALSE]
plot_data$dataset_label <- factor(plot_names[plot_data$dataset],
                                  levels = unname(plot_names[plot_datasets]))
plot_data$method_label <- factor(plot_data$method_label,
                                 levels = c(tsne_methods, umap_methods))

base_theme <- theme_minimal(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    legend.text = element_text(size = 7.2),
    legend.key.width = grid::unit(0.8, "lines"),
    legend.title = element_blank(),
    plot.title = element_text(face = "bold", size = 10),
    plot.margin = margin(4, 5, 4, 5)
  )

runtime_panel <- function(family, methods, title) {
  d <- plot_data[plot_data$family == family &
                   plot_data$method_label %in% methods, , drop = FALSE]
  ggplot(d, aes(dataset_label, runtime, color = method_label, group = method_label)) +
    geom_linerange(aes(ymin = runtime_q1, ymax = runtime_q3),
                   position = position_dodge(width = 0.55), linewidth = 0.5) +
    geom_point(position = position_dodge(width = 0.55), size = 2) +
    scale_y_log10() +
    scale_color_manual(values = palette, drop = TRUE) +
    labs(x = NULL, y = "Total runtime (s, log scale)", title = title) +
    guides(color = guide_legend(nrow = if (length(methods) > 3L) 2 else 1,
                                byrow = TRUE)) +
    base_theme
}

quality_panel <- function(family, methods, title) {
  d <- plot_data[plot_data$family == family &
                   plot_data$method_label %in% methods, , drop = FALSE]
  ggplot(d, aes(dataset_label, trust, color = method_label, group = method_label)) +
    geom_point(position = position_dodge(width = 0.55), size = 2) +
    scale_color_manual(values = palette, drop = TRUE) +
    coord_cartesian(ylim = c(min(0.70, min(d$trust, na.rm = TRUE) - 0.02), 1)) +
    labs(x = NULL, y = "Trustworthiness", title = title) +
    base_theme +
    theme(legend.position = "none")
}

publication_figure <- (
  runtime_panel("t-SNE", tsne_methods, "A  t-SNE total runtime") |
    quality_panel("t-SNE", tsne_methods, "B  t-SNE quality")
) / (
  runtime_panel("UMAP", umap_methods, "C  UMAP total runtime") |
    quality_panel("UMAP", umap_methods, "D  UMAP quality")
) +
  plot_layout(heights = c(1, 1))

ggsave(
  file.path(figure_dir, "hpc_runtime_quality.png"),
  publication_figure,
  width = 10.5, height = 8.2, dpi = 320, bg = "white"
)
ggsave(
  file.path(figure_dir, "hpc_runtime_quality.pdf"),
  publication_figure,
  width = 10.5, height = 8.2, device = cairo_pdf
)

landmark_success <- landmark[landmark$status == "success", , drop = FALSE]
if (nrow(landmark_success)) {
  landmark_success$runtime <- clean_number(landmark_success$total_runtime_sec_median)
  landmark_success$trust <- clean_number(landmark_success$trustworthiness_median)
  landmark_success$preserve30 <- clean_number(landmark_success$knn_preservation_30_median)
  landmark_success$n_landmarks <- clean_number(landmark_success$n_landmarks_median)
  write.csv(landmark_success, file.path(output_dir, "successful_landmark_results.csv"),
            row.names = FALSE, na = "")
}

find_plot <- function(dataset, profile, method) {
  base <- file.path(results_root, dataset, "standard", profile)
  candidates <- list.files(
    base,
    pattern = paste0("^", dataset, "_", method, "_.*seed4\\.png$"),
    recursive = TRUE,
    full.names = TRUE
  )
  if (!length(candidates)) return(NA_character_)
  sort(candidates)[[1L]]
}

panels <- data.frame(
  dataset = c("MNIST", "MNIST", "FashionMNIST", "TabulaMuris", "flow18", "mass41"),
  profile = rep("cuda", 6L),
  method = c(
    "fastEmbedR_opentsne_cuda_full",
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_opentsne_cuda_full",
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_opentsne_cuda_full"
  ),
  title = c(
    "A  MNIST: openTSNE",
    "B  MNIST: fuzzy UMAP",
    "C  Fashion-MNIST: fuzzy UMAP",
    "D  Tabula Muris: openTSNE",
    "E  flow18: fuzzy UMAP",
    "F  mass41: openTSNE"
  ),
  stringsAsFactors = FALSE
)
panels$path <- mapply(find_plot, panels$dataset, panels$profile, panels$method,
                      USE.NAMES = FALSE)
write.csv(panels, file.path(output_dir, "representative_embedding_sources.csv"),
          row.names = FALSE, na = "")

if (all(file.exists(panels$path)) && requireNamespace("png", quietly = TRUE)) {
  png(file.path(figure_dir, "hpc_representative_embeddings.png"),
      width = 4800, height = 3200, res = 300, bg = "white")
  par(mfrow = c(2, 3), mar = c(0.2, 0.2, 2.2, 0.2), oma = c(0, 0, 0, 0),
      xaxs = "i", yaxs = "i")
  for (i in seq_len(nrow(panels))) {
    image <- png::readPNG(panels$path[[i]])
    plot.new()
    plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
    rasterImage(image, 0, 0, 1, 1, interpolate = TRUE)
    title(main = panels$title[[i]], line = 0.45, cex.main = 1.1,
          font.main = 2, adj = 0)
  }
  dev.off()
}

cat("Wrote publication outputs to:", output_dir, "\n")
print(claim_summary)
