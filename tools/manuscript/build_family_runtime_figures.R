#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
generated_dir <- if (length(args) >= 1L) args[[1L]] else
  "manuscript/mloss/generated"
figure_dir <- if (length(args) >= 2L) args[[2L]] else
  "manuscript/mloss/figures"

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("scales", quietly = TRUE)) {
  stop("The ggplot2 and scales packages are required.")
}

standard_path <- file.path(generated_dir, "all_methods_all_datasets_standard.csv")
python_path <- file.path(generated_dir, "python_summary_median.csv")
if (!file.exists(standard_path) || !file.exists(python_path)) {
  stop("Run the manuscript result aggregation before building runtime figures.")
}

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
standard <- read.csv(
  standard_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
python <- read.csv(
  python_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

dataset_order <- c(
  "COIL20", "USPS", "FashionMNIST",
  "FlowRepository_FR-FCM-ZYRM_files", "flow18", "MNIST", "imagenet",
  "MetRef", "mass41", "TabulaMuris", "Macosko2015_retina"
)
dataset_labels <- c(
  "COIL20" = "COIL-20",
  "USPS" = "USPS",
  "FashionMNIST" = "Fashion-MNIST",
  "FlowRepository_FR-FCM-ZYRM_files" = "FlowRepository",
  "flow18" = "flow18",
  "MNIST" = "MNIST",
  "imagenet" = "ImageNet",
  "MetRef" = "MetRef",
  "mass41" = "mass41",
  "TabulaMuris" = "Tabula Muris",
  "Macosko2015_retina" = "Retina"
)

standard_spec <- data.frame(
  method = c(
    "Rtsne_full",
    "KlugerLab_FItSNE",
    "fastEmbedR_opentsne_cpu_full",
    "fastEmbedR_opentsne_cuda_full",
    "umap_package",
    "uwot_default",
    "uwot_fast_sgd",
    "fastEmbedR_umap_cpu_fuzzy_full",
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_umap_cpu_binary_full",
    "fastEmbedR_umap_cuda_binary_full"
  ),
  method_label = c(
    "Rtsne",
    "FIt-SNE",
    "fastEmbedR openTSNE",
    "fastEmbedR openTSNE",
    "umap",
    "uwot default",
    "uwot fast SGD",
    "fastEmbedR fuzzy",
    "fastEmbedR fuzzy",
    "fastEmbedR binary",
    "fastEmbedR binary"
  ),
  family = c(
    rep("t-SNE", 4L),
    rep("UMAP", 7L)
  ),
  profile = c(
    "cpu4", "cpu4", "cpu4", "cuda",
    "cpu4", "cpu4", "cpu4", "cpu4", "cuda", "cpu4", "cuda"
  ),
  backend = c(
    "cpu", "cpu", "cpu", "cuda",
    "cpu", "cpu", "cpu", "cpu", "cuda", "cpu", "cuda"
  ),
  timing_interface = "R public function",
  timing_scope = "r_public_function_total_call",
  runtime_measure = "r_public_function_total_call_sec",
  stringsAsFactors = FALSE
)

standard$key <- paste(
  standard$method, standard$profile, standard$backend, sep = "\r"
)
standard_spec$key <- paste(
  standard_spec$method, standard_spec$profile, standard_spec$backend,
  sep = "\r"
)
standard_rows <- merge(
  standard_spec,
  standard,
  by = "key",
  all.x = FALSE,
  all.y = FALSE,
  suffixes = c(".spec", "")
)
standard_rows <- standard_rows[
  standard_rows$status == "success",
  ,
  drop = FALSE
]
standard_plot <- data.frame(
  dataset = standard_rows$dataset,
  method = standard_rows$method.spec,
  method_label = standard_rows$method_label,
  family = standard_rows$family.spec,
  backend = standard_rows$backend.spec,
  profile = standard_rows$profile.spec,
  timing_interface = standard_rows$timing_interface,
  timing_scope = standard_rows$timing_scope,
  runtime_measure = standard_rows$runtime_measure,
  n_runs = suppressWarnings(as.integer(standard_rows$n_runs)),
  runtime = suppressWarnings(as.numeric(
    standard_rows$total_runtime_sec_median
  )),
  runtime_q1 = suppressWarnings(as.numeric(
    standard_rows$total_runtime_sec_q1
  )),
  runtime_q3 = suppressWarnings(as.numeric(
    standard_rows$total_runtime_sec_q3
  )),
  r_public_function_total_call_sec = suppressWarnings(as.numeric(
    standard_rows$total_runtime_sec_median
  )),
  r_mediated_total_call_sec = NA_real_,
  direct_python_fit_sec = NA_real_,
  direct_python_process_total_sec = NA_real_,
  stringsAsFactors = FALSE
)

python_spec <- data.frame(
  method = c(
    "python_opentsne_fft",
    "python_opentsne_fft_direct",
    "rapids_cuml_tsne_full",
    "rapids_cuml_tsne_full_direct",
    "python_umap_learn",
    "python_umap_learn_direct",
    "rapids_cuml_umap_full",
    "rapids_cuml_umap_full_direct"
  ),
  method_label = c(
    "Python openTSNE via R (total call)",
    "Python openTSNE direct (fit only)",
    "RAPIDS cuML t-SNE via R (total call)",
    "RAPIDS cuML t-SNE direct (fit only)",
    "Python umap-learn via R (total call)",
    "Python umap-learn direct (fit only)",
    "RAPIDS cuML UMAP via R (total call)",
    "RAPIDS cuML UMAP direct (fit only)"
  ),
  family = c(rep("t-SNE", 4L), rep("UMAP", 4L)),
  profile = c("cpu4", "cpu4", "cuda", "cuda",
              "cpu4", "cpu4", "cuda", "cuda"),
  backend = c("cpu", "cpu", "cuda", "cuda",
              "cpu", "cpu", "cuda", "cuda"),
  timing_interface = c(
    "R/reticulate", "direct Python",
    "R/reticulate", "direct Python",
    "R/reticulate", "direct Python",
    "R/reticulate", "direct Python"
  ),
  timing_scope = c(
    "r_mediated_total_call", "direct_python_fit",
    "r_mediated_total_call", "direct_python_fit",
    "r_mediated_total_call", "direct_python_fit",
    "r_mediated_total_call", "direct_python_fit"
  ),
  runtime_measure = c(
    "r_mediated_total_call_sec", "direct_python_fit_sec",
    "r_mediated_total_call_sec", "direct_python_fit_sec",
    "r_mediated_total_call_sec", "direct_python_fit_sec",
    "r_mediated_total_call_sec", "direct_python_fit_sec"
  ),
  stringsAsFactors = FALSE
)

python$key <- paste(python$method, python$profile, python$backend, sep = "\r")
python_spec$key <- paste(
  python_spec$method, python_spec$profile, python_spec$backend, sep = "\r"
)
python_rows <- merge(
  python_spec,
  python,
  by = "key",
  all.x = FALSE,
  all.y = FALSE,
  suffixes = c(".spec", "")
)
python_runtime <- ifelse(
  python_rows$timing_scope == "direct_python_fit",
  suppressWarnings(as.numeric(python_rows$direct_python_fit_sec_median)),
  suppressWarnings(as.numeric(python_rows$r_mediated_total_call_sec_median))
)
python_runtime_q1 <- ifelse(
  python_rows$timing_scope == "direct_python_fit",
  suppressWarnings(as.numeric(python_rows$direct_python_fit_sec_q1)),
  suppressWarnings(as.numeric(python_rows$r_mediated_total_call_sec_q1))
)
python_runtime_q3 <- ifelse(
  python_rows$timing_scope == "direct_python_fit",
  suppressWarnings(as.numeric(python_rows$direct_python_fit_sec_q3)),
  suppressWarnings(as.numeric(python_rows$r_mediated_total_call_sec_q3))
)
python_plot <- data.frame(
  dataset = python_rows$dataset,
  method = python_rows$method.spec,
  method_label = python_rows$method_label,
  family = python_rows$family,
  backend = python_rows$backend.spec,
  profile = python_rows$profile.spec,
  timing_interface = python_rows$timing_interface,
  timing_scope = python_rows$timing_scope.spec,
  runtime_measure = python_rows$runtime_measure.spec,
  n_runs = suppressWarnings(as.integer(python_rows$n_runs)),
  runtime = python_runtime,
  runtime_q1 = python_runtime_q1,
  runtime_q3 = python_runtime_q3,
  r_public_function_total_call_sec = NA_real_,
  r_mediated_total_call_sec = suppressWarnings(as.numeric(
    python_rows$r_mediated_total_call_sec_median
  )),
  direct_python_fit_sec = suppressWarnings(as.numeric(
    python_rows$direct_python_fit_sec_median
  )),
  direct_python_process_total_sec = suppressWarnings(as.numeric(
    python_rows$direct_python_process_total_sec_median
  )),
  stringsAsFactors = FALSE
)

plot_data <- rbind(standard_plot, python_plot)
plot_data <- plot_data[
  is.finite(plot_data$runtime) & plot_data$runtime >= 0,
  ,
  drop = FALSE
]
plot_data$dataset_label <- factor(
  unname(dataset_labels[plot_data$dataset]),
  levels = unname(dataset_labels[dataset_order])
)
plot_data$display_label <- paste0(
  plot_data$method_label,
  " [",
  ifelse(plot_data$backend == "cuda", "CUDA", "CPU"),
  "]"
)

write.csv(
  plot_data[plot_data$family == "t-SNE", , drop = FALSE],
  file.path(generated_dir, "runtime_tsne_all_methods.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  plot_data[plot_data$family == "UMAP", , drop = FALSE],
  file.path(generated_dir, "runtime_umap_all_methods.csv"),
  row.names = FALSE,
  na = ""
)

tsne_levels <- c(
  "Rtsne [CPU]", "FIt-SNE [CPU]",
  "fastEmbedR openTSNE [CPU]", "fastEmbedR openTSNE [CUDA]",
  "Python openTSNE via R (total call) [CPU]",
  "Python openTSNE direct (fit only) [CPU]",
  "RAPIDS cuML t-SNE via R (total call) [CUDA]",
  "RAPIDS cuML t-SNE direct (fit only) [CUDA]"
)
tsne_colors <- c(
  "Rtsne [CPU]" = "#4D4D4D",
  "FIt-SNE [CPU]" = "#9E9E9E",
  "fastEmbedR openTSNE [CPU]" = "#0072B2",
  "fastEmbedR openTSNE [CUDA]" = "#56B4E9",
  "Python openTSNE via R (total call) [CPU]" = "#7B3294",
  "Python openTSNE direct (fit only) [CPU]" = "#C2A5CF",
  "RAPIDS cuML t-SNE via R (total call) [CUDA]" = "#D55E00",
  "RAPIDS cuML t-SNE direct (fit only) [CUDA]" = "#E69F00"
)
umap_levels <- c(
  "umap [CPU]", "uwot default [CPU]", "uwot fast SGD [CPU]",
  "fastEmbedR fuzzy [CPU]", "fastEmbedR fuzzy [CUDA]",
  "fastEmbedR binary [CPU]", "fastEmbedR binary [CUDA]",
  "Python umap-learn via R (total call) [CPU]",
  "Python umap-learn direct (fit only) [CPU]",
  "RAPIDS cuML UMAP via R (total call) [CUDA]",
  "RAPIDS cuML UMAP direct (fit only) [CUDA]"
)
umap_colors <- c(
  "umap [CPU]" = "#777777",
  "uwot default [CPU]" = "#A6A6A6",
  "uwot fast SGD [CPU]" = "#2F2F2F",
  "fastEmbedR fuzzy [CPU]" = "#0072B2",
  "fastEmbedR fuzzy [CUDA]" = "#56B4E9",
  "fastEmbedR binary [CPU]" = "#004B6B",
  "fastEmbedR binary [CUDA]" = "#8FD3EA",
  "Python umap-learn via R (total call) [CPU]" = "#7B3294",
  "Python umap-learn direct (fit only) [CPU]" = "#C2A5CF",
  "RAPIDS cuML UMAP via R (total call) [CUDA]" = "#D55E00",
  "RAPIDS cuML UMAP direct (fit only) [CUDA]" = "#E69F00"
)

publication_theme <- ggplot2::theme_minimal(base_size = 10.5) +
  ggplot2::theme(
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    axis.text.x = ggplot2::element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(size = 8.2),
    strip.text = ggplot2::element_text(face = "bold", size = 10),
    strip.background = ggplot2::element_rect(
      fill = "#F1F1F1", color = NA
    ),
    plot.title = ggplot2::element_text(face = "bold"),
    plot.subtitle = ggplot2::element_text(size = 8.8),
    plot.margin = ggplot2::margin(5, 8, 5, 8)
  )

build_plot <- function(data, levels, colors, title, legend_rows = 3L) {
  data$display_label <- factor(data$display_label, levels = levels)
  ggplot2::ggplot(
    data,
    ggplot2::aes(dataset_label, runtime, fill = display_label)
  ) +
    ggplot2::geom_col(
      position = ggplot2::position_dodge(width = 0.84),
      width = 0.76
    ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = runtime_q1, ymax = runtime_q3),
      position = ggplot2::position_dodge(width = 0.84),
      width = 0.16,
      linewidth = 0.3
    ) +
    ggplot2::scale_y_continuous(
      trans = scales::pseudo_log_trans(base = 10, sigma = 0.05),
      breaks = c(0, 0.1, 1, 10, 100, 1000, 10000),
      labels = scales::label_number()
    ) +
    ggplot2::scale_fill_manual(values = colors, drop = TRUE) +
    ggplot2::guides(
      fill = ggplot2::guide_legend(nrow = legend_rows, byrow = TRUE)
    ) +
    ggplot2::labs(
      x = NULL,
      y = "Median reported time (seconds, pseudo-log scale)",
      title = title,
      subtitle = paste0(
        "CPU and CUDA share one panel; labels identify the backend.\n",
        "R-mediated Python rows report total-call time; direct Python rows report fit time only.\n",
        "The pseudo-log axis retains a true zero baseline; medians and IQRs use three seeds."
      )
    ) +
    publication_theme
}

tsne_plot <- build_plot(
  plot_data[plot_data$family == "t-SNE", , drop = FALSE],
  tsne_levels,
  tsne_colors,
  "t-SNE runtime by timing boundary",
  legend_rows = 3L
)
umap_plot <- build_plot(
  plot_data[plot_data$family == "UMAP", , drop = FALSE],
  umap_levels,
  umap_colors,
  "UMAP runtime by timing boundary",
  legend_rows = 4L
)

save_plot <- function(plot, stem, width, height) {
  ggplot2::ggsave(
    file.path(figure_dir, paste0(stem, ".png")),
    plot,
    width = width,
    height = height,
    dpi = 360,
    bg = "white"
  )
  ggplot2::ggsave(
    file.path(figure_dir, paste0(stem, ".pdf")),
    plot,
    width = width,
    height = height,
    device = grDevices::cairo_pdf
  )
}

save_plot(tsne_plot, "runtime_tsne_all_methods", 9.8, 7.4)
save_plot(umap_plot, "runtime_umap_all_methods", 10.2, 8.0)
