#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))

root <- normalizePath(getwd())
out_dir <- file.path(root, "results", "presentation_mnist70k_runtime")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tsne_file <- file.path(root, "manuscript", "mloss", "generated",
                       "runtime_tsne_all_methods.csv")
umap_file <- file.path(root, "manuscript", "mloss", "generated",
                       "runtime_umap_all_methods.csv")
metal_file <- file.path(root, "manuscript", "mloss", "generated",
                        "local_full_pipeline_runtime_quality.csv")

stopifnot(file.exists(tsne_file), file.exists(umap_file), file.exists(metal_file))

tsne <- read.csv(tsne_file, check.names = FALSE)
umap <- read.csv(umap_file, check.names = FALSE)
metal <- read.csv(metal_file, check.names = FALSE)

select_generated <- function(x, methods) {
  x <- x[x$dataset == "MNIST" & x$method %in% methods, , drop = FALSE]
  data.frame(
    family = x$family,
    method = x$method,
    runtime = x$runtime,
    q1 = x$runtime_q1,
    q3 = x$runtime_q3,
    stringsAsFactors = FALSE
  )
}

select_metal <- function(methods) {
  x <- metal[
    metal$dataset == "MNIST" &
      metal$backend == "metal" &
      metal$timing_scope == "full_pipeline" &
      metal$method %in% methods,
    , drop = FALSE
  ]
  data.frame(
    family = x$family,
    method = x$method,
    runtime = x$total_runtime_sec_median,
    q1 = x$total_runtime_sec_q1,
    q3 = x$total_runtime_sec_q3,
    stringsAsFactors = FALSE
  )
}

plot_data <- rbind(
  select_generated(tsne, c(
    "fastEmbedR_opentsne_cuda_full",
    "fastEmbedR_opentsne_cpu_full",
    "rapids_cuml_tsne_full",
    "Rtsne_full",
    "KlugerLab_FItSNE",
    "python_opentsne_fft"
  )),
  select_metal("fastEmbedR_opentsne_metal_full"),
  select_generated(umap, c(
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_umap_cuda_binary_full",
    "fastEmbedR_umap_cpu_fuzzy_full",
    "fastEmbedR_umap_cpu_binary_full",
    "rapids_cuml_umap_full",
    "uwot_fast_sgd",
    "uwot_default",
    "python_umap_learn",
    "umap_package"
  )),
  select_metal(c(
    "fastEmbedR_umap_metal_fuzzy_full",
    "fastEmbedR_umap_metal_binary_full"
  ))
)

method_labels <- c(
  fastEmbedR_opentsne_cuda_full = "fastEmbedR openTSNE - CUDA",
  fastEmbedR_opentsne_metal_full = "fastEmbedR openTSNE - Metal",
  fastEmbedR_opentsne_cpu_full = "fastEmbedR openTSNE - CPU (4 cores)",
  rapids_cuml_tsne_full = "RAPIDS cuML t-SNE via R - CUDA",
  Rtsne_full = "Rtsne - CPU (4 cores)",
  KlugerLab_FItSNE = "FIt-SNE - CPU (4 cores)",
  python_opentsne_fft = "Python openTSNE via R - CPU (4 cores)",
  fastEmbedR_umap_cuda_fuzzy_full = "fastEmbedR fuzzy UMAP - CUDA",
  fastEmbedR_umap_cuda_binary_full = "fastEmbedR binary UMAP - CUDA",
  fastEmbedR_umap_metal_fuzzy_full = "fastEmbedR fuzzy UMAP - Metal",
  fastEmbedR_umap_metal_binary_full = "fastEmbedR binary UMAP - Metal",
  fastEmbedR_umap_cpu_fuzzy_full = "fastEmbedR fuzzy UMAP - CPU (4 cores)",
  fastEmbedR_umap_cpu_binary_full = "fastEmbedR binary UMAP - CPU (4 cores)",
  rapids_cuml_umap_full = "RAPIDS cuML UMAP via R - CUDA",
  uwot_fast_sgd = "uwot fast SGD - CPU (4 cores)",
  uwot_default = "uwot default - CPU (4 cores)",
  python_umap_learn = "Python umap-learn via R - CPU (4 cores)",
  umap_package = "umap R package - CPU (4 cores)"
)

plot_data$label <- unname(method_labels[plot_data$method])
plot_data$backend_group <- ifelse(
  grepl("cuda|rapids", plot_data$method, ignore.case = TRUE), "CUDA",
  ifelse(grepl("metal", plot_data$method, ignore.case = TRUE), "Metal",
         ifelse(grepl("^fastEmbedR", plot_data$method), "fastEmbedR CPU",
                "Reference CPU"))
)
plot_data$family <- factor(plot_data$family, levels = c("t-SNE", "UMAP"))

if (anyNA(plot_data$label) || anyNA(plot_data$runtime)) {
  stop("A selected benchmark row or display label is missing")
}

plot_data <- plot_data[order(plot_data$family, -plot_data$runtime), ]
plot_data$label <- factor(plot_data$label, levels = unique(plot_data$label))

write.csv(
  plot_data[, c("family", "method", "label", "backend_group", "runtime", "q1", "q3")],
  file.path(out_dir, "mnist70k_runtime_linear_plot_data.csv"),
  row.names = FALSE
)

palette <- c(
  "CUDA" = "#009E73",
  "Metal" = "#CC79A7",
  "fastEmbedR CPU" = "#0072B2",
  "Reference CPU" = "#6B7280"
)

make_plot <- function(data, subtitle) {
  fastest <- ave(data$runtime, data$family, FUN = min)
  ratio <- data$runtime / fastest
  data$speed_label <- ifelse(
    ratio < 10,
    paste0(format(round(ratio, 1), nsmall = 1, trim = TRUE), "x"),
    paste0(round(ratio), "x")
  )
  data$label_position <- pmax(data$runtime, data$q3, na.rm = TRUE)

  ggplot(data, aes(x = label, y = runtime, fill = backend_group)) +
    geom_col(width = 0.72, colour = "white", linewidth = 0.25) +
    geom_errorbar(aes(ymin = q1, ymax = q3), width = 0.18,
                  linewidth = 0.55, colour = "#202020") +
    geom_text(
      aes(y = label_position, label = speed_label),
      hjust = -0.22, size = 4.2, fontface = "bold", colour = "#111827"
    ) +
    coord_flip() +
    facet_wrap(~family, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = palette, name = NULL) +
    scale_y_continuous(
      limits = c(0, NA),
      expand = expansion(mult = c(0, 0.12)),
      breaks = scales::breaks_pretty(n = 6)
    ) +
    labs(
      title = "MNIST 70k: end-to-end embedding runtime",
      subtitle = subtitle,
      x = NULL,
      y = "Median elapsed time (seconds)",
      caption = "Bar-end labels show how many times faster the fastest method in each panel is (1x = fastest)."
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", size = 21),
      plot.subtitle = element_text(size = 11.5, colour = "#4B5563"),
      strip.text = element_text(face = "bold", size = 16),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 11, colour = "#111827"),
      axis.text.x = element_text(size = 11, colour = "#374151"),
      legend.position = "bottom",
      legend.text = element_text(size = 11),
      plot.caption = element_text(size = 10.5, colour = "#4B5563", hjust = 0),
      plot.margin = margin(12, 20, 12, 12)
    )
}

focus <- plot_data[plot_data$method != "umap_package", , drop = FALSE]
focus$label <- droplevels(focus$label)

p_focus <- make_plot(
  focus,
  paste0(
    "Full public-function calls; medians and IQR across 3 seeds. Linear axes begin at zero. ",
    "CPU/CUDA: Xeon Gold 6442Y and NVIDIA L40S; Metal: Apple M3."
  )
)
p_full <- make_plot(
  plot_data,
  paste0(
    "Full range including the umap R package; medians and IQR across 3 seeds. Linear axes begin at zero. ",
    "CPU/CUDA: Xeon Gold 6442Y and NVIDIA L40S; Metal: Apple M3."
  )
)

ggsave(file.path(out_dir, "mnist70k_runtime_linear_slide.png"), p_focus,
       width = 16, height = 9, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "mnist70k_runtime_linear_slide.pdf"), p_focus,
       width = 16, height = 9, device = cairo_pdf, bg = "white")
ggsave(file.path(out_dir, "mnist70k_runtime_linear_full_range.png"), p_full,
       width = 16, height = 9, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "mnist70k_runtime_linear_full_range.pdf"), p_full,
       width = 16, height = 9, device = cairo_pdf, bg = "white")

cat("Wrote presentation figures to:", out_dir, "\n")
