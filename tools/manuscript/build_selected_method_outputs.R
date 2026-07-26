#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
results_root <- if (length(args) >= 1L) args[[1L]] else
  "/Users/stefano/Documents/fastEmbedR-results"
output_dir <- if (length(args) >= 2L) args[[2L]] else
  "/Users/stefano/Documents/umap/manuscript/mloss/generated"
figure_dir <- if (length(args) >= 3L) args[[3L]] else
  file.path(dirname(output_dir), "figures")
embedding_dir <- file.path(figure_dir, "embedding_comparisons")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(embedding_dir, recursive = TRUE, showWarnings = FALSE)

required_packages <- c("ggplot2", "png")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Missing required packages: ", paste(missing_packages, collapse = ", "))
}

clean_number <- function(x) suppressWarnings(as.numeric(x))

read_csv_safe <- function(path) {
  tryCatch(
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) NULL
  )
}

relative_parts <- function(path) {
  root <- normalizePath(results_root, mustWork = TRUE)
  full <- normalizePath(path, mustWork = TRUE)
  relative <- substring(full, nchar(root) + 2L)
  strsplit(relative, "/", fixed = TRUE)[[1L]]
}

summary_files <- list.files(
  results_root,
  pattern = "^benchmark_summary_median_variability\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)
summary_files <- summary_files[
  grepl("/standard/", summary_files, fixed = TRUE) &
    !grepl("/kodama/", summary_files, fixed = TRUE)
]
if (!length(summary_files)) {
  stop("No standard benchmark summaries found under ", results_root)
}

summary_rows <- lapply(summary_files, function(path) {
  x <- read_csv_safe(path)
  if (is.null(x) || !nrow(x)) return(NULL)
  parts <- relative_parts(path)
  x$suite <- parts[[2L]]
  x$profile <- parts[[3L]]
  x$run_dir <- dirname(path)
  x$source_file <- path
  x
})
summary_rows <- Filter(Negate(is.null), summary_rows)
all_standard <- do.call(rbind, summary_rows)

write.csv(
  all_standard,
  file.path(output_dir, "all_methods_all_datasets_standard.csv"),
  row.names = FALSE,
  na = ""
)

method_spec <- data.frame(
  method = c(
    "Rtsne_full",
    "KlugerLab_FItSNE",
    "fastEmbedR_opentsne_cpu_full",
    "fastEmbedR_opentsne_cuda_full",
    "uwot_fast_sgd",
    "fastEmbedR_umap_cpu_fuzzy_full",
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_umap_cpu_binary_full",
    "fastEmbedR_umap_cuda_binary_full"
  ),
  display = c(
    "Rtsne::Rtsne",
    "FIt-SNE",
    "fastEmbedR::opentsne",
    "fastEmbedR::opentsne",
    "uwot::umap (fast_sgd = TRUE)",
    "fastEmbedR::umap (fuzzy)",
    "fastEmbedR::umap (fuzzy)",
    "fastEmbedR::umap (binary)",
    "fastEmbedR::umap (binary)"
  ),
  implementation = c(
    "Rtsne CPU (4 threads)",
    "FIt-SNE CPU (4 threads)",
    "fastEmbedR openTSNE CPU (4 threads)",
    "fastEmbedR openTSNE CUDA",
    "uwot UMAP fast SGD CPU (4 threads)",
    "fastEmbedR fuzzy UMAP CPU (4 threads)",
    "fastEmbedR fuzzy UMAP CUDA",
    "fastEmbedR binary UMAP CPU (4 threads)",
    "fastEmbedR binary UMAP CUDA"
  ),
  family = c("t-SNE", "t-SNE", "t-SNE", "t-SNE",
             "UMAP", "UMAP", "UMAP", "UMAP", "UMAP"),
  profile = c("cpu4", "cpu4", "cpu4", "cuda",
              "cpu4", "cpu4", "cuda", "cpu4", "cuda"),
  backend = c("cpu", "cpu", "cpu", "cuda",
              "cpu", "cpu", "cuda", "cpu", "cuda"),
  stringsAsFactors = FALSE
)

dataset_order <- c(
  "COIL20", "USPS", "FashionMNIST", "FlowRepository_FR-FCM-ZYRM_files",
  "flow18", "MNIST", "imagenet", "MetRef", "mass41", "TabulaMuris",
  "Macosko2015_retina"
)
dataset_order <- dataset_order[dataset_order %in% unique(all_standard$dataset)]
extra_datasets <- setdiff(sort(unique(all_standard$dataset)), dataset_order)
dataset_order <- c(dataset_order, extra_datasets)

expected <- merge(
  data.frame(dataset = dataset_order, stringsAsFactors = FALSE),
  method_spec,
  by = NULL
)

selected_columns <- c(
  "dataset", "method", "family", "backend", "profile", "status",
  "n_runs", "n_success", "n", "p", "k", "perplexity", "input_type",
  "requested_threads", "effective_threads",
  "total_runtime_sec_median", "total_runtime_sec_q1",
  "total_runtime_sec_q3", "total_runtime_sec_iqr",
  "peak_ram_gb_median", "peak_gpu_delta_mb_median",
  "trustworthiness_median", "knn_preservation_15_median",
  "knn_preservation_30_median", "knn_preservation_50_median",
  "silhouette_median", "label_knn_accuracy_median", "tsne_kl_median",
  "run_dir", "source_file"
)
available_columns <- intersect(selected_columns, names(all_standard))
observed <- all_standard[
  all_standard$method %in% method_spec$method,
  available_columns,
  drop = FALSE
]

key <- function(x) paste(x$dataset, x$method, x$profile, x$backend, sep = "\r")
observed <- observed[!duplicated(key(observed)), , drop = FALSE]
expected$key <- key(expected)
observed$key <- key(observed)
selected <- merge(
  expected,
  observed,
  by = "key",
  all.x = TRUE,
  suffixes = c("", ".observed"),
  sort = FALSE
)

for (name in c("dataset", "method", "family", "profile", "backend")) {
  observed_name <- paste0(name, ".observed")
  if (observed_name %in% names(selected)) {
    selected[[observed_name]] <- NULL
  }
}
selected$status[is.na(selected$status) | !nzchar(selected$status)] <- "not_run"
selected$dataset_index <- match(selected$dataset, dataset_order)
selected$method_index <- match(selected$method, method_spec$method)
selected <- selected[
  order(selected$dataset_index, selected$method_index),
  ,
  drop = FALSE
]
selected$dataset_index <- NULL
selected$method_index <- NULL

write.csv(
  selected,
  file.path(output_dir, "selected_methods_all_datasets.csv"),
  row.names = FALSE,
  na = ""
)

success <- selected[selected$status == "success", , drop = FALSE]
success$runtime <- clean_number(success$total_runtime_sec_median)
success$runtime_q1 <- clean_number(success$total_runtime_sec_q1)
success$runtime_q3 <- clean_number(success$total_runtime_sec_q3)
success$trust <- clean_number(success$trustworthiness_median)
success$preserve30 <- clean_number(success$knn_preservation_30_median)
success$label_accuracy <- clean_number(success$label_knn_accuracy_median)
success$peak_ram_gb <- clean_number(success$peak_ram_gb_median)

cpu4 <- success[success$profile == "cpu4", , drop = FALSE]
write.csv(
  cpu4,
  file.path(output_dir, "selected_methods_cpu4_success.csv"),
  row.names = FALSE,
  na = ""
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
missing_labels <- setdiff(dataset_order, names(dataset_labels))
dataset_labels[missing_labels] <- missing_labels

method_colors <- c(
  "Rtsne CPU (4 threads)" = "#D55E00",
  "FIt-SNE CPU (4 threads)" = "#CC79A7",
  "fastEmbedR openTSNE CPU (4 threads)" = "#0072B2",
  "fastEmbedR openTSNE CUDA" = "#009E73",
  "uwot UMAP fast SGD CPU (4 threads)" = "#D55E00",
  "fastEmbedR fuzzy UMAP CPU (4 threads)" = "#0072B2",
  "fastEmbedR fuzzy UMAP CUDA" = "#009E73",
  "fastEmbedR binary UMAP CPU (4 threads)" = "#56B4E9",
  "fastEmbedR binary UMAP CUDA" = "#E69F00"
)

publication_theme <- ggplot2::theme_minimal(base_size = 11) +
  ggplot2::theme(
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    axis.text.x = ggplot2::element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(size = 8.5),
    strip.text = ggplot2::element_text(face = "bold"),
    plot.title = ggplot2::element_text(face = "bold"),
    plot.subtitle = ggplot2::element_text(size = 9),
    plot.margin = ggplot2::margin(5, 8, 5, 8)
  )

runtime_plot <- function(data, title, subtitle) {
  data$dataset_label <- factor(
    unname(dataset_labels[data$dataset]),
    levels = unname(dataset_labels[dataset_order])
  )
  data$implementation <- factor(
    data$implementation,
    levels = method_spec$implementation
  )
  ggplot2::ggplot(
    data,
    ggplot2::aes(dataset_label, runtime, fill = implementation)
  ) +
    ggplot2::geom_col(
      position = ggplot2::position_dodge(width = 0.82),
      width = 0.72
    ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = runtime_q1, ymax = runtime_q3),
      position = ggplot2::position_dodge(width = 0.82),
      width = 0.18,
      linewidth = 0.35
    ) +
    ggplot2::facet_grid(family ~ ., scales = "free_y") +
    ggplot2::scale_y_continuous(
      trans = scales::pseudo_log_trans(base = 10, sigma = 0.05),
      breaks = c(0, 0.1, 1, 10, 100, 1000, 10000),
      labels = scales::label_number()
    ) +
    ggplot2::scale_fill_manual(values = method_colors, drop = TRUE) +
    ggplot2::guides(
      fill = ggplot2::guide_legend(nrow = 3L, byrow = TRUE)
    ) +
    ggplot2::labs(
      x = NULL,
      y = "Median total runtime (seconds, pseudo-log scale)",
      title = title,
      subtitle = subtitle
    ) +
    publication_theme
}

cpu_plot <- runtime_plot(
  cpu4,
  "Total runtime across benchmark datasets",
  "Four requested CPU threads; bars show medians and IQRs over seeds 4, 17, and 42"
)
ggplot2::ggsave(
  file.path(figure_dir, "selected_speed_cpu4_all_datasets.png"),
  cpu_plot,
  width = 10.5,
  height = 7.4,
  dpi = 320,
  bg = "white"
)
ggplot2::ggsave(
  file.path(figure_dir, "selected_speed_cpu4_all_datasets.pdf"),
  cpu_plot,
  width = 10.5,
  height = 7.4,
  device = grDevices::cairo_pdf
)

all_backend_plot <- runtime_plot(
  success,
  "Total runtime across CPU and CUDA implementations",
  "Full public-function time; unavailable and failed method-dataset pairs are omitted from bars but retained in the result table"
)
ggplot2::ggsave(
  file.path(figure_dir, "selected_speed_cpu4_cuda_all_datasets.png"),
  all_backend_plot,
  width = 10.8,
  height = 7.8,
  dpi = 320,
  bg = "white"
)
ggplot2::ggsave(
  file.path(figure_dir, "selected_speed_cpu4_cuda_all_datasets.pdf"),
  all_backend_plot,
  width = 10.8,
  height = 7.8,
  device = grDevices::cairo_pdf
)

paired_pareto_rows <- function(reference_method, comparison_methods,
                               quality_column, panel_label) {
  reference <- success[
    success$method == reference_method,
    c("dataset", "runtime", quality_column),
    drop = FALSE
  ]
  names(reference)[names(reference) == "runtime"] <- "reference_runtime"
  names(reference)[names(reference) == quality_column] <- "reference_quality"

  comparison <- success[
    success$method %in% comparison_methods,
    c("dataset", "method", "implementation", "backend", "runtime",
      quality_column),
    drop = FALSE
  ]
  names(comparison)[names(comparison) == quality_column] <- "quality"
  paired <- merge(comparison, reference, by = "dataset", all = FALSE)
  paired <- paired[
    is.finite(paired$runtime) & paired$runtime > 0 &
      is.finite(paired$reference_runtime) & paired$reference_runtime > 0 &
      is.finite(paired$quality) & is.finite(paired$reference_quality),
    ,
    drop = FALSE
  ]
  paired$speedup <- paired$reference_runtime / paired$runtime
  paired$quality_delta <- paired$quality - paired$reference_quality
  paired$panel <- panel_label
  paired
}

pareto_tsne <- paired_pareto_rows(
  "Rtsne_full",
  c(
    "KlugerLab_FItSNE",
    "fastEmbedR_opentsne_cpu_full",
    "fastEmbedR_opentsne_cuda_full"
  ),
  "trust",
  "t-SNE: trustworthiness difference vs Rtsne"
)
pareto_umap <- paired_pareto_rows(
  "uwot_fast_sgd",
  c(
    "fastEmbedR_umap_cpu_fuzzy_full",
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_umap_cpu_binary_full",
    "fastEmbedR_umap_cuda_binary_full"
  ),
  "preserve30",
  "UMAP: Preserve@30 difference vs uwot fast SGD"
)
pareto <- rbind(pareto_tsne, pareto_umap)
pareto$dataset_label <- unname(dataset_labels[pareto$dataset])
pareto$implementation <- factor(
  pareto$implementation,
  levels = method_spec$implementation
)
pareto_labels <- c(
  "FIt-SNE CPU (4 threads)" = "FIt-SNE CPU4",
  "fastEmbedR openTSNE CPU (4 threads)" = "openTSNE CPU4",
  "fastEmbedR openTSNE CUDA" = "openTSNE CUDA",
  "fastEmbedR fuzzy UMAP CPU (4 threads)" = "fuzzy UMAP CPU4",
  "fastEmbedR fuzzy UMAP CUDA" = "fuzzy UMAP CUDA",
  "fastEmbedR binary UMAP CPU (4 threads)" = "binary UMAP CPU4",
  "fastEmbedR binary UMAP CUDA" = "binary UMAP CUDA"
)
pareto$pareto_method <- factor(
  unname(pareto_labels[as.character(pareto$implementation)]),
  levels = unname(pareto_labels)
)
pareto$panel <- factor(
  pareto$panel,
  levels = c(
    "t-SNE: trustworthiness difference vs Rtsne",
    "UMAP: Preserve@30 difference vs uwot fast SGD"
  )
)

write.csv(
  pareto,
  file.path(output_dir, "selected_runtime_quality_pareto.csv"),
  row.names = FALSE,
  na = ""
)

pareto_groups <- split(
  pareto,
  interaction(pareto$panel, pareto$pareto_method, drop = TRUE)
)
pareto_medians <- do.call(rbind, lapply(pareto_groups, function(x) {
  x[1L, c("panel", "pareto_method", "backend"), drop = FALSE] |>
    transform(
      speedup = median(x$speedup, na.rm = TRUE),
      quality_delta = median(x$quality_delta, na.rm = TRUE),
      n_datasets = nrow(x)
    )
}))
rownames(pareto_medians) <- NULL

pareto_plot <- ggplot2::ggplot(
  pareto,
  ggplot2::aes(speedup, quality_delta, color = pareto_method)
) +
  ggplot2::geom_hline(
    yintercept = 0,
    linewidth = 0.45,
    color = "grey40"
  ) +
  ggplot2::geom_vline(
    xintercept = 1,
    linewidth = 0.45,
    color = "grey40"
  ) +
  ggplot2::geom_point(size = 2.15, alpha = 0.58) +
  ggplot2::geom_point(
    data = pareto_medians,
    shape = 18,
    size = 4.4,
    alpha = 1
  ) +
  ggplot2::facet_wrap(~panel, nrow = 2L, scales = "free_y") +
  ggplot2::scale_x_log10(
    breaks = c(0.25, 0.5, 1, 2, 10, 50, 100),
    labels = scales::label_number()
  ) +
  ggplot2::scale_color_manual(
    values = setNames(
      unname(method_colors[names(pareto_labels)]),
      unname(pareto_labels)
    ),
    drop = TRUE
  ) +
  ggplot2::guides(
    color = ggplot2::guide_legend(nrow = 2L, byrow = TRUE)
  ) +
  ggplot2::labs(
    x = "Speedup over reference (reference time / method time; log scale)",
    y = "Quality difference from reference",
    title = "Runtime-quality trade-offs on matched datasets",
    subtitle = paste(
      "Small points are individual datasets; diamonds are medians.",
      "Right of 1 is faster and above 0 is higher quality."
    )
  ) +
  publication_theme +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 0, hjust = 0.5),
    panel.grid.major.x = ggplot2::element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    panel.grid.major.y = ggplot2::element_line(
      color = "grey90",
      linewidth = 0.3
    )
  )

ggplot2::ggsave(
  file.path(figure_dir, "selected_runtime_quality_pareto.png"),
  pareto_plot,
  width = 8.5,
  height = 7.2,
  dpi = 360,
  bg = "white"
)
ggplot2::ggsave(
  file.path(figure_dir, "selected_runtime_quality_pareto.pdf"),
  pareto_plot,
  width = 8.5,
  height = 7.2,
  device = grDevices::cairo_pdf
)

image_methods <- data.frame(
  method = c(
    "Rtsne_full",
    "KlugerLab_FItSNE",
    "fastEmbedR_opentsne_cpu_full",
    "uwot_fast_sgd",
    "fastEmbedR_umap_cpu_fuzzy_full",
    "fastEmbedR_umap_cpu_binary_full"
  ),
  fallback_method = c(
    NA_character_,
    NA_character_,
    "fastEmbedR_opentsne_cuda_full",
    NA_character_,
    "fastEmbedR_umap_cuda_fuzzy_full",
    "fastEmbedR_umap_cuda_binary_full"
  ),
  display = c(
    "Rtsne::Rtsne",
    "FIt-SNE",
    "fastEmbedR::opentsne",
    "uwot::umap (fast_sgd = TRUE)",
    "fastEmbedR::umap (fuzzy)",
    "fastEmbedR::umap (binary)"
  ),
  stringsAsFactors = FALSE
)

run_files <- list.files(
  results_root,
  pattern = "^benchmark_runs\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)
run_files <- run_files[
  grepl("/standard/(cpu4|cuda)/", run_files) &
    !grepl("/kodama/", run_files, fixed = TRUE)
]
run_rows <- lapply(run_files, function(path) {
  x <- read_csv_safe(path)
  if (is.null(x) || !nrow(x)) return(NULL)
  x$run_dir <- dirname(path)
  x$profile <- relative_parts(path)[[3L]]
  x
})
run_rows <- Filter(Negate(is.null), run_rows)
all_runs <- do.call(rbind, run_rows)
seed4 <- all_runs[
  all_runs$seed == 4L &
    all_runs$method %in% c(image_methods$method, image_methods$fallback_method),
  ,
  drop = FALSE
]

local_plot_path <- function(run_dir, recorded_path) {
  if (is.na(recorded_path) || !nzchar(recorded_path)) return(NA_character_)
  candidate <- file.path(run_dir, "plots", basename(recorded_path))
  if (file.exists(candidate)) candidate else NA_character_
}
seed4$local_plot <- mapply(
  local_plot_path,
  seed4$run_dir,
  seed4$plot_file,
  USE.NAMES = FALSE
)

draw_placeholder <- function(label, status) {
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
  rect(0.02, 0.02, 0.98, 0.98, border = "#BDBDBD", col = "#F7F7F7")
  text(0.5, 0.54, label, cex = 1.05, font = 2)
  text(0.5, 0.44, status, cex = 0.9, col = "#666666")
}

source_manifest <- list()
for (dataset in dataset_order) {
  dataset_rows <- vector("list", nrow(image_methods))
  for (i in seq_len(nrow(image_methods))) {
    method <- image_methods$method[[i]]
    hit <- seed4[
      seed4$dataset == dataset & seed4$method == method &
        seed4$profile == "cpu4",
      ,
      drop = FALSE
    ]
    if (nrow(hit) > 1L) hit <- hit[1L, , drop = FALSE]
    fallback_used <- FALSE
    if ((!nrow(hit) || hit$status[[1L]] != "success" ||
         is.na(hit$local_plot[[1L]])) &&
        !is.na(image_methods$fallback_method[[i]])) {
      fallback <- seed4[
        seed4$dataset == dataset &
          seed4$method == image_methods$fallback_method[[i]] &
          seed4$profile == "cuda",
        ,
        drop = FALSE
      ]
      if (nrow(fallback) > 1L) fallback <- fallback[1L, , drop = FALSE]
      if (nrow(fallback) && fallback$status[[1L]] == "success" &&
          !is.na(fallback$local_plot[[1L]])) {
        hit <- fallback
        fallback_used <- TRUE
      }
    }
    status <- if (nrow(hit)) hit$status[[1L]] else "not_run"
    path <- if (nrow(hit)) hit$local_plot[[1L]] else NA_character_
    profile <- if (nrow(hit)) hit$profile[[1L]] else "cpu4"
    source_method <- if (nrow(hit)) hit$method[[1L]] else method
    dataset_rows[[i]] <- data.frame(
      dataset = dataset,
      requested_method = method,
      source_method = source_method,
      display = image_methods$display[[i]],
      seed = 4L,
      profile = profile,
      fallback_used = fallback_used,
      status = status,
      image_path = path,
      stringsAsFactors = FALSE
    )
  }
  manifest <- do.call(rbind, dataset_rows)
  source_manifest[[dataset]] <- manifest

  output_path <- file.path(
    embedding_dir,
    paste0(gsub("[^A-Za-z0-9]+", "_", dataset), "_six_methods.png")
  )
  grDevices::png(
    output_path,
    width = 2700,
    height = 1875,
    res = 300,
    bg = "white"
  )
  old_par <- par(
    mfrow = c(2, 3),
    mar = c(0.15, 0.15, 2.0, 0.15),
    oma = c(0, 0, 2.6, 0),
    xaxs = "i",
    yaxs = "i"
  )
  for (i in seq_len(nrow(manifest))) {
    image_path <- manifest$image_path[[i]]
    if (!is.na(image_path) && file.exists(image_path)) {
      image <- png::readPNG(image_path)
      plot.new()
      plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
      rasterImage(image, 0, 0, 1, 1, interpolate = TRUE)
    } else {
      draw_placeholder(manifest$display[[i]], manifest$status[[i]])
    }
    title_text <- manifest$display[[i]]
    if (isTRUE(manifest$fallback_used[[i]])) {
      title_text <- paste0(title_text, " [CUDA]")
    }
    title(
      main = title_text,
      line = 0.35,
      cex.main = 1.0,
      font.main = 2
    )
  }
  dataset_context <- if (any(manifest$fallback_used)) {
    "seed 4; CPU profile unless a CUDA fallback is marked"
  } else {
    "seed 4, CPU profile"
  }
  mtext(
    paste0(unname(dataset_labels[dataset]), " (", dataset_context, ")"),
    side = 3,
    outer = TRUE,
    line = 0.5,
    cex = 1.25,
    font = 2
  )
  par(old_par)
  dev.off()
}

source_manifest <- do.call(rbind, source_manifest)
write.csv(
  source_manifest,
  file.path(output_dir, "selected_embedding_image_manifest.csv"),
  row.names = FALSE,
  na = ""
)

latex_escape <- function(x) {
  slash <- intToUtf8(92L)
  x <- gsub("_", paste0(slash, "_"), x, fixed = TRUE)
  x <- gsub("&", paste0(slash, "&"), x, fixed = TRUE)
  x <- gsub("%", paste0(slash, "%"), x, fixed = TRUE)
  x
}

format_value <- function(x, digits = 3L) {
  x <- clean_number(x)
  ifelse(is.na(x), "--", formatC(x, format = "f", digits = digits))
}

table_rows <- selected
table_dataset_labels <- dataset_labels
table_dataset_labels["FlowRepository_FR-FCM-ZYRM_files"] <- "FlowRepo."
table_rows$dataset_label <- unname(table_dataset_labels[table_rows$dataset])
table_rows$runtime_text <- ifelse(
  table_rows$status == "success",
  format_value(table_rows$total_runtime_sec_median, 3L),
  paste0("\\textit{", latex_escape(table_rows$status), "}")
)
table_rows$trust_text <- ifelse(
  table_rows$status == "success",
  format_value(table_rows$trustworthiness_median, 3L),
  "--"
)
table_rows$preserve_text <- ifelse(
  table_rows$status == "success",
  format_value(table_rows$knn_preservation_30_median, 3L),
  "--"
)
table_rows$accuracy_text <- ifelse(
  table_rows$status == "success",
  format_value(table_rows$label_knn_accuracy_median, 3L),
  "--"
)

tex <- c(
  "{\\scriptsize",
  "\\setlength{\\tabcolsep}{2.2pt}",
  "\\begin{longtable}{p{0.11\\textwidth}p{0.27\\textwidth}llrrrr}",
  "\\caption{Complete selected-method benchmark results. Runtime is the median total public-function time over three seeds; unavailable and failed rows are retained explicitly. FlowRepository and ImageNet contain unmatched fastEmbedR rows that are descriptive only and excluded from comparative summaries.}\\label{tab:allselected}\\\\",
  "\\toprule",
  "Dataset & Method & Backend & Profile & Seconds & Trust. & Preserve@30 & Label acc.\\\\",
  "\\midrule",
  "\\endfirsthead",
  "\\toprule",
  "Dataset & Method & Backend & Profile & Seconds & Trust. & Preserve@30 & Label acc.\\\\",
  "\\midrule",
  "\\endhead"
)
previous_dataset <- NA_character_
for (i in seq_len(nrow(table_rows))) {
  row <- table_rows[i, ]
  if (!is.na(previous_dataset) && row$dataset != previous_dataset) {
    tex <- c(tex, "\\addlinespace[2pt]")
  }
  dataset_cell <- if (identical(row$dataset, previous_dataset)) {
    ""
  } else {
    latex_escape(row$dataset_label)
  }
  tex <- c(tex, sprintf(
    "%s & %s & %s & %s & %s & %s & %s & %s\\\\",
    dataset_cell,
    latex_escape(row$display),
    toupper(latex_escape(row$backend)),
    latex_escape(row$profile),
    row$runtime_text,
    row$trust_text,
    row$preserve_text,
    row$accuracy_text
  ))
  previous_dataset <- row$dataset
}
tex <- c(tex, "\\bottomrule", "\\end{longtable}", "}")
writeLines(tex, file.path(output_dir, "selected_methods_all_datasets.tex"))

embedding_tex <- c(
  "\\section{Selected Embedding Visualizations}",
  "",
  "Each figure uses seed 4 and the four-thread CPU benchmark profile. When the CPU run is absent but a successful CUDA fastEmbedR run exists, that panel is retained and explicitly marked CUDA. Labels determine point color only and were not used during fitting. A labeled placeholder is retained when the archived run was unavailable, failed, or timed out. FlowRepository and ImageNet contain unmatched fastEmbedR results; those panels demonstrate feasibility only and are excluded from comparative claims."
)
for (dataset in dataset_order) {
  file_stem <- paste0(gsub("[^A-Za-z0-9]+", "_", dataset), "_six_methods.png")
  label_stem <- tolower(gsub("[^A-Za-z0-9]+", "", dataset))
  comparison_note <- if (dataset %in% c(
    "FlowRepository_FR-FCM-ZYRM_files", "imagenet"
  )) {
    " This dataset has no successful matched reference row; displayed fastEmbedR results are descriptive only."
  } else {
    ""
  }
  embedding_tex <- c(
    embedding_tex,
    "",
    "\\begin{figure}[p]",
    "\\centering",
    paste0("\\includegraphics[width=\\textwidth]{figures/embedding_comparisons/", file_stem, "}"),
    paste0(
      "\\caption{", latex_escape(unname(dataset_labels[dataset])),
      " embeddings for the six selected methods (seed 4; CPU profile unless explicitly marked CUDA). ",
      "The upper row shows Rtsne, FIt-SNE, and \\pkg{} openTSNE; ",
      "the lower row shows uwot UMAP, \\pkg{} fuzzy UMAP, and \\pkg{} binary UMAP.",
      comparison_note,
      "}"
    ),
    paste0("\\label{fig:embedding", label_stem, "}"),
    "\\end{figure}",
    "\\clearpage"
  )
}
writeLines(
  embedding_tex,
  file.path(output_dir, "selected_embedding_figures.tex")
)

landmark_files <- list.files(
  results_root,
  pattern = "^benchmark_summary_median_variability\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)
landmark_files <- landmark_files[
  grepl("/landmark/", landmark_files, fixed = TRUE) &
    !grepl("/kodama/", landmark_files, fixed = TRUE)
]
landmark_rows <- lapply(landmark_files, function(path) {
  x <- read_csv_safe(path)
  if (is.null(x) || !nrow(x)) return(NULL)
  parts <- relative_parts(path)
  x$profile <- parts[[3L]]
  x
})
landmark_rows <- Filter(Negate(is.null), landmark_rows)
if (length(landmark_rows)) {
  landmark_all <- do.call(rbind, landmark_rows)
  landmark_pairs <- list(
    list(
      family = "t-SNE",
      profile = "cpu4",
      full = "fastEmbedR_opentsne_cpu_full",
      landmark = "fastEmbedR_opentsne_cpu_landmark"
    ),
    list(
      family = "t-SNE",
      profile = "cuda",
      full = "fastEmbedR_opentsne_cuda_full",
      landmark = "fastEmbedR_opentsne_cuda_landmark"
    ),
    list(
      family = "UMAP",
      profile = "cpu4",
      full = "fastEmbedR_umap_cpu_binary_full",
      landmark = "fastEmbedR_umap_cpu_binary_landmark"
    ),
    list(
      family = "UMAP",
      profile = "cuda",
      full = "fastEmbedR_umap_cuda_binary_full",
      landmark = "fastEmbedR_umap_cuda_binary_landmark"
    )
  )
  pair_rows <- lapply(landmark_pairs, function(spec) {
    full_rows <- landmark_all[
      landmark_all$profile == spec$profile &
        landmark_all$method == spec$full &
        landmark_all$status == "success",
      ,
      drop = FALSE
    ]
    landmark_fit <- landmark_all[
      landmark_all$profile == spec$profile &
        landmark_all$method == spec$landmark &
        landmark_all$status == "success",
      ,
      drop = FALSE
    ]
    joined <- merge(
      full_rows,
      landmark_fit,
      by = "dataset",
      suffixes = c("_full", "_landmark")
    )
    if (!nrow(joined)) return(NULL)
    data.frame(
      dataset = joined$dataset,
      family = spec$family,
      profile = spec$profile,
      landmark_fraction = clean_number(joined$landmark_fraction_landmark),
      full_runtime_sec = clean_number(joined$total_runtime_sec_median_full),
      landmark_runtime_sec = clean_number(joined$total_runtime_sec_median_landmark),
      full_over_landmark_speedup =
        clean_number(joined$total_runtime_sec_median_full) /
        clean_number(joined$total_runtime_sec_median_landmark),
      trustworthiness_full =
        clean_number(joined$trustworthiness_median_full),
      trustworthiness_landmark =
        clean_number(joined$trustworthiness_median_landmark),
      trustworthiness_delta =
        clean_number(joined$trustworthiness_median_landmark) -
        clean_number(joined$trustworthiness_median_full),
      preservation30_full =
        clean_number(joined$knn_preservation_30_median_full),
      preservation30_landmark =
        clean_number(joined$knn_preservation_30_median_landmark),
      preservation30_delta =
        clean_number(joined$knn_preservation_30_median_landmark) -
        clean_number(joined$knn_preservation_30_median_full),
      stringsAsFactors = FALSE
    )
  })
  pair_rows <- Filter(Negate(is.null), pair_rows)
  if (length(pair_rows)) {
    landmark_comparison <- do.call(rbind, pair_rows)
    write.csv(
      landmark_comparison,
      file.path(output_dir, "landmark_20pct_comparison.csv"),
      row.names = FALSE,
      na = ""
    )
    grouping <- interaction(
      landmark_comparison$family,
      landmark_comparison$profile,
      drop = TRUE
    )
    landmark_summary <- do.call(
      rbind,
      lapply(split(landmark_comparison, grouping), function(x) {
        data.frame(
          family = x$family[[1L]],
          profile = x$profile[[1L]],
          matched_datasets = nrow(x),
          landmark_fraction_median = median(x$landmark_fraction, na.rm = TRUE),
          speedup_median = median(x$full_over_landmark_speedup, na.rm = TRUE),
          speedup_q1 = unname(quantile(
            x$full_over_landmark_speedup, 0.25, na.rm = TRUE
          )),
          speedup_q3 = unname(quantile(
            x$full_over_landmark_speedup, 0.75, na.rm = TRUE
          )),
          speedup_min = min(x$full_over_landmark_speedup, na.rm = TRUE),
          speedup_max = max(x$full_over_landmark_speedup, na.rm = TRUE),
          trustworthiness_delta_median =
            median(x$trustworthiness_delta, na.rm = TRUE),
          preservation30_delta_median =
            median(x$preservation30_delta, na.rm = TRUE)
        )
      })
    )
    write.csv(
      landmark_summary,
      file.path(output_dir, "landmark_20pct_summary.csv"),
      row.names = FALSE,
      na = ""
    )
  }
}

cat("Wrote selected-method outputs to", output_dir, "\n")
cat("Embedding figures:", embedding_dir, "\n")
cat("Selected table rows:", nrow(selected), "\n")
cat("Successful rows:", nrow(success), "\n")
