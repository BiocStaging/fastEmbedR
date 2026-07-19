#!/usr/bin/env Rscript

parse_args <- function(x) {
  out <- list()
  for (arg in x) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", kv[[1L]])]] <- if (length(kv) > 1L) {
      paste(kv[-1L], collapse = "=")
    } else "TRUE"
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x
inputs <- trimws(strsplit(args$inputs %||% "", ",", fixed = TRUE)[[1L]])
inputs <- inputs[nzchar(inputs)]
if (length(inputs) < 2L) {
  stop("Pass at least two benchmark folders with --inputs=DIR1,DIR2.", call. = FALSE)
}
inputs <- normalizePath(inputs, mustWork = TRUE)
out_dir <- normalizePath(
  args$out_dir %||% file.path(dirname(inputs[[1L]]), "combined_backend_agreement"),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]), mustWork = TRUE)
source(file.path(dirname(script_path), "publication_metrics.R"), local = TRUE)

read_runs <- function(folder) {
  path <- file.path(folder, "benchmark_runs.csv")
  if (!file.exists(path)) stop("Missing benchmark_runs.csv in ", folder, call. = FALSE)
  x <- read.csv(path, stringsAsFactors = FALSE)
  x$source_folder <- folder
  x
}
runs <- do.call(rbind, lapply(inputs, read_runs))
write.csv(runs, file.path(out_dir, "combined_benchmark_runs.csv"), row.names = FALSE)

canonical_method <- function(x) {
  x <- gsub("_(cpu|cuda|metal)(_|$)", "_", x)
  x <- gsub("__+", "_", x)
  sub("_$", "", x)
}
runs$canonical_method <- canonical_method(runs$method)

load_layout <- function(path) {
  if (is.na(path) || !file.exists(path)) return(NULL)
  value <- tryCatch(readRDS(path), error = function(e) NULL)
  if (is.null(value)) return(NULL)
  publication_layout_matrix(value$layout %||% value)
}

layout_rows <- list()
successful <- runs[
  runs$status == "success" & runs$family %in% c("t-SNE", "UMAP", "PCA"),
  , drop = FALSE
]
reference <- successful[successful$backend == "cpu", , drop = FALSE]
if (nrow(reference)) {
  max_threads <- tapply(reference$requested_threads, reference$dataset, max, na.rm = TRUE)
  keep <- vapply(seq_len(nrow(reference)), function(i) {
    reference$requested_threads[[i]] == max_threads[[reference$dataset[[i]]]]
  }, logical(1))
  reference <- reference[keep, , drop = FALSE]
}
candidate <- successful[successful$backend %in% c("metal", "cuda"), , drop = FALSE]

for (i in seq_len(nrow(candidate))) {
  row <- candidate[i, , drop = FALSE]
  matches <- reference[
    reference$dataset == row$dataset &
      reference$canonical_method == row$canonical_method &
      reference$seed == row$seed &
      reference$timing_scope == row$timing_scope,
    , drop = FALSE
  ]
  if (!nrow(matches)) next
  ref <- matches[1L, , drop = FALSE]
  ref_layout <- load_layout(ref$layout_file)
  candidate_layout <- load_layout(row$layout_file)
  if (is.null(ref_layout) || is.null(candidate_layout) ||
      nrow(ref_layout) != nrow(candidate_layout)) next
  sample_rows <- publication_sample_rows(nrow(ref_layout), min(3000L, nrow(ref_layout)), 1013L)
  agreement <- publication_procrustes(
    ref_layout[sample_rows, , drop = FALSE],
    candidate_layout[sample_rows, , drop = FALSE]
  )
  neighbor_agreement <- publication_knn_overlap(
    publication_exact_knn(ref_layout[sample_rows, , drop = FALSE], 15L),
    publication_exact_knn(candidate_layout[sample_rows, , drop = FALSE], 15L),
    15L
  )
  layout_rows[[length(layout_rows) + 1L]] <- data.frame(
    dataset = row$dataset, canonical_method = row$canonical_method,
    timing_scope = row$timing_scope, seed = row$seed,
    reference_backend = "cpu", candidate_backend = row$backend,
    reference_threads = ref$requested_threads,
    candidate_threads = row$requested_threads,
    procrustes_rmsd = agreement$rmsd,
    procrustes_correlation = agreement$correlation,
    embedding_neighbor_agreement_15 = neighbor_agreement,
    stringsAsFactors = FALSE
  )
}
layout_agreement <- if (length(layout_rows)) do.call(rbind, layout_rows) else data.frame()
write.csv(
  layout_agreement,
  file.path(out_dir, "cpu_metal_cuda_layout_agreement.csv"),
  row.names = FALSE
)

find_validation_knn <- function(folder, dataset, backend) {
  path <- file.path(
    folder, "validation_knn",
    sprintf("%s_%s_knn.rds", gsub("[^A-Za-z0-9_.-]+", "_", dataset), backend)
  )
  if (file.exists(path)) path else NA_character_
}

knn_rows <- list()
datasets <- unique(runs$dataset)
for (dataset in datasets) {
  cpu_path <- NA_character_
  cpu_folder <- NA_character_
  for (folder in inputs) {
    candidate_path <- find_validation_knn(folder, dataset, "cpu")
    if (!is.na(candidate_path)) {
      cpu_path <- candidate_path
      cpu_folder <- folder
      break
    }
  }
  if (is.na(cpu_path)) next
  cpu_knn <- readRDS(cpu_path)
  cpu_affinity <- publication_sparse_affinities(cpu_knn, min(30L, ncol(cpu_knn$indices)))
  for (backend in c("metal", "cuda")) {
    candidate_path <- NA_character_
    for (folder in inputs) {
      path <- find_validation_knn(folder, dataset, backend)
      if (!is.na(path)) {
        candidate_path <- path
        break
      }
    }
    if (is.na(candidate_path)) next
    candidate_knn <- readRDS(candidate_path)
    affinity <- publication_edge_agreement(
      cpu_affinity,
      publication_sparse_affinities(candidate_knn, min(30L, ncol(candidate_knn$indices)))
    )
    for (graph_mode in c("fuzzy", "binary")) {
      graph <- publication_edge_agreement(
        publication_umap_edges(cpu_knn, graph_mode = graph_mode, n_threads = 1L),
        publication_umap_edges(candidate_knn, graph_mode = graph_mode, n_threads = 1L)
      )
      knn_rows[[length(knn_rows) + 1L]] <- data.frame(
        dataset = dataset, reference_backend = "cpu", candidate_backend = backend,
        graph_mode = graph_mode,
        knn_overlap = publication_knn_overlap(cpu_knn, candidate_knn),
        affinity_edge_jaccard = affinity$edge_jaccard,
        affinity_weight_pearson = affinity$weight_pearson,
        affinity_weight_spearman = affinity$weight_spearman,
        affinity_weight_l1_similarity = affinity$weight_l1_similarity,
        umap_graph_edge_jaccard = graph$edge_jaccard,
        umap_graph_weight_pearson = graph$weight_pearson,
        umap_graph_weight_spearman = graph$weight_spearman,
        umap_graph_weight_l1_similarity = graph$weight_l1_similarity,
        stringsAsFactors = FALSE
      )
    }
  }
}
knn_agreement <- if (length(knn_rows)) do.call(rbind, knn_rows) else data.frame()
write.csv(
  knn_agreement,
  file.path(out_dir, "cpu_metal_cuda_knn_affinity_graph_agreement.csv"),
  row.names = FALSE
)

manifest <- c(
  paste0("generated_at=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  paste0("input_folders=", paste(inputs, collapse = ";")),
  "reference_backend=cpu",
  "layout_alignment=orthogonal Procrustes on reproducible sample",
  "knn_affinity_graph_comparison=shared validation rows"
)
writeLines(manifest, file.path(out_dir, "combine_manifest.txt"))
cat("Combined backend agreement written to:", out_dir, "\n")
