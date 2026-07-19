#!/usr/bin/env Rscript

## Export the already-computed June 18 Tabula Muris PCA50 result.
## This script does not normalize, select genes, or recompute PCA.

workspace_file <- "/Users/stefano/Desktop/Tabula.RData"
output_dir <- "/Users/stefano/HPC-firenze/NN/Data/TabulaMuris"

if (!requireNamespace("SeuratObject", quietly = TRUE)) {
  stop("The SeuratObject package is required.", call. = FALSE)
}
if (!requireNamespace("float", quietly = TRUE)) {
  stop("The float package is required.", call. = FALSE)
}

loaded <- new.env(parent = emptyenv())
load(workspace_file, envir = loaded)

required <- c("tabula_seurat_qc", "pca_data")
missing <- setdiff(required, ls(loaded))
if (length(missing)) {
  stop("Missing object(s) in recovered workspace: ", paste(missing, collapse = ", "), call. = FALSE)
}

seurat_object <- loaded$tabula_seurat_qc
pca_data <- loaded$pca_data

if (!is.matrix(pca_data) || !identical(dim(pca_data), c(100102L, 50L))) {
  stop("Recovered PCA must be a 100102 x 50 matrix.", call. = FALSE)
}
labels <- factor(as.character(seurat_object$tissue))
if (length(labels) != nrow(pca_data) || anyNA(labels)) {
  stop("The `tissue` labels are missing or misaligned.", call. = FALSE)
}

## The recovered PCA matrix was saved without row names. Its order is proven
## against the June 18 benchmark export, which was created in the same R session
## by assigning pca_data and the Seurat tissue vector together.
existing_path <- file.path(output_dir, "TabulaMuris.RData")
if (is.null(rownames(pca_data))) {
  if (!file.exists(existing_path)) {
    stop("Cannot prove the row order because the June 18 benchmark export is missing.", call. = FALSE)
  }
  existing <- new.env(parent = emptyenv())
  load(existing_path, envir = existing)
  if (!identical(pca_data, existing$dataset$data) ||
      !identical(labels, existing$dataset$labels)) {
    stop("Recovered PCA order does not match the June 18 benchmark export.", call. = FALSE)
  }
  rownames(pca_data) <- colnames(seurat_object)
} else if (!identical(rownames(pca_data), colnames(seurat_object))) {
  stop("PCA rows are not aligned with Seurat cells.", call. = FALSE)
}

colnames(pca_data) <- paste0("PC_", seq_len(ncol(pca_data)))
seurat_object[["pca"]] <- SeuratObject::CreateDimReducObject(
  embeddings = pca_data,
  key = "PC_",
  assay = "RNA"
)

metadata <- list(
  name = "Tabula Muris merged droplet and FACS PCA50",
  n = nrow(pca_data),
  p = ncol(pca_data),
  label = "tissue",
  label_levels = nlevels(labels),
  source = "Tabula Muris Consortium legacy Seurat R objects",
  source_url = "https://doi.org/10.6084/m9.figshare.5821263.v1",
  original_cells = 100605L,
  retained_cells = 100102L,
  genes = 23341L,
  droplet_objects = 12L,
  facs_objects = 20L,
  preprocessing = paste(
    "nFeature_RNA > 200; nFeature_RNA < 7000; percent.mt < 20;",
    "library-size normalization to 10000; log1p; top 2000 genes by variance;",
    "per-gene centering and scaling; irlba PCA50 (seed 123)"
  ),
  recovered_workspace = workspace_file,
  recovered_script = "/Users/stefano/Documents/umap/tools/tabula_muris/tabula_original_20260618.R"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seurat_path <- file.path(output_dir, "TabulaMuris_merged_QC_Seurat_PCA50.rds")
classic_path <- file.path(output_dir, "TabulaMuris.RData")
float_path <- file.path(output_dir, "TabulaMuris_float32.RData")

saveRDS(seurat_object, seurat_path, compress = TRUE)

dataset <- list(data = pca_data, labels = labels, metadata = metadata)
save(dataset, file = classic_path, compress = TRUE)

dataset$data <- float::fl(pca_data)
save(dataset, file = float_path, compress = TRUE)

classic_check <- new.env(parent = emptyenv())
load(classic_path, envir = classic_check)
float_check <- new.env(parent = emptyenv())
load(float_path, envir = float_check)

stopifnot(
  identical(dim(classic_check$dataset$data), c(100102L, 50L)),
  identical(classic_check$dataset$labels, labels),
  inherits(float_check$dataset$data, "float32"),
  identical(float_check$dataset$labels, labels)
)

cat("Seurat PCA50:", seurat_path, "\n")
cat("Classic PCA50:", classic_path, "\n")
cat("Float32 PCA50:", float_path, "\n")
cat("Rows:", nrow(pca_data), " Columns:", ncol(pca_data), " Labels:", nlevels(labels), "\n")
