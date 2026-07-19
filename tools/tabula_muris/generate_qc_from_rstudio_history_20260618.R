## ============================================================
## Tabula Muris old Seurat objects -> one modern Seurat object
## Then QC + normalization + PCA
## ============================================================

library(Seurat)
library(Matrix)
library(ggplot2)

## ------------------------------------------------------------
## 1. Folder with the downloaded .Robj files
## ------------------------------------------------------------

data_dir <- "/Users/stefano/Downloads/5821263"

files <- list.files(
  data_dir,
  pattern = "\\.Robj$",
  full.names = TRUE
)

files
length(files)
basename(files)

if (length(files) == 0) {
  stop("No .Robj files found in: ", data_dir)
}

## Optional: use only droplet or only FACS
## Recommended: start with only droplet OR only FACS, not both together

## All files:
files_to_use <- files

## Only droplet:
## files_to_use <- files[grepl("^droplet_", basename(files))]

## Only FACS:
## files_to_use <- files[grepl("^facs_", basename(files))]

basename(files_to_use)
length(files_to_use)

## ------------------------------------------------------------
## 2. Convert old Seurat raw.data data.frame to sparse matrix
##    WITHOUT converting the whole thing to dense
## ------------------------------------------------------------

dataframe_to_sparse <- function(df, chunk_size = 250) {
  genes <- rownames(df)
  cells <- colnames(df)

  i_all <- integer()
  j_all <- integer()
  x_all <- numeric()

  for (start in seq(1, ncol(df), by = chunk_size)) {
    end <- min(start + chunk_size - 1, ncol(df))

    message("  converting cells ", start, " to ", end, " of ", ncol(df))

    chunk <- df[, start:end, drop = FALSE]
    chunk <- as.matrix(chunk)

    nz <- which(chunk != 0, arr.ind = TRUE)

    if (nrow(nz) > 0) {
      i_all <- c(i_all, nz[, 1])
      j_all <- c(j_all, nz[, 2] + start - 1)
      x_all <- c(x_all, chunk[nz])
    }

    rm(chunk, nz)
    gc()
  }

  sparseMatrix(
    i = i_all,
    j = j_all,
    x = x_all,
    dims = c(length(genes), length(cells)),
    dimnames = list(genes, cells)
  )
}

## ------------------------------------------------------------
## 3. Load one old Tabula file and convert to modern Seurat
## ------------------------------------------------------------

load_old_tabula_as_seurat <- function(f) {
  message("\n======================================")
  message("Loading: ", basename(f))

  e <- new.env()
  loaded_names <- load(f, envir = e)
  obj_old <- get(loaded_names[1], envir = e)

  message("Old class: ", paste(class(obj_old), collapse = ", "))

  tissue_name <- basename(f)
  tissue_name <- sub("_seurat_tiss\\.Robj$", "", tissue_name)

  method_name <- ifelse(
    grepl("^droplet_", tissue_name),
    "droplet",
    ifelse(grepl("^facs_", tissue_name), "facs", "unknown")
  )

  ## Old Seurat v2 count matrix and metadata
  counts_old <- obj_old@raw.data
  meta <- obj_old@meta.data

  message("Genes: ", nrow(counts_old))
  message("Cells in counts: ", ncol(counts_old))
  message("Cells in metadata: ", nrow(meta))

  ## Keep only common cells
  common_cells <- intersect(colnames(counts_old), rownames(meta))
  message("Common cells: ", length(common_cells))

  counts_old <- counts_old[, common_cells, drop = FALSE]
  meta <- meta[common_cells, , drop = FALSE]

  ## Convert to sparse matrix safely
  counts_sparse <- dataframe_to_sparse(counts_old, chunk_size = 250)

  ## Create modern Seurat object
  seu <- CreateSeuratObject(
    counts = counts_sparse,
    meta.data = meta,
    project = tissue_name,
    min.cells = 0,
    min.features = 0
  )

  seu$tissue <- tissue_name
  seu$method <- method_name
  seu$source_file <- basename(f)

  ## Make cell names unique before merge
  seu <- RenameCells(
    seu,
    add.cell.id = tissue_name
  )

  rm(e, obj_old, counts_old, counts_sparse, meta)
  gc()

  return(seu)
}

## ------------------------------------------------------------
## 4. Convert all files to modern Seurat objects
## ------------------------------------------------------------

seurat_list <- list()

for (f in files_to_use) {
  tissue_name <- basename(f)
  tissue_name <- sub("_seurat_tiss\\.Robj$", "", tissue_name)

  seurat_list[[tissue_name]] <- load_old_tabula_as_seurat(f)

  message("Finished: ", tissue_name)
  gc()
}

length(seurat_list)
names(seurat_list)

## ------------------------------------------------------------
## 5. Merge all Seurat objects
## ------------------------------------------------------------

if (length(seurat_list) == 0) {
  stop("No Seurat objects were created.")
}

if (length(seurat_list) == 1) {
  tabula_seurat <- seurat_list[[1]]
} else {
  tabula_seurat <- merge(
    x = seurat_list[[1]],
    y = seurat_list[-1],
    project = "TabulaMuris"
  )
}

tabula_seurat

## Save immediately
saveRDS(
  tabula_seurat,
  file = file.path(data_dir, "TabulaMuris_merged_raw_Seurat.rds")
)

## ------------------------------------------------------------
## 6. Basic QC metrics
## ------------------------------------------------------------

## Mouse mitochondrial genes usually start with mt-
tabula_seurat[["percent.mt"]] <- PercentageFeatureSet(
  tabula_seurat,
  pattern = "^mt-"
)

## Ribosomal genes, optional
tabula_seurat[["percent.ribo"]] <- PercentageFeatureSet(
  tabula_seurat,
  pattern = "^Rp[sl]"
)

## Check metadata
head(tabula_seurat@meta.data)

## ------------------------------------------------------------
## 7. QC plots before filtering
## ------------------------------------------------------------

VlnPlot(
  tabula_seurat,
  features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
  group.by = "tissue",
  pt.size = 0
)

VlnPlot(
  tabula_seurat,
  features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
  group.by = "method",
  pt.size = 0
)

FeatureScatter(
  tabula_seurat,
  feature1 = "nCount_RNA",
  feature2 = "nFeature_RNA",
  group.by = "tissue"
)

FeatureScatter(
  tabula_seurat,
  feature1 = "nCount_RNA",
  feature2 = "percent.mt",
  group.by = "tissue"
)

## ------------------------------------------------------------
## 8. QC filtering
## ------------------------------------------------------------

## Conservative generic filtering.
## Adjust after looking at the violin plots.
tabula_seurat_qc <- subset(
  tabula_seurat,
  subset =
    nFeature_RNA > 200 &
    nFeature_RNA < 7000 &
    percent.mt < 20
)

tabula_seurat
tabula_seurat_qc
table(tabula_seurat$tissue)
table(tabula_seurat_qc$tissue)

## Save QC object
saveRDS(
  tabula_seurat_qc,
  file = file.path(data_dir, "TabulaMuris_merged_QC_Seurat.rds")
)
