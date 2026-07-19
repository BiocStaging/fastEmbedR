library(Seurat)
library(Matrix)

data_dir <- "/Users/stefano/Downloads/5821263"

files <- list.files(
  data_dir,
  pattern = "\\.Robj$",
  full.names = TRUE
)

## Optional: use only one technology
## files <- files[grepl("^droplet_", basename(files))]
## files <- files[grepl("^facs_", basename(files))]

dataframe_to_sparse <- function(df, chunk_size = 200) {
  
  genes <- rownames(df)
  cells <- colnames(df)
  
  out <- vector("list", ceiling(ncol(df) / chunk_size))
  k <- 1
  
  for (start in seq(1, ncol(df), by = chunk_size)) {
    
    end <- min(start + chunk_size - 1, ncol(df))
    message("Converting ", start, " to ", end, " of ", ncol(df))
    
    chunk <- df[, start:end, drop = FALSE]
    out[[k]] <- Matrix(as.matrix(chunk), sparse = TRUE)
    
    k <- k + 1
    
    rm(chunk)
    gc()
  }
  
  mat <- do.call(cbind, out)
  rownames(mat) <- genes
  colnames(mat) <- cells
  
  mat
}

counts_list <- list()
meta_list <- list()

for (f in files) {
  
  message("\nLoading ", basename(f))
  
  env <- new.env()
  loaded <- load(f, envir = env)
  old <- get(loaded[1], envir = env)
  
  tissue <- basename(f)
  tissue <- sub("_seurat_tiss\\.Robj$", "", tissue)
  
  method <- ifelse(
    grepl("^droplet_", tissue),
    "droplet",
    ifelse(grepl("^facs_", tissue), "facs", "unknown")
  )
  
  counts <- old@raw.data
  meta <- old@meta.data
  
  common <- intersect(colnames(counts), rownames(meta))
  
  counts <- counts[, common, drop = FALSE]
  meta <- meta[common, , drop = FALSE]
  
  new_cells <- paste(tissue, common, sep = "_")
  
  colnames(counts) <- new_cells
  rownames(meta) <- new_cells
  
  meta$tissue <- tissue
  meta$method <- method
  meta$source_file <- basename(f)
  
  counts_sparse <- dataframe_to_sparse(counts, chunk_size = 200)
  
  counts_list[[tissue]] <- counts_sparse
  meta_list[[tissue]] <- meta
  
  rm(env, old, counts, meta, counts_sparse)
  gc()
}

common_genes <- Reduce(intersect, lapply(counts_list, rownames))

counts_list <- lapply(counts_list, function(x) {
  x[common_genes, , drop = FALSE]
})

counts_merged <- do.call(cbind, counts_list)
library(dplyr)

meta_merged <- bind_rows(meta_list)
meta_merged <- as.data.frame(meta_merged)

rownames(meta_merged) <- meta_merged$cell_id


tabula_seurat <- CreateSeuratObject(
  counts = counts_merged,
  meta.data = meta_merged,
  project = "TabulaMuris",
  min.cells = 0,
  min.features = 0
)

tabula_seurat














library(Seurat)
library(Matrix)
library(irlba)
library(Rtsne)
library(ggplot2)

## ------------------------------------------------------------
## Input: tabula_seurat_qc already exists
## ------------------------------------------------------------

DefaultAssay(tabula_seurat_qc) <- "RNA"

out_dir <- "/Users/stefano/Downloads/5821263"

## ------------------------------------------------------------
## 1. Get sparse counts
## ------------------------------------------------------------

counts <- GetAssayData(
  tabula_seurat_qc,
  assay = "RNA",
  layer = "counts"
)

counts <- as(counts, "dgCMatrix")

dim(counts)

## ------------------------------------------------------------
## 2. Library size
## ------------------------------------------------------------

lib_size <- Matrix::colSums(counts)

keep_cells <- lib_size > 0

counts <- counts[, keep_cells, drop = FALSE]
lib_size <- lib_size[keep_cells]

tabula_seurat_qc <- subset(
  tabula_seurat_qc,
  cells = colnames(counts)
)

## ------------------------------------------------------------
## 3. Select top 2000 variable genes without NormalizeData()
## ------------------------------------------------------------

gene_block_size <- 500

gene_means <- numeric(nrow(counts))
gene_vars <- numeric(nrow(counts))

names(gene_means) <- rownames(counts)
names(gene_vars) <- rownames(counts)

for (start in seq(1, nrow(counts), by = gene_block_size)) {
  
  end <- min(start + gene_block_size - 1, nrow(counts))
  
  message("Processing genes ", start, " to ", end, " of ", nrow(counts))
  
  block <- counts[start:end, , drop = FALSE]
  
  ## sparse log-normalization:
  ## log1p(count / library_size * 10000)
  block_norm <- t(t(block) / lib_size) * 10000
  block_norm@x <- log1p(block_norm@x)
  
  gene_means[start:end] <- Matrix::rowMeans(block_norm)
  
  block_sq <- block_norm
  block_sq@x <- block_sq@x^2
  
  mean_sq <- Matrix::rowMeans(block_sq)
  
  gene_vars[start:end] <- mean_sq - gene_means[start:end]^2
  
  rm(block, block_norm, block_sq, mean_sq)
  gc()
}

## Remove genes with zero/NA variance
gene_vars[is.na(gene_vars)] <- 0
gene_vars[gene_means == 0] <- 0

top2000 <- names(sort(gene_vars, decreasing = TRUE))[1:2000]

top2000 <- top2000[!is.na(top2000)]

length(top2000)
head(top2000, 20)








## DO NOT RUN THIS:
## VariableFeatures(tabula_seurat_qc) <- top2000

length(top2000)
head(top2000)

## ------------------------------------------------------------
## Build log-normalized matrix only for top 2000 genes
## ------------------------------------------------------------

library(Matrix)
library(irlba)
library(Rtsne)
library(Seurat)
library(ggplot2)

out_dir <- "/Users/stefano/Downloads/5821263"

counts <- GetAssayData(
  tabula_seurat_qc,
  assay = "RNA",
  layer = "counts"
)

counts <- as(counts, "dgCMatrix")

lib_size <- Matrix::colSums(counts)
keep_cells <- lib_size > 0

counts <- counts[, keep_cells, drop = FALSE]
lib_size <- lib_size[keep_cells]

tabula_seurat_qc <- subset(
  tabula_seurat_qc,
  cells = colnames(counts)
)

top2000 <- intersect(top2000, rownames(counts))

message("Using ", length(top2000), " genes")

x <- counts[top2000, , drop = FALSE]

## Sparse log-normalization only for top2000
x <- t(t(x) / lib_size) * 10000
x@x <- log1p(x@x)

## Dense matrix is only cells x 2000 genes
x_dense <- t(as.matrix(x))

rm(x)
gc()

## ------------------------------------------------------------
## Manual scaling
## ------------------------------------------------------------

gene_center <- colMeans(x_dense)

gene_scale <- apply(x_dense, 2, sd)
gene_scale[gene_scale == 0] <- 1
gene_scale[is.na(gene_scale)] <- 1

x_scaled <- scale(
  x_dense,
  center = gene_center,
  scale = gene_scale
)

rm(x_dense)
gc()

## ------------------------------------------------------------
## PCA 50 PCs using irlba
## ------------------------------------------------------------

set.seed(123)

pca_res <- irlba(
  x_scaled,
  nv = 50,
  center = FALSE,
  scale = FALSE
)

pca_embeddings <- pca_res$u %*% diag(pca_res$d)

rownames(pca_embeddings) <- colnames(counts)
colnames(pca_embeddings) <- paste0("PC_", 1:50)

rm(x_scaled)
gc()

## ------------------------------------------------------------
## Add PCA to Seurat object
## ------------------------------------------------------------

tabula_seurat_qc[["pca"]] <- CreateDimReducObject(
  embeddings = pca_embeddings,
  key = "PC_",
  assay = "RNA"
)

## ------------------------------------------------------------
## t-SNE from PCA
## ------------------------------------------------------------

set.seed(123)

tsne_res <- Rtsne(
  pca_embeddings[, 1:50],
  dims = 2,
  perplexity = 30,
  theta = 0.5,
  pca = FALSE,
  check_duplicates = FALSE,
  verbose = TRUE,
  max_iter = 1000
)

tsne_embeddings <- tsne_res$Y

rownames(tsne_embeddings) <- rownames(pca_embeddings)
colnames(tsne_embeddings) <- c("tSNE_1", "tSNE_2")

tabula_seurat_qc[["tsne"]] <- CreateDimReducObject(
  embeddings = tsne_embeddings,
  key = "tSNE_",
  assay = "RNA"
)

## ------------------------------------------------------------
## Plot
## ------------------------------------------------------------

p_tissue <- DimPlot(
  tabula_seurat_qc,
  reduction = "tsne",
  group.by = "tissue",
  label = TRUE,
  repel = TRUE,
  pt.size = 0.2
) +
  ggtitle("Tabula Muris t-SNE by tissue")

print(p_tissue)

p_method <- DimPlot(
  tabula_seurat_qc,
  reduction = "tsne",
  group.by = "method",
  label = TRUE,
  repel = TRUE,
  pt.size = 0.2
) +
  ggtitle("Tabula Muris t-SNE by method")

print(p_method)

if ("cell_ontology_class" %in% colnames(tabula_seurat_qc@meta.data)) {
  
  p_celltype <- DimPlot(
    tabula_seurat_qc,
    reduction = "tsne",
    group.by = "cell_ontology_class",
    label = TRUE,
    repel = TRUE,
    pt.size = 0.2
  ) +
    ggtitle("Tabula Muris t-SNE by cell ontology class")
  
  print(p_celltype)
}

tabula_seurat_qc@meta.data$tissue[1:10]
pca_data[1:10,1:10]

load("/Users/stefano/HPC-firenze/NN/Data/TabulaMuris/TabulaMuris.RData")

dataset$data=pca_data
dataset$labels=as.factor(tabula_seurat_qc@meta.data$tissue)
dataset$metadata$n=nrow(pca_data)

save(dataset,file="/Users/stefano/HPC-firenze/NN/Data/TabulaMuris/TabulaMuris.RData")



