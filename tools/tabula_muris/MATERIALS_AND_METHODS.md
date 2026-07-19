# Tabula Muris

## Data source

Tabula Muris is a single-cell transcriptomic atlas of 100,605 cells isolated
from 20 mouse organs and tissues using microfluidic droplet capture and
FACS/Smart-seq2. The legacy Seurat R objects distributed by the Tabula Muris
Consortium were obtained from Figshare
([doi:10.6084/m9.figshare.5821263.v1](https://doi.org/10.6084/m9.figshare.5821263.v1)).
The present dataset combines all 32 available tissue-by-protocol objects: 12
droplet objects and 20 FACS objects. The source matrices collectively contained
23,341 genes and 100,605 cells.

## Object conversion and quality control

For each legacy Seurat object, the raw count matrix (`raw.data`) and cell
metadata (`meta.data`) were extracted, restricted to shared cell identifiers,
and converted to a sparse `dgCMatrix` in blocks. A modern Seurat object was
created for each source file. Cell identifiers were prefixed with the source
name before the objects were merged, and the metadata fields `tissue`, `method`,
and `source_file` were recorded. In this merged object, `tissue` is the exact
source identifier and therefore includes the acquisition protocol prefix, for
example `droplet_Lung` or `facs_Lung`.

Mitochondrial and ribosomal percentages were calculated using gene-name
patterns `^mt-` and `^Rp[sl]`, respectively. Cells were retained when they had
more than 200 and fewer than 7,000 detected genes and less than 20% mitochondrial
counts. This removed 503 cells and retained 100,102 cells. Ribosomal percentage
was recorded but was not used as a filtering criterion.

## Normalization, feature selection, and PCA

Library sizes were computed from the retained raw counts. For each cell, counts
were divided by its library size, multiplied by 10,000, and transformed with
`log1p`. Gene means and variances were computed from the sparse normalized
matrix in 500-gene blocks, using the second-moment identity
`variance = mean(x^2) - mean(x)^2`. Genes with zero or undefined variance were
excluded, and the 2,000 genes with the largest variance were retained.

The selected expression matrix was converted to cells by genes, and every gene
was centered and scaled by its sample standard deviation. PCA was computed with
`irlba` using 50 components and random seed 123; because centering and scaling
had already been applied, both `center` and `scale` were disabled in the
`irlba` call. Cell scores were calculated as `U %*% diag(d)`. No t-SNE or UMAP
coordinates are included in the benchmark input.

## Labels and saved representations

The benchmark labels are the 32-level factor taken directly from the merged
Seurat object's `tissue` metadata field. Row order is identical in all saved
representations.

- `TabulaMuris_merged_QC_Seurat_PCA50.rds`: QC-filtered Seurat object with the
  `100102 x 50` PCA scores stored as the `pca` dimensional reduction.
- `TabulaMuris.RData`: list named `dataset`; `dataset$data` is the conventional
  double-precision `100102 x 50` PCA matrix and `dataset$labels` contains the
  tissue factor.
- `TabulaMuris_float32.RData`: the same list and labels, with `dataset$data`
  stored as a `float::float32` matrix for float32 fastEmbedR validation.

The float32 file changes storage precision only; normalization, selected genes,
PCA scores, sample order, and labels are otherwise shared with the conventional
R representation.

## Reproducibility

The exact QC command sequence recovered from the timestamped RStudio history is
archived as `tools/tabula_muris/generate_qc_from_rstudio_history_20260618.R`.
The unchanged working notebook used for PCA50 is archived as
`tools/tabula_muris/tabula_original_20260618.R`. The export script does not
recompute preprocessing; it reads the completed June 18 workspace and writes
the three validated representations described above.

## Reference

The Tabula Muris Consortium. Single-cell transcriptomics of 20 mouse organs
creates a Tabula Muris. *Nature*. 2018;562:367-372.
doi:10.1038/s41586-018-0590-4.
