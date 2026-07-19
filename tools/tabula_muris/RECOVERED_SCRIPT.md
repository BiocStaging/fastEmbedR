# Recovered Tabula Muris preprocessing

Two original records are preserved:

- `generate_qc_from_rstudio_history_20260618.R` is the exact command block
  that created the raw and QC Seurat objects. It was recovered from RStudio's
  timestamped command history.
- `tabula_original_20260618.R` is the unchanged saved working notebook used
  subsequently for variable-gene selection, PCA50, t-SNE, and benchmark-data
  preparation.

Provenance:

- Original path: `/Users/stefano/Desktop/tabula.R`
- Original birth time: 2026-06-18 18:43:14
- Original modification time: 2026-06-18 19:05:44
- Corresponding RStudio command history:
  `/Users/stefano/.local/share/rstudio/history_database.1`
- Generated workspace: `/Users/stefano/Desktop/Tabula.RData`
- Saved QC object: `/Users/stefano/Downloads/5821263/TabulaMuris_merged_QC_Seurat.rds`

The QC save command is timestamped `2026-06-18 17:47:04 CAT`, exactly matching
the QC file's filesystem birth time. The saved object has 32 count layers, as
expected from the recovered Seurat merge. The later notebook is archived
unchanged rather than rewritten as a new pipeline.
