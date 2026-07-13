#' fastEmbedR: fast KNN, UMAP, and native openTSNE from KNN
#'
#' fastEmbedR combines native CPU/Metal nearest-neighbour search with native
#' UMAP/openTSNE-style embeddings from data or precomputed KNN matrices. Use
#' [umap()] or [opentsne()] for one-call workflows, or compute a reusable KNN
#' object with `faissR::nn()` and pass it to [umap_knn()] or [opentsne_knn()].
#' Score results with [evaluate_embedding()].
#'
#' The package intentionally does not export the earlier legacy t-SNE,
#' InfoTSNE, PaCMAP, TriMap, or LocalMAP reducers.
#'
#' @keywords internal
"_PACKAGE"

utils::globalVariables(c(
  "landmark_refinement_epoch_count",
  "landmark_tsne_transform_resident_cuda_cpp",
  "landmark_tsne_transform_resident_metal_cpp",
  "run_native_knn_optimizer",
  "transform_tsne_cuda_cpp"
))
