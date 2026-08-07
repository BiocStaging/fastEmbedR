#' fastEmbedR: native UMAP and openTSNE embeddings
#'
#' fastEmbedR provides native float32 UMAP and openTSNE-style embeddings from
#' data or precomputed nearest-neighbor matrices. Use [umap()] or [opentsne()]
#' for one-call workflows, [precompute_knn()] to run the package-native CPU,
#' Metal, or CUDA search separately, or pass reusable KNN `indices` and
#' `distances` to [umap_knn()] or [opentsne_knn()].
#' Use [pca()] for reusable backend-native PCA and request
#' `opentsne_init = TRUE` when an openTSNE-ready initialization is needed.
#' Score embeddings with [evaluate_embedding()]. Optional downstream graph
#' utilities are available through [knn_graph()] and [graph_cluster()].
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
