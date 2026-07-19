#' fastEmbedR: native embeddings and graph clustering from KNN
#'
#' fastEmbedR combines native CPU/Metal nearest-neighbour search with native
#' UMAP/openTSNE-style embeddings and native graph clustering from data or
#' precomputed KNN matrices. Use
#' [umap()] or [opentsne()] for one-call workflows, [precompute_knn()] to run
#' the package-native CPU, Metal, or CUDA search separately, or pass a reusable
#' list of KNN `indices` and `distances` to [umap_knn()] or [opentsne_knn()].
#' Use [pca()] for reusable backend-native PCA and request
#' `opentsne_init = TRUE` when an openTSNE-ready initialization is needed.
#' Build a clustering graph with [knn_graph()] and apply native Louvain,
#' Leiden, or Walktrap with [graph_cluster()]. Score embeddings with
#' [evaluate_embedding()].
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
