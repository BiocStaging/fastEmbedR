#' Inspect the supported public API
#'
#' `fastEmbedR_api()` returns the package's machine-readable public API map.
#' The map distinguishes canonical workflows from advanced reusable-state
#' interfaces, diagnostics, secondary graph utilities, and compatibility
#' helpers. It also records accepted input classes, returned classes, backend
#' support, device-residency behavior, related S3 methods, and lifecycle
#' status.
#'
#' The inventory describes the public contract of the installed package. A
#' backend marked as supported still requires that backend to have been
#' compiled and to be available at runtime; use [fastEmbedR_capabilities()] to
#' inspect the current installation.
#'
#' UMAP rows describe fastEmbedR's fixed optimizer policy, not a general-purpose
#' parameter-sweep interface. See [umap()] for the public/internal control
#' boundary and the resolved values recorded by each fit.
#'
#' @return A data frame with one row per exported function.
#' @examples
#' api <- fastEmbedR_api()
#' api[api$tier == "canonical", c("function", "purpose", "return_class")]
#' @export
fastEmbedR_api <- function() {
  row <- function(function_name,
                  tier,
                  purpose,
                  input_classes,
                  return_class,
                  cpu,
                  metal,
                  cuda,
                  operation,
                  device_residency,
                  related_methods = "none",
                  lifecycle = "stable") {
    data.frame(
      `function` = function_name,
      tier = tier,
      purpose = purpose,
      input_classes = input_classes,
      return_class = return_class,
      cpu = cpu,
      metal = metal,
      cuda = cuda,
      operation = operation,
      device_residency = device_residency,
      related_methods = related_methods,
      lifecycle = lifecycle,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }

  rows <- list(
    row(
      "fastEmbedR_backend", "canonical", "Get or set the session backend.",
      "NULL or backend string", "character scalar", "yes", "yes", "yes",
      "configuration", "not applicable"
    ),
    row(
      "fastEmbedR_capabilities", "canonical",
      "Report compiled and runtime backend capabilities.", "none",
      "data.frame", "yes", "yes", "yes", "diagnostic",
      "not applicable"
    ),
    row(
      "fastEmbedR_api", "canonical", "Return this public API inventory.",
      "none", "data.frame", "yes", "yes", "yes", "diagnostic",
      "not applicable"
    ),
    row(
      "pca", "canonical", "Backend-native randomized PCA/TSVD.",
      "matrix, data.frame, or float32", "fastEmbedR_pca", "yes", "yes",
      "yes", "randomized approximation",
      "accelerator intermediates; host R result"
    ),
    row(
      "precompute_knn", "canonical",
      "Compute reusable non-self nearest neighbors.",
      "matrix, data.frame, or float32", "fastEmbedR_knn", "yes", "yes",
      "yes", "backend-dependent exact or approximate search",
      "CUDA result remains device-resident; CPU/Metal result is host",
      "print"
    ),
    row(
      "opentsne", "canonical", "Run the complete native t-SNE workflow.",
      "matrix, data.frame, float32, or fastEmbedR_knn",
      "fastEmbedR_embedding", "yes", "yes", "yes",
      "sparse affinities with exact or FFT-grid repulsion",
      "backend intermediates may remain resident; final layout is host",
      "print, plot"
    ),
    row(
      "opentsne_knn", "canonical", "Run t-SNE from reusable neighbors.",
      paste(
        "index/distance matrices, fastEmbedR_knn, fastEmbedR_gpu_knn,",
        "or fastEmbedR_opentsne_prepared"
      ),
      "numeric or float32 layout matrix", "yes", "yes", "yes",
      "sparse affinities with exact or FFT-grid repulsion",
      "resident CUDA KNN can be consumed on device; final layout is host"
    ),
    row(
      "umap", "canonical", "Run the opinionated fixed-policy UMAP workflow.",
      "matrix, data.frame, float32, or fastEmbedR_knn",
      "fastEmbedR_embedding", "yes", "yes", "yes",
      "fixed-policy stochastic fuzzy or binary graph embedding",
      "backend intermediates may remain resident; final layout is host",
      "print, plot"
    ),
    row(
      "umap_knn", "canonical", "Run fixed-policy UMAP from reusable neighbors.",
      paste(
        "index/distance matrices, fastEmbedR_knn, fastEmbedR_gpu_knn,",
        "fastEmbedR_umap_prepared, or fastEmbedR_umap_initialization"
      ),
      "numeric or float32 layout matrix", "yes", "yes", "yes",
      "fixed-policy stochastic fuzzy or binary graph embedding",
      "resident CUDA KNN can be consumed on device; final layout is host"
    ),
    row(
      "select_landmarks", "canonical", "Select a reusable reference subset.",
      "matrix, data.frame, or float32", "fastEmbedR_landmark_selection",
      "yes", "host", "host", "deterministic seeded approximation",
      "host result"
    ),
    row(
      "fit_landmark_model", "canonical",
      "Fit a UMAP or t-SNE reference embedding.",
      "data matrix plus fastEmbedR_landmark_selection",
      "fastEmbedR_landmark_model", "yes", "yes", "yes",
      "reference-subset approximation",
      "backend intermediates may remain resident; model is host"
    ),
    row(
      "project_landmark_model", "canonical",
      "Project original or new rows into a fixed reference.",
      "fastEmbedR_landmark_model plus compatible data matrix",
      "fastEmbedR_embedding", "yes", "yes", "yes",
      "fixed-reference approximation",
      "CUDA transform may remain resident internally; final layout is host",
      "print, plot"
    ),
    row(
      "precompute_query_knn", "advanced",
      "Compute query-to-reference nearest neighbors.",
      "reference and query matrices/data frames/float32",
      "fastEmbedR_knn", "yes", "yes", "yes",
      "backend-dependent exact or approximate search",
      "CUDA result remains device-resident; CPU/Metal result is host",
      "print"
    ),
    row(
      "prepare_opentsne_knn", "advanced",
      "Normalize and cache t-SNE neighbor support.",
      "host KNN object or index/distance matrices",
      "fastEmbedR_opentsne_prepared", "yes", "yes", "yes",
      "exact preparation given supplied KNN", "host prepared object"
    ),
    row(
      "prepare_umap_knn", "advanced",
      "Build and cache a UMAP CSR graph and edge schedule.",
      "host KNN object or index/distance matrices",
      "fastEmbedR_umap_prepared", "yes", "yes", "yes",
      "exact graph construction given supplied KNN", "host prepared object"
    ),
    row(
      "umap_init", "advanced",
      "Build a reusable UMAP graph and initialization.",
      "host KNN, prepared UMAP object, or index/distance matrices",
      "fastEmbedR_umap_initialization", "yes", "yes", "yes",
      "diagnostic fixed-boundary initialization", "host diagnostic object",
      "print"
    ),
    row(
      "transform_tsne", "advanced",
      "Transform new observations into a fixed t-SNE layout.",
      "reference layout plus KNN, or reference/query data matrices",
      "numeric or float32 layout matrix", "yes", "yes", "yes",
      "fixed-reference approximation",
      "backend intermediates may remain resident; final layout is host"
    ),
    row(
      "opentsne_pca_init", "advanced",
      "Create and optionally cache t-SNE-scaled PCA coordinates.",
      "matrix, data.frame, or float32", "numeric or float32 matrix", "yes",
      "yes", "yes", "randomized approximation",
      "accelerator intermediates; host R result"
    ),
    row(
      "landmark_tsne", "convenience",
      "Run selection, t-SNE reference fitting, and projection in one call.",
      "matrix, data.frame, or float32", "fastEmbedR_embedding", "yes",
      "yes", "yes", "landmark approximation",
      "backend intermediates may remain resident; final layout is host",
      "print, plot"
    ),
    row(
      "landmark_umap", "convenience",
      "Run selection, UMAP reference fitting, and projection in one call.",
      "matrix, data.frame, or float32", "fastEmbedR_embedding", "yes",
      "yes", "yes", "landmark approximation",
      "backend intermediates may remain resident; final layout is host",
      "print, plot"
    ),
    row(
      "evaluate_embedding", "diagnostic",
      "Compute embedding quality and preservation metrics.",
      "high-dimensional matrix, layout matrix, and optional labels/KNN",
      "one-row data.frame", "yes", "conditional", "conditional",
      "diagnostic", "host result"
    ),
    row(
      "knn_graph", "secondary",
      "Build a weighted graph from data, embedding, or KNN.",
      "matrix, fastEmbedR_embedding, fastEmbedR_knn, or fastEmbedR_graph",
      "fastEmbedR_graph", "yes", "yes", "yes",
      "exact graph construction after backend-routed KNN", "host graph",
      "print"
    ),
    row(
      "graph_cluster", "secondary",
      "Run Louvain, Leiden, or Walktrap community detection.",
      "fastEmbedR_graph or compatible weighted edge list",
      "fastEmbedR_graph_cluster", "yes", "Louvain/Leiden", "Louvain/Leiden",
      "heuristic Louvain/Leiden; exact Walktrap", "host result", "print"
    ),
    row(
      "embed_knn", "compatibility",
      "Dispatch a KNN object to UMAP or t-SNE.",
      "KNN, prepared UMAP/t-SNE object, or index/distance matrices",
      "numeric or float32 layout matrix", "yes", "yes", "yes",
      "compatibility dispatcher", "same as selected KNN embedding function",
      lifecycle = "stable compatibility; prefer umap_knn or opentsne_knn"
    )
  )

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
