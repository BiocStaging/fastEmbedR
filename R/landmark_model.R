#' Select representative landmark observations
#'
#' `select_landmarks()` separates row indices into a representative landmark
#' reference and its complementary query set. It does not embed or project the
#' data.
#'
#' @param data Numeric matrix, data frame, or `float::float32` matrix.
#' @param landmarks `TRUE` for an automatic count, a fraction in `(0, 1)`, a
#'   landmark count, or explicit row indices.
#' @param seed Random seed used by projection-based landmark selection.
#' @param n.cores Number of CPU cores used for float32 selection features.
#' @return A `fastEmbedR_landmark_selection` object containing `indices` and
#'   `query_indices`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' selection <- select_landmarks(x, landmarks = 0.5, seed = 1)
#' selection$indices
#' @export
select_landmarks <- function(data,
                             landmarks = TRUE,
                             seed = 4L,
                             n.cores = NULL) {
  n_threads <- n.cores
  prepared <- prepare_embedding_data(
    data,
    standardize = FALSE,
    pca_dims = NULL,
    seed = seed,
    backend = "cpu"
  )
  x <- prepared$data
  indices <- resolve_landmarks(
    landmarks,
    x,
    seed,
    n_threads = n_threads
  )
  if (is.null(indices)) indices <- seq_len(nrow(x))
  query_indices <- setdiff(seq_len(nrow(x)), indices)
  method <- attr(indices, "selection_method", exact = TRUE)
  if (is.null(method)) {
    method <- if (length(indices) == nrow(x)) "all_rows" else "indices"
  }
  out <- list(
    indices = as.integer(indices),
    query_indices = as.integer(query_indices),
    n = as.integer(nrow(x)),
    p = as.integer(ncol(x)),
    n_landmarks = as.integer(length(indices)),
    landmark_fraction = length(indices) / nrow(x),
    method = method,
    seed = as.integer(seed)
  )
  class(out) <- c("fastEmbedR_landmark_selection", "list")
  out
}

normalize_landmark_selection <- function(selection, data) {
  n <- nrow(data)
  selection_method <- NULL
  selection_seed <- NA_integer_
  if (inherits(selection, "fastEmbedR_landmark_selection")) {
    if (!identical(as.integer(selection$n), as.integer(n))) {
      stop(
        "`selection` was created for a different number of rows.",
        call. = FALSE
      )
    }
    indices <- selection$indices
    selection_method <- selection$method
    selection_seed <- selection$seed
  } else if (is.numeric(selection)) {
    indices <- sort(unique(as.integer(selection)))
  } else {
    stop(
      "`selection` must be returned by select_landmarks() or contain row indices.",
      call. = FALSE
    )
  }
  if (length(indices) < 2L || anyNA(indices) ||
      any(indices < 1L) || any(indices > n)) {
    stop("Landmark indices must identify at least two valid rows.", call. = FALSE)
  }
  query_indices <- setdiff(seq_len(n), indices)
  method <- selection_method %||%
    attr(indices, "selection_method") %||% "indices"
  out <- list(
    indices = as.integer(indices),
    query_indices = as.integer(query_indices),
    n = as.integer(n),
    p = as.integer(ncol(data)),
    n_landmarks = as.integer(length(indices)),
    landmark_fraction = length(indices) / n,
    method = method,
    seed = as.integer(selection_seed %||% NA_integer_)
  )
  class(out) <- c("fastEmbedR_landmark_selection", "list")
  out
}

#' Fit an embedding model on selected landmarks
#'
#' `fit_landmark_model()` runs the ordinary production [umap()] or
#' [opentsne()] implementation on the selected reference rows. The returned
#' object stores the reference data and embedding needed to project the
#' complementary rows or genuinely new observations.
#'
#' @param data Data in the analysis space used for KNN. Preprocess once before
#'   calling this staged API when centering, scaling, or PCA is required.
#' @param selection A result from [select_landmarks()] or explicit landmark row
#'   indices.
#' @param method `"umap"` or `"opentsne"`.
#' @param n_neighbors UMAP neighborhood size. For openTSNE this is the
#'   precomputed KNN support width; `NULL` derives it from `perplexity`.
#' @param perplexity openTSNE perplexity.
#' @param n_components Embedding dimensionality.
#' @param metric KNN metric.
#' @param seed Random seed.
#' @param backend `"cpu"`, `"cuda"`, or `"metal"`.
#' @param n.cores Number of CPU cores.
#' @param graph_mode UMAP graph mode, passed unchanged to [umap()]. The
#'   standard `"fuzzy"` graph is the default.
#' @param keep_knn Retain reference KNN output.
#' @param verbose Print optimizer progress.
#' @param ... Additional optimizer arguments passed to [opentsne()].
#' @return A `fastEmbedR_landmark_model`.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' selection <- select_landmarks(x, 0.5, seed = 1)
#' model <- fit_landmark_model(
#'   x, selection, method = "umap", n_neighbors = 10,
#'   graph_mode = "fuzzy", seed = 1
#' )
#' @export
fit_landmark_model <- function(data,
                               selection,
                               method = c("umap", "opentsne"),
                               n_neighbors = NULL,
                               perplexity = NULL,
                               n_components = 2L,
                               metric = c(
                                 "euclidean", "cosine", "correlation",
                                 "inner_product"
                               ),
                               seed = 4L,
                               backend = NULL,
                               n.cores = NULL,
                               graph_mode = c("fuzzy", "binary"),
                               keep_knn = FALSE,
                               verbose = FALSE,
                               ...) {
  n_threads <- n.cores
  method <- match.arg(method)
  backend <- resolve_embedding_backend(backend)
  graph_mode <- match.arg(graph_mode)
  metric <- resolve_embedding_metric(metric, data)
  n_components <- validate_n_components(n_components)
  prepared <- prepare_embedding_data(
    data,
    standardize = FALSE,
    pca_dims = NULL,
    seed = seed,
    backend = backend
  )
  x <- prepared$data
  selection <- normalize_landmark_selection(selection, x)
  reference_data <- if (length(selection$query_indices) == 0L) {
    x
  } else if (is_float32_matrix(x)) {
    split_float32_rows_cpp(
      x,
      selection$indices,
      selection$query_indices,
      as.integer(normalize_nn_threads(n_threads))
    )$landmarks
  } else {
    x[selection$indices, , drop = FALSE]
  }
  n_reference <- nrow(reference_data)
  if (n_reference < 2L) {
    stop("At least two landmark rows are required.", call. = FALSE)
  }

  elapsed <- system.time({
    if (identical(method, "umap")) {
      if (is.null(n_neighbors)) {
        n_neighbors <- auto_embedding_k(nrow(x), method = "umap")
      }
      n_neighbors <- as.integer(min(n_neighbors, n_reference - 1L))
      reference_knn <- precompute_knn(
        reference_data,
        k = n_neighbors,
        metric = metric,
        backend = backend,
        n.cores = n_threads
      )
      fit <- umap(
        reference_data,
        n_neighbors = n_neighbors,
        n_components = n_components,
        standardize = FALSE,
        metric = metric,
        nn = reference_knn,
        seed = seed,
        backend = backend,
        n.cores = n_threads,
        keep_knn = keep_knn,
        graph_mode = graph_mode,
        verbose = verbose
      )
    } else {
      if (is.null(n_neighbors)) {
        n_neighbors <- opentsne_neighbor_policy(
          nrow(x),
          perplexity = perplexity
        )$n_neighbors
      }
      n_neighbors <- as.integer(min(n_neighbors, n_reference - 1L))
      reference_knn <- precompute_knn(
        reference_data,
        k = n_neighbors,
        metric = metric,
        backend = backend,
        n.cores = n_threads
      )
      fit <- opentsne(
        reference_data,
        perplexity = perplexity,
        n_components = n_components,
        init_data = reference_data,
        standardize = FALSE,
        metric = metric,
        nn = reference_knn,
        seed = seed,
        backend = backend,
        keep_knn = keep_knn,
        verbose = verbose,
        n.cores = n_threads,
        ...
      )
      perplexity <- fit$parameters$perplexity %||% perplexity
    }
  })

  out <- list(
    method = method,
    fit = fit,
    reference_data = reference_data,
    selection = selection,
    n_total = as.integer(nrow(x)),
    p = as.integer(ncol(x)),
    n_components = as.integer(n_components),
    n_neighbors = as.integer(n_neighbors),
    perplexity = perplexity,
    metric = metric,
    graph_mode = if (identical(method, "umap")) graph_mode else NA_character_,
    backend = backend,
    seed = as.integer(seed),
    n.cores = normalize_nn_threads(n_threads),
    elapsed_sec = unname(elapsed[["elapsed"]])
  )
  class(out) <- c("fastEmbedR_landmark_model", "list")
  out
}

landmark_query_data <- function(model, data, query_indices = NULL) {
  prepared <- prepare_embedding_data(
    data,
    standardize = FALSE,
    pca_dims = NULL,
    seed = model$seed,
    backend = model$backend
  )
  x <- prepared$data
  full_input <- is.null(query_indices) && nrow(x) == model$n_total
  if (full_input) query_indices <- model$selection$query_indices
  if (is.null(query_indices)) {
    query <- x
    query_indices <- seq_len(nrow(x))
  } else {
    query_indices <- as.integer(query_indices)
    if (anyNA(query_indices) || any(query_indices < 1L) ||
        any(query_indices > nrow(x))) {
      stop("`query_indices` contains invalid rows.", call. = FALSE)
    }
    query <- if (length(query_indices) == nrow(x)) {
      x
    } else if (is_float32_matrix(x)) {
      split_float32_rows_cpp(
        x,
        query_indices,
        setdiff(seq_len(nrow(x)), query_indices),
        as.integer(model$n.cores)
      )$landmarks
    } else {
      x[query_indices, , drop = FALSE]
    }
  }
  if (ncol(query) != ncol(model$reference_data)) {
    stop("Query data are not in the model's reference feature space.", call. = FALSE)
  }
  list(
    data = query,
    indices = query_indices,
    full_input = full_input,
    input = x
  )
}

landmark_model_projection_k <- function(model,
                                        transform_k,
                                        transform_perplexity = 5) {
  if (is.null(transform_k)) {
    transform_k <- if (identical(model$method, "umap")) {
      model$n_neighbors
    } else {
      max(25L, ceiling(3 * transform_perplexity))
    }
  }
  transform_embedding_k(
    min(as.integer(transform_k), nrow(model$reference_data)),
    nrow(model$reference_data)
  )
}

project_landmark_umap <- function(model,
                                  query,
                                  projection_knn,
                                  refinement_epochs,
                                  n_threads,
                                  verbose) {
  n_reference <- nrow(model$reference_data)
  n_query <- nrow(query)
  reference_layout <- model$fit$layout
  params <- model$fit$parameters
  min_dist <- as.numeric(params$min_dist %||% 0.01)
  negative_sample_rate <- as.integer(params$negative_sample_rate %||% 5L)
  learning_rate <- as.numeric(params$learning_rate %||% 1)
  repulsion_strength <- as.numeric(params$repulsion_strength %||% 1)

  if (identical(model$backend, "cuda") &&
      fastembedr_is_gpu_knn(projection_knn)) {
    result <- landmark_umap_project_refine_cuda_gpu_cpp(
      projection_knn,
      model$reference_data,
      query,
      reference_layout,
      seq_len(n_reference),
      n_reference + seq_len(n_query),
      n_reference + n_query,
      as.integer(refinement_epochs),
      min_dist,
      negative_sample_rate,
      learning_rate,
      repulsion_strength,
      as.integer(model$seed + 2003L),
      12L,
      1e-3,
      2.5
    )
    return(list(
      layout = result$layout[n_reference + seq_len(n_query), , drop = FALSE],
      backend = "cuda"
    ))
  }

  affine <- landmark_affine_projection(
    model$reference_data,
    query,
    reference_layout,
    projection_knn,
    n_threads = n_threads,
    backend = model$backend
  )
  if (!embedding_layout_dims_match(
    affine, n_query, ncol(reference_layout)
  )) {
    affine <- project_embedding_knn_cpp(
      reference_layout,
      projection_knn$indices,
      projection_knn$distances
    )
  }
  if (refinement_epochs <= 0L) {
    return(list(layout = affine, backend = attr(affine, "projection_backend") %||% "cpu"))
  }
  combined <- rbind(
    embedding_dense_double_matrix(reference_layout),
    embedding_dense_double_matrix(affine)
  )
  query_rows <- n_reference + seq_len(n_query)
  if (identical(model$backend, "metal")) {
    combined <- knn_umap_refine_rows_metal_cpp(
      projection_knn$indices,
      projection_knn$distances,
      as.integer(query_rows),
      combined,
      as.integer(refinement_epochs),
      min_dist,
      negative_sample_rate,
      learning_rate,
      repulsion_strength,
      as.integer(model$seed + 2003L)
    )
    used_backend <- "metal"
  } else {
    combined <- knn_umap_refine_rows_cpp(
      projection_knn$indices,
      projection_knn$distances,
      as.integer(query_rows),
      combined,
      as.integer(refinement_epochs),
      min_dist,
      negative_sample_rate,
      learning_rate,
      repulsion_strength,
      as.integer(normalize_nn_threads(n_threads)),
      as.integer(model$seed + 2003L),
      isTRUE(verbose)
    )
    used_backend <- "cpu"
  }
  combined[seq_len(n_reference), ] <-
    embedding_dense_double_matrix(reference_layout)
  list(
    layout = combined[query_rows, , drop = FALSE],
    backend = used_backend
  )
}

project_landmark_tsne <- function(model,
                                  query,
                                  projection_knn,
                                  transform_perplexity,
                                  transform_iter,
                                  transform_early_exaggeration_iter,
                                  transform_n_negatives,
                                  initialization,
                                  n_threads,
                                  verbose,
                                  dots) {
  n_reference <- nrow(model$reference_data)
  reference_layout <- model$fit$layout
  exact_threshold <- as.integer(dots$exact_repulsion_threshold %||% 4096L)
  n_negatives <- transform_n_negatives %||% if (
    n_reference <= exact_threshold
  ) n_reference else min(256L, n_reference)
  learning_rate <- as.numeric(dots$transform_learning_rate %||% 0.1)
  early_exaggeration <- as.numeric(dots$transform_early_exaggeration %||% 4)
  exaggeration <- as.numeric(dots$transform_exaggeration %||% 1.5)
  initial_momentum <- as.numeric(dots$transform_initial_momentum %||% 0.8)
  final_momentum <- as.numeric(dots$transform_final_momentum %||% 0.8)
  max_grad_norm <- as.numeric(dots$transform_max_grad_norm %||% 0.25)
  max_step_norm <- as.numeric(dots$transform_max_step_norm %||% Inf)

  if (identical(model$backend, "cuda") &&
      fastembedr_is_gpu_knn(projection_knn)) {
    result <- landmark_tsne_transform_cuda_gpu_cpp(
      projection_knn,
      model$reference_data,
      query,
      reference_layout,
      as.numeric(transform_perplexity),
      as.integer(transform_iter),
      as.integer(transform_early_exaggeration_iter),
      learning_rate,
      early_exaggeration,
      exaggeration,
      initial_momentum,
      final_momentum,
      max_grad_norm,
      max_step_norm,
      as.integer(n_negatives),
      exact_threshold,
      as.integer(model$seed + 1009L),
      12L,
      1e-3,
      2.5
    )
    return(list(layout = result$Y, backend = "cuda"))
  }

  initial <- landmark_affine_projection(
    model$reference_data,
    query,
    reference_layout,
    projection_knn,
    n_threads = n_threads,
    backend = model$backend
  )
  if (!embedding_layout_dims_match(
    initial, nrow(query), ncol(reference_layout)
  )) {
    initial <- NULL
  }
  layout <- transform_tsne(
    reference_layout,
    knn = projection_knn,
    perplexity = transform_perplexity,
    initialization = initialization,
    Y_init = initial,
    n_iter = transform_iter,
    early_exaggeration_iter = transform_early_exaggeration_iter,
    learning_rate = learning_rate,
    early_exaggeration = early_exaggeration,
    exaggeration = exaggeration,
    initial_momentum = initial_momentum,
    final_momentum = final_momentum,
    max_grad_norm = max_grad_norm,
    max_step_norm = max_step_norm,
    n_negatives = n_negatives,
    exact_repulsion_threshold = exact_threshold,
    n.cores = n_threads,
    seed = model$seed + 1009L,
    backend = model$backend,
    verbose = verbose
  )
  list(layout = layout, backend = attr(layout, "backend") %||% model$backend)
}

#' Project observations into a fitted landmark embedding
#'
#' `project_landmark_model()` computes query-to-reference KNN with the same
#' native backend policy used by the full embedding functions, then applies the
#' method-specific fixed-reference transform. Landmark coordinates remain
#' fixed.
#'
#' @param model A model returned by [fit_landmark_model()].
#' @param data The original full data matrix, or a matrix containing only new
#'   query observations in the same feature space.
#' @param query_indices Optional rows of `data` to project.
#' @param transform_k Number of reference neighbors.
#' @param refinement_epochs Fixed-reference UMAP refinement epochs.
#' @param transform_perplexity Perplexity of the openTSNE transform.
#' @param transform_iter Number of openTSNE transform iterations.
#' @param transform_early_exaggeration_iter Transform exaggeration iterations.
#' @param transform_n_negatives Number of reference negatives for openTSNE.
#' @param initialization openTSNE query initialization.
#' @param keep_knn Retain query-to-reference KNN output.
#' @param n.cores Number of CPU cores.
#' @param verbose Print optimizer progress.
#' @param ... Low-level openTSNE transform controls.
#' @return A `fastEmbedR_embedding`. When `data` is the original full matrix,
#'   `layout` is reassembled in original row order. Otherwise it contains only
#'   the supplied query rows.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' selection <- select_landmarks(x, 0.5, seed = 1)
#' model <- fit_landmark_model(
#'   x, selection, method = "umap", n_neighbors = 10, seed = 1
#' )
#' fit <- project_landmark_model(model, x, refinement_epochs = 2)
#' @export
project_landmark_model <- function(model,
                                   data,
                                   query_indices = NULL,
                                   transform_k = NULL,
                                   refinement_epochs = 50L,
                                   transform_perplexity = 5,
                                   transform_iter = 250L,
                                   transform_early_exaggeration_iter = 0L,
                                   transform_n_negatives = NULL,
                                   initialization = c(
                                     "median", "weighted", "random"
                                   ),
                                   keep_knn = FALSE,
                                   n.cores = NULL,
                                   verbose = FALSE,
                                   ...) {
  if (!inherits(model, "fastEmbedR_landmark_model")) {
    stop("`model` must be returned by fit_landmark_model().", call. = FALSE)
  }
  if (is.null(query_indices) &&
      length(model$selection$query_indices) == 0L &&
      nrow(data) == model$n_total) {
    return(model$fit)
  }
  initialization <- match.arg(initialization)
  n_threads <- normalize_nn_threads(n.cores %||% model$n.cores)
  query_info <- landmark_query_data(model, data, query_indices)
  query <- query_info$data
  if (nrow(query) < 1L) {
    layout <- model$fit$layout
    return(model$fit)
  }
  transform_k <- landmark_model_projection_k(
    model,
    transform_k,
    transform_perplexity = transform_perplexity
  )
  knn_time <- system.time({
    projection_knn <- precompute_query_knn(
      model$reference_data,
      query,
      k = transform_k,
      metric = model$metric,
      backend = model$backend,
      n.cores = n_threads
    )
  })
  projection_time <- system.time({
    projected <- if (identical(model$method, "umap")) {
      refinement_epochs <- as.integer(refinement_epochs)
      if (length(refinement_epochs) != 1L || is.na(refinement_epochs) ||
          refinement_epochs < 0L) {
        stop("`refinement_epochs` must be non-negative.", call. = FALSE)
      }
      project_landmark_umap(
        model,
        query,
        projection_knn,
        refinement_epochs,
        n_threads,
        verbose
      )
    } else {
      project_landmark_tsne(
        model,
        query,
        projection_knn,
        transform_perplexity,
        as.integer(transform_iter),
        as.integer(transform_early_exaggeration_iter),
        transform_n_negatives,
        initialization,
        n_threads,
        verbose,
        list(...)
      )
    }
  })
  query_layout <- finalize_embedding_layout(
    projected$layout,
    if (identical(model$method, "umap")) "UMAP" else "openTSNE",
    return_float32 = is_float32_matrix(query)
  )
  if (query_info$full_input) {
    layout <- assemble_landmark_layout(
      model$fit$layout,
      query_layout,
      model$selection$indices,
      model$selection$query_indices,
      model$n_total,
      prefix = if (identical(model$method, "umap")) "UMAP" else "openTSNE",
      return_float32 = is_float32_matrix(query_info$input)
    )
  } else {
    layout <- query_layout
  }
  timings <- rbind(
    reference_embedding = model$fit$timings["embedding", , drop = FALSE],
    projection_knn = knn_time,
    projection_transform = projection_time
  )
  metrics <- data.frame(
    method = paste0("landmark_", model$method),
    n = nrow(layout),
    p = model$p,
    elapsed = model$elapsed_sec + knn_time[["elapsed"]] +
      projection_time[["elapsed"]],
    reference_embedding_elapsed = model$elapsed_sec,
    projection_knn_elapsed = knn_time[["elapsed"]],
    projection_transform_elapsed = projection_time[["elapsed"]],
    stringsAsFactors = FALSE
  )
  approximation <- attr(projection_knn, "approximation", exact = TRUE)
  parameters <- list(
    method = paste0("landmark_", model$method),
    backend = model$backend,
    metric = model$metric,
    n_neighbors = model$n_neighbors,
    perplexity = model$perplexity,
    graph_mode = model$graph_mode,
    seed = model$seed,
    landmark = TRUE,
    n_landmarks = nrow(model$reference_data),
    landmark_fraction = model$selection$landmark_fraction,
    landmark_selection = model$selection$method,
    transform_k = transform_k,
    projection_nn_backend =
      projection_knn$execution_backend %||%
      attr(projection_knn, "backend") %||% model$backend,
    projection_strategy =
      approximation$strategy %||%
      projection_knn$engine %||%
      projection_knn$method %||%
      attr(projection_knn, "method"),
    projection_backend = projected$backend,
    refinement_epochs = if (identical(model$method, "umap")) {
      as.integer(refinement_epochs)
    } else {
      NA_integer_
    },
    transform_iter = if (identical(model$method, "opentsne")) {
      as.integer(transform_iter)
    } else {
      NA_integer_
    }
  )
  out <- list(
    layout = layout,
    query_layout = query_layout,
    labels = NULL,
    method = paste0("landmark_", model$method),
    model = model,
    metrics = metrics,
    parameters = parameters,
    timings = timings,
    knn = if (isTRUE(keep_knn)) projection_knn else NULL,
    landmarks = list(
      indices = model$selection$indices,
      layout = model$fit$layout,
      reference_fit = model$fit,
      projection_knn = if (isTRUE(keep_knn)) projection_knn else NULL
    )
  )
  class(out) <- "fastEmbedR_embedding"
  out
}
