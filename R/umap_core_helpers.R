# Internal orchestration helpers for KNN-input UMAP.

# Prepared objects bypass graph reconstruction and retain their stored layout.

dispatch_prepared_umap_input <- function(indices,
                                         distances,
                                         n_components,
                                         seed,
                                         verbose,
                                         backend,
                                         n_threads,
                                         n_epochs) {
  if (inherits(indices, "fastEmbedR_umap_initialization")) {
    if (!is.null(distances)) {
      stop(
        "Do not pass `distances` when `indices` is a UMAP initialization object.",
        call. = FALSE
      )
    }
    prepared <- indices$prepared
    prepared$initialization <- indices$layout
    prepared$initialization_parameters <- indices$parameters
    layout <- fast_knn_umap_prepared_core(
      prepared,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend,
      n_threads = n_threads,
      n_epochs = n_epochs
    )
    return(list(handled = TRUE, layout = layout))
  }

  if (inherits(indices, "fastEmbedR_umap_prepared")) {
    if (!is.null(distances)) {
      stop(
        "Do not pass `distances` when `indices` is a prepared UMAP object.",
        call. = FALSE
      )
    }
    layout <- fast_knn_umap_prepared_core(
      indices,
      n_components = n_components,
      seed = seed,
      verbose = verbose,
      backend = backend,
      n_threads = n_threads,
      n_epochs = n_epochs
    )
    return(list(handled = TRUE, layout = layout))
  }

  list(handled = FALSE, layout = NULL)
}

# Parameter profiling is isolated from graph construction and optimization.
profile_umap_knn_parameters <- function(cfg, indices, distances, knn) {
  auto_policy <- tryCatch(
    {
      use_range <- knn$col_start != 0L ||
        knn$n_neighbors != ncol(distances)
      policy_distances <- if (use_range) {
        materialize_knn_range(
          indices,
          distances,
          knn$col_start,
          knn$n_neighbors
        )$distances
      } else {
        distances
      }
      if (is_float32_matrix(policy_distances)) {
        stop(
          "float32 KNN auto-policy uses default parameters",
          call. = FALSE
        )
      }
      umap_auto_parameters_cpp(
        policy_distances,
        as.integer(knn$n_neighbors),
        as.character(cfg$backend)
      )
    },
    error = function(e) list(error = conditionMessage(e))
  )

  if (!is.null(auto_policy$error)) {
    cfg$auto_parameter_backend <- "r_size_rule_fallback"
    cfg$auto_parameter_error <- auto_policy$error
    return(cfg)
  }

  cfg$n_epochs <- as.integer(auto_policy$n_epochs)
  cfg$min_dist <- as.numeric(auto_policy$min_dist)
  cfg$negative_sample_rate <- as.integer(auto_policy$negative_sample_rate)
  cfg$learning_rate <- as.numeric(auto_policy$learning_rate)
  cfg$spectral_n_iter <- as.integer(auto_policy$spectral_n_iter)
  cfg$init_scale <- as.numeric(auto_policy$init_scale)
  cfg$auto_parameter_backend <- "cpp_knn_distance_profile"
  cfg$auto_parameter_rule <- as.character(auto_policy$rule)
  cfg$knn_distance_cv <- as.numeric(auto_policy$knn_distance_cv)
  cfg$knn_distance_ratio_30_15 <- as.numeric(
    auto_policy$knn_distance_ratio_30_15
  )
  cfg$knn_distance_ratio_50_15 <- as.numeric(
    auto_policy$knn_distance_ratio_50_15
  )
  cfg
}

apply_umap_thread_override <- function(cfg,
                                       n_threads,
                                       argument = "n_threads") {
  if (is.null(n_threads)) {
    return(cfg)
  }
  n_threads <- as.integer(n_threads)
  invalid <- length(n_threads) != 1L ||
    is.na(n_threads) ||
    !is.finite(n_threads) ||
    n_threads < 1L
  if (invalid) {
    stop(
      sprintf("`%s` must be NULL or a positive integer.", argument),
      call. = FALSE
    )
  }
  cfg$n_threads <- as.integer(max(1L, min(4L, n_threads)))
  cfg
}

configure_host_knn_umap <- function(indices,
                                    distances,
                                    knn,
                                    backend,
                                    n_threads,
                                    n_epochs,
                                    config_override,
                                    graph_mode,
                                    seed) {
  cfg <- fast_knn_umap_config(
    n = nrow(indices),
    k = knn$n_neighbors,
    backend = backend
  )
  cfg <- profile_umap_knn_parameters(cfg, indices, distances, knn)
  cfg <- apply_umap_thread_override(cfg, n_threads)
  cfg$input_had_self <- isTRUE(knn$has_self)
  cfg$knn_col_start <- as.integer(knn$col_start)
  cfg$knn_n_neighbors <- as.integer(knn$n_neighbors)
  cfg$knn_materialized <- isTRUE(knn$materialized)
  cfg$knn_backend <- knn$input_backend
  cfg$graph_mode <- graph_mode
  cfg$sgd_loop <- "csr_float32_contiguous_inplace"

  if (!is.null(n_epochs)) {
    cfg$n_epochs <- validate_epoch_count(n_epochs)
    cfg$preset <- "internal_epoch_override"
    cfg$epoch_source <- "internal_override"
  }

  cfg <- apply_umap_connectivity_spectral_rule(
    cfg,
    indices,
    col_start = knn$col_start,
    n_neighbors = knn$n_neighbors
  )

  should_pilot <- fast_knn_umap_should_auto_pilot(
    cfg = cfg,
    indices = indices,
    config_override = config_override,
    n_epochs = n_epochs
  )
  if (should_pilot) {
    pilot <- tryCatch(
      auto_umap_knn_pilot_tune(
        indices,
        distances,
        seed = seed,
        full_n = nrow(indices),
        pilot_min_n = fast_knn_umap_auto_pilot_min_n(),
        pilot_max_n = fast_knn_umap_auto_pilot_max_n(),
        pilot_max_configs = fast_knn_umap_auto_pilot_max_configs(),
        use_cache = fast_knn_umap_auto_pilot_use_cache(),
        cache_dir = getOption("fastEmbedR.knn_pilot_cache_dir", NULL),
        force_recompute = isTRUE(
          getOption("fastEmbedR.knn_pilot_force_recompute", FALSE)
        )
      ),
      error = function(e) {
        list(status = "failed", reason = conditionMessage(e))
      }
    )
    cfg$auto_knn_pilot_status <- pilot$status
    cfg$auto_knn_pilot_reason <- pilot$reason
    if (identical(pilot$status, "success") &&
        !is.null(pilot$config_override)) {
      cfg <- apply_fast_knn_umap_config_override(
        cfg,
        pilot$config_override
      )
    }
  } else {
    cfg$auto_knn_pilot_status <- "skipped"
    cfg$auto_knn_pilot_reason <- fast_knn_umap_auto_pilot_skip_reason(
      cfg = cfg,
      indices = indices,
      config_override = config_override,
      n_epochs = n_epochs
    )
  }

  apply_fast_knn_umap_config_override(cfg, config_override)
}

# GPU host-input paths materialize the selected KNN range exactly once.
materialize_umap_gpu_knn <- function(indices, distances, knn, cfg) {
  needs_materialization <- knn$col_start != 0L ||
    knn$n_neighbors != ncol(indices)
  if (!needs_materialization) {
    cfg$knn_materialized_for_gpu <- FALSE
    return(list(indices = indices, distances = distances, cfg = cfg))
  }

  materialized <- materialize_knn_range(
    indices,
    distances,
    knn$col_start,
    knn$n_neighbors
  )
  cfg$knn_materialized_for_gpu <- TRUE
  cfg$knn_col_start <- 0L
  cfg$knn_n_neighbors <- ncol(materialized$indices)
  list(
    indices = materialized$indices,
    distances = materialized$distances,
    cfg = cfg
  )
}

initialize_umap_gpu_host_knn <- function(indices,
                                         distances,
                                         cfg,
                                         graph_mode,
                                         seed) {
  if (identical(graph_mode, "binary")) {
    cfg$init_backend <- "pending_binary_csr"
    cfg$init_backend_reason <- paste(
      "Binary UMAP initializes from the same unit-weight CSR graph used",
      "by the optimizer."
    )
    return(list(init = NULL, cfg = cfg))
  }

  init_distances <- if (is_float32_matrix(distances)) {
    matrix(
      as.numeric(distances),
      nrow = nrow(indices),
      ncol = ncol(indices)
    )
  } else {
    distances
  }
  init <- spectral_knn_init(
    indices,
    init_distances,
    n_components = 2L,
    min_dist = cfg$min_dist,
    spectral_n_iter = cfg$spectral_n_iter,
    seed = seed,
    backend = "cpu",
    n_threads = cfg$n_threads
  )
  init <- scale_embedding_sdev_r(init, cfg$init_scale)
  cfg$init_backend <- attr(init, "backend")
  cfg$init_backend_reason <- paste(
    "CPU initialization avoids a separate GPU initialization round trip;",
    "the native optimizer uploads KNN and initialization once."
  )
  list(init = init, cfg = cfg)
}

# CSR graph construction and initialization are shared by all host-KNN paths.
build_umap_csr_state <- function(indices,
                                 distances,
                                 col_start,
                                 n_neighbors,
                                 cfg,
                                 graph_mode,
                                 n_components,
                                 seed,
                                 init = NULL) {
  graph <- umap_build_csr_graph(
    indices,
    distances,
    as.integer(col_start),
    as.integer(n_neighbors),
    as.integer(n_neighbors),
    as.integer(cfg$n_threads),
    graph_mode = graph_mode
  )
  if (is.null(init)) {
    init <- umap_init_from_csr_graph(
      graph,
      n_components = n_components,
      cfg = cfg,
      seed = seed,
      verbose = FALSE
    )
    cfg$init_backend <- attr(init, "backend")
  }
  cfg$graph_nnz <- as.integer(graph$nnz)
  cfg$graph_max_weight <- as.numeric(graph$max_weight)
  cfg$graph_cuda_like_width <- graph$cuda_like_width
  cfg$graph_builder <- graph$graph_builder
  list(graph = graph, init = init, cfg = cfg)
}

# Backend dispatch is intentionally separate from graph and policy setup.
optimize_umap_gpu_csr <- function(graph, init, cfg, backend, seed) {
  if (identical(backend, "metal")) {
    out <- knn_embed_metal_csr_cpp(
      graph$offsets,
      graph$neighbors,
      graph$weights,
      init,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$min_dist,
      as.numeric(graph$max_weight),
      cfg$repulsion_strength,
      as.integer(seed),
      1L
    )
    cfg$metal_graph_input <- attr(out, "metal_graph_input")
    cfg$metal_csr_width <- attr(out, "metal_csr_width")
    cfg$metal_truncated_edges <- attr(out, "metal_truncated_edges")
    return(list(layout = out, cfg = cfg))
  }

  if (identical(backend, "cuda")) {
    out <- umap_cuda_optimize_csr_cpp(
      graph$offsets,
      graph$neighbors,
      graph$weights,
      graph$epochs_per_sample,
      init,
      as.integer(cfg$n_epochs),
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$min_dist,
      cfg$repulsion_strength,
      as.integer(seed),
      0L
    )
    return(list(layout = out, cfg = cfg))
  }

  stop("Unsupported native GPU UMAP backend.", call. = FALSE)
}

run_umap_gpu_host_knn <- function(indices,
                                  distances,
                                  knn,
                                  cfg,
                                  n_components,
                                  graph_mode,
                                  seed) {
  if (n_components != 2L) {
    stop(
      "Native GPU embedding backends currently support only `n_components = 2`.",
      call. = FALSE
    )
  }

  materialized <- materialize_umap_gpu_knn(
    indices,
    distances,
    knn,
    cfg
  )
  indices <- materialized$indices
  distances <- materialized$distances
  cfg <- materialized$cfg
  gpu_backend <- cfg$backend

  initialized <- initialize_umap_gpu_host_knn(
    indices,
    distances,
    cfg,
    graph_mode,
    seed
  )
  init <- initialized$init
  cfg <- initialized$cfg
  cfg$graph_prep_backend <- if (identical(graph_mode, "binary")) {
    "cpu_binary_csr"
  } else {
    "cpu_fuzzy_csr"
  }
  cfg$graph_storage <- if (identical(gpu_backend, "cuda")) {
    "cpu_csr_uploaded_to_cuda"
  } else {
    "cpu_csr_packed_to_metal"
  }
  cfg$gpu_transfer_policy <- "single_upload_optimizer"
  cfg$gpu_optimizer_mode <- if (identical(gpu_backend, "metal")) {
    "clean_linear_atomic_inplace"
  } else {
    "atomic_coo"
  }
  cfg$gpu_optimizer_update_rule <- if (identical(gpu_backend, "metal")) {
    "native_metal_csr_clean_linear_atomic_inplace_edge_update"
  } else {
    "native_cuda_atomic_coo_umap_schedule"
  }
  cfg$gpu_optimizer_schedule <- if (identical(gpu_backend, "metal")) {
    "clean_edge_probability_linear_decay"
  } else {
    "coo_epochs_per_sample"
  }
  cfg$gpu_epochs_per_command <- if (identical(gpu_backend, "metal")) {
    64L
  } else {
    NA_integer_
  }
  cfg$gpu_initial_backend <- gpu_backend
  cfg$optimizer_backend <- gpu_backend
  cfg <- add_gpu_transfer_metadata(
    cfg,
    indices,
    distances,
    init = init,
    n = nrow(indices),
    n_components = 2L,
    objective = "umap"
  )
  cfg$backend <- cfg$optimizer_backend
  cfg$gpu_umap_path <- if (identical(gpu_backend, "cuda")) {
    "cuda_fuzzy_graph_atomic"
  } else {
    "metal_atomic_inplace"
  }
  cfg$gpu_initial_epochs <- as.integer(cfg$n_epochs)

  state <- build_umap_csr_state(
    indices,
    distances,
    col_start = 0L,
    n_neighbors = ncol(indices),
    cfg = cfg,
    graph_mode = graph_mode,
    n_components = 2L,
    seed = seed,
    init = init
  )
  optimized <- optimize_umap_gpu_csr(
    state$graph,
    state$init,
    state$cfg,
    gpu_backend,
    seed
  )
  layout <- finalize_embedding_layout(
    optimized$layout,
    "UMAP",
    return_float32 = is_float32_matrix(distances)
  )
  attr(layout, "fastEmbedR_config") <- public_core_config(optimized$cfg)
  layout
}

run_umap_cpu_host_knn <- function(indices,
                                  distances,
                                  knn,
                                  cfg,
                                  n_components,
                                  graph_mode,
                                  seed,
                                  verbose) {
  cfg$graph_prep_backend <- "cpu"
  if (n_components != 2L) {
    cfg <- add_gpu_transfer_metadata(
      cfg,
      indices,
      distances,
      n = nrow(indices),
      n_components = n_components,
      objective = "umap"
    )
    cfg$init_backend <- "cpu"
    cfg$optimizer_mode <- "legacy_range_optimizer_non_2d"
    layout <- fast_knn_umap_range_cpp(
      indices,
      distances,
      as.integer(knn$col_start),
      as.integer(knn$n_neighbors),
      n_components,
      as.integer(cfg$n_epochs),
      cfg$min_dist,
      as.integer(cfg$negative_sample_rate),
      cfg$learning_rate,
      cfg$repulsion_strength,
      as.integer(cfg$spectral_n_iter),
      as.integer(cfg$n_threads),
      cfg$init_scale,
      as.integer(seed),
      isTRUE(verbose)
    )
    layout <- finalize_embedding_layout(
      layout,
      "UMAP",
      return_float32 = is_float32_matrix(distances)
    )
    attr(layout, "fastEmbedR_config") <- public_core_config(cfg)
    return(layout)
  }

  state <- build_umap_csr_state(
    indices,
    distances,
    col_start = knn$col_start,
    n_neighbors = knn$n_neighbors,
    cfg = cfg,
    graph_mode = graph_mode,
    n_components = n_components,
    seed = seed
  )
  cfg <- state$cfg
  init_backend <- cfg$init_backend
  cfg$init_backend <- NULL
  cfg$init_backend <- init_backend
  cfg$graph_prep_backend <- if (identical(graph_mode, "binary")) {
    "cpu_binary_csr"
  } else {
    "cpu_fuzzy_csr"
  }
  cfg$graph_storage <- cfg$graph_prep_backend
  cfg$optimizer_mode <- "csr_epoch_schedule"
  cfg$optimizer_schedule <- if (identical(graph_mode, "binary")) {
    "epochs_per_sample_binary_csr"
  } else {
    "epochs_per_sample_fuzzy_csr"
  }
  layout <- fast_knn_umap_csr_init_cpp(
    state$graph$offsets,
    state$graph$neighbors,
    state$graph$weights,
    state$init,
    as.integer(cfg$n_epochs),
    cfg$min_dist,
    as.integer(cfg$negative_sample_rate),
    cfg$learning_rate,
    cfg$repulsion_strength,
    as.integer(cfg$n_threads),
    as.integer(seed),
    isTRUE(verbose)
  )
  layout <- finalize_embedding_layout(
    layout,
    "UMAP",
    return_float32 = is_float32_matrix(distances)
  )
  attr(layout, "fastEmbedR_config") <- public_core_config(cfg)
  layout
}
