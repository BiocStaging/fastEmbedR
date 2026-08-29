gpu_transfer_value <- function(value, default) {
    if (is.null(value) || length(value) == 0L || is.na(value[[
        1L
    ]])) {
        default
    } else {
        value
    }
}

gpu_transfer_matrix_bytes <- function(x, bytes_per_value) {
    if (is.null(x)) {
        return(0)
    }
    dims <- dim(x)
    n_values <- if (is.null(dims)) length(x) else prod(dims)
    as.numeric(n_values) * as.numeric(bytes_per_value)
}

gpu_transfer_layout_bytes <- function(n, n_components, bytes_per_value = 4L) {
    as.numeric(n) * as.numeric(n_components) * as.numeric(bytes_per_value)
}

gpu_transfer_dimensions <- function(indices, init, n, n_components) {
    if (is.null(n)) {
        n <- if (!is.null(indices)) {
            nrow(indices)
        } else if (!is.null(init)) {
            nrow(init)
        } else {
            NA_integer_
        }
    }
    if (is.null(n_components)) {
        n_components <- if (!is.null(init)) ncol(init) else 2L
    }
    list(n = as.integer(n), n_components = as.integer(n_components))
}

gpu_transfer_empty_plan <- function(backend) {
    list(
        gpu_transfer_policy = "not_applicable",
        gpu_transfer_backend = backend,
        gpu_transfer_host_to_device_count = 0L,
        gpu_transfer_device_to_host_count = 0L,
        gpu_transfer_host_to_device_bytes = 0,
        gpu_transfer_device_to_host_bytes = 0,
        gpu_transfer_host_to_device_mb = 0,
        gpu_transfer_device_to_host_mb = 0,
        gpu_transfer_knn_uploaded_once = NA,
        gpu_transfer_init_uploaded_once = NA,
        gpu_transfer_embedding_returned_only_at_end = NA,
        gpu_transfer_init_roundtrip = NA,
        gpu_transfer_graph_metadata_roundtrip = NA,
        gpu_transfer_graph_prepared_on_device = NA,
        gpu_transfer_note = paste0(
            "CPU or unsupported backend; no GPU transfer ",
            "accounting."
        )
    )
}

gpu_transfer_optimizer_bytes <- function(
    backend, indices, distances, init, n, n_components, init_backend
) {
    index_bytes <- gpu_transfer_matrix_bytes(indices, 4L)
    distance_bytes <- if (identical(backend, "cuda")) {
        gpu_transfer_matrix_bytes(distances, 8L)
    } else {
        # Metal graph weights are prepared as float buffers before upload.
        gpu_transfer_matrix_bytes(distances, 4L)
    }
    init_on_device <- identical(
        init_backend,
        paste0(backend, "_fused_spectral")
    )
    init_bytes <- if (isTRUE(init_on_device)) {
        0
    } else if (!is.null(init)) {
        gpu_transfer_matrix_bytes(init, 4L)
    } else {
        gpu_transfer_layout_bytes(n, n_components, 4L)
    }
    layout_bytes <- gpu_transfer_layout_bytes(n, n_components, 4L)
    h2d_count <- if (isTRUE(init_on_device)) 2L else 3L
    d2h_count <- 1L
    h2d_bytes <- index_bytes + distance_bytes + init_bytes
    d2h_bytes <- layout_bytes
    init_roundtrip <- identical(init_backend, backend)

    if (init_roundtrip) {
        h2d_count <- h2d_count + 2L
        d2h_count <- d2h_count + 1L
        h2d_bytes <- h2d_bytes + index_bytes + distance_bytes
        d2h_bytes <- d2h_bytes + init_bytes
    }
    list(
        init_on_device = init_on_device,
        init_roundtrip = init_roundtrip,
        h2d_count = h2d_count,
        d2h_count = d2h_count,
        h2d_bytes = h2d_bytes,
        d2h_bytes = d2h_bytes
    )
}

gpu_transfer_optimizer_note <- function(backend, objective, init_on_device) {
    if (identical(backend, "cuda")) {
        if (isTRUE(init_on_device) && identical(objective, "umap")) {
            paste(
                "CUDA fused UMAP uploads KNN once, computes spectral",
                "initialization",
                "and CSR graph on device, and returns only the final embedding."
            )
        } else if (identical(objective, "umap")) {
            paste(
                "CUDA uploads KNN and initialization once for optimization;",
                "CSR graph offsets and weights stay on device."
            )
        } else {
            paste(
                "CUDA uploads KNN and initialization once for optimization",
                "and returns only the final embedding."
            )
        }
    } else {
        paste(
            "Metal uses shared Metal buffers; graph preparation is CPU-side",
            "unless explicitly reported otherwise."
        )
    }
}

gpu_transfer_optimizer_plan <- function(
    backend, objective, init_backend, graph_prep_backend, bytes
) {
    graph_on_device <- identical(graph_prep_backend, backend) ||
        identical(graph_prep_backend, paste0(backend, "_exact")) ||
        startsWith(graph_prep_backend, paste0(backend, "_fused"))
    list(
        gpu_transfer_policy = "single_upload_optimizer",
        gpu_transfer_backend = backend,
        gpu_transfer_host_to_device_count = as.integer(bytes$h2d_count),
        gpu_transfer_device_to_host_count = as.integer(bytes$d2h_count),
        gpu_transfer_host_to_device_bytes = as.numeric(bytes$h2d_bytes),
        gpu_transfer_device_to_host_bytes = as.numeric(bytes$d2h_bytes),
        gpu_transfer_host_to_device_mb = as.numeric(bytes$h2d_bytes) / 1024^2,
        gpu_transfer_device_to_host_mb = as.numeric(bytes$d2h_bytes) / 1024^2,
        gpu_transfer_knn_uploaded_once = !bytes$init_roundtrip,
        gpu_transfer_init_uploaded_once = !isTRUE(bytes$init_on_device),
        gpu_transfer_init_computed_on_device = isTRUE(bytes$init_on_device),
        gpu_transfer_embedding_returned_only_at_end = TRUE,
        gpu_transfer_init_roundtrip = bytes$init_roundtrip,
        gpu_transfer_graph_metadata_roundtrip = FALSE,
        gpu_transfer_graph_prepared_on_device = graph_on_device,
        gpu_transfer_note = gpu_transfer_optimizer_note(
            backend,
            objective,
            bytes$init_on_device
        )
    )
}

gpu_transfer_plan_knn_optimizer <- function(
    backend, indices, distances, init = NULL, n = NULL,
    n_components = NULL, objective = "umap", init_backend = "cpu",
    graph_prep_backend = backend
) {
    backend <- as.character(gpu_transfer_value(backend, "cpu"))
    if (!backend %in% c("cuda", "metal")) {
        return(gpu_transfer_empty_plan(backend))
    }
    dimensions <- gpu_transfer_dimensions(indices, init, n, n_components)
    objective <- as.character(gpu_transfer_value(objective, "umap"))
    init_backend <- as.character(gpu_transfer_value(init_backend, "cpu"))
    graph_prep_backend <- as.character(
        gpu_transfer_value(graph_prep_backend, backend)
    )
    bytes <- gpu_transfer_optimizer_bytes(
        backend, indices, distances, init, dimensions$n,
        dimensions$n_components, init_backend
    )
    gpu_transfer_optimizer_plan(
        backend, objective, init_backend, graph_prep_backend, bytes
    )
}

add_gpu_transfer_metadata <- function(cfg,
                                        indices,
                                        distances,
                                        init = NULL,
                                        n = NULL,
                                        n_components = NULL,
                                        objective = "umap") {
    plan <- gpu_transfer_plan_knn_optimizer(
        backend = gpu_transfer_value(cfg$backend, "cpu"),
        indices = indices,
        distances = distances,
        init = init,
        n = n,
        n_components = n_components,
        objective = objective,
        init_backend = gpu_transfer_value(cfg$init_backend, "cpu"),
        graph_prep_backend = gpu_transfer_value(
            cfg$graph_prep_backend,
            gpu_transfer_value(cfg$backend, "cpu")
        )
    )
    utils::modifyList(cfg, plan)
}
