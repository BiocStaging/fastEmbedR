silhouette_score <- function(labels, layout) {
    layout <- embedding_dense_double_matrix(layout)
    labels <- as.factor(labels)
    if (length(labels) != nrow(layout)) {
        stop("`labels` must have one entry per row of `layout`.", call. = FALSE)
    }
    silhouette_score_cpp(layout, as.integer(labels))
}

cuda_metric_requested <- function(backend) {
    identical(backend, "cuda") || identical(backend, "gpu")
}

cuda_metric_available <- function() {
    isTRUE(embedding_cuda_available_cpp())
}

metal_metric_requested <- function(backend) {
    identical(backend, "metal")
}

metal_metric_available <- function() {
    isTRUE(embedding_metal_available_cpp())
}

resolve_metric_backend <- function(backend) {
    backend <- as.character(backend)[1L]
    if (length(backend) != 1L || is.na(backend) || !nzchar(backend)) {
        backend <- "auto"
    }
    backend <- tolower(backend)
    if (backend %in% c("auto", "cpu")) {
        return(list(backend = "cpu", reason = NA_character_))
    }
    if (identical(backend, "gpu")) {
        selected <- available_native_gpu_backend(need_embedding = TRUE)
        if (identical(selected, "cuda") && cuda_metric_available()) {
            return(list(backend = selected, reason = NA_character_))
        }
        if (identical(selected, "metal")) {
            return(list(
                backend = "cpu",
                reason = "metal_knn_metric_backend_unavailable"
            ))
        }
        return(list(backend = "cpu", reason = "gpu_metric_backend_unavailable"))
    }
    if (identical(backend, "metal")) {
        return(list(
            backend = "cpu",
            reason = "metal_knn_metric_backend_unavailable"
        ))
    }
    if (identical(backend, "cuda")) {
        if (cuda_metric_available()) {
            return(list(backend = "cuda", reason = NA_character_))
        }
        return(list(
            backend = "cpu",
            reason = "cuda_metric_backend_unavailable"
        ))
    }
    list(backend = "cpu", reason = paste0(
        backend,
        "_metric_backend_not_supported"
    ))
}

structure_metric_limit_reason <- function(backend, layout, preserve_k) {
    if (backend == "cuda" && !cuda_metric_available()) {
        return("cuda_scoring_unavailable")
    }
    if (backend == "metal" && !metal_metric_available()) {
        return("metal_scoring_unavailable")
    }
    if (ncol(layout) != 2L) {
        return(paste0(backend, "_scoring_requires_2d_layout"))
    }
    if (preserve_k > 64L) {
        return(paste0(backend, "_scoring_supports_at_most_64_neighbors"))
    }
    NA_character_
}

run_structure_metric_backend <- function(
    backend, layout, indices, keep, preserve_k, labels_int, n_label_levels
) {
    reason <- structure_metric_limit_reason(backend, layout, preserve_k)
    if (!is.na(reason)) {
        return(list(value = NULL, reason = reason))
    }
    call <- if (backend == "cuda") {
        knn_structure_score_cuda_cpp
    } else {
        knn_structure_score_metal_cpp
    }
    attempt <- capture_error(call(
        layout, indices, keep, preserve_k, labels_int,
        as.integer(n_label_levels)
    ))
    list(value = attempt$value, reason = attempt$error)
}

structure_score_with_backend <- function(
    layout, indices, keep, preserve_k, labels_int, n_label_levels,
    backend = "cpu"
) {
    layout <- embedding_dense_double_matrix(layout)
    if (!is.matrix(indices)) indices <- as.matrix(indices)
    if (!is.integer(indices)) storage.mode(indices) <- "integer"
    keep <- as.integer(keep)
    labels_int <- if (length(labels_int) == 0L) {
        integer(0L)
    } else {
        as.integer(labels_int)
    }
    preserve_k <- as.integer(preserve_k)
    reason <- NA_character_
    requested <- if (cuda_metric_requested(backend)) {
        "cuda"
    } else if (metal_metric_requested(backend)) {
        "metal"
    } else {
        NA_character_
    }
    if (!is.na(requested)) {
        attempt <- run_structure_metric_backend(
            requested, layout, indices, keep, preserve_k, labels_int,
            n_label_levels
        )
        if (!is.null(attempt$value)) {
            return(list(
                values = attempt$value,
                backend = requested,
                reason = NA_character_
            ))
        }
        reason <- attempt$reason
    }
    out <- knn_structure_score_cpp(
        layout, indices, keep, preserve_k, labels_int,
        as.integer(n_label_levels)
    )
    list(values = out, backend = "cpu", reason = reason)
}

silhouette_metric_limit_reason <- function(backend, layout, n_levels) {
    if (backend == "cuda" && !cuda_metric_available()) {
        return("cuda_scoring_unavailable")
    }
    if (backend == "metal" && !metal_metric_available()) {
        return("metal_scoring_unavailable")
    }
    if (ncol(layout) != 2L) {
        return(paste0(backend, "_silhouette_requires_2d_layout"))
    }
    if (n_levels > 128L) {
        return(paste0(
            backend,
            "_silhouette_supports_at_most_128_label_levels"
        ))
    }
    NA_character_
}

run_silhouette_metric_backend <- function(
    backend, layout, labels_int, n_label_levels
) {
    reason <- silhouette_metric_limit_reason(
        backend,
        layout,
        n_label_levels
    )
    if (!is.na(reason)) {
        return(list(value = NULL, reason = reason))
    }
    call <- if (backend == "cuda") {
        silhouette_score_cuda_cpp
    } else {
        silhouette_score_metal_cpp
    }
    attempt <- capture_error(call(
        layout,
        labels_int,
        as.integer(n_label_levels)
    ))
    list(value = attempt$value, reason = attempt$error)
}

silhouette_score_with_backend <- function(
    labels_int, layout, n_label_levels, backend = "cpu"
) {
    if (length(labels_int) == 0L || nrow(layout) < 2L || n_label_levels < 2L) {
        return(list(value = NA_real_, backend = "none", reason = NA_character_))
    }
    layout <- embedding_dense_double_matrix(layout)
    labels_int <- as.integer(labels_int)
    reason <- NA_character_
    requested <- if (cuda_metric_requested(backend)) {
        "cuda"
    } else if (metal_metric_requested(backend)) {
        "metal"
    } else {
        NA_character_
    }
    if (!is.na(requested)) {
        attempt <- run_silhouette_metric_backend(
            requested,
            layout,
            labels_int,
            n_label_levels
        )
        if (!is.null(attempt$value)) {
            return(list(
                value = attempt$value,
                backend = requested,
                reason = NA_character_
            ))
        }
        reason <- attempt$reason
    }
    list(
        value = silhouette_score_cpp(layout, labels_int),
        backend = "cpu",
        reason = reason
    )
}

sample_indices <- function(n, sample_size = NULL, seed = 4L) {
    if (is.null(sample_size)) {
        return(integer(0L))
    }
    sample_size <- as.integer(sample_size)
    invalid_sample_size <- length(sample_size) != 1L ||
        is.na(sample_size) ||
        !is.finite(sample_size) ||
        sample_size < 1L
    if (invalid_sample_size) {
        return(integer(0L))
    }
    if (sample_size >= n) {
        return(seq_len(n))
    }
    restore_seed <- set_local_seed(seed)
    on.exit(restore_seed(), add = TRUE)
    sort(sample.int(n, sample_size))
}
