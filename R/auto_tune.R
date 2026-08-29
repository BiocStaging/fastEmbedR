auto_umap_optimizer_pilot_configs <- function(full_n,
                                                k,
                                                max_configs) {
    max_configs <- as.integer(max(1L, max_configs))
    base <- data.frame(
        k = as.integer(k),
        spectral_n_iter = auto_umap_default_pilot_spectral(k),
        n_epochs = auto_umap_pilot_quick_epochs(full_n),
        init_scale = auto_umap_default_pilot_init_scale(full_n),
        stringsAsFactors = FALSE
    )
    refined <- auto_umap_refined_pilot_configs(
        full_n = full_n,
        k = k,
        max_configs = max(0L, max_configs - 1L)
    )
    grid <- unique(rbind(base, refined))
    grid[seq_len(min(nrow(grid), max_configs)), , drop = FALSE]
}





auto_umap_pilot_cache_version <- function() {
    3L
}

auto_umap_pilot_cache_dir <- function(cache_dir = NULL) {
    if (!is.null(cache_dir) && nzchar(cache_dir)) {
        return(cache_dir)
    }
    opt <- getOption("fastEmbedR.pilot_cache_dir", NULL)
    if (!is.null(opt) && nzchar(opt)) {
        return(opt)
    }
    if ("R_user_dir" %in% getNamespaceExports("tools")) {
        return(file.path(tools::R_user_dir("fastEmbedR", "cache"), "pilot"))
    }
    file.path(tempdir(), "fastEmbedR_pilot_cache")
}

auto_umap_pilot_cache_key <- function(dataset_hash, k, seed) {
    paste0("hash", dataset_hash, "_k", as.integer(k), "_seed", as.integer(seed))
}

auto_umap_pilot_cache_path <- function(cache_dir,
                                        dataset_hash,
                                        k,
                                        seed,
                                        kind = "data") {
    key <- auto_umap_pilot_cache_key(dataset_hash, k, seed)
    file.path(cache_dir, paste0("umap_pilot_", kind, "_", key, ".rds"))
}

auto_umap_init_key <- function(init_scale) {
    init_scale <- as.numeric(init_scale)
    if (length(init_scale) != 1L || !is.finite(init_scale)) {
        return("NA")
    }
    format(signif(init_scale, 12L), scientific = FALSE, trim = TRUE)
}

auto_umap_read_pilot_cache_row <- function(cache_dir,
                                            dataset_hash,
                                            k,
                                            seed,
                                            spectral_n_iter,
                                            n_epochs,
                                            init_scale,
                                            kind = "data",
                                            use_cache = TRUE,
                                            force_recompute = FALSE) {
    if (!isTRUE(use_cache) || isTRUE(force_recompute)) {
        return(NULL)
    }
    path <- auto_umap_pilot_cache_path(cache_dir, dataset_hash, k, seed,
        kind = kind
    )
    if (!file.exists(path)) {
        return(NULL)
    }
    cached <- tryCatch(readRDS(path), error = function(e) NULL)
    if (!is.list(cached) || !identical(
        cached$version,
        auto_umap_pilot_cache_version()
    )) {
        return(NULL)
    }
    rows <- cached$rows
    if (!is.data.frame(rows) || nrow(rows) == 0L) {
        return(NULL)
    }
    init_key <- auto_umap_init_key(init_scale)
    row_init <- vapply(rows$init_scale, auto_umap_init_key, character(1L))
    keep <- rows$spectral_n_iter == as.integer(spectral_n_iter) &
        rows$n_epochs == as.integer(n_epochs) &
        row_init == init_key &
        rows$status == "success"
    if (!any(keep, na.rm = TRUE)) {
        return(NULL)
    }
    out <- rows[which(keep)[1L], , drop = FALSE]
    out$cache_hit <- TRUE
    out
}

auto_umap_write_pilot_cache_row <- function(
    row, cache_dir, dataset_hash, k, seed,
    kind = "data", use_cache = TRUE
) {
    if (!isTRUE(use_cache) || !is.data.frame(row) || nrow(row) != 1L) {
        return(invisible(FALSE))
    }
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    path <- auto_umap_pilot_cache_path(cache_dir, dataset_hash, k, seed,
        kind = kind
    )
    cached <- if (file.exists(path)) {
        tryCatch(readRDS(path), error = function(e) NULL)
    } else {
        NULL
    }
    rows <- if (is.list(cached) && identical(
        cached$version,
        auto_umap_pilot_cache_version()
    ) &&
        is.data.frame(cached$rows)) {
        cached$rows
    } else {
        row[0L, , drop = FALSE]
    }
    init_key <- auto_umap_init_key(row$init_scale)
    if (nrow(rows) > 0L) {
        row_init <- vapply(rows$init_scale, auto_umap_init_key, character(1L))
        keep <- !(rows$spectral_n_iter == row$spectral_n_iter &
            rows$n_epochs == row$n_epochs &
            row_init == init_key)
        rows <- rows[keep, , drop = FALSE]
    }
    row$cache_hit <- FALSE
    rows <- rbind(rows, row)
    saveRDS(
        list(
            version = auto_umap_pilot_cache_version(),
            dataset_hash = dataset_hash,
            k = as.integer(k),
            seed = as.integer(seed),
            kind = kind,
            rows = rows
        ),
        path
    )
    invisible(TRUE)
}


auto_umap_knn_hash <- function(indices,
                                distances,
                                max_rows = 512L,
                                max_cols = 64L) {
    indices <- as.matrix(indices)
    distances <- as.matrix(distances)
    n <- nrow(indices)
    p <- ncol(indices)
    rows <- unique(as.integer(round(seq(1L, n, length.out = min(n, max_rows)))))
    cols <- unique(as.integer(round(seq(1L, p, length.out = min(p, max_cols)))))
    auto_umap_raw_hash(serialize(
        list(
            dim = c(n, p),
            rows = rows,
            cols = cols,
            indices = as.integer(indices[rows, cols, drop = FALSE]),
            distances = signif(
                as.numeric(distances[rows, cols, drop = FALSE]),
                12L
            )
        ),
        NULL,
        version = 2L
    ))
}

auto_umap_raw_hash <- function(raw) {
    bytes <- as.integer(raw)
    if (length(bytes) == 0L) {
        return("0000000000000000")
    }
    pos <- seq_along(bytes)
    h1 <- sum((bytes + 1) * ((pos %% 104729L) + 1L)) %% 2147483647
    h2 <- sum((bytes + 3) * (((pos * 13L) %% 130363L) + 1L)) %% 2147483647
    paste0(
        sprintf("%08x", as.integer(h1)),
        sprintf("%08x", as.integer(h2))
    )
}


auto_umap_pilot_size <- function(full_n,
                                    available_n,
                                    pilot_min_n,
                                    pilot_max_n) {
    full_n <- as.integer(full_n)
    available_n <- as.integer(available_n)
    pilot_min_n <- as.integer(pilot_min_n)
    pilot_max_n <- as.integer(pilot_max_n)
    if (full_n < pilot_min_n || available_n < 50L) {
        return(0L)
    }
    if (full_n <= pilot_max_n) {
        return(as.integer(min(available_n, full_n)))
    }
    as.integer(min(
        available_n,
        max(pilot_min_n, min(pilot_max_n, floor(full_n * 0.10)))
    ))
}




auto_umap_default_pilot_spectral <- function(k) {
    as.integer(if (k <= 15L) 20L else 10L)
}

auto_umap_pilot_quick_epochs <- function(full_n) {
    if (full_n >= 10000L) 60L else 120L
}

auto_umap_default_pilot_init_scale <- function(full_n) {
    if (full_n >= 10000L) 10 else NA_real_
}

auto_umap_refined_pilot_configs <- function(full_n,
                                            k,
                                            max_configs) {
    if (max_configs <= 0L) {
        return(data.frame())
    }
    spectral <- unique(as.integer(c(
        auto_umap_default_pilot_spectral(k),
        if (k <= 15L) 30L else 20L
    )))
    epochs <- if (full_n >= 10000L) c(80L, 100L) else c(150L, 300L)
    init_scale <- if (full_n >= 10000L) c(10, 5) else c(NA_real_, 5)
    grid <- expand.grid(
        k = as.integer(k),
        spectral_n_iter = spectral,
        n_epochs = as.integer(epochs),
        init_scale = init_scale,
        KEEP.OUT.ATTRS = FALSE,
        stringsAsFactors = FALSE
    )
    grid <- unique(grid)
    grid[seq_len(min(nrow(grid), max_configs)), , drop = FALSE]
}

auto_umap_failed_pilot_row <- function(stage,
                                        k,
                                        spectral_n_iter,
                                        n_epochs,
                                        init_scale,
                                        elapsed,
                                        error) {
    data.frame(
        stage = stage,
        k = as.integer(k),
        spectral_n_iter = as.integer(spectral_n_iter),
        n_epochs = as.integer(n_epochs),
        init_scale = as.numeric(init_scale),
        elapsed = as.numeric(elapsed),
        local_trustworthiness = NA_real_,
        knn_preservation = NA_real_,
        local_continuity = NA_real_,
        knn_preservation_50 = NA_real_,
        local_continuity_50 = NA_real_,
        embedding_knn_accuracy = NA_real_,
        rare_class_recall = NA_real_,
        structure_score = NA_real_,
        pilot_score = NA_real_,
        low_dimensional = NA,
        tabular_like = NA,
        highly_imbalanced = NA,
        avoid_aggressive_k_reduction = NA,
        cache_hit = FALSE,
        status = "failed",
        error = error,
        stringsAsFactors = FALSE
    )
}


auto_umap_pilot_quality_values <- function(structure, proxy, labels) {
    trust <- unname(structure["local_trustworthiness"])
    preserve <- unname(structure["knn_preservation"])
    continuity <- unname(structure["local_continuity"])
    label_acc <- if (is.null(labels)) {
        NA_real_
    } else {
        labels$label_knn_accuracy
    }
    rare_recall <- if (is.null(labels)) {
        NA_real_
    } else {
        labels$rare_class_recall
    }
    if (is.finite(label_acc) || is.finite(rare_recall)) {
        return(list(
            values = c(
                trust = trust, preserve = preserve,
                continuity = continuity, label = label_acc,
                rare = rare_recall
            ),
            weights = c(
                trust = 0.25, preserve = 0.20, continuity = 0.15,
                label = 0.25, rare = 0.15
            )
        ))
    }
    list(
        values = c(
            trust = trust,
            preserve50 = unname(proxy["knn_preservation"]),
            continuity50 = unname(proxy["local_continuity"]),
            preserve = preserve
        ),
        weights = c(
            trust = 0.25, preserve50 = 0.35,
            continuity50 = 0.25, preserve = 0.15
        )
    )
}

adjust_auto_umap_pilot_quality <- function(quality, profile, k) {
    if (!isTRUE(profile$avoid_aggressive_k_reduction)) {
        return(quality)
    }
    base <- auto_embedding_k(
        profile$full_n,
        "umap",
        include_self = FALSE
    )
    if (is.finite(k) && k < base) {
        quality <- quality - 0.03 * (1 - as.numeric(k) / max(1, base))
    }
    quality
}

auto_umap_pilot_score <- function(structure, elapsed,
                                    label_scores = NULL,
                                    proxy_structure = NULL,
                                    dataset_profile = NULL,
                                    k = NA_integer_) {
    proxy <- if (is.null(proxy_structure)) {
        structure
    } else {
        proxy_structure
    }
    quality <- auto_umap_pilot_quality_values(
        structure,
        proxy,
        label_scores
    )
    finite <- is.finite(quality$values)
    if (!any(finite)) {
        return(NA_real_)
    }
    score <- sum(
        quality$values[finite] * quality$weights[finite]
    ) / sum(quality$weights[finite])
    score <- adjust_auto_umap_pilot_quality(
        score,
        dataset_profile,
        k
    )
    penalty <- if (is.finite(elapsed)) {
        0.01 * log1p(as.numeric(elapsed))
    } else {
        0
    }
    score - penalty
}


auto_umap_knn_pilot_skip <- function(reason) {
    list(status = "skipped", reason = reason)
}

normalize_auto_knn_pilot_input <- function(indices, distances) {
    indices <- as.matrix(indices)
    distances <- as.matrix(distances)
    if (!is.integer(indices)) {
        storage.mode(indices) <- "integer"
    }
    if (!identical(typeof(distances), "double")) {
        storage.mode(distances) <- "double"
    }
    list(indices = indices, distances = distances)
}

auto_umap_knn_pilot_candidates <- function(
    indices,
    full_n,
    pilot_min_n,
    pilot_max_n
) {
    if (full_n < pilot_min_n || ncol(indices) < 2L) {
        return(auto_umap_knn_pilot_skip(
            "KNN pilot skipped below threshold or with too few neighbors"
        ))
    }
    pilot_n <- auto_umap_pilot_size(
        full_n = full_n,
        available_n = nrow(indices),
        pilot_min_n = pilot_min_n,
        pilot_max_n = pilot_max_n
    )
    if (pilot_n < 50L) {
        return(auto_umap_knn_pilot_skip(
            "KNN pilot skipped below minimum pilot size"
        ))
    }
    candidates <- auto_umap_knn_pilot_k_candidates(
        full_n = full_n,
        pilot_n = pilot_n,
        supplied_k = ncol(indices)
    )
    if (!length(candidates)) {
        return(auto_umap_knn_pilot_skip(
            "KNN pilot skipped because no valid k was available"
        ))
    }
    list(
        status = "ready",
        pilot_n = pilot_n,
        candidates = candidates,
        supplied_k = as.integer(ncol(indices))
    )
}

prepare_auto_knn_pilot_graph <- function(input, candidate, seed) {
    rows <- auto_knn_pilot_rows(
        input$indices,
        candidate$pilot_n,
        seed
    )
    graph <- auto_knn_pilot_subgraph(
        indices = input$indices,
        distances = input$distances,
        rows = rows,
        max_k = max(candidate$candidates)
    )
    if (nrow(graph$indices) < 50L) {
        return(auto_umap_knn_pilot_skip(
            "KNN pilot skipped because the sampled graph was too small"
        ))
    }
    list(
        status = "ready",
        graph = graph,
        score_keep = sample_indices(
            nrow(graph$indices),
            min(1000L, max(50L, nrow(graph$indices))),
            seed + 307L
        ),
        hash = auto_umap_knn_hash(
            graph$indices,
            graph$distances
        )
    )
}

auto_knn_pilot_cached_row <- function(context, config) {
    cached <- auto_umap_read_pilot_cache_row(
        cache_dir = context$cache_dir,
        dataset_hash = context$hash,
        k = config$k,
        seed = context$seed,
        spectral_n_iter = config$spectral_n_iter,
        n_epochs = config$n_epochs,
        init_scale = config$init_scale,
        kind = "knn",
        use_cache = context$use_cache,
        force_recompute = context$force_recompute
    )
    if (!is.null(cached)) {
        cached$stage <- config$stage
        cached$cache_hit <- TRUE
    }
    cached
}

capture_auto_knn_pilot_layout <- function(
    indices,
    distances,
    context,
    config
) {
    capture_error(
        fast_knn_umap_core(
            indices,
            distances,
            n_components = 2L,
            seed = context$seed,
            verbose = FALSE,
            backend = "cpu",
            config_override = list(
                n_epochs = as.integer(config$n_epochs),
                spectral_n_iter =
                    as.integer(config$spectral_n_iter),
                init_scale = config$init_scale,
                tuning_source = "knn_pilot"
            )
        )
    )
}

run_auto_knn_pilot_layout <- function(context, config) {
    indices <- context$graph$indices[
        , seq_len(config$k),
        drop = FALSE
    ]
    distances <- context$graph$distances[
        , seq_len(config$k),
        drop = FALSE
    ]
    result <- timed_do_call(
        capture_auto_knn_pilot_layout,
        list(indices, distances, context, config)
    )
    list(
        layout = result$value$value,
        error = result$value$error,
        elapsed = result$time[["elapsed"]],
        indices = indices
    )
}

score_auto_knn_pilot_layout <- function(layout, indices, keep) {
    attempt <- capture_error(knn_structure_score_cpp(
        layout,
        indices,
        keep,
        as.integer(min(50L, ncol(indices))),
        integer(0L),
        0L
    ))
    if (is.null(attempt$value)) {
        return(list(
            values = rep(NA_real_, 5L),
            error = attempt$error
        ))
    }
    list(values = attempt$value, error = NA_character_)
}

auto_knn_pilot_success_row <- function(config, layout, structure) {
    values <- structure$values
    data.frame(
        stage = config$stage,
        k = as.integer(config$k),
        spectral_n_iter = as.integer(config$spectral_n_iter),
        n_epochs = as.integer(config$n_epochs),
        init_scale = as.numeric(config$init_scale),
        elapsed = as.numeric(layout$elapsed),
        local_trustworthiness =
            unname(values["local_trustworthiness"]),
        knn_preservation = unname(values["knn_preservation"]),
        local_continuity = unname(values["local_continuity"]),
        knn_preservation_50 = unname(values["knn_preservation"]),
        local_continuity_50 = unname(values["local_continuity"]),
        embedding_knn_accuracy =
            unname(values["embedding_knn_accuracy"]),
        rare_class_recall = NA_real_,
        structure_score = unname(values["structure_score"]),
        pilot_score = as.numeric(
            auto_umap_pilot_score(values, layout$elapsed)
        ),
        low_dimensional = NA,
        tabular_like = NA,
        highly_imbalanced = NA,
        avoid_aggressive_k_reduction = NA,
        cache_hit = FALSE,
        status = "success",
        error = structure$error,
        stringsAsFactors = FALSE
    )
}

write_auto_knn_pilot_row <- function(row, context, config) {
    auto_umap_write_pilot_cache_row(
        row,
        cache_dir = context$cache_dir,
        dataset_hash = context$hash,
        k = config$k,
        seed = context$seed,
        kind = "knn",
        use_cache = context$use_cache
    )
    row
}

run_auto_knn_pilot_config <- function(context, config) {
    cached <- auto_knn_pilot_cached_row(context, config)
    if (!is.null(cached)) {
        return(cached)
    }
    layout <- run_auto_knn_pilot_layout(context, config)
    if (is.null(layout$layout)) {
        return(auto_umap_failed_pilot_row(
            config$stage,
            config$k,
            config$spectral_n_iter,
            config$n_epochs,
            config$init_scale,
            layout$elapsed,
            layout$error
        ))
    }
    structure <- score_auto_knn_pilot_layout(
        layout$layout,
        layout$indices,
        context$score_keep
    )
    row <- auto_knn_pilot_success_row(
        config,
        layout,
        structure
    )
    write_auto_knn_pilot_row(row, context, config)
}

auto_knn_pilot_config_row <- function(grid, row, stage) {
    list(
        k = grid$k[[row]],
        spectral_n_iter = grid$spectral_n_iter[[row]],
        n_epochs = grid$n_epochs[[row]],
        init_scale = grid$init_scale[[row]],
        stage = stage
    )
}

run_auto_knn_pilot_grid <- function(context, full_n, max_configs) {
    base_k <- as.integer(context$candidates[[1L]])
    grid <- auto_umap_optimizer_pilot_configs(
        full_n = full_n,
        k = base_k,
        max_configs = max_configs
    )
    rows <- lapply(seq_len(nrow(grid)), function(i) {
        run_auto_knn_pilot_config(
            context,
            auto_knn_pilot_config_row(grid, i, "config")
        )
    })
    do.call(rbind, rows)
}

select_auto_knn_pilot_result <- function(scores) {
    successful <- scores[
        is.finite(scores$pilot_score), ,
        drop = FALSE
    ]
    if (!nrow(successful)) {
        return(list(
            status = "failed",
            reason = "KNN pilot failed for all optimizer candidates",
            scores = scores
        ))
    }
    order <- order(
        -successful$pilot_score,
        successful$elapsed,
        successful$k
    )
    successful[order, , drop = FALSE][1L, ]
}

auto_knn_pilot_override <- function(best, context, sample_n, scores) {
    list(
        n_neighbors = as.integer(best$k),
        n_epochs = as.integer(best$n_epochs),
        spectral_n_iter = as.integer(best$spectral_n_iter),
        init_scale = as.numeric(best$init_scale),
        tuning_source = "knn_pilot",
        pilot_sample_n = as.integer(sample_n),
        pilot_score = as.numeric(best$pilot_score),
        pilot_cache_key = auto_umap_pilot_cache_key(
            context$hash,
            best$k,
            context$seed
        ),
        pilot_cache_hit = any(
            scores$cache_hit %in% TRUE,
            na.rm = TRUE
        )
    )
}

finish_auto_knn_pilot <- function(best, context, supplied_k, scores) {
    sample_n <- nrow(context$graph$indices)
    cache_hit <- any(scores$cache_hit %in% TRUE, na.rm = TRUE)
    list(
        status = "success",
        n_neighbors = as.integer(best$k),
        spectral_n_iter = as.integer(best$spectral_n_iter),
        n_epochs = as.integer(best$n_epochs),
        init_scale = as.numeric(best$init_scale),
        selected_score = as.numeric(best$pilot_score),
        pilot_sample_n = as.integer(sample_n),
        config_override = auto_knn_pilot_override(
            best,
            context,
            sample_n,
            scores
        ),
        scores = scores,
        reason = paste0(
            "KNN pilot sampled ", sample_n,
            " graph rows, kept supplied k=", supplied_k,
            ", spectral_n_iter=", best$spectral_n_iter,
            ", epochs=", best$n_epochs,
            ", init_scale=", format(best$init_scale, trim = TRUE),
            if (cache_hit) "; reused cached pilot score(s)" else ""
        )
    )
}

auto_umap_knn_pilot_tune <- function(
    indices, distances, seed = 4L, full_n = nrow(indices),
    pilot_min_n = 2000L, pilot_max_n = 5000L,
    pilot_max_configs = 6L, use_cache = TRUE, cache_dir = NULL,
    force_recompute = FALSE
) {
    input <- normalize_auto_knn_pilot_input(indices, distances)
    full_n <- as.integer(full_n)
    candidate <- auto_umap_knn_pilot_candidates(
        input$indices,
        full_n,
        pilot_min_n,
        pilot_max_n
    )
    if (candidate$status != "ready") {
        return(candidate)
    }
    prepared <- prepare_auto_knn_pilot_graph(input, candidate, seed)
    if (prepared$status != "ready") {
        return(prepared)
    }
    context <- list(
        graph = prepared$graph,
        score_keep = prepared$score_keep,
        hash = prepared$hash,
        cache_dir = auto_umap_pilot_cache_dir(cache_dir),
        seed = seed,
        candidates = candidate$candidates,
        use_cache = use_cache,
        force_recompute = force_recompute
    )
    scores <- run_auto_knn_pilot_grid(
        context,
        full_n,
        pilot_max_configs
    )
    best <- select_auto_knn_pilot_result(scores)
    if (is.list(best) && identical(best$status, "failed")) {
        return(best)
    }
    finish_auto_knn_pilot(
        best,
        context,
        candidate$supplied_k,
        scores
    )
}

auto_umap_knn_pilot_k_candidates <- function(full_n,
                                                pilot_n,
                                                supplied_k) {
    candidates <- supplied_k
    candidates <- unique(as.integer(candidates))
    candidates <- candidates[is.finite(candidates) & candidates >= 2L]
    candidates <- candidates[candidates <= supplied_k & candidates < pilot_n]
    candidates
}

auto_knn_indices_one_based <- function(indices) {
    if (identical(knn_index_base(indices, nrow(indices)), "zero")) {
        indices + 1L
    } else {
        indices
    }
}

auto_knn_pilot_rows <- function(indices,
                                sample_size,
                                seed) {
    n <- nrow(indices)
    sample_size <- as.integer(min(max(1L, sample_size), n))
    if (sample_size >= n) {
        return(seq_len(n))
    }
    idx <- auto_knn_indices_one_based(indices)
    selected <- rep(FALSE, n)
    rows <- integer(0L)
    restore_seed <- set_local_seed(seed + 1709L)
    on.exit(restore_seed(), add = TRUE)
    queue <- sample.int(n, min(32L, n))
    cursor <- 1L
    while (length(rows) < sample_size) {
        if (cursor > length(queue)) {
            remaining <- which(!selected)
            if (length(remaining) == 0L) break
            queue <- c(queue, sample(remaining, 1L))
        }
        row <- queue[cursor]
        cursor <- cursor + 1L
        if (selected[row]) next
        selected[row] <- TRUE
        rows <- c(rows, row)
        nb <- idx[row, ]
        nb <- nb[is.finite(nb) & nb >= 1L & nb <= n & !selected[nb]]
        if (length(nb) > 0L) {
            nb <- sample(unique(as.integer(nb)), min(length(unique(nb)), 12L))
            queue <- c(queue, nb)
        }
    }
    sort(as.integer(rows[seq_len(min(length(rows), sample_size))]))
}

auto_knn_pilot_fallback_distance <- function(distances) {
    finite_dist <- distances[is.finite(distances) & distances >= 0]
    value <- if (!length(finite_dist)) {
        1
    } else {
        stats::median(finite_dist)
    }
    if (!is.finite(value) || value <= 0) 1 else value
}

auto_knn_pilot_subgraph_row <- function(
    global_i, local_i, idx, distances, map,
    n_sub, max_k, fallback
) {
    cols <- seq_len(min(ncol(idx), max_k))
    row_idx <- idx[global_i, cols, drop = TRUE]
    row_dst <- distances[global_i, cols, drop = TRUE]
    valid <- is.finite(row_idx) &
        row_idx >= 1L &
        row_idx <= length(map)
    row_idx <- row_idx[valid]
    row_dst <- row_dst[valid]
    local_nb <- map[row_idx]
    keep <- local_nb > 0L &
        local_nb != local_i &
        is.finite(row_dst) &
        row_dst >= 0
    local_nb <- local_nb[keep]
    row_dst <- row_dst[keep]
    dedup <- !duplicated(local_nb)
    local_nb <- local_nb[dedup]
    row_dst <- row_dst[dedup]
    if (length(local_nb) < max_k) {
        fill <- setdiff(seq_len(n_sub), c(local_i, local_nb))
        fill <- utils::head(fill, max_k - length(local_nb))
        fill_distance <- if (length(row_dst)) {
            max(row_dst) * 1.25
        } else {
            fallback * 1.25
        }
        local_nb <- c(local_nb, fill)
        row_dst <- c(row_dst, rep(fill_distance, length(fill)))
    }
    list(
        indices = as.integer(utils::head(local_nb, max_k)),
        distances = as.numeric(utils::head(row_dst, max_k))
    )
}

auto_knn_pilot_subgraph <- function(indices, distances, rows, max_k) {
    rows <- sort(unique(as.integer(rows)))
    n_sub <- length(rows)
    max_k <- as.integer(min(max_k, n_sub - 1L, ncol(indices)))
    idx <- auto_knn_indices_one_based(indices)
    map <- integer(nrow(indices))
    map[rows] <- seq_along(rows)
    out_idx <- matrix(0L, n_sub, max_k)
    out_dst <- matrix(0, n_sub, max_k)
    fallback <- auto_knn_pilot_fallback_distance(distances)
    for (local_i in seq_len(n_sub)) {
        row <- auto_knn_pilot_subgraph_row(
            rows[[local_i]], local_i, idx, distances, map,
            n_sub, max_k, fallback
        )
        out_idx[local_i, ] <- row$indices
        out_dst[local_i, ] <- row$distances
    }
    list(indices = out_idx, distances = out_dst)
}






validate_epoch_count <- function(n_epochs) {
    n_epochs <- as.integer(n_epochs)
    if (length(n_epochs) != 1L || is.na(n_epochs) || !is.finite(n_epochs) ||
        n_epochs < 1L) {
        stop("`n_epochs` must be a positive integer.", call. = FALSE)
    }
    n_epochs
}
