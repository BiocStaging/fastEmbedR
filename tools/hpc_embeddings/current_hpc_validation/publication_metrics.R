`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L ||
      (length(x) == 1L && is.na(x))) y else x
}

publication_sample_rows <- function(n, size = 3000L, seed = 4L) {
  size <- min(as.integer(size), as.integer(n))
  if (size >= n) return(seq_len(n))
  set.seed(as.integer(seed))
  sort(sample.int(n, size))
}

publication_layout_matrix <- function(x) {
  if (is.list(x) && !is.null(x$layout)) x <- x$layout
  if (is.list(x) && !is.null(x$Y)) x <- x$Y
  if (inherits(x, "float32")) {
    if (!requireNamespace("float", quietly = TRUE)) {
      stop("The float package is required to materialize a float32 layout.",
           call. = FALSE)
    }
    x <- float::dbl(x)
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (ncol(x) < 2L) stop("An embedding needs at least two columns.", call. = FALSE)
  x[, 1:2, drop = FALSE]
}

publication_knn_host <- function(knn) {
  if (inherits(knn, "fastEmbedR_gpu_knn")) {
    knn <- fastEmbedR:::fastembedr_gpu_knn_to_host(knn)
  }
  if (is.null(knn$indices) || is.null(knn$distances)) {
    stop("KNN object does not contain host indices and distances.", call. = FALSE)
  }
  list(
    indices = matrix(as.integer(knn$indices), nrow = nrow(knn$indices)),
    distances = matrix(as.numeric(knn$distances), nrow = nrow(knn$distances))
  )
}

publication_exact_knn <- function(x, k) {
  x <- as.matrix(x)
  n <- nrow(x)
  k <- min(as.integer(k), n - 1L)
  d <- as.matrix(stats::dist(x))
  diag(d) <- Inf
  indices <- matrix(NA_integer_, n, k)
  distances <- matrix(NA_real_, n, k)
  for (i in seq_len(n)) {
    ord <- order(d[i, ], decreasing = FALSE)[seq_len(k)]
    indices[i, ] <- ord
    distances[i, ] <- d[i, ord]
  }
  list(indices = indices, distances = distances)
}

publication_conditional_probabilities <- function(distances,
                                                   perplexity,
                                                   tolerance = 1e-5,
                                                   max_iter = 64L) {
  distances <- as.matrix(distances)
  n <- nrow(distances)
  k <- ncol(distances)
  target <- log(min(as.numeric(perplexity), k))
  out <- matrix(0, n, k)
  for (i in seq_len(n)) {
    d2 <- pmax(as.numeric(distances[i, ]), 0)^2
    beta <- 1
    beta_min <- -Inf
    beta_max <- Inf
    p <- rep.int(1 / k, k)
    for (iteration in seq_len(max_iter)) {
      p <- exp(-d2 * beta)
      p[!is.finite(p)] <- 0
      z <- sum(p)
      if (!is.finite(z) || z <= .Machine$double.xmin) {
        p <- rep.int(1 / k, k)
        entropy <- log(k)
      } else {
        p <- p / z
        entropy <- -sum(p[p > 0] * log(p[p > 0]))
      }
      delta <- entropy - target
      if (abs(delta) <= tolerance) break
      if (delta > 0) {
        beta_min <- beta
        beta <- if (is.finite(beta_max)) (beta + beta_max) / 2 else beta * 2
      } else {
        beta_max <- beta
        beta <- if (is.finite(beta_min)) (beta + beta_min) / 2 else beta / 2
      }
    }
    out[i, ] <- p
  }
  out
}

publication_sparse_affinities <- function(knn, perplexity) {
  knn <- publication_knn_host(knn)
  idx <- knn$indices
  n <- nrow(idx)
  p <- publication_conditional_probabilities(knn$distances, perplexity)
  i <- rep(seq_len(n), each = ncol(idx))
  j <- as.vector(t(idx))
  w <- as.vector(t(p))
  valid <- is.finite(w) & w > 0 & j >= 1L & j <= n & i != j
  i <- i[valid]
  j <- j[valid]
  w <- w[valid]
  lo <- pmin.int(i, j)
  hi <- pmax.int(i, j)
  key <- paste0(lo, ":", hi)
  summed <- rowsum(w, key, reorder = FALSE)
  keys <- rownames(summed)
  pieces <- strsplit(keys, ":", fixed = TRUE)
  edge <- data.frame(
    i = as.integer(vapply(pieces, `[[`, character(1), 1L)),
    j = as.integer(vapply(pieces, `[[`, character(1), 2L)),
    weight = as.numeric(summed[, 1L]) / n,
    stringsAsFactors = FALSE
  )
  total <- sum(edge$weight)
  if (is.finite(total) && total > 0) edge$weight <- edge$weight / total
  edge$key <- keys
  edge
}

publication_sparse_triplet_edges <- function(i, j, weight, one_based = TRUE) {
  i <- as.integer(i)
  j <- as.integer(j)
  weight <- as.numeric(weight)
  if (!one_based) {
    i <- i + 1L
    j <- j + 1L
  }
  valid <- i != j & i > 0L & j > 0L & is.finite(weight) & weight > 0
  i <- i[valid]
  j <- j[valid]
  weight <- weight[valid]
  lo <- pmin.int(i, j)
  hi <- pmax.int(i, j)
  key <- paste0(lo, ":", hi)
  summed <- rowsum(weight, key, reorder = FALSE)
  keys <- rownames(summed)
  pieces <- strsplit(keys, ":", fixed = TRUE)
  edge <- data.frame(
    i = as.integer(vapply(pieces, `[[`, character(1), 1L)),
    j = as.integer(vapply(pieces, `[[`, character(1), 2L)),
    weight = as.numeric(summed[, 1L]),
    stringsAsFactors = FALSE
  )
  total <- sum(edge$weight)
  if (is.finite(total) && total > 0) edge$weight <- edge$weight / total
  edge$key <- keys
  edge
}

publication_edge_agreement <- function(reference, candidate) {
  all_keys <- union(reference$key, candidate$key)
  rw <- numeric(length(all_keys))
  cw <- numeric(length(all_keys))
  rw[match(reference$key, all_keys)] <- reference$weight
  cw[match(candidate$key, all_keys)] <- candidate$weight
  common <- intersect(reference$key, candidate$key)
  common_ref <- reference$weight[match(common, reference$key)]
  common_candidate <- candidate$weight[match(common, candidate$key)]
  data.frame(
    edge_jaccard = if (length(all_keys)) length(common) / length(all_keys) else NA_real_,
    weight_pearson = if (length(all_keys) > 2L) {
      suppressWarnings(stats::cor(rw, cw, method = "pearson"))
    } else NA_real_,
    weight_spearman = if (length(common) > 2L) {
      suppressWarnings(stats::cor(common_ref, common_candidate, method = "spearman"))
    } else NA_real_,
    weight_l1_similarity = if (length(all_keys)) {
      max(0, 1 - sum(abs(rw - cw)) / 2)
    } else NA_real_,
    common_edges = length(common),
    union_edges = length(all_keys),
    stringsAsFactors = FALSE
  )
}

publication_tsne_kl <- function(layout, affinity_edges) {
  layout <- publication_layout_matrix(layout)
  if (nrow(layout) < 2L || !nrow(affinity_edges)) return(NA_real_)
  q <- 1 / (1 + as.numeric(stats::dist(layout))^2)
  z <- sum(q)
  if (!is.finite(z) || z <= 0) return(NA_real_)
  q <- q / z
  n <- nrow(layout)
  pair_offset <- function(i, j) {
    # Index in stats::dist's lower-triangle vector for i < j.
    n * (i - 1L) - (i - 1L) * i / 2L + (j - i)
  }
  pos <- pair_offset(affinity_edges$i, affinity_edges$j)
  q_edge <- pmax(q[pos], .Machine$double.xmin)
  p_edge <- pmax(affinity_edges$weight, .Machine$double.xmin)
  sum(p_edge * log(p_edge / q_edge))
}

publication_umap_edges <- function(knn, graph_mode = "fuzzy", n_threads = 1L) {
  prepared <- fastEmbedR::prepare_umap_knn(
    knn,
    graph_mode = graph_mode,
    backend = "cpu",
    n_threads = as.integer(n_threads)
  )
  graph <- prepared$graph
  offsets <- as.integer(graph$offsets)
  counts <- diff(offsets)
  i <- rep(seq_along(counts), counts)
  j <- as.integer(graph$neighbors) + 1L
  w <- as.numeric(graph$weights)
  valid <- i != j & is.finite(w) & w > 0
  i <- i[valid]
  j <- j[valid]
  w <- w[valid]
  lo <- pmin.int(i, j)
  hi <- pmax.int(i, j)
  key <- paste0(lo, ":", hi)
  # The prepared CSR may contain both directions. The graph weight is symmetric;
  # max avoids counting an edge twice while preserving its fuzzy strength.
  split_w <- split(w, key)
  weight <- vapply(split_w, max, numeric(1))
  keys <- names(weight)
  pieces <- strsplit(keys, ":", fixed = TRUE)
  out <- data.frame(
    i = as.integer(vapply(pieces, `[[`, character(1), 1L)),
    j = as.integer(vapply(pieces, `[[`, character(1), 2L)),
    weight = as.numeric(weight),
    key = keys,
    stringsAsFactors = FALSE
  )
  total <- sum(out$weight)
  if (is.finite(total) && total > 0) out$weight <- out$weight / total
  out
}

publication_sparse_matrix_edges <- function(graph) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Matrix is required to inspect a reference sparse graph.", call. = FALSE)
  }
  values <- Matrix::summary(graph)
  valid <- values$i != values$j & is.finite(values$x) & values$x > 0
  i <- as.integer(values$i[valid])
  j <- as.integer(values$j[valid])
  w <- as.numeric(values$x[valid])
  lo <- pmin.int(i, j)
  hi <- pmax.int(i, j)
  key <- paste0(lo, ":", hi)
  split_w <- split(w, key)
  weight <- vapply(split_w, max, numeric(1))
  keys <- names(weight)
  pieces <- strsplit(keys, ":", fixed = TRUE)
  out <- data.frame(
    i = as.integer(vapply(pieces, `[[`, character(1), 1L)),
    j = as.integer(vapply(pieces, `[[`, character(1), 2L)),
    weight = as.numeric(weight), key = keys,
    stringsAsFactors = FALSE
  )
  total <- sum(out$weight)
  if (is.finite(total) && total > 0) out$weight <- out$weight / total
  out
}

publication_procrustes <- function(reference, candidate) {
  reference <- publication_layout_matrix(reference)
  candidate <- publication_layout_matrix(candidate)
  if (!identical(dim(reference), dim(candidate))) {
    return(data.frame(rmsd = NA_real_, correlation = NA_real_))
  }
  reference <- scale(reference, center = TRUE, scale = FALSE)
  candidate <- scale(candidate, center = TRUE, scale = FALSE)
  ref_norm <- sqrt(sum(reference^2))
  cand_norm <- sqrt(sum(candidate^2))
  if (!is.finite(ref_norm) || !is.finite(cand_norm) ||
      ref_norm <= 0 || cand_norm <= 0) {
    return(data.frame(rmsd = NA_real_, correlation = NA_real_))
  }
  reference <- reference / ref_norm
  candidate <- candidate / cand_norm
  rotation <- tryCatch({
    dec <- svd(crossprod(candidate, reference))
    dec$u %*% t(dec$v)
  }, error = function(e) NULL)
  if (is.null(rotation)) {
    return(data.frame(rmsd = NA_real_, correlation = NA_real_))
  }
  aligned <- candidate %*% rotation
  data.frame(
    rmsd = sqrt(mean(rowSums((aligned - reference)^2))),
    correlation = suppressWarnings(stats::cor(as.vector(aligned), as.vector(reference))),
    stringsAsFactors = FALSE
  )
}

publication_knn_overlap <- function(reference, candidate, k = NULL) {
  reference <- publication_knn_host(reference)$indices
  candidate <- publication_knn_host(candidate)$indices
  if (is.null(k)) k <- min(ncol(reference), ncol(candidate))
  k <- min(as.integer(k), ncol(reference), ncol(candidate))
  if (nrow(reference) != nrow(candidate) || k < 1L) return(NA_real_)
  reference <- reference[, seq_len(k), drop = FALSE]
  candidate <- candidate[, seq_len(k), drop = FALSE]
  mean(vapply(seq_len(nrow(reference)), function(i) {
    length(intersect(reference[i, ], candidate[i, ])) / k
  }, numeric(1)))
}

publication_clean_plot <- function(layout,
                                   labels,
                                   path,
                                   cex = NULL,
                                   landmark_indices = integer()) {
  layout <- publication_layout_matrix(layout)
  if (is.null(cex)) {
    cex <- if (nrow(layout) < 2000L) 0.75 else if (nrow(layout) < 20000L) 0.35 else 0.16
  }
  if (is.null(labels)) {
    colors <- rep.int("#1F77B4", nrow(layout))
  } else {
    labels <- as.factor(labels)
    palette <- grDevices::hcl.colors(max(3L, nlevels(labels)), "Dark 3")
    colors <- palette[as.integer(labels)]
  }
  landmark_indices <- unique(as.integer(landmark_indices))
  landmark_indices <- landmark_indices[
    is.finite(landmark_indices) &
      landmark_indices >= 1L &
      landmark_indices <= nrow(layout)
  ]
  projected_indices <- if (length(landmark_indices)) {
    setdiff(seq_len(nrow(layout)), landmark_indices)
  } else {
    seq_len(nrow(layout))
  }
  grDevices::png(path, width = 1800, height = 1800, res = 220,
                 bg = "white")
  on.exit(grDevices::dev.off(), add = TRUE)
  old <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old), add = TRUE)
  graphics::par(mar = rep(0, 4L), xaxs = "i", yaxs = "i", bty = "n")
  padded_limits <- function(values, fraction = 0.04) {
    limits <- range(values, finite = TRUE)
    span <- diff(limits)
    if (!is.finite(span) || span <= 0) {
      center <- if (all(is.finite(limits))) mean(limits) else 0
      span <- max(1, abs(center) * 0.08)
      limits <- center + c(-0.5, 0.5) * span
    }
    limits + c(-1, 1) * span * fraction
  }
  graphics::plot(
    layout[, 1L], layout[, 2L],
    type = "n",
    xlim = padded_limits(layout[, 1L]),
    ylim = padded_limits(layout[, 2L]),
    axes = FALSE, ann = FALSE, frame.plot = FALSE
  )
  if (length(landmark_indices)) {
    graphics::points(
      layout[landmark_indices, 1L],
      layout[landmark_indices, 2L],
      pch = 16,
      cex = cex,
      col = "#D3D3D3"
    )
  }
  if (length(projected_indices)) {
    graphics::points(
      layout[projected_indices, 1L],
      layout[projected_indices, 2L],
      pch = 16,
      cex = cex,
      col = colors[projected_indices]
    )
  }
  invisible(path)
}

publication_median_iqr <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) {
    return(c(median = NA_real_, q1 = NA_real_, q3 = NA_real_,
             iqr = NA_real_, sd = NA_real_, min = NA_real_, max = NA_real_))
  }
  q <- stats::quantile(x, c(0.25, 0.5, 0.75), names = FALSE, type = 7)
  c(
    median = q[[2L]], q1 = q[[1L]], q3 = q[[3L]],
    iqr = q[[3L]] - q[[1L]], sd = if (length(x) > 1L) stats::sd(x) else 0,
    min = min(x), max = max(x)
  )
}
