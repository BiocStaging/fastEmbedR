normalize_nn_threads <- function(n_threads) {
  n_threads <- suppressWarnings(as.integer(n_threads))
  if (length(n_threads) != 1L || is.na(n_threads) || n_threads < 1L) {
    n_threads <- 1L
  }
  n_threads
}

.fastembedr_faissr_cache <- new.env(parent = emptyenv())

fastembedr_minimum_faissr_version <- function() numeric_version("0.99.3")

fastembedr_optional_namespace_available <- function(package) {
  requireNamespace(package, quietly = TRUE)
}

fastembedr_optional_export <- function(package, name) {
  getExportedValue(package, name)
}

fastembedr_faissr_function <- function(name) {
  fn <- .fastembedr_faissr_cache[[name]]
  if (!is.null(fn)) return(fn)
  package <- "faissR"
  if (!fastembedr_optional_namespace_available(package)) {
    stop(
      "Package `faissR` is required for the validated one-call CUDA KNN route. ",
      "Install faissR with FAISS GPU support, or pass a ",
      "precomputed KNN object to `opentsne_knn()` ",
      "or `umap_knn()`.",
      call. = FALSE
    )
  }
  installed_version <- utils::packageVersion(package)
  minimum_version <- fastembedr_minimum_faissr_version()
  if (installed_version < minimum_version) {
    stop(
      "fastEmbedR requires faissR >= ", minimum_version,
      " for the requested external KNN route; installed version is ", installed_version, ".",
      call. = FALSE
    )
  }
  fn <- fastembedr_optional_export(package, name)
  .fastembedr_faissr_cache[[name]] <- fn
  fn
}

fastembedr_faissr_function_available <- function(name) {
  fn <- .fastembedr_faissr_cache[[name]]
  if (!is.null(fn)) return(TRUE)
  package <- "faissR"
  if (!fastembedr_optional_namespace_available(package)) return(FALSE)
  name %in% getNamespaceExports(package)
}

fastembedr_faissr_nn <- function(data,
                                 points = data,
                                 k,
                                 backend = "cpu",
                                 n_threads = NULL,
                                 ...) {
  fn <- fastembedr_faissr_function("nn")
  args <- list(
    data = data,
    points = points,
    k = k,
    backend = backend,
    n_threads = n_threads,
    ...
  )
  do.call(fn, args)
}

fastembedr_exact_knn_fallback <- function(data,
                                          points = data,
                                          k,
                                          include_self = identical(data, points)) {
  data <- as.matrix(data)
  points <- as.matrix(points)
  storage.mode(data) <- "double"
  storage.mode(points) <- "double"
  if (ncol(data) != ncol(points)) {
    stop("KNN fallback requires data and query matrices with the same number of columns.", call. = FALSE)
  }
  pair_count <- as.double(nrow(data)) * as.double(nrow(points))
  if (pair_count > 1e7) {
    stop(
      "The base exact-KNN fallback would allocate an unsafe dense distance matrix. ",
      "Install faissR, provide precomputed neighbors, or evaluate a smaller subset.",
      call. = FALSE
    )
  }
  k <- as.integer(k)
  k <- min(max(1L, k), nrow(data))
  data_norm <- rowSums(data * data)
  point_norm <- rowSums(points * points)
  d2 <- outer(point_norm, data_norm, "+") - 2 * tcrossprod(points, data)
  d2[d2 < 0 & d2 > -1e-8] <- 0
  if (!isTRUE(include_self) && nrow(data) == nrow(points) && isTRUE(all.equal(data, points, tolerance = 0))) {
    diag(d2) <- Inf
  }
  indices <- matrix(NA_integer_, nrow(points), k)
  distances <- matrix(NA_real_, nrow(points), k)
  for (i in seq_len(nrow(points))) {
    ord <- order(d2[i, ], na.last = NA)
    ord <- ord[seq_len(min(k, length(ord)))]
    indices[i, seq_along(ord)] <- as.integer(ord)
    distances[i, seq_along(ord)] <- sqrt(pmax(d2[i, ord], 0))
  }
  list(indices = indices, distances = distances, backend = "base_exact")
}

fastembedr_call_supported <- function(fn, args) {
  formal_names <- names(formals(fn))
  if (!("..." %in% formal_names)) {
    args <- args[names(args) %in% formal_names]
  }
  do.call(fn, args)
}

fastembedr_supports_formal <- function(fn, name) {
  name %in% names(formals(fn))
}

fastembedr_faissr_method_available <- function(fn_name, method) {
  if (!fastembedr_faissr_function_available(fn_name)) return(FALSE)
  fn <- fastembedr_faissr_function(fn_name)
  method_formal <- formals(fn)$method
  if (is.null(method_formal)) return(FALSE)
  choices <- tryCatch(eval(method_formal, envir = baseenv()), error = function(e) NULL)
  if (is.null(choices)) return(TRUE)
  method %in% as.character(choices)
}

fastembedr_has_gpu_knn_shape <- function(x) {
  is.list(x) &&
    all(c("indices_ptr", "distances_ptr", "n_query", "k") %in% names(x)) &&
    identical(as.character(x$result_residency %||% attr(x, "result_residency") %||% ""), "cuda")
}

fastembedr_as_gpu_knn <- function(x) {
  if (!fastembedr_has_gpu_knn_shape(x) ||
      inherits(x, "faissR_gpu_knn") ||
      inherits(x, "fastEmbedR_gpu_knn")) {
    return(x)
  }
  class(x) <- unique(c("fastEmbedR_gpu_knn_contract", class(x), "list"))
  x
}

fastembedr_is_gpu_knn <- function(x) {
  inherits(x, "faissR_gpu_knn") ||
    inherits(x, "fastEmbedR_gpu_knn") ||
    fastembedr_has_gpu_knn_shape(x)
}

fastembedr_gpu_knn_to_host <- function(knn) {
  if (!fastembedr_is_gpu_knn(knn)) return(knn)
  knn <- fastembedr_as_gpu_knn(knn)
  gpu_backend <- knn$backend_used %||% attr(knn, "backend") %||% "cuda"
  gpu_metric <- knn$metric %||% attr(knn, "metric") %||% NA_character_
  native_provider <- inherits(knn, "fastEmbedR_gpu_knn") ||
    identical(knn$gpu_provider %||% "", "fastEmbedR_native_cuvs")
  out <- if (native_provider) {
    native_cuda_knn_to_host_cpp(knn)
  } else {
    fastembedr_faissr_function("gpu_knn_to_host")(knn)
  }
  attr(out, "gpu_resident_source") <- TRUE
  attr(out, "gpu_backend_used") <- gpu_backend
  attr(out, "backend") <- attr(out, "backend") %||% paste0(gpu_backend, "_host")
  if (!is.na(gpu_metric)) {
    attr(out, "metric") <- gpu_metric
  }
  out
}

fastembedr_gpu_knn_info <- function(knn) {
  if (!fastembedr_is_gpu_knn(knn)) {
    stop("Expected a CUDA GPU-resident KNN object.", call. = FALSE)
  }
  n <- as.integer(knn$n_query %||% knn$n %||% NA_integer_)
  k <- as.integer(knn$k %||% NA_integer_)
  if (length(n) != 1L || is.na(n) || n < 2L ||
      length(k) != 1L || is.na(k) || k < 1L) {
    stop("Invalid CUDA GPU-resident KNN dimensions.", call. = FALSE)
  }
  list(
    n = n,
    k = k,
    has_self = !isTRUE(knn$exclude_self %||% attr(knn, "exclude_self") %||% FALSE),
    input_backend = knn$backend_used %||% attr(knn, "backend_used") %||%
      attr(knn, "backend") %||% "cuda",
    metric = knn$metric %||% attr(knn, "metric") %||% NA_character_,
    distance_type = knn$distance_type %||% attr(knn, "distance_type") %||% "float32",
    result_residency = knn$result_residency %||% attr(knn, "result_residency") %||% "cuda"
  )
}

fastembedr_convert_knn_distances <- function(knn, output) {
  if (!identical(output, "float")) return(knn)
  if (!is.list(knn) || !("distances" %in% names(knn))) return(knn)
  if (is_float32_matrix(knn$distances)) return(knn)
  if (requireNamespace("float", quietly = TRUE)) {
    knn$distances <- float::fl(knn$distances)
    attr(knn, "distance_type") <- "float32"
  }
  knn
}

fastembedr_embedding_nn_policy <- function(embedding_backend, n = NULL) {
  embedding_backend <- resolve_embedding_backend(embedding_backend)
  n <- suppressWarnings(as.integer(n %||% NA_integer_))
  small_enough_for_exact <- length(n) == 1L && !is.na(n) && n < 100000L
  method <- if (isTRUE(small_enough_for_exact)) "exact" else "ivf"
  if (identical(embedding_backend, "cuda")) {
    return(list(
      backend = "cuda",
      method = method,
      tuning = "auto",
      target_recall = 0.99
    ))
  }
  if (identical(embedding_backend, "metal")) {
    return(list(
      backend = "metal",
      method = if (length(n) == 1L && !is.na(n) && n < 4096L) "exact" else "ivf",
      tuning = "auto",
      target_recall = 0.99
    ))
  }
  list(
    backend = "cpu",
    method = "hnsw",
    tuning = "auto",
    target_recall = 0.99
  )
}

fastembedr_nn_policy_engine <- function(policy, keep_gpu = FALSE) {
  if (is.list(policy) && identical(policy$backend, "cuda")) {
    provider <- if (identical(policy$method %||% "auto", "ivf")) {
      "native_cuvs"
    } else {
      "native_faiss"
    }
    prefix <- if (isTRUE(keep_gpu)) {
      paste0(provider, "_gpu_")
    } else {
      paste0(provider, "_cuda_host_")
    }
    paste0(prefix, policy$method %||% "auto")
  } else if (is.list(policy) && identical(policy$backend, "cpu")) {
    paste0("native_cpu_", policy$method %||% "hnsw")
  } else if (is.list(policy) && identical(policy$backend, "metal")) {
    paste0("native_metal_", policy$method %||% "ivf")
  } else {
    "faissR"
  }
}

fastembedr_nn_without_self <- function(data,
                                       k,
                                       backend,
                                       method = "auto",
                                       metric = "euclidean",
                                       output = "double",
                                       n_threads = NULL,
                                       tuning = "auto",
                                       target_recall = NULL,
                                       keep_gpu = FALSE) {
  k <- as.integer(k)
  target_recall <- target_recall %||% 0.99
  if (identical(backend, "cuda")) {
    if (!method %in% c("auto", "exact", "flat", "bruteforce", "ivf")) {
      stop(
        "CUDA one-call KNN supports methods `auto`, `exact`, and `ivf`.",
        call. = FALSE
      )
    }
    data_n <- if (is_float32_matrix(data)) {
      nrow(methods::slot(data, "Data"))
    } else {
      nrow(data)
    }
    exact_route <- method %in% c("exact", "flat", "bruteforce") ||
      (identical(method, "auto") && data_n < 100000L)
    if (!isTRUE(native_cuda_knn_available_cpp())) {
      stop(
        "Native CUDA KNN is unavailable in this fastEmbedR build; no CPU ",
        "fallback was used.",
        call. = FALSE
      )
    }
    if (isTRUE(exact_route) && !isTRUE(native_cuda_faiss_gpu_available_cpp())) {
      stop(
        "The validated CUDA exact KNN route requires a fastEmbedR build linked ",
        "directly to FAISS GPU. Reinstall with FASTEMBEDR_USE_FAISS_GPU=1; ",
        "no slower cuVS or CPU fallback was used.",
        call. = FALSE
      )
    }
    out <- native_cuda_knn_cpp(
      data,
      k = k,
      method = method,
      metric = metric,
      target_recall = target_recall,
      keep_gpu = isTRUE(keep_gpu)
    )
    if (isTRUE(keep_gpu)) return(out)
    return(fastembedr_convert_knn_distances(out, output))
  }
  if (identical(backend, "cpu") && method %in% c("auto", "hnsw") &&
      metric %in% c("euclidean", "cosine", "correlation")) {
    out <- native_hnsw_knn_cpp(
      data,
      k = k,
      n_threads = normalize_nn_threads(n_threads),
      metric = metric,
      target_recall = target_recall
    )
    attr(out, "backend") <- "cpu"
    attr(out, "method") <- "native_hnsw"
    attr(out, "exclude_self") <- TRUE
    return(fastembedr_convert_knn_distances(out, output))
  }
  if (identical(backend, "metal") && method %in% c("auto", "exact", "ivf") &&
      metric %in% c("euclidean", "cosine", "correlation")) {
    if (!isTRUE(native_metal_knn_available_cpp())) {
      stop("Native Metal KNN was requested but is unavailable in this build.", call. = FALSE)
    }
    out <- native_metal_knn_cpp(
      data,
      k = k,
      method = method,
      metric = metric,
      target_recall = target_recall
    )
    if (identical(out$method, "native_metal_ivf") && !isTRUE(out$target_met)) {
      stop(
        "Native Metal IVF could not meet the requested KNN recall target; ",
        "no CPU fallback was used.",
        call. = FALSE
      )
    }
    attr(out, "backend") <- "metal"
    attr(out, "exclude_self") <- TRUE
    return(fastembedr_convert_knn_distances(out, output))
  }
  fn <- fastembedr_faissr_function("nn")
  use_exclude_self <- fastembedr_supports_formal(fn, "exclude_self")
  args <- list(
    data = data,
    k = if (use_exclude_self) k else k + 1L,
    backend = backend,
    method = method,
    metric = metric,
    tuning = tuning,
    output = output,
    n_threads = n_threads
  )
  if (use_exclude_self) {
    args$exclude_self <- TRUE
  }
  if (!is.null(target_recall)) {
    args$target_recall <- target_recall
  }
  out <- fastembedr_call_supported(fn, args)
  if (!use_exclude_self && is.list(out) && !is.null(out$indices) && !is.null(out$distances)) {
    idx <- out$indices
    dst <- out$distances
    n <- nrow(idx)
    if (n > 0L && ncol(idx) > k) {
      new_idx <- matrix(NA_integer_, n, k)
      new_dst <- matrix(NA_real_, n, k)
      for (i in seq_len(n)) {
        keep <- which(idx[i, ] != i)
        if (length(keep) < k) keep <- seq_len(ncol(idx))
        keep <- keep[seq_len(min(k, length(keep)))]
        new_idx[i, seq_along(keep)] <- as.integer(idx[i, keep, drop = TRUE])
        new_dst[i, seq_along(keep)] <- as.numeric(dst[i, keep, drop = TRUE])
      }
      out$indices <- new_idx
      out$distances <- new_dst
    } else if (ncol(idx) > k) {
      out$indices <- idx[, seq_len(k), drop = FALSE]
      out$distances <- dst[, seq_len(k), drop = FALSE]
    }
  }
  fastembedr_convert_knn_distances(out, output)
}
