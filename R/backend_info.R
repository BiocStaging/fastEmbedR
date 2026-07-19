# Internal backend summary used by tests and diagnostics.
backend_info <- function() {
  cuda_knn <- backend_flag(native_cuda_knn_available_cpp)
  cuda_embedding <- backend_flag(embedding_cuda_available_cpp)
  metal_knn <- backend_flag(native_metal_knn_available_cpp)
  metal_embedding <- backend_flag(embedding_metal_available_cpp)
  knn_available <- c(TRUE, cuda_knn, cuda_knn, metal_knn)
  embedding_available <- c(TRUE, FALSE, cuda_embedding, metal_embedding)

  data.frame(
    backend = c("cpu", "cuvs", "cuda", "metal"),
    available = knn_available | embedding_available,
    knn_available = knn_available,
    embedding_available = embedding_available,
    explicit_backend = c("cpu", "cuda", "cuda", "metal"),
    device = c(cpu_summary(), NA_character_, NA_character_, NA_character_),
    runtime = rep(R.version$platform, 4L),
    note = c(
      "Package-native CPU HNSW and CPU embedding are available.",
      if (cuda_knn) {
        "Package-native CUDA KNN uses the linked RAPIDS cuVS C API."
      } else {
        "RAPIDS cuVS KNN is unavailable in this build."
      },
      if (cuda_embedding) {
        "Package-native CUDA embedding kernels are available."
      } else {
        "CUDA embedding kernels are unavailable in this build."
      },
      if (metal_knn && metal_embedding) {
        "Package-native Metal KNN and embedding kernels are available."
      } else {
        "One or more Metal components are unavailable in this build."
      }
    ),
    stringsAsFactors = FALSE
  )
}

backend_flag <- function(fn) {
  tryCatch(isTRUE(fn()), error = function(e) FALSE)
}

isTRUE_VECTOR <- function(x) {
  x <- as.logical(x)
  x[is.na(x)] <- FALSE
  x
}

resolve_native_gpu_backend <- function(need_knn = FALSE,
                                       need_embedding = FALSE) {
  backend <- available_native_gpu_backend(
    need_knn = need_knn,
    need_embedding = need_embedding
  )
  if (!is.na(backend)) return(backend)

  need <- c(
    if (isTRUE(need_knn)) "KNN" else NULL,
    if (isTRUE(need_embedding)) "embedding" else NULL
  )
  if (length(need) == 0L) need <- "requested"
  stop(
    "No native GPU backend is available for ",
    paste(need, collapse = " and "),
    ". Rebuild fastEmbedR with the requested native backend enabled.",
    call. = FALSE
  )
}

available_native_gpu_backend <- function(need_knn = FALSE,
                                         need_embedding = FALSE) {
  cuda_knn_available <- backend_flag(native_cuda_knn_available_cpp)
  cuda_ok <- (!isTRUE(need_knn) || cuda_knn_available) &&
    (!isTRUE(need_embedding) || backend_flag(embedding_cuda_available_cpp))
  if (cuda_ok) return("cuda")

  metal_ok <- (!isTRUE(need_knn) || backend_flag(native_metal_knn_available_cpp)) &&
    (!isTRUE(need_embedding) || backend_flag(embedding_metal_available_cpp))
  if (metal_ok) return("metal")

  NA_character_
}

resolve_backend_request <- function(backend,
                                    need_knn = FALSE,
                                    need_embedding = FALSE) {
  if (identical(backend, "gpu")) {
    resolve_native_gpu_backend(
      need_knn = need_knn,
      need_embedding = need_embedding
    )
  } else {
    backend
  }
}

embedding_backend_choices <- function() {
  c("cpu", "cuda", "metal")
}

resolve_embedding_backend <- function(backend) {
  backend <- match.arg(backend, embedding_backend_choices())
  backend
}

embedding_knn_backend <- function(backend) {
  backend <- resolve_embedding_backend(backend)
  backend
}

fixed_embedding_knn_backend <- function(backend) {
  embedding_knn_backend(backend)
}

cpu_summary <- function() {
  cores <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    "CPU"
  } else {
    paste0("CPU (", cores, " logical cores)")
  }
}
