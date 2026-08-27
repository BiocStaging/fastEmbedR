#' Report compiled fastEmbedR backend capabilities
#'
#' Reports whether the installed fastEmbedR build can provide nearest-neighbor
#' search, embedding optimization, and graph clustering through each native
#' execution backend. This function performs capability checks only; it does
#' not change the backend selected by [fastEmbedR_backend()].
#'
#' The `cuvs` row describes the RAPIDS cuVS nearest-neighbor component used by
#' the CUDA backend. It is diagnostic information and is not itself a value
#' accepted by public `backend` arguments.
#'
#' @return A data frame with one row per backend or backend component. In
#'   addition to logical availability columns, the result records the public
#'   backend name, detected device, native KNN engine, internal precision,
#'   linked runtime families, and an explanation when a component is
#'   unavailable. Device-name queries are best effort and do not change the
#'   capability decision made by the compiled native probes.
#' @examples
#' fastEmbedR_capabilities()
#' @export
fastEmbedR_capabilities <- function() {
  cuda_knn <- backend_flag(native_cuda_knn_available_cpp)
  cuda_embedding <- backend_flag(embedding_cuda_available_cpp)
  cuda_clustering <- backend_flag(graph_clustering_cuda_available_cpp)
  metal_knn <- backend_flag(native_metal_knn_available_cpp)
  metal_embedding <- backend_flag(embedding_metal_available_cpp)
  metal_clustering <- backend_flag(graph_clustering_metal_available_cpp)
  knn_available <- c(TRUE, cuda_knn, cuda_knn, metal_knn)
  embedding_available <- c(TRUE, FALSE, cuda_embedding, metal_embedding)
  clustering_available <- c(TRUE, FALSE, cuda_clustering, metal_clustering)
  cuda_available <- cuda_knn || cuda_embedding || cuda_clustering
  metal_available <- metal_knn || metal_embedding || metal_clustering
  cuda_device <- cuda_device_summary(cuda_available)
  metal_device <- metal_device_summary(metal_available)
  unavailable_reason <- c(
    NA_character_,
    if (cuda_knn) NA_character_ else
      "The installed build did not expose a usable RAPIDS cuVS KNN component.",
    if (cuda_available) NA_character_ else
      "The installed build did not expose a usable native CUDA component.",
    if (metal_available) NA_character_ else
      "The installed build did not expose a usable native Metal component."
  )

  data.frame(
    backend = c("cpu", "cuvs", "cuda", "metal"),
    available = knn_available | embedding_available | clustering_available,
    knn_available = knn_available,
    embedding_available = embedding_available,
    clustering_available = clustering_available,
    explicit_backend = c("cpu", "cuda", "cuda", "metal"),
    device = c(cpu_summary(), cuda_device, cuda_device, metal_device),
    knn_engine = c(
      "package-native HNSW",
      "RAPIDS cuVS IVF-Flat",
      "FAISS GPU exact or RAPIDS cuVS IVF-Flat",
      "package-native exact or IVF-Flat"
    ),
    precision = c(
      "float32 native computation; host return follows the public API",
      "float32 device KNN",
      "float32 device computation; host return follows the public API",
      "float32 Metal computation; host return follows the public API"
    ),
    runtime_libraries = c(
      "R, Rcpp, package-native C++17",
      "CUDA runtime, RAPIDS cuVS C API",
      "CUDA runtime, FAISS GPU/cuVS, cuFFT; optional RAFT/RMM TSVD",
      "Foundation, Accelerate, Metal, MPS, MPSGraph"
    ),
    unavailable_reason = unavailable_reason,
    runtime = rep(R.version$platform, 4L),
    note = c(
      "Package-native CPU HNSW and CPU embedding are available.",
      if (cuda_knn) {
        "Package-native CUDA KNN uses the linked RAPIDS cuVS C API."
      } else {
        "RAPIDS cuVS KNN is unavailable in this build."
      },
      if (cuda_embedding && cuda_clustering) {
        "Package-native CUDA embedding and graph-clustering kernels are available."
      } else {
        "One or more CUDA embedding/clustering components are unavailable."
      },
      if (metal_knn && metal_embedding && metal_clustering) {
        "Package-native Metal KNN, embedding, and graph-clustering kernels are available."
      } else {
        "One or more Metal components are unavailable in this build."
      }
    ),
    stringsAsFactors = FALSE
  )
}

# Private compatibility alias for package-internal diagnostics.
backend_info <- function() fastEmbedR_capabilities()

#' Configure the default fastEmbedR execution backend
#'
#' @param backend Optional backend: `"cpu"`, `"cuda"`, or `"metal"`.
#' @return The active backend. Setting returns the previous option invisibly.
#' @examples
#' old_backend <- getOption("fastEmbedR.backend")
#' fastEmbedR_backend("cpu")
#' fastEmbedR_backend()
#' options(fastEmbedR.backend = old_backend)
#' @export
fastEmbedR_backend <- function(backend = NULL) {
  if (is.null(backend)) return(resolve_embedding_backend(NULL))
  backend <- validate_environment_backend(backend, "backend")
  old <- getOption("fastEmbedR.backend", NULL)
  options(fastEmbedR.backend = backend)
  invisible(old)
}

validate_environment_backend <- function(backend, label = "backend") {
  backend <- tolower(as.character(backend))
  if (length(backend) != 1L || is.na(backend) || !nzchar(backend) ||
      !backend %in% embedding_backend_choices()) {
    stop("`", label, "` must be one of \"cpu\", \"cuda\", or \"metal\".", call. = FALSE)
  }
  backend
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
  if (!is.null(backend) && length(backend) == 1L) {
    return(validate_environment_backend(backend))
  }
  option <- getOption("fastEmbedR.backend", NULL)
  if (!is.null(option)) return(validate_environment_backend(option, "option fastEmbedR.backend"))
  environment <- Sys.getenv("FASTEMBEDR_BACKEND", unset = "")
  if (nzchar(environment)) return(validate_environment_backend(environment, "FASTEMBEDR_BACKEND"))
  "cpu"
}

embedding_knn_backend <- function(backend) {
  backend <- resolve_embedding_backend(backend)
  backend
}

fixed_embedding_knn_backend <- function(backend) {
  embedding_knn_backend(backend)
}

cpu_summary <- function() {
  cores <- parallel::detectCores(logical = TRUE)
  if (length(cores) != 1L || is.na(cores) || !is.finite(cores)) {
    "CPU"
  } else {
    paste0("CPU (", cores, " logical cores)")
  }
}

cuda_device_summary <- function(available) {
  if (!isTRUE(available)) return(NA_character_)
  executable <- Sys.which("nvidia-smi")
  if (!nzchar(executable)) return("CUDA device available (name query unavailable)")
  value <- tryCatch(
    suppressWarnings(system2(
      executable,
      c("--query-gpu=name", "--format=csv,noheader"),
      stdout = TRUE,
      stderr = FALSE
    )),
    error = function(e) character()
  )
  value <- trimws(value[nzchar(trimws(value))])
  if (length(value)) value[[1L]] else
    "CUDA device available (name query unavailable)"
}

metal_device_summary <- function(available) {
  if (!isTRUE(available)) return(NA_character_)
  executable <- Sys.which("system_profiler")
  if (!nzchar(executable)) return("Metal device available (name query unavailable)")
  value <- tryCatch(
    suppressWarnings(system2(
      executable,
      "SPDisplaysDataType",
      stdout = TRUE,
      stderr = FALSE
    )),
    error = function(e) character()
  )
  chipset <- grep("^[[:space:]]*Chipset Model:", value, value = TRUE)
  chipset <- trimws(sub("^[[:space:]]*Chipset Model:[[:space:]]*", "", chipset))
  if (length(chipset) && nzchar(chipset[[1L]])) chipset[[1L]] else
    "Metal device available (name query unavailable)"
}
