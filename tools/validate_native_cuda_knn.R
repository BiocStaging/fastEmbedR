args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args)) args[[1L]] else "native_cuda_knn_validation"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

library(fastEmbedR)

if (!isTRUE(fastEmbedR:::native_cuda_knn_available_cpp())) {
  stop("This fastEmbedR build does not expose native RAPIDS cuVS KNN.")
}

exact_reference <- function(x, k) {
  squared_norm <- rowSums(x * x)
  d2 <- outer(squared_norm, squared_norm, "+") - 2 * tcrossprod(x)
  diag(d2) <- Inf
  indices <- t(vapply(seq_len(nrow(x)), function(i) {
    order(d2[i, ], method = "radix")[seq_len(k)]
  }, integer(k)))
  list(indices = indices)
}

recall_at_k <- function(observed, expected) {
  k <- ncol(expected$indices)
  mean(vapply(seq_len(nrow(expected$indices)), function(i) {
    length(intersect(observed$indices[i, ], expected$indices[i, ])) / k
  }, numeric(1)))
}

set.seed(4)
x_small <- matrix(rnorm(2048L * 32L), nrow = 2048L)
truth_small <- exact_reference(x_small, 15L)
exact_time <- system.time({
  exact_device <- fastEmbedR:::native_cuda_knn_cpp(
    x_small, 15L, "exact", "euclidean", 0.99, TRUE
  )
})[["elapsed"]]
exact_host <- fastEmbedR:::native_cuda_knn_to_host_cpp(exact_device)
if (!inherits(exact_device, "fastEmbedR_gpu_knn")) {
  stop("Unexpected native CUDA KNN class: ", paste(class(exact_device), collapse = ", "))
}
stopifnot(
  identical(exact_device$result_residency, "cuda"),
  identical(exact_device$distance_type, "float32"),
  isTRUE(all.equal(recall_at_k(exact_host, truth_small), 1))
)

set.seed(5)
x_ivf <- matrix(rnorm(10000L * 32L), nrow = 10000L)
ivf_truth_device <- fastEmbedR:::native_cuda_knn_cpp(
  x_ivf, 15L, "exact", "euclidean", 0.99, TRUE
)
ivf_truth <- fastEmbedR:::native_cuda_knn_to_host_cpp(ivf_truth_device)
ivf_time <- system.time({
  ivf_device <- fastEmbedR:::native_cuda_knn_cpp(
    x_ivf, 15L, "ivf", "euclidean", 0.99, TRUE
  )
})[["elapsed"]]
ivf_host <- fastEmbedR:::native_cuda_knn_to_host_cpp(ivf_device)
ivf_recall <- recall_at_k(ivf_host, ivf_truth)
if (ivf_recall < 0.99) {
  stop(sprintf("Native IVF recall %.6f is below target 0.99.", ivf_recall))
}
if (!isTRUE(ivf_device$target_met)) {
  stop("Native IVF metadata did not record the requested recall tier as met.")
}

results <- data.frame(
  route = c("exact", "ivf_flat"),
  provider = c(exact_device$backend_used, ivf_device$backend_used),
  n = c(nrow(x_small), nrow(x_ivf)),
  p = c(ncol(x_small), ncol(x_ivf)),
  k = 15L,
  elapsed_sec = c(exact_time, ivf_time),
  recall_at_15 = c(1, ivf_recall),
  pilot_recall = c(1, ivf_device$pilot_recall),
  nprobe = c(NA_integer_, ivf_device$nprobe),
  tuning_attempts = c(0L, ivf_device$tuning_attempts),
  resident_result_bytes = c(
    exact_device$resident_result_bytes,
    ivf_device$resident_result_bytes
  ),
  peak_temporary_search_bytes = c(
    exact_device$peak_temporary_search_bytes,
    ivf_device$peak_temporary_search_bytes
  ),
  stringsAsFactors = FALSE
)

utils::write.csv(
  results,
  file.path(output_dir, "native_cuda_knn_smoke.csv"),
  row.names = FALSE
)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
print(results)
