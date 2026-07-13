exact_knn_reference <- function(x, k) {
  squared_norm <- rowSums(x * x)
  d2 <- outer(squared_norm, squared_norm, "+") - 2 * tcrossprod(x)
  diag(d2) <- Inf
  indices <- t(vapply(seq_len(nrow(x)), function(i) {
    order(d2[i, ], method = "radix")[seq_len(k)]
  }, integer(k)))
  list(indices = indices, distances = sqrt(pmax(0, matrix(d2[cbind(
    rep(seq_len(nrow(x)), each = k), as.vector(t(indices))
  )], nrow(x), k, byrow = TRUE))))
}

knn_recall_test <- function(observed, expected) {
  k <- ncol(expected$indices)
  mean(vapply(seq_len(nrow(expected$indices)), function(i) {
    length(intersect(observed$indices[i, ], expected$indices[i, ])) / k
  }, numeric(1)))
}

test_that("native CPU HNSW reaches its recall tier", {
  set.seed(4)
  x <- matrix(rnorm(600 * 12), nrow = 600)
  truth <- exact_knn_reference(x, 10L)
  observed <- native_hnsw_knn_cpp(x, 10L, 2L, "euclidean", 0.99)

  expect_identical(dim(observed$indices), c(600L, 10L))
  expect_gte(knn_recall_test(observed, truth), 0.99)
  expect_identical(attr(observed, "backend"), "cpu")
  expect_identical(observed$method, "native_hnsw")
})

test_that("native KNN consumes float32 input without a double input copy", {
  skip_if_not_installed("float")
  set.seed(4)
  x <- float::fl(matrix(rnorm(500 * 8), nrow = 500))
  observed <- native_hnsw_knn_cpp(x, 8L, 2L, "euclidean", 0.99)

  expect_identical(dim(observed$indices), c(500L, 8L))
  expect_true(all(is.finite(observed$distances)))
})

test_that("native Metal exact and IVF searches are recall checked", {
  skip_if_not(isTRUE(native_metal_knn_available_cpp()), "Metal is unavailable")
  set.seed(4)
  exact_data <- matrix(rnorm(400 * 10), nrow = 400)
  truth <- exact_knn_reference(exact_data, 10L)
  exact <- native_metal_knn_cpp(exact_data, 10L, "exact", "euclidean", 0.99)
  expect_equal(knn_recall_test(exact, truth), 1, tolerance = 1e-8)

  ivf_data <- matrix(rnorm(2000 * 32), nrow = 2000)
  ivf_truth <- exact_knn_reference(ivf_data, 10L)
  ivf <- native_metal_knn_cpp(ivf_data, 10L, "ivf", "euclidean", 0.99)
  expect_true(isTRUE(ivf$target_met))
  expect_gte(knn_recall_test(ivf, ivf_truth), 0.99)
  expect_identical(attr(ivf, "backend"), "metal")
})

test_that("one-call routing uses native CPU and Metal KNN", {
  expect_identical(fastembedr_embedding_nn_policy("cpu", 70000L)$method, "hnsw")
  expect_identical(fastembedr_embedding_nn_policy("metal", 70000L)$method, "ivf")
  expect_identical(fastembedr_embedding_nn_policy("metal", 1000L)$method, "exact")
  expect_identical(fastembedr_embedding_nn_policy("cuda", 70000L)$method, "exact")
  expect_identical(fastembedr_embedding_nn_policy("cuda", 100000L)$method, "ivf")
  expect_identical(
    fastembedr_nn_policy_engine(
      fastembedr_embedding_nn_policy("cuda", 70000L),
      keep_gpu = TRUE
    ),
    "native_faiss_gpu_exact"
  )
})

test_that("native CUDA KNN never silently falls back", {
  expect_type(native_cuda_knn_available_cpp(), "logical")
  if (!isTRUE(native_cuda_knn_available_cpp())) {
    expect_error(
      fastembedr_nn_without_self(
        matrix(rnorm(64), nrow = 16),
        k = 3L,
        backend = "cuda",
        method = "auto",
        metric = "euclidean",
        keep_gpu = TRUE
      ),
      "Native CUDA KNN"
    )
  }
})

test_that("native CUDA exact KNN remains resident and matches the exact oracle", {
  skip_if_not(isTRUE(native_cuda_knn_available_cpp()), "native CUDA KNN is unavailable")
  skip_if_not(isTRUE(native_cuda_faiss_gpu_available_cpp()), "native FAISS GPU KNN is unavailable")
  set.seed(4)
  x <- matrix(rnorm(256 * 12), nrow = 256)
  truth <- exact_knn_reference(x, 10L)
  device_knn <- native_cuda_knn_cpp(
    x, 10L, "exact", "euclidean", 0.99, keep_gpu = TRUE
  )

  expect_s3_class(device_knn, "fastEmbedR_gpu_knn")
  expect_identical(device_knn$result_residency, "cuda")
  expect_identical(device_knn$layout, "column_major_query_by_k")
  expect_identical(device_knn$distance_type, "float32")
  expect_identical(device_knn$device_to_host_result_copies, 0)
  expect_false(isTRUE(device_knn$cpu_fallback))
  expect_equal(device_knn$resident_result_bytes, 256 * 10 * 8)
  expect_lt(
    device_knn$peak_temporary_search_bytes,
    256 * 11 * 16
  )

  observed <- native_cuda_knn_to_host_cpp(device_knn)
  expect_identical(dim(observed$indices), c(256L, 10L))
  expect_equal(knn_recall_test(observed, truth), 1, tolerance = 1e-8)
  expect_true(all(is.finite(observed$distances)))
})
