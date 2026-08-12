test_that("fastEmbedR backend precedence is explicit, option, environment, CPU", {
  old_option <- getOption("fastEmbedR.backend", NULL)
  old_env <- Sys.getenv("FASTEMBEDR_BACKEND", unset = NA_character_)
  on.exit({
    options(fastEmbedR.backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("FASTEMBEDR_BACKEND") else Sys.setenv(FASTEMBEDR_BACKEND = old_env)
  }, add = TRUE)
  options(fastEmbedR.backend = NULL); Sys.unsetenv("FASTEMBEDR_BACKEND")
  expect_identical(fastEmbedR_backend(), "cpu")
  Sys.setenv(FASTEMBEDR_BACKEND = "metal"); expect_identical(fastEmbedR_backend(), "metal")
  options(fastEmbedR.backend = "cuda"); expect_identical(fastEmbedR_backend(), "cuda")
  expect_identical(fastEmbedR:::resolve_embedding_backend("cpu"), "cpu")
  expect_error(fastEmbedR:::resolve_embedding_backend("auto"), "must be one of")
})

test_that("principal public functions defer omitted backends", {
  functions <- list(umap, opentsne, pca, knn_graph, graph_cluster,
                    precompute_knn, precompute_query_knn, embed_knn)
  expect_true(all(vapply(functions, function(fn) is.null(formals(fn)$backend), logical(1))))
})
