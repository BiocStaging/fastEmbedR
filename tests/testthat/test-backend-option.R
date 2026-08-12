test_that("fastEmbedR backend precedence is explicit, option, environment, CPU", {
  old_global_option <- getOption("backend", NULL)
  old_option <- getOption("fastEmbedR.backend", NULL)
  old_global_env <- Sys.getenv("BACKEND", unset = NA_character_)
  old_env <- Sys.getenv("FASTEMBEDR_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_global_option)
    options(fastEmbedR.backend = old_option)
    if (is.na(old_global_env)) Sys.unsetenv("BACKEND") else Sys.setenv(BACKEND = old_global_env)
    if (is.na(old_env)) Sys.unsetenv("FASTEMBEDR_BACKEND") else Sys.setenv(FASTEMBEDR_BACKEND = old_env)
  }, add = TRUE)
  options(backend = NULL, fastEmbedR.backend = NULL); Sys.unsetenv(c("BACKEND", "FASTEMBEDR_BACKEND"))
  expect_identical(fastEmbedR_backend(), "cpu")
  Sys.setenv(BACKEND = "metal"); expect_identical(fastEmbedR_backend(), "metal")
  options(backend = "cuda", fastEmbedR.backend = "cpu"); expect_identical(fastEmbedR_backend(), "cuda")
  expect_identical(fastEmbedR:::resolve_embedding_backend("cpu"), "cpu")
  expect_error(fastEmbedR:::resolve_embedding_backend("auto"), "must be one of")
})

test_that("principal public functions defer omitted backends", {
  functions <- list(umap, opentsne, pca, knn_graph, graph_cluster,
                    precompute_knn, precompute_query_knn, embed_knn)
  expect_true(all(vapply(functions, function(fn) is.null(formals(fn)$backend), logical(1))))
})
