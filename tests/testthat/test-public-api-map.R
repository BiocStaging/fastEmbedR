test_that("the API map covers every export exactly once", {
    api <- fastEmbedR_api()
    exports <- sort(getNamespaceExports("fastEmbedR"))

    expect_s3_class(api, "data.frame")
    expect_identical(anyDuplicated(api[["function"]]), 0L)
    expect_setequal(api[["function"]], exports)
    expect_true(all(nzchar(api$purpose)))
    expect_true(all(nzchar(api$input_classes)))
    expect_true(all(nzchar(api$return_class)))
    expect_true(all(nzchar(api$device_residency)))
    expect_true(all(api$tier %in% c(
        "canonical", "advanced", "convenience", "diagnostic", "secondary",
        "compatibility"
    )))
})

test_that("the canonical manuscript example executes from its installed file", {
    path <- system.file(
        "examples", "mloss_canonical_api.R",
        package = "fastEmbedR"
    )
    expect_true(nzchar(path))
    example_env <- new.env(parent = globalenv())
    expect_silent(sys.source(path, envir = example_env))
    expect_s3_class(example_env$fit_umap, "fastEmbedR_embedding")
    expect_s3_class(example_env$knn, "fastEmbedR_knn")
    expect_equal(dim(example_env$layout_tsne), c(150L, 2L))
    expect_equal(dim(example_env$projected$layout), c(5L, 2L))
})
