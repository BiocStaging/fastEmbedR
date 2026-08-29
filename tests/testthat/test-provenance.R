test_that("installed provenance inventory is machine readable", {
    skip_if_not_installed("jsonlite")

    path <- system.file(
        "THIRD_PARTY_DEPENDENCIES.json",
        package = "fastEmbedR"
    )
    expect_true(nzchar(path))
    expect_true(file.exists(path))

    inventory <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    expect_identical(inventory$package$name, "fastEmbedR")
    expect_identical(inventory$package$license_spdx, "MIT")

    ids <- vapply(inventory$components, `[[`, character(1L), "id")
    expect_false(anyDuplicated(ids) > 0L)
    expect_true(all(c(
        "faiss", "faiss-mlx", "faissr", "dlpack", "applesiliconfft",
        "cuvs", "raft", "cuda-toolkit", "apple-frameworks"
    ) %in% ids))

    relationships <- vapply(
        inventory$components,
        `[[`,
        character(1L),
        "classification"
    )
    expect_true(all(
        relationships %in% names(inventory$classification_definitions)
    ))
})
