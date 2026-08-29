test_that("device summaries tolerate failed system queries", {
    local_mocked_bindings(
        device_query_executable = function(command) {
            stats::setNames("/mock/device-query", command)
        },
        device_query_output = function(...) warning("query failed"),
        .package = "fastEmbedR"
    )

    expect_identical(
        fastEmbedR:::cuda_device_summary(TRUE),
        "CUDA device available (name query unavailable)"
    )
    expect_identical(
        fastEmbedR:::metal_device_summary(TRUE),
        "Metal device available (name query unavailable)"
    )
})

test_that("device summaries do not query unavailable backends", {
    expect_identical(fastEmbedR:::cuda_device_summary(FALSE), NA_character_)
    expect_identical(fastEmbedR:::metal_device_summary(FALSE), NA_character_)
})
