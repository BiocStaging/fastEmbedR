#!/usr/bin/env Rscript

# Produce machine-readable coverage evidence without implying that covr can
# observe code executed inside Metal or CUDA kernels.

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag, default = "") {
  hit <- which(args == flag)
  if (!length(hit) || hit[[1L]] == length(args)) return(default)
  args[[hit[[1L]] + 1L]]
}

output_dir <- value_after("--output-dir", file.path("coverage-evidence", "current"))
coverage_rds <- value_after("--coverage-rds", "")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("covr", quietly = TRUE)) {
  stop("Install `covr` before generating coverage evidence.", call. = FALSE)
}

run_command <- paste(c("Rscript", commandArgs(trailingOnly = FALSE)), collapse = " ")
if (nzchar(coverage_rds)) {
  if (!file.exists(coverage_rds)) {
    stop("Coverage RDS does not exist: ", coverage_rds, call. = FALSE)
  }
  coverage <- readRDS(coverage_rds)
  coverage_source <- normalizePath(coverage_rds)
} else {
  coverage <- covr::package_coverage(type = "tests", quiet = FALSE)
  coverage_source <- "generated_by_this_command"
  saveRDS(coverage, file.path(output_dir, "coverage.rds"))
}

tallied <- covr:::tally_coverage(coverage)
if (!nrow(tallied)) stop("No instrumented lines were reported by covr.")
tallied$covered <- tallied$value > 0

summarize_lines <- function(data) {
  total <- nrow(data)
  covered <- sum(data$covered)
  data.frame(
    covered_lines = covered,
    instrumented_lines = total,
    coverage_percent = if (total) 100 * covered / total else NA_real_,
    stringsAsFactors = FALSE
  )
}

file_parts <- split(tallied, tallied$filename)
by_file <- do.call(rbind, lapply(names(file_parts), function(path) {
  cbind(
    data.frame(file = path, stringsAsFactors = FALSE),
    summarize_lines(file_parts[[path]])
  )
}))
row.names(by_file) <- NULL

classify_file <- function(path) {
  if (startsWith(path, "R/")) return("R")
  if (grepl("metal", path, ignore.case = TRUE)) return("Metal host")
  if (grepl("cuda", path, ignore.case = TRUE)) return("CUDA host")
  if (startsWith(path, "src/")) return("portable native")
  "other"
}
by_file$component <- vapply(by_file$file, classify_file, character(1))
by_file <- by_file[order(by_file$component, by_file$file), ]

component_parts <- split(tallied, vapply(tallied$filename, classify_file, character(1)))
by_component <- do.call(rbind, lapply(names(component_parts), function(component) {
  cbind(
    data.frame(component = component, measurement = "covr line coverage",
               stringsAsFactors = FALSE),
    summarize_lines(component_parts[[component]]),
    data.frame(
      status = "measured",
      limitation = if (component %in% c("Metal host", "CUDA host")) {
        "Host-side compiled code only; device kernels are not observed by covr."
      } else {
        "Coverage records instrumented source lines executed by the R test suite."
      },
      stringsAsFactors = FALSE
    )
  )
}))

unmeasured <- data.frame(
  component = c("Metal GPU kernels", "CUDA GPU kernels"),
  measurement = "device-kernel coverage",
  covered_lines = NA_integer_,
  instrumented_lines = NA_integer_,
  coverage_percent = NA_real_,
  status = "not_measured",
  limitation = c(
    "Requires a real Apple Metal runner and a device-aware kernel coverage method.",
    "Requires a real NVIDIA CUDA runner and a device-aware kernel coverage method."
  ),
  stringsAsFactors = FALSE
)
by_component <- rbind(by_component, unmeasured)

overall <- cbind(
  data.frame(component = "overall instrumented host code",
             measurement = "covr line coverage", stringsAsFactors = FALSE),
  summarize_lines(tallied),
  data.frame(
    status = "measured",
    limitation = "Does not include Metal or CUDA device-kernel execution coverage.",
    stringsAsFactors = FALSE
  )
)
by_component <- rbind(overall, by_component)

git_output <- function(arguments) {
  out <- system2("git", arguments, stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(out, "status")) && attr(out, "status") != 0L) return(NA_character_)
  trimws(paste(out, collapse = "\n"))
}
commit <- git_output(c("rev-parse", "HEAD"))
status <- git_output(c("status", "--porcelain"))
worktree_state <- if (is.na(status)) "unknown" else if (nzchar(status)) "dirty" else "clean"
description <- read.dcf("DESCRIPTION", fields = c("Package", "Version"))
metadata <- data.frame(
  field = c(
    "package", "package_version", "package_commit", "worktree_state",
    "coverage_source", "generated_utc", "R_version", "platform", "command"
  ),
  value = c(
    description[[1L, "Package"]], description[[1L, "Version"]], commit,
    worktree_state, coverage_source,
    format(Sys.time(), tz = "UTC", usetz = TRUE), R.version.string,
    R.version$platform, run_command
  ),
  stringsAsFactors = FALSE
)

write.csv(by_file, file.path(output_dir, "coverage_by_file.csv"), row.names = FALSE)
write.csv(by_component, file.path(output_dir, "coverage_by_component.csv"),
          row.names = FALSE)
write.csv(metadata, file.path(output_dir, "coverage_metadata.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))

cat(sprintf(
  "Coverage evidence written to %s (overall host coverage %.2f%%; worktree %s).\n",
  normalizePath(output_dir), overall$coverage_percent, worktree_state
))
if (!identical(worktree_state, "clean")) {
  cat("This diagnostic result is not immutable release evidence.\n")
}
