#!/usr/bin/env Rscript

# Write a reproducibility bundle for manuscript and benchmark runs.
# The script is intentionally small and dependency-light. If jsonlite is
# installed, it also writes a JSON manifest; otherwise it writes text files.

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L || is.na(x)) y else x

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[gsub("-", "_", kv[[1L]])]] <- if (length(kv) > 1L) paste(kv[-1L], collapse = "=") else TRUE
  }
  out
}

run_cmd <- function(command, args = character()) {
  path <- Sys.which(command)
  if (!nzchar(path)) return(NA_character_)
  out <- tryCatch(
    system2(path, args, stdout = TRUE, stderr = TRUE),
    warning = function(w) conditionMessage(w),
    error = function(e) conditionMessage(e)
  )
  paste(out, collapse = "\n")
}

git_value <- function(args) {
  out <- run_cmd("git", args)
  if (length(out) != 1L || is.na(out)) NA_character_ else trimws(out)
}

pkg_version <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(pkg))
}

backend_capture <- function() {
  if (!requireNamespace("faissR", quietly = TRUE)) return("faissR not installed")
  paste(capture.output(print(try(faissR::backend_info(), silent = TRUE))), collapse = "\n")
}

cuda_probe <- function() {
  list(
    nvidia_smi = run_cmd(
      "nvidia-smi",
      c("--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader")
    ),
    nvidia_smi_full = run_cmd("nvidia-smi"),
    nvcc_version = run_cmd("nvcc", "--version"),
    cuda_home = Sys.getenv("CUDA_HOME", unset = NA_character_),
    ld_library_path = Sys.getenv("LD_LIBRARY_PATH", unset = NA_character_),
    faissR_cuda_available = if (requireNamespace("faissR", quietly = TRUE)) {
      paste(capture.output(print(try(faissR::cuda_available(), silent = TRUE))), collapse = "\n")
    } else {
      "faissR not installed"
    },
    faissR_cuvs_available = if (requireNamespace("faissR", quietly = TRUE)) {
      paste(capture.output(print(try(faissR::cuvs_available(), silent = TRUE))), collapse = "\n")
    } else {
      "faissR not installed"
    }
  )
}

benchmark_commands <- function(seed, k, perplexity, threads_cpu, timeout) {
  list(
    github_mnist70k_example = paste(
      "Rscript tools/benchmark_github_mnist70k.R",
      "--n=70000",
      paste0("--k=", k),
      paste0("--perplexity=", perplexity),
      paste0("--threads=", threads_cpu),
      "--run-metal=true",
      "--run-cuda=false",
      "--run-references=true",
      paste0("--out-dir=results/mnist70k_github_benchmark_seed", seed)
    ),
    reference_validation = paste(
      "Rscript tools/validate_reference_implementations.R",
      "--out-dir=results/reference_validation_current",
      "--threads=2",
      "--seed=4",
      "--perplexity=10",
      "--k=31"
    ),
    hpc_cpu12 = paste(
      "DATASETS=MNIST,FashionMNIST,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,MetRef",
      paste0("K=", k),
      paste0("PERPLEXITY=", perplexity),
      paste0("TIMEOUT=", timeout),
      "sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh"
    ),
    hpc_cuda = paste(
      "DATASETS=MNIST,FashionMNIST,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,MetRef",
      paste0("K=", k),
      paste0("PERPLEXITY=", perplexity),
      paste0("TIMEOUT=", timeout),
      "sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh"
    )
  )
}

write_text_list <- function(x, path, indent = "") {
  con <- file(path, open = "wt")
  on.exit(close(con), add = TRUE)
  recurse <- function(obj, prefix = "") {
    if (is.list(obj)) {
      for (nm in names(obj)) {
        cat(prefix, nm, ":\n", sep = "", file = con)
        recurse(obj[[nm]], paste0(prefix, "  "))
      }
    } else {
      val <- paste(as.character(obj), collapse = "\n")
      if (!nzchar(val) || is.na(val)) val <- "NA"
      lines <- strsplit(val, "\n", fixed = TRUE)[[1L]]
      for (line in lines) cat(prefix, line, "\n", sep = "", file = con)
    }
  }
  recurse(x, indent)
  invisible(path)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
out_dir <- normalizePath(
  args$out_dir %||% file.path("results", paste0("manuscript_reproducibility_", format(Sys.time(), "%Y%m%d_%H%M%S"))),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

seed <- as.integer(args$seed %||% 4L)
k <- as.integer(args$k %||% 30L)
perplexity <- as.numeric(args$perplexity %||% 15)
threads_cpu <- as.integer(args$threads %||% 12L)
timeout <- as.integer(args$timeout %||% 10800L)

manifest <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  repository = "https://github.com/tkcaccia/fastEmbedR",
  git_commit = git_value(c("rev-parse", "HEAD")),
  git_describe = git_value(c("describe", "--tags", "--always", "--dirty")),
  git_status_short = git_value(c("status", "--short")),
  manuscript_release_tag = Sys.getenv("FASTEMBEDR_MANUSCRIPT_TAG", unset = "v0.1.0-manuscript"),
  archival_snapshot = Sys.getenv("FASTEMBEDR_ZENODO_DOI", unset = "Zenodo DOI to be minted from the manuscript release tag before submission"),
  random_seed = seed,
  benchmark_parameters = list(k = k, perplexity = perplexity, cpu_threads = threads_cpu, timeout_seconds = timeout),
  benchmark_commands = benchmark_commands(seed, k, perplexity, threads_cpu, timeout),
  hardware = list(
    system = Sys.info(),
    cpu = run_cmd("sysctl", c("-n", "machdep.cpu.brand_string")),
    linux_cpu = run_cmd("lscpu"),
    memory = run_cmd("free", "-h")
  ),
  r = list(
    version = R.version.string,
    platform = R.version$platform,
    home = R.home(),
    library_paths = .libPaths(),
    session_info = paste(capture.output(utils::sessionInfo()), collapse = "\n")
  ),
  packages = list(
    fastEmbedR = pkg_version("fastEmbedR"),
    faissR = pkg_version("faissR"),
    Rcpp = pkg_version("Rcpp"),
    float = pkg_version("float"),
    Rtsne = pkg_version("Rtsne"),
    uwot = pkg_version("uwot"),
    umap = pkg_version("umap"),
    jsonlite = pkg_version("jsonlite")
  ),
  backends = list(
    faissR_backend_info = backend_capture(),
    cuda = cuda_probe()
  ),
  environment_files = list(
    r_package_installation = "docs/installation.md",
    backend_capabilities = "docs/backend-capabilities.md",
    lightweight_r_environment = "tools/reproducibility/benchmark_environment.yml",
    hpc_cpu_wrapper = "tools/hpc_embeddings/benchmark_embeddings_float32_cpu12.sh",
    hpc_cuda_wrapper = "tools/hpc_embeddings/benchmark_embeddings_float32_cuda.sh",
    hpc_r_driver = "tools/hpc_embeddings/benchmark_embeddings_float32_publication.R"
  )
)

writeLines(manifest$r$session_info, file.path(out_dir, "sessionInfo.txt"))
write_text_list(manifest, file.path(out_dir, "reproducibility_manifest.txt"))
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(manifest, file.path(out_dir, "reproducibility_manifest.json"), auto_unbox = TRUE, pretty = TRUE)
}

cat("Wrote reproducibility bundle to:", normalizePath(out_dir), "\n")
