cache_file <- function(cache_dir, prefix, dataset, n, p, key) {
  safe_dataset <- gsub("[^A-Za-z0-9_.-]+", "_", dataset)
  file.path(cache_dir, sprintf("%s_%s_n%s_p%s_%s.rds", prefix, safe_dataset, n, p, key))
}

set_local_seed <- function(seed) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  seeded_state <- withr::with_seed(
    as.integer(seed),
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  )
  assign(".Random.seed", seeded_state, envir = .GlobalEnv)
  function() {
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
    invisible(NULL)
  }
}
