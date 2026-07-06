args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else file.path('/Users/stefano/Documents/umap/results', paste0('local_cpu_metal_opentsne_', format(Sys.time(), '%Y%m%d_%H%M%S')))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(fastEmbedR)
  library(faissR)
})

message('fastEmbedR: ', find.package('fastEmbedR'))
message('faissR: ', find.package('faissR'))
message('Metal available: ', fastEmbedR:::embedding_metal_available_cpp())
message('Metal openTSNE native: ', fastEmbedR:::metal_opentsne_native_available())

load('/Users/stefano/Documents/fastEmbedR/Data/MetRef/MetRef.RData')
if (!exists('dataset')) stop('dataset object not found')
x <- dataset$data
labels <- dataset$labels
if (!is.matrix(x)) x <- as.matrix(x)
storage.mode(x) <- 'double'
labels <- as.factor(labels)

# Mean-center only for consistency with current benchmark convention.
x_center <- sweep(x, 2L, colMeans(x, na.rm = TRUE), '-')

set.seed(4)
k <- 30L
perplexity <- 15
threads <- 4L

knn_time <- system.time({
  knn <- faissR::nn(x_center, k = k + 1L, backend = 'cpu')
})
# Remove self if faissR returned it. opentsne_knn can also normalize, but keeping explicit here.
idx <- knn$indices
dst <- knn$distances
if (ncol(idx) > k) {
  idx <- idx[, seq_len(k + 1L), drop = FALSE]
  dst <- dst[, seq_len(k + 1L), drop = FALSE]
  keep_idx <- matrix(NA_integer_, nrow(idx), k)
  keep_dst <- matrix(NA_real_, nrow(idx), k)
  for (i in seq_len(nrow(idx))) {
    keep <- idx[i, ] != i
    keep_idx[i, ] <- idx[i, keep][seq_len(k)]
    keep_dst[i, ] <- dst[i, keep][seq_len(k)]
  }
  knn <- list(indices = keep_idx, distances = keep_dst)
}

Y_init <- fastEmbedR::opentsne_pca_init(x_center, seed = 4L, backend = 'cpu')

run_one <- function(label, backend) {
  gc()
  t <- system.time({
    y <- fastEmbedR::opentsne_knn(
      knn,
      perplexity = perplexity,
      Y_init = Y_init,
      seed = 4L,
      backend = backend,
      n_threads = threads,
      negative_gradient_method = 'auto'
    )
  })
  cfg <- attr(y, 'fastEmbedR_config')
  list(
    label = label,
    backend = backend,
    layout = y,
    sec = unname(t[['elapsed']]),
    status = 'success',
    optimizer = cfg$optimizer %||% NA_character_,
    repulsion = cfg$repulsion %||% NA_character_,
    actual_iter = cfg$max_iter_actual %||% NA_real_,
    error = NA_character_
  )
}

`%||%` <- function(a, b) if (is.null(a)) b else a
res <- list()
res[[1]] <- tryCatch(run_one('fastEmbedR openTSNE CPU', 'cpu'), error = function(e) list(label='fastEmbedR openTSNE CPU', backend='cpu', layout=NULL, sec=NA_real_, status='failed', optimizer=NA_character_, repulsion=NA_character_, actual_iter=NA_real_, error=conditionMessage(e)))
res[[2]] <- tryCatch(run_one('fastEmbedR openTSNE Metal', 'metal'), error = function(e) list(label='fastEmbedR openTSNE Metal', backend='metal', layout=NULL, sec=NA_real_, status='failed', optimizer=NA_character_, repulsion=NA_character_, actual_iter=NA_real_, error=conditionMessage(e)))

tab <- do.call(rbind, lapply(res, function(z) data.frame(
  method = z$label,
  backend = z$backend,
  status = z$status,
  n = nrow(x_center),
  p = ncol(x_center),
  k = k,
  perplexity = perplexity,
  knn_sec = unname(knn_time[['elapsed']]),
  embed_sec = z$sec,
  optimizer = z$optimizer,
  repulsion = z$repulsion,
  actual_iter = z$actual_iter,
  error = z$error,
  stringsAsFactors = FALSE
)))
write.csv(tab, file.path(out_dir, 'cpu_metal_opentsne_timing.csv'), row.names = FALSE)
print(tab)

cols <- grDevices::rainbow(length(levels(labels)))[as.integer(labels)]
png(file.path(out_dir, 'cpu_metal_opentsne_plots.png'), width = 1800, height = 900, res = 140)
par(mfrow = c(1, 2), mar = c(3.5, 3.5, 3, 1), bg = 'white')
for (z in res) {
  if (is.null(z$layout)) {
    plot.new(); title(main = paste0(z$label, '\nFAILED')); text(0.5, 0.5, z$error, cex = 0.9)
  } else {
    plot(z$layout, pch = 20, cex = 1.0, col = cols, xlab = 'openTSNE1', ylab = 'openTSNE2', main = sprintf('%s\nembed %.3fs', z$label, z$sec))
  }
}
dev.off()

png(file.path(out_dir, 'cpu_metal_opentsne_speed.png'), width = 1100, height = 750, res = 140)
ok <- tab$status == 'success'
barplot(tab$embed_sec[ok], names.arg = tab$method[ok], las = 2, ylab = 'Embedding seconds', col = c('#4C78A8', '#F58518')[seq_len(sum(ok))], main = 'openTSNE CPU vs Metal')
dev.off()

message('OUT_DIR=', out_dir)
