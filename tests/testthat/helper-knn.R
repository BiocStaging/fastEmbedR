test_exact_knn <- function(data,
                           query = NULL,
                           k = NULL,
                           exclude_self = FALSE,
                           backend = NULL,
                           ...) {
  data <- as.matrix(data)
  same_data <- is.null(query)
  if (same_data) query <- data
  query <- as.matrix(query)
  if (is.null(k)) {
    k <- fastEmbedR:::auto_k(data, include_self = !isTRUE(exclude_self))
  }
  k <- as.integer(k)

  d2 <- outer(rowSums(query * query), rowSums(data * data), "+") -
    2 * tcrossprod(query, data)
  d2[d2 < 0] <- 0
  if (same_data) {
    diag(d2) <- if (isTRUE(exclude_self)) Inf else 0
  }

  indices <- t(apply(d2, 1L, order))[, seq_len(k), drop = FALSE]
  distances <- matrix(
    sqrt(d2[cbind(rep(seq_len(nrow(query)), each = k), as.vector(t(indices)))]),
    nrow = nrow(query),
    byrow = TRUE
  )
  structure(
    list(indices = indices, distances = distances),
    backend = "test_exact",
    exclude_self = isTRUE(exclude_self)
  )
}
