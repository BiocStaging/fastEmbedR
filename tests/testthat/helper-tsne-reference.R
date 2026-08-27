tsne_reference_probabilities <- function(indices, distances, perplexity) {
  indices <- as.matrix(indices)
  distances <- as.matrix(distances)
  n <- nrow(indices)
  k <- ncol(indices)
  if (min(indices) == 0L) indices <- indices + 1L

  directed <- matrix(0, n, n)
  target <- log(perplexity)
  for (i in seq_len(n)) {
    beta <- 1
    beta_min <- -Inf
    beta_max <- Inf
    d2 <- distances[i, ]^2
    shifted <- d2 - min(d2)
    if (max(shifted) <= .Machine$double.eps * max(1, max(d2))) {
      directed[i, indices[i, ]] <- 1 / k
      next
    }
    for (iteration in seq_len(200L)) {
      unnormalized <- exp(-beta * shifted)
      mass <- sum(unnormalized)
      entropy <- beta * sum(shifted * unnormalized) / mass + log(mass)
      difference <- entropy - target
      if (abs(difference) < 1e-5) break
      if (difference > 0) {
        beta_min <- beta
        beta <- if (is.infinite(beta_max)) 2 * beta else
          0.5 * (beta + beta_max)
      } else {
        beta_max <- beta
        beta <- if (is.infinite(beta_min)) 0.5 * beta else
          0.5 * (beta + beta_min)
      }
    }
    probability <- unnormalized / sum(unnormalized)
    for (j in seq_len(k)) {
      neighbor <- indices[i, j]
      if (neighbor != i) {
        directed[i, neighbor] <- directed[i, neighbor] + probability[j]
      }
    }
  }
  symmetric <- directed + t(directed)
  symmetric / sum(symmetric)
}

tsne_reference_forces <- function(probability, layout, exaggeration = 1) {
  layout <- as.matrix(layout)
  n <- nrow(layout)
  difference <- array(0, dim = c(n, n, ncol(layout)))
  distance2 <- matrix(0, n, n)
  for (dimension in seq_len(ncol(layout))) {
    difference[, , dimension] <- outer(
      layout[, dimension], layout[, dimension], "-"
    )
    distance2 <- distance2 + difference[, , dimension]^2
  }
  numerator <- 1 / (1 + distance2)
  diag(numerator) <- 0
  sum_q <- sum(numerator)

  attractive <- matrix(0, n, ncol(layout))
  repulsive <- matrix(0, n, ncol(layout))
  for (dimension in seq_len(ncol(layout))) {
    attractive[, dimension] <- rowSums(
      exaggeration * probability * numerator * difference[, , dimension]
    )
    repulsive[, dimension] <- rowSums(
      -(numerator^2 / sum_q) * difference[, , dimension]
    )
  }
  optimizer_gradient <- attractive + repulsive
  positive <- probability > 0
  normalized_q <- numerator / sum_q
  kl <- sum(probability[positive] * log(
    probability[positive] / normalized_q[positive]
  ))
  list(
    probability = probability,
    attractive_force = attractive,
    repulsive_force = repulsive,
    optimizer_gradient = optimizer_gradient,
    objective_gradient = 4 * optimizer_gradient,
    sum_q = sum_q,
    kl = kl
  )
}

tsne_relative_l2 <- function(observed, reference) {
  sqrt(sum((observed - reference)^2)) /
    max(sqrt(sum(reference^2)), .Machine$double.eps)
}

tsne_csr_probability_matrix <- function(diagnostic) {
  row_ptr <- diagnostic$affinity_row_ptr0
  columns <- diagnostic$affinity_col1
  weights <- diagnostic$affinity_weight
  n <- length(row_ptr) - 1L
  probability <- matrix(0, n, n)
  for (row in seq_len(n)) {
    begin <- row_ptr[row] + 1L
    end <- row_ptr[row + 1L]
    if (begin <= end) {
      positions <- seq.int(begin, end)
      probability[row, columns[positions]] <- weights[positions]
    }
  }
  probability
}
