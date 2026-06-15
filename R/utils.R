`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

auto_k <- function(x, include_self = FALSE) {
  n <- if (length(x) == 1L && is.numeric(x)) {
    as.integer(x)
  } else {
    nrow(x)
  }
  if (length(n) != 1L || is.na(n) || !is.finite(n) || n < 2L) {
    stop("`x` must describe at least two observations.", call. = FALSE)
  }

  k <- if (n < 500L) {
    15L
  } else if (n < 10000L) {
    30L
  } else {
    50L
  }
  k <- max(1L, min(k, n - 1L))
  if (isTRUE(include_self)) k + 1L else k
}

select_landmark_rows <- function(x, count, seed) {
  n <- nrow(x)
  count <- as.integer(min(max(2L, count), n))
  if (count >= n) {
    return(seq_len(n))
  }

  z <- landmark_selection_features(x, seed)
  if (count <= 2000L) {
    candidate_count <- min(n, max(count, min(12000L, max(500L, 4L * count))))
    candidates <- projection_quantile_rows(z, candidate_count, seed)
    picked <- farthest_landmark_subset(z[candidates, , drop = FALSE], count)
    selected <- candidates[picked]
    method <- "projected_farthest"
  } else {
    selected <- projection_quantile_rows(z, count, seed)
    method <- "multi_projection_quantiles"
  }
  selected <- sort(unique(as.integer(selected)))
  if (length(selected) < count) {
    selected <- fill_landmark_rows(selected, n, count, seed)
  }
  selected <- sort(selected[seq_len(count)])
  attr(selected, "selection_method") <- method
  selected
}

landmark_selection_features <- function(x, seed) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)
  direct <- x[, seq_len(min(4L, p)), drop = FALSE]
  n_random <- min(4L, p)

  set.seed(seed)
  directions <- matrix(stats::rnorm(p * n_random), nrow = p, ncol = n_random)
  norms <- sqrt(colSums(directions * directions))
  norms[!is.finite(norms) | norms == 0] <- 1
  directions <- sweep(directions, 2L, norms, "/")
  z <- cbind(direct, x %*% directions)
  z <- as.matrix(z)
  storage.mode(z) <- "double"

  center <- colMeans(z)
  z <- sweep(z, 2L, center, "-")
  scale <- sqrt(colSums(z * z) / max(1L, n - 1L))
  keep <- is.finite(scale) & scale > 0
  if (!any(keep)) {
    return(matrix(seq_len(n), ncol = 1L))
  }
  sweep(z[, keep, drop = FALSE], 2L, scale[keep], "/")
}

projection_quantile_rows <- function(z, count, seed) {
  n <- nrow(z)
  count <- as.integer(min(max(1L, count), n))
  n_axes <- ncol(z)
  per_axis <- max(2L, ceiling(1.25 * count / max(1L, n_axes)))
  selected <- integer(0)

  center_order <- order(rowSums(z * z), seq_len(n))
  selected <- c(selected, center_order[1L])
  for (axis in seq_len(n_axes)) {
    ordered <- order(z[, axis], seq_len(n))
    positions <- unique(round(seq(1, n, length.out = per_axis)))
    selected <- c(selected, ordered[positions])
  }
  selected <- unique(as.integer(selected))

  if (length(selected) < count) {
    selected <- fill_landmark_rows(selected, n, count, seed)
  }
  if (length(selected) > count) {
    ordered <- order(z[selected, 1L], selected)
    positions <- unique(round(seq(1, length(selected), length.out = count)))
    selected <- selected[ordered[positions]]
  }
  sort(unique(as.integer(selected)))[seq_len(count)]
}

farthest_landmark_subset <- function(z, count) {
  n <- nrow(z)
  count <- as.integer(min(max(1L, count), n))
  if (count >= n) {
    return(seq_len(n))
  }

  z_norm <- rowSums(z * z)
  selected <- integer(count)
  selected[1L] <- which.min(z_norm)
  min_dist <- rep(Inf, n)

  for (i in seq_len(count)) {
    if (i > 1L) {
      selected[i] <- which.max(min_dist)
    }
    center <- z[selected[i], , drop = FALSE]
    dist <- z_norm + z_norm[selected[i]] - 2 * drop(z %*% t(center))
    min_dist <- pmin(min_dist, pmax(0, dist))
    min_dist[selected[seq_len(i)]] <- -Inf
  }
  selected
}

fill_landmark_rows <- function(selected, n, count, seed) {
  selected <- unique(as.integer(selected))
  if (length(selected) >= count) {
    return(selected[seq_len(count)])
  }
  set.seed(seed + 1009L)
  remaining <- setdiff(seq_len(n), selected)
  need <- count - length(selected)
  c(selected, sort(sample(remaining, need)))
}
