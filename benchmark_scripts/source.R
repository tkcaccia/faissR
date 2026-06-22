benchmark_knn_recall <- function(approx, exact, k = NULL) {
  approx_idx <- if (is.list(approx)) approx$indices else approx
  exact_idx <- if (is.list(exact)) exact$indices else exact
  approx_idx <- as.matrix(approx_idx)
  exact_idx <- as.matrix(exact_idx)
  if (nrow(approx_idx) != nrow(exact_idx)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k_is_auto <- is.null(k)
  k <- if (k_is_auto) {
    min(ncol(approx_idx), ncol(exact_idx))
  } else {
    suppressWarnings(as.numeric(k))
  }
  if (length(k) != 1L || is.na(k) || !is.finite(k) || (!k_is_auto && (
      k < 1L || abs(k - round(k)) > sqrt(.Machine$double.eps)))) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- as.integer(round(k))
  k <- min(k, ncol(approx_idx), ncol(exact_idx))
  if (k < 1L) {
    stop("KNN matrices must have at least one neighbour column.", call. = FALSE)
  }
  recall <- numeric(nrow(approx_idx))
  for (i in seq_len(nrow(approx_idx))) {
    approx_row <- approx_idx[i, seq_len(k)]
    exact_row <- exact_idx[i, seq_len(k)]
    approx_row <- approx_row[!is.na(approx_row) & is.finite(approx_row)]
    exact_row <- exact_row[!is.na(exact_row) & is.finite(exact_row)]
    recall[[i]] <- if (length(exact_row)) {
      sum(approx_row %in% exact_row) / length(exact_row)
    } else {
      NA_real_
    }
  }
  recall <- recall[is.finite(recall)]
  data.frame(
    k = k,
    recall_at_k = if (length(recall)) mean(recall) else NA_real_,
    median_recall_at_k = if (length(recall)) median(recall) else NA_real_,
    min_recall_at_k = if (length(recall)) min(recall) else NA_real_,
    stringsAsFactors = FALSE
  )
}

benchmark_finite_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

benchmark_knn_rank_correlation <- function(approx, exact, k = NULL) {
  approx_idx <- if (is.list(approx)) approx$indices else approx
  exact_idx <- if (is.list(exact)) exact$indices else exact
  approx_idx <- as.matrix(approx_idx)
  exact_idx <- as.matrix(exact_idx)
  if (nrow(approx_idx) != nrow(exact_idx)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k_is_auto <- is.null(k)
  k <- if (k_is_auto) {
    min(ncol(approx_idx), ncol(exact_idx))
  } else {
    suppressWarnings(as.numeric(k))
  }
  if (length(k) != 1L || is.na(k) || !is.finite(k) || (!k_is_auto && (
      k < 1L || abs(k - round(k)) > sqrt(.Machine$double.eps)))) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- as.integer(round(k))
  k <- min(k, ncol(approx_idx), ncol(exact_idx))
  if (k < 1L) {
    stop("KNN matrices must have at least one neighbour column.", call. = FALSE)
  }
  vals <- numeric(nrow(exact_idx))
  vals[] <- NA_real_
  for (i in seq_len(nrow(exact_idx))) {
    approx_row <- approx_idx[i, seq_len(k)]
    exact_row <- exact_idx[i, seq_len(k)]
    approx_row <- approx_row[!is.na(approx_row) & is.finite(approx_row)]
    exact_row <- exact_row[!is.na(exact_row) & is.finite(exact_row)]
    if (!length(approx_row) || !length(exact_row)) next
    universe <- unique(c(approx_row, exact_row))
    approx_rank <- match(universe, approx_row)
    exact_rank <- match(universe, exact_row)
    approx_rank[is.na(approx_rank)] <- k + 1L
    exact_rank[is.na(exact_rank)] <- k + 1L
    if (length(unique(approx_rank)) > 1L && length(unique(exact_rank)) > 1L) {
      vals[[i]] <- suppressWarnings(stats::cor(approx_rank, exact_rank, method = "spearman"))
    }
  }
  benchmark_finite_mean(vals)
}

benchmark_knn_distance_error <- function(approx, exact, k = NULL) {
  approx_dist <- if (is.list(approx)) approx$distances else NULL
  exact_dist <- if (is.list(exact)) exact$distances else NULL
  if (is.null(approx_dist) || is.null(exact_dist)) return(NA_real_)
  approx_dist <- as.matrix(approx_dist)
  exact_dist <- as.matrix(exact_dist)
  if (nrow(approx_dist) != nrow(exact_dist)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k_is_auto <- is.null(k)
  k <- if (k_is_auto) {
    min(ncol(approx_dist), ncol(exact_dist))
  } else {
    suppressWarnings(as.numeric(k))
  }
  if (length(k) != 1L || is.na(k) || !is.finite(k) || (!k_is_auto && (
      k < 1L || abs(k - round(k)) > sqrt(.Machine$double.eps)))) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- as.integer(round(k))
  k <- min(k, ncol(approx_dist), ncol(exact_dist))
  if (k < 1L) {
    stop("KNN matrices must have at least one neighbour column.", call. = FALSE)
  }
  approx_dist <- approx_dist[, seq_len(k), drop = FALSE]
  exact_dist <- exact_dist[, seq_len(k), drop = FALSE]
  abs_ref <- benchmark_finite_mean(abs(exact_dist))
  abs_err <- benchmark_finite_mean(abs(approx_dist - exact_dist))
  if (is.finite(abs_ref) && abs_ref > 0 && is.finite(abs_err)) {
    abs_err / abs_ref
  } else {
    NA_real_
  }
}

benchmark_knn_quality <- function(approx, exact, k = NULL) {
  recall <- benchmark_knn_recall(approx, exact, k = k)
  k <- recall$k[[1L]]
  data.frame(
    k = k,
    recall_at_k = recall$recall_at_k[[1L]],
    median_recall_at_k = recall$median_recall_at_k[[1L]],
    min_recall_at_k = recall$min_recall_at_k[[1L]],
    mean_relative_distance_error = benchmark_knn_distance_error(approx, exact, k = k),
    rank_correlation = benchmark_knn_rank_correlation(approx, exact, k = k),
    stringsAsFactors = FALSE
  )
}

benchmark_adjusted_rand_index <- function(labels, clusters) {
  if (is.null(labels) || is.null(clusters)) return(NA_real_)
  labels <- as.vector(labels)
  clusters <- as.vector(clusters)
  if (length(labels) != length(clusters)) {
    stop("`labels` and `clusters` must have the same length.", call. = FALSE)
  }
  keep <- !is.na(labels) & !is.na(clusters)
  labels <- labels[keep]
  clusters <- clusters[keep]
  n <- length(labels)
  if (n < 2L || length(unique(labels)) < 2L || length(unique(clusters)) < 1L) {
    return(NA_real_)
  }

  choose2 <- function(x) x * (x - 1) / 2
  tab <- table(labels, clusters)
  nij <- as.numeric(tab)
  ai <- rowSums(tab)
  bj <- colSums(tab)
  total_pairs <- choose2(n)
  if (!is.finite(total_pairs) || total_pairs <= 0) return(NA_real_)

  index <- sum(choose2(nij))
  row_pairs <- sum(choose2(ai))
  col_pairs <- sum(choose2(bj))
  expected <- row_pairs * col_pairs / total_pairs
  max_index <- (row_pairs + col_pairs) / 2
  denom <- max_index - expected
  if (!is.finite(denom) || abs(denom) < .Machine$double.eps) {
    if (abs(index - expected) < .Machine$double.eps) return(1)
    return(NA_real_)
  }
  as.numeric((index - expected) / denom)
}
