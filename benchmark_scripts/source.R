benchmark_knn_recall <- function(approx, exact, k = NULL) {
  approx_idx <- if (is.list(approx)) approx$indices else approx
  exact_idx <- if (is.list(exact)) exact$indices else exact
  approx_idx <- as.matrix(approx_idx)
  exact_idx <- as.matrix(exact_idx)
  if (nrow(approx_idx) != nrow(exact_idx)) {
    stop("Approximate and exact KNN must have the same number of rows.", call. = FALSE)
  }
  k <- if (is.null(k)) min(ncol(approx_idx), ncol(exact_idx)) else as.integer(k)
  if (length(k) != 1L || is.na(k) || !is.finite(k) || k < 1L) {
    stop("`k` must be a positive integer.", call. = FALSE)
  }
  k <- min(k, ncol(approx_idx), ncol(exact_idx))
  recall <- numeric(nrow(approx_idx))
  for (i in seq_len(nrow(approx_idx))) {
    recall[[i]] <- mean(approx_idx[i, seq_len(k)] %in% exact_idx[i, seq_len(k)])
  }
  data.frame(
    k = k,
    recall_at_k = mean(recall),
    median_recall_at_k = median(recall),
    min_recall_at_k = min(recall),
    stringsAsFactors = FALSE
  )
}
