parse_metric_list <- function(value) {
  value <- trimws(value)
  if (identical(value, "unsupported")) return(character())
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

test_that("NN methods documentation metric table agrees with nn_capabilities", {
  docs_file <- test_path("../../docs/nn-methods.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  table_start <- grep("^\\| Method \\| CPU metrics \\| CUDA metrics \\| Notes \\|$", lines)
  expect_length(table_start, 1L)
  rows <- character()
  for (line in lines[(table_start + 2L):length(lines)]) {
    if (!grepl("^\\| `\"", line)) break
    rows <- c(rows, line)
  }
  expect_gt(length(rows), 0L)

  documented <- do.call(rbind, lapply(rows, function(row) {
    cells <- trimws(strsplit(row, "\\|")[[1L]])
    cells <- cells[nzchar(cells)]
    method <- gsub("`|\"", "", cells[[1L]])
    data.frame(
      method = method,
      backend = rep(c("cpu", "cuda"), c(length(parse_metric_list(cells[[2L]])), length(parse_metric_list(cells[[3L]])))),
      metric = c(parse_metric_list(cells[[2L]]), parse_metric_list(cells[[3L]])),
      stringsAsFactors = FALSE
    )
  }))

  caps <- nn_capabilities()
  supported <- caps[caps$supported, c("method", "backend", "metric")]
  supported <- supported[order(supported$method, supported$backend, supported$metric), , drop = FALSE]
  documented <- documented[order(documented$method, documented$backend, documented$metric), , drop = FALSE]
  row.names(supported) <- NULL
  row.names(documented) <- NULL

  expect_equal(documented, supported)
})

test_that("usage API graph_cluster signature shows the live default method", {
  docs_file <- test_path("../../docs/usage-api.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  signature_line <- grep("^graph_cluster\\(graph, method =", lines, value = TRUE)
  expect_length(signature_line, 1L)

  documented_method <- sub('^.*method = "([^"]+)".*$', "\\1", signature_line)
  live_method <- eval(formals(graph_cluster)$method, envir = baseenv())[[1L]]
  expect_equal(documented_method, live_method)
})

test_that("benchmark documentation describes canonical metric aliases once", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  expect_length(grep("^## NN Metrics$", lines), 0L)
  expect_length(grep("^## NN Metric Cycles$", lines), 1L)
  expect_length(grep("^## NN Metrics File Layout$", lines), 1L)
  prose <- paste(lines, collapse = " ")
  expect_true(grepl('"l2".*"cor".*"pearson".*"ip"', prose))
})

test_that("autotuning method settings table keeps public and implementation labels separate", {
  docs_file <- test_path("../../docs/autotuning.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  header <- grep("^\\| Public method \\| Resolved implementation route \\| Role \\| Current tuning rule \\|$", lines)
  expect_length(header, 1L)
  separator_cells <- trimws(strsplit(lines[[header + 1L]], "\\|")[[1L]])
  separator_cells <- separator_cells[nzchar(separator_cells)]
  expect_length(separator_cells, 4L)
  expect_true(any(grepl("not separate public", lines, fixed = TRUE)))
})
