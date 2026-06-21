parse_metric_list <- function(value) {
  value <- trimws(value)
  if (identical(value, "unsupported")) return(character())
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

rd_file_text <- function(topic) {
  rd_file <- test_path("../../man", paste0(topic, ".Rd"))
  if (!file.exists(rd_file)) {
    skip("Manual files are not available in this installed-package test context.")
  }
  paste(readLines(rd_file, warn = FALSE), collapse = "\n")
}

expect_rd_documents_formals <- function(topic, fun) {
  rd <- rd_file_text(topic)
  args <- names(formals(fun))
  for (arg in args) {
    expect_true(
      grepl(paste0("\\item{", arg, "}"), rd, fixed = TRUE),
      info = paste(topic, "argument", arg)
    )
    if (!identical(arg, "...")) {
      expect_true(
        grepl(arg, rd, fixed = TRUE),
        info = paste(topic, "usage/formal", arg)
      )
    }
  }
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

test_that("public NN method and metric labels are unique canonical labels", {
  methods <- faissR:::nn_method_labels()
  metrics <- faissR:::nn_metric_labels()

  expect_equal(methods, unique(methods))
  expect_equal(metrics, unique(metrics))
  expect_equal(methods, unname(vapply(methods, faissR:::normalize_nn_method, character(1L))))
  expect_equal(metrics, unname(vapply(metrics, faissR:::normalize_nn_metric, character(1L))))
  expect_false(any(grepl("^faiss_|^cuda_|^cpu_", methods)))
  expect_false("manhattan" %in% metrics)
})

test_that("public API excludes retired wrapper and platform-specific helper names", {
  namespace_exports <- getNamespaceExports("faissR")
  expected_exports <- c(
    "backend_info",
    "candidate_knn",
    "cuda_available",
    "cugraph_available",
    "cuvs_available",
    "faiss_available",
    "fast_kmeans",
    "graph_cluster",
    "knn",
    "knn_graph",
    "nn",
    "nn_capabilities",
    "nn_without_self"
  )
  retired_exports <- c(
    "knn_fit",
    "faiss.fit",
    "cuvs.fit",
    "predict_proba",
    "knn_recall",
    "metal_available"
  )

  expect_setequal(namespace_exports, expected_exports)
  expect_false(any(retired_exports %in% namespace_exports))

  man_dir <- test_path("../../man")
  if (!dir.exists(man_dir)) {
    skip("Manual files are not available in this installed-package test context.")
  }
  man_topics <- sub("\\.Rd$", "", basename(list.files(man_dir, pattern = "\\.Rd$")))
  expect_false(any(retired_exports %in% man_topics))
  expect_false(any(grepl("metal", man_topics, ignore.case = TRUE)))
})

test_that("reference manual documents live public function arguments", {
  expect_rd_documents_formals("nn", nn)
  expect_rd_documents_formals("nn_without_self", nn_without_self)
  expect_rd_documents_formals("knn", knn)
  expect_rd_documents_formals("knn_graph", knn_graph)
  expect_rd_documents_formals("graph_cluster", graph_cluster)
  expect_rd_documents_formals("fast_kmeans", fast_kmeans)
  expect_rd_documents_formals("candidate_knn", candidate_knn)
  expect_rd_documents_formals(
    "predict.faissR_knn_model",
    getS3method("predict", "faissR_knn_model")
  )
})

test_that("reference manual keeps probability prediction inside predict", {
  knn_rd <- rd_file_text("knn")
  predict_rd <- rd_file_text("predict.faissR_knn_model")
  manual_text <- paste(
    vapply(
      list.files(test_path("../../man"), pattern = "\\.Rd$", full.names = TRUE),
      function(path) paste(readLines(path, warn = FALSE), collapse = "\n"),
      character(1L)
    ),
    collapse = "\n"
  )

  expect_true(grepl('type = c\\("response", "prob"\\)', knn_rd))
  expect_true(grepl('type = c\\("response", "prob"\\)', predict_rd))
  expect_false(grepl("predict_proba", manual_text, fixed = TRUE))
})

test_that("README describes public NN metrics and correlation semantics", {
  readme_file <- test_path("../../README.md")
  if (!file.exists(readme_file)) {
    skip("README is not available in this installed-package test context.")
  }

  prose <- paste(readLines(readme_file, warn = FALSE), collapse = " ")
  expect_true(grepl('"euclidean".*"cosine".*"correlation".*"inner_product"', prose))
  expect_true(grepl("Correlation is centered cosine similarity", prose, fixed = TRUE))
  expect_true(grepl("inner product is the raw dot product", prose, fixed = TRUE))
  expect_true(grepl("distance choices belong in `metric`", prose, fixed = TRUE))
  expect_true(grepl("zero-normalized rows have distance `0`", prose, fixed = TRUE))
  expect_true(grepl("explicit CUDA routes remain on CUDA", prose, fixed = TRUE))
})

test_that("GitHub and reference docs describe normalized zero-row metric semantics", {
  docs_files <- c(
    test_path("../../README.md"),
    test_path("../../docs", c(
      "backend-capabilities.md",
      "implementation.md",
      "nn-methods.md",
      "usage-api.md"
    )),
    test_path("../../man/nn.Rd")
  )
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(
      grepl("zero-normalized", prose, fixed = TRUE),
      info = basename(docs_file)
    )
  }
})

test_that("GitHub documentation pages do not duplicate headings", {
  docs_files <- c(
    test_path("../../README.md"),
    list.files(test_path("../../docs"), pattern = "\\.md$", full.names = TRUE)
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    lines <- readLines(docs_file, warn = FALSE)
    headings <- grep("^#{1,3}[[:space:]]+", lines, value = TRUE)
    duplicated_headings <- unique(headings[duplicated(headings)])
    expect_equal(
      duplicated_headings,
      character(),
      info = basename(docs_file)
    )
  }
})

test_that("usage API includes all exported public workflow sections", {
  docs_file <- test_path("../../docs/usage-api.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  expected_sections <- paste0("## `", c(
    "nn",
    "nn_without_self",
    "candidate_knn",
    "knn_graph",
    "graph_cluster",
    "fast_kmeans",
    "knn",
    "predict"
  ), "()`")

  for (section in expected_sections) {
    expect_equal(
      sum(lines == section),
      1L,
      info = section
    )
  }
})

test_that("usage API NN method documentation agrees with live signatures", {
  docs_file <- test_path("../../docs/usage-api.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  method_rows <- grep("^\\| `method` \\| Algorithm selector:", lines, value = TRUE)
  expect_length(method_rows, 1L)
  documented <- regmatches(method_rows, gregexpr('`"[^"]+"`', method_rows))[[1L]]
  documented <- gsub('`|"', "", documented)

  live <- eval(formals(nn)$method, envir = baseenv())
  expect_equal(documented, live)

  nn_without_self_rows <- grep("^\\| `method` \\| Same algorithm selector as `nn\\(\\)`\\.", lines, value = TRUE)
  expect_length(nn_without_self_rows, 1L)
  expect_equal(eval(formals(nn_without_self)$method, envir = baseenv()), live)
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

test_that("graph cluster target documentation states integer and graph-size constraints", {
  docs_files <- c(
    test_path("../../docs", c(
      "usage-api.md",
      "implementation.md",
      "benchmarks.md"
    )),
    test_path("../../man", c("knn_graph.Rd", "graph_cluster.Rd"))
  )
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("Graph documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(grepl("positive integer", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("graph vertices", prose, fixed = TRUE) || grepl("graph vertex count", prose, fixed = TRUE),
                info = basename(docs_file))
  }
})

test_that("backend auto documentation states the CUDA runtime requirement", {
  docs_files <- test_path("../../docs", c(
    "backend-capabilities.md",
    "implementation.md",
    "nn-methods.md",
    "usage-api.md"
  ))
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(
      grepl("CUDA/cuVS runtime[[:space:]]+support is[[:space:]]+available", prose),
      info = basename(docs_file)
    )
  }
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
  expect_false(grepl("non-inner-product metrics", prose, fixed = TRUE))
  expect_true(grepl("four public metrics L2/Euclidean, cosine, correlation, and inner product", prose, fixed = TRUE))
  expect_true(grepl("--metrics=euclidean,cosine,correlation,inner_product", prose, fixed = TRUE))
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
