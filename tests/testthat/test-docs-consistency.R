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

usage_section_lines <- function(lines, section) {
  start <- match(section, lines)
  expect_false(is.na(start), label = section)
  next_heading <- grep("^## `", lines[(start + 1L):length(lines)])
  end <- if (length(next_heading)) start + next_heading[[1L]] - 1L else length(lines)
  lines[start:end]
}

usage_argument_rows <- function(lines, section) {
  section_lines <- usage_section_lines(lines, section)
  header <- grep("^\\| Argument \\| Description \\|$", section_lines)
  expect_length(header, 1L)
  rows <- character()
  for (line in section_lines[(header + 2L):length(section_lines)]) {
    if (!grepl("^\\|", line)) break
    if (!grepl("^\\| `[^`]+` \\|", line)) next
    rows <- c(rows, line)
  }
  sub("^\\| `([^`]+)` \\|.*$", "\\1", rows)
}

extract_reference_numbers <- function(text) {
  hits <- unlist(regmatches(text, gregexpr("\\[[0-9][0-9, -]*\\]", text)))
  if (!length(hits)) return(integer())
  hits <- gsub("\\[|\\]|[[:space:]]", "", hits)
  refs <- integer()
  for (hit in hits) {
    pieces <- strsplit(hit, ",", fixed = TRUE)[[1L]]
    for (piece in pieces) {
      if (grepl("-", piece, fixed = TRUE)) {
        bounds <- suppressWarnings(as.integer(strsplit(piece, "-", fixed = TRUE)[[1L]]))
        if (length(bounds) == 2L && all(is.finite(bounds)) && bounds[[1L]] <= bounds[[2L]]) {
          refs <- c(refs, seq.int(bounds[[1L]], bounds[[2L]]))
        }
      } else {
        value <- suppressWarnings(as.integer(piece))
        if (is.finite(value)) refs <- c(refs, value)
      }
    }
  }
  sort(unique(refs))
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
  supported <- supported[supported$backend %in% c("cpu", "cuda"), , drop = FALSE]
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
    "faiss_gpu_available",
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

test_that("R source documentation and function signatures do not contain duplicate arguments", {
  r_dir <- test_path("../../R")
  if (!dir.exists(r_dir)) {
    skip("R source files are not available in this installed-package test context.")
  }

  r_files <- list.files(r_dir, pattern = "[.]R$", full.names = TRUE)
  expect_gt(length(r_files), 0L)
  for (file in r_files) {
    lines <- readLines(file, warn = FALSE)
    block <- character()
    block_start <- NA_integer_
    for (i in seq_along(lines)) {
      line <- lines[[i]]
      if (startsWith(line, "#'")) {
        if (!length(block)) block_start <- i
        block <- c(block, line)
        next
      }
      if (length(block)) {
        param_lines <- block[grepl("^#'\\s*@param\\s+", block)]
        params <- sub("^#'\\s*@param\\s+([^[:space:]]+).*$", "\\1", param_lines)
        duplicates <- unique(params[duplicated(params)])
        expect_false(
          length(duplicates) > 0L,
          info = sprintf(
            "%s roxygen block starting at line %d duplicates @param: %s",
            basename(file),
            block_start,
            paste(duplicates, collapse = ", ")
          )
        )
      }
      block <- character()
      block_start <- NA_integer_
    }
  }

  ns <- asNamespace("faissR")
  function_names <- ls(ns, all.names = TRUE)
  for (name in function_names) {
    object <- get(name, envir = ns)
    if (!is.function(object)) next
    args <- names(formals(object))
    duplicates <- unique(args[duplicated(args)])
    expect_false(
      length(duplicates) > 0L,
      info = sprintf("%s duplicates formal argument(s): %s", name, paste(duplicates, collapse = ", "))
    )
  }
})

test_that("native routines use registered symbols without public helper leakage", {
  root <- test_path("../../")
  namespace_file <- file.path(root, "NAMESPACE")
  rcpp_exports_file <- file.path(root, "src", "RcppExports.cpp")
  docs_files <- c(
    list.files(file.path(root, "man"), pattern = "\\.Rd$", full.names = TRUE),
    list.files(file.path(root, "docs"), pattern = "\\.md$", full.names = TRUE),
    file.path(root, "README.md")
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!file.exists(namespace_file) || !file.exists(rcpp_exports_file) || !length(docs_files)) {
    skip("Source and documentation files are not available in this installed-package test context.")
  }

  namespace_text <- paste(readLines(namespace_file, warn = FALSE), collapse = "\n")
  rcpp_exports_text <- paste(readLines(rcpp_exports_file, warn = FALSE), collapse = "\n")
  r_exports_file <- file.path(root, "R", "RcppExports.R")
  embedding_utils_file <- file.path(root, "src", "embedding_utils.cpp")
  r_exports_text <- if (file.exists(r_exports_file)) {
    paste(readLines(r_exports_file, warn = FALSE), collapse = "\n")
  } else {
    ""
  }
  embedding_utils_text <- if (file.exists(embedding_utils_file)) {
    paste(readLines(embedding_utils_file, warn = FALSE), collapse = "\n")
  } else {
    ""
  }
  docs_text <- paste(
    vapply(docs_files, function(path) paste(readLines(path, warn = FALSE), collapse = "\n"), character(1L)),
    collapse = "\n"
  )

  expect_true(grepl("useDynLib(faissR, .registration = TRUE)", namespace_text, fixed = TRUE))
  expect_true(grepl("R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);", rcpp_exports_text, fixed = TRUE))
  expect_true(grepl("R_useDynamicSymbols(dll, FALSE);", rcpp_exports_text, fixed = TRUE))
  expect_false(grepl("knn_recall_cpp", docs_text, fixed = TRUE))
  expect_false(grepl("knn_recall_cpp", r_exports_text, fixed = TRUE))
  expect_false(grepl("knn_recall_cpp", rcpp_exports_text, fixed = TRUE))
  expect_false(grepl("_faissR_knn_recall_cpp", rcpp_exports_text, fixed = TRUE))
  expect_false(grepl("knn_recall_cpp", embedding_utils_text, fixed = TRUE))
})

test_that("repository text excludes private benchmark machine details", {
  root <- test_path("../../")
  files <- list.files(
    root,
    pattern = "\\.(R|Rd|md|cpp|h|hpp|in)$|^(DESCRIPTION|NAMESPACE|LICENSE|CITATION)$",
    recursive = TRUE,
    full.names = TRUE
  )
  files <- files[!grepl("(^|/)\\.git(/|$)|(^|/)faissR\\.Rcheck(/|$)", files)]
  files <- files[!grepl("(^|/)tests/testthat/test-docs-consistency\\.R$", files)]
  expect_gt(length(files), 0L)

  forbidden <- c(
    "chiamaka",
    "137\\.158\\.",
    "\\$Life_2025\\$",
    "/mnt/sata_ssd",
    "fastEmbedR_BENCHMARK",
    "fastEmbedR/Data",
    "micromamba/envs",
    "On this machine, full ImageNet"
  )
  text <- vapply(
    files,
    function(path) paste(readLines(path, warn = FALSE), collapse = "\n"),
    character(1L)
  )
  for (pattern in forbidden) {
    hits <- names(text)[grepl(pattern, text, ignore.case = TRUE)]
    expect_equal(hits, character(), label = pattern)
  }
})

test_that("GitHub docs list all public availability helpers", {
  files <- test_path("../../", c(
    "README.md",
    "docs/backend-capabilities.md",
    "docs/implementation.md",
    "docs/installation.md",
    "docs/usage-api.md"
  ))
  missing_files <- files[!file.exists(files)]
  if (length(missing_files)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  availability_helpers <- c(
    "backend_info()",
    "faiss_available()",
    "faiss_gpu_available()",
    "cuda_available()",
    "cuvs_available()",
    "cugraph_available()"
  )
  for (file in files) {
    prose <- paste(readLines(file, warn = FALSE), collapse = " ")
    for (helper in availability_helpers) {
      expect_true(
        grepl(helper, prose, fixed = TRUE),
        info = paste(basename(file), helper)
      )
    }
  }
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
  expect_true(grepl("backend, method, metric, tuning", knn_rd, fixed = TRUE))
  expect_true(grepl("The fitted model's method and metric are always[[:space:]]+reused", predict_rd))
  expect_false(grepl("predict_proba", manual_text, fixed = TRUE))
})

test_that("fast_kmeans docs describe effective tuning metadata", {
  docs_files <- c(
    test_path("../../docs", c("implementation.md", "usage-api.md")),
    test_path("../../man/fast_kmeans.Rd")
  )
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(grepl("tuning$effective", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("effective_max_iter", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("effective_n_init", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("effective_tol", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("final values", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("direct cuVS C API", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("does not expose an explicit seed", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("centers = 1", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("single_cluster_exact_mean", prose, fixed = TRUE), info = basename(docs_file))
  }

  readme_file <- test_path("../../README.md")
  if (file.exists(readme_file)) {
    readme <- paste(readLines(readme_file, warn = FALSE), collapse = " ")
    expect_true(grepl("fast_kmeans()", readme, fixed = TRUE))
    expect_true(grepl("deterministic shape-aware defaults", readme, fixed = TRUE))
    expect_true(grepl('tuning = "auto"', readme, fixed = TRUE))
  }
})

test_that("fast_kmeans source docs describe single-cluster exact rule", {
  files <- c(
    test_path("../../R/kmeans.R"),
    test_path("../../man/fast_kmeans.Rd"),
    test_path("../../docs/implementation.md"),
    test_path("../../docs/usage-api.md")
  )
  missing <- !file.exists(files)
  if (any(missing)) {
    skip("Source, GitHub, or reference documentation files are not available in this installed-package test context.")
  }

  for (docs_file in files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(grepl("centers = 1", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("single_cluster_exact_mean", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("exact", prose, fixed = TRUE), info = basename(docs_file))
  }
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
  expect_true(grepl("best returned dot product", prose, fixed = TRUE))
  expect_true(grepl("smaller-is-better", prose, fixed = TRUE))
  expect_true(grepl("distance choices belong in `metric`", prose, fixed = TRUE))
  expect_true(grepl("zero-normalized rows have distance `0`", prose, fixed = TRUE))
  expect_true(grepl("explicit CUDA routes remain on CUDA", prose, fixed = TRUE))
})

test_that("GitHub NN docs describe requested and resolved result metadata", {
  docs_files <- c(
    test_path("../../docs", c("backend-capabilities.md", "implementation.md", "nn-methods.md", "usage-api.md")),
    test_path("../../man/nn.Rd")
  )
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  required_terms <- c(
    "requested_backend",
    "requested_method",
    "tuning",
    "resolved_backend"
  )
  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    for (term in required_terms) {
      expect_true(grepl(term, prose, fixed = TRUE), info = paste(basename(docs_file), term))
    }
  }
})

test_that("benchmark docs describe graph method and metric reuse keys", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub benchmark documentation is not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(
    grepl(
      "dataset/cycle/k/graph-backend/graph-method/metric/weight combination",
      prose,
      fixed = TRUE
    )
  )
  expect_true(grepl("--k_values=5,10,15,50,100", prose, fixed = TRUE))
})

test_that("supervised kNN docs describe prediction route metadata", {
  docs_files <- c(
    test_path("../../docs", c("implementation.md", "usage-api.md")),
    test_path("../../man", c("knn.Rd", "predict.faissR_knn_model.Rd"))
  )
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(grepl("faissR_nn", prose, fixed = TRUE), info = basename(docs_file))
  }
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

test_that("GitHub and reference docs describe direct cuVS brute-force metric scope", {
  docs_files <- c(
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
      grepl("direct cuVS brute force", prose, ignore.case = TRUE) &&
        grepl("Euclidean/L2", prose, fixed = TRUE) &&
        grepl("FAISS GPU Flat", prose, fixed = TRUE),
      info = basename(docs_file)
    )
  }
})

test_that("GitHub and reference docs describe grid cosine/correlation support", {
  docs_files <- c(
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
      grepl("grid", prose, fixed = TRUE) &&
        grepl("cosine", prose, fixed = TRUE) &&
        grepl("correlation", prose, fixed = TRUE),
      info = basename(docs_file)
    )
    expect_false(
      grepl("grid for large 2D/3D Euclidean self-search", prose, fixed = TRUE) ||
        grepl("grid search for large 2D/3D Euclidean self-KNN", prose, fixed = TRUE),
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

test_that("GitHub references are cited outside the reference list", {
  refs_file <- test_path("../../docs/references.md")
  docs_dir <- test_path("../../docs")
  if (!file.exists(refs_file) || !dir.exists(docs_dir)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  refs <- readLines(refs_file, warn = FALSE)
  listed <- suppressWarnings(as.integer(sub("^([0-9]+)\\..*$", "\\1", grep("^[0-9]+\\.", refs, value = TRUE))))
  listed <- listed[is.finite(listed)]
  expect_gt(length(listed), 0L)

  docs_files <- c(test_path("../../README.md"), list.files(docs_dir, pattern = "\\.md$", full.names = TRUE))
  docs_files <- docs_files[file.exists(docs_files)]
  docs_files <- docs_files[
    normalizePath(docs_files, mustWork = FALSE) != normalizePath(refs_file, mustWork = FALSE)
  ]
  text <- paste(
    vapply(docs_files, function(path) paste(readLines(path, warn = FALSE), collapse = "\n"), character(1L)),
    collapse = "\n"
  )
  cited <- extract_reference_numbers(text)
  expect_equal(setdiff(listed, cited), integer())
})

test_that("usage API argument tables document live function formals", {
  docs_file <- test_path("../../docs/usage-api.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  lines <- readLines(docs_file, warn = FALSE)
  sections <- list(
    "## `nn()`" = names(formals(nn)),
    "## `nn_without_self()`" = names(formals(nn_without_self)),
    "## `candidate_knn()`" = names(formals(candidate_knn)),
    "## `knn_graph()`" = names(formals(knn_graph)),
    "## `graph_cluster()`" = names(formals(graph_cluster)),
    "## `fast_kmeans()`" = names(formals(fast_kmeans)),
    "## `knn()`" = names(formals(knn)),
    "## `predict()`" = names(formals(getS3method("predict", "faissR_knn_model")))
  )

  for (section in names(sections)) {
    documented <- usage_argument_rows(lines, section)
    expected <- sections[[section]]
    expect_equal(sort(documented), sort(expected))
    expect_equal(documented, unique(documented))
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

test_that("knn_graph reference documents benchmark graph metadata", {
  rd <- rd_file_text("knn_graph")
  required_metadata <- c(
    "target_n_clusters",
    "nearest-neighbour method",
    "metric",
    "tuning policy",
    "requested/resolved KNN backends"
  )

  for (field in required_metadata) {
    expect_true(grepl(field, rd, fixed = TRUE), info = field)
  }
})

test_that("graph_cluster docs describe clustered graph size metadata", {
  docs_files <- c(
    test_path("../../docs", c("implementation.md", "usage-api.md")),
    test_path("../../man/graph_cluster.Rd")
  )
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(grepl("parameters$n_vertices", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("parameters$n_edges", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("target_n_clusters", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("selected_resolution", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("target_gap", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("resolution_selection", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("resolution_search", prose, fixed = TRUE), info = basename(docs_file))
  }
})

test_that("graph docs describe inherited inner-product distance semantics", {
  docs_files <- c(
    test_path("../../docs", c("usage-api.md", "implementation.md")),
    test_path("../../man", c("knn_graph.Rd", "graph_cluster.Rd"))
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }
  for (docs_file in docs_files) {
    prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
    expect_true(grepl("Inner-product graph construction", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("larger raw dot product", prose, fixed = TRUE), info = basename(docs_file))
    expect_true(grepl("shifted smaller-is-better", prose, fixed = TRUE), info = basename(docs_file))
  }
})

test_that("benchmark docs describe deterministic ARI recommendation tie-breaks", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("Benchmark documentation is not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("higher median ARI and then higher median modularity", prose, fixed = TRUE))
  expect_true(grepl("higher median ARI and then lower median total within-cluster sum of squares", prose, fixed = TRUE))
})

test_that("benchmark docs describe deterministic NN recall recommendation tie-breaks", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("Benchmark documentation is not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("higher median recall, minimum recall, and median minimum recall", prose, fixed = TRUE))
  expect_true(grepl("below-threshold median-recall ties", prose, fixed = TRUE))
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

test_that("benchmark documentation describes saved dataset selector universe", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("available_datasets", prose, fixed = TRUE))
  expect_true(grepl("validated real plus simulated dataset names", prose, fixed = TRUE))
  expect_true(grepl("--datasets", prose, fixed = TRUE))
})

test_that("benchmark documentation describes NN expected-skip reason labels", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("nn_metric_benchmark_results.csv", prose, fixed = TRUE))
  expect_true(grepl("expected_skip_reason", prose, fixed = TRUE))
  expect_true(grepl("without parsing the prose error message", prose, fixed = TRUE))
})

test_that("benchmark documentation describes deterministic NN tuning metadata", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("route_parameters", prose, fixed = TRUE))
  expect_true(grepl("tuning_rule", prose, fixed = TRUE))
  expect_true(grepl("FAISS CPU HNSW", prose, fixed = TRUE))
  expect_true(grepl("deterministic no-pilot", prose, fixed = TRUE))
})

test_that("benchmark documentation describes graph expected-skip reason labels", {
  docs_files <- c(
    test_path("../../docs/benchmarks.md"),
    test_path("../../benchmark_scripts/benchmark_graph_clustering.R")
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(unlist(lapply(docs_files, readLines, warn = FALSE)), collapse = " ")
  expect_true(grepl("graph_cluster_benchmark_results.csv", prose, fixed = TRUE))
  expect_true(grepl("expected_skip_reason", prose, fixed = TRUE))
  expect_true(grepl("runtime, shape, and input-type skips", prose, fixed = TRUE))
  expect_true(grepl("bounded deterministic resolution grid", prose, fixed = TRUE))
})

test_that("benchmark documentation describes graph route parameter metadata", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("graph_route_parameters", prose, fixed = TRUE))
  expect_true(grepl("KNN route", prose, fixed = TRUE))
  expect_true(grepl("tuning_rule", prose, fixed = TRUE))
  expect_true(grepl("ef_search", prose, fixed = TRUE))
})

test_that("benchmark documentation describes k-means runtime reason codes", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("kmeans_runtime_capabilities.csv", prose, fixed = TRUE))
  expect_true(grepl("runtime_reason", prose, fixed = TRUE))
  expect_true(grepl("runtime_notes", prose, fixed = TRUE))
  expect_true(grepl("missing_cuda_runtime", prose, fixed = TRUE))
  expect_true(grepl("missing_gpu_kmeans_backend", prose, fixed = TRUE))
})

test_that("benchmark documentation describes k-means tuning rule details", {
  docs_files <- c(
    test_path("../../docs/benchmarks.md"),
    test_path("../../benchmark_scripts/benchmark_kmeans.R")
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(unlist(lapply(docs_files, readLines, warn = FALSE)), collapse = " ")
  expect_true(grepl("tuning_rule", prose, fixed = TRUE))
  expect_true(grepl("tuning_rule_detail", prose, fixed = TRUE))
  expect_true(grepl("small_low_work_multistart", prose, fixed = TRUE))
  expect_true(grepl("large_fast_convergence", prose, fixed = TRUE))
  expect_true(grepl("single_cluster_exact_mean", prose, fixed = TRUE))
  expect_true(grepl("exact CPU column-mean solution", prose, fixed = TRUE))
})

test_that("NN capability docs describe runtime reason codes", {
  docs_files <- c(
    test_path("../../docs", c("backend-capabilities.md", "nn-methods.md", "benchmarks.md")),
    test_path("../../man/nn_capabilities.Rd")
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  prose <- paste(unlist(lapply(docs_files, readLines, warn = FALSE)), collapse = " ")
  expect_true(grepl("runtime_reason", prose, fixed = TRUE))
  expect_true(grepl("runtime_notes", prose, fixed = TRUE))
  expect_true(grepl("missing_faiss", prose, fixed = TRUE))
  expect_true(grepl("missing_faiss_gpu", prose, fixed = TRUE))
  expect_true(grepl("missing_cuda", prose, fixed = TRUE))
  expect_true(grepl("missing_cuvs", prose, fixed = TRUE))
  expect_true(grepl("unsupported_combination", prose, fixed = TRUE))
})

test_that("candidate KNN docs describe CUDA inner-product support", {
  docs_files <- c(
    test_path("../../docs", c("usage-api.md", "implementation.md", "backend-capabilities.md")),
    test_path("../../man/candidate_knn.Rd")
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  prose <- paste(unlist(lapply(docs_files, readLines, warn = FALSE)), collapse = " ")
  expect_true(grepl("CUDA candidate", prose, fixed = TRUE))
  expect_true(grepl("inner-product", prose, fixed = TRUE))
  expect_false(grepl("raw inner-product CUDA candidate scoring is not exposed", prose, fixed = TRUE))
})

test_that("benchmark documentation distinguishes requested and actual graph cluster counts", {
  docs_file <- test_path("../../docs/benchmarks.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("n_clusters_requested", prose, fixed = TRUE))
  expect_true(grepl("n_communities", prose, fixed = TRUE))
  expect_true(grepl("convenience target, not a hard guarantee", prose, fixed = TRUE))
})

test_that("GitHub docs describe the bounded target-cluster resolution grid", {
  docs_files <- c(
    test_path("../../README.md"),
    test_path("../../docs", c("usage-api.md", "implementation.md")),
    test_path("../../man/graph_cluster.Rd")
  )
  docs_files <- docs_files[file.exists(docs_files)]
  if (!length(docs_files)) {
    skip("GitHub or reference documentation files are not available in this installed-package test context.")
  }

  prose <- paste(unlist(lapply(docs_files, readLines, warn = FALSE)), collapse = " ")
  expect_true(grepl("bounded deterministic", prose, fixed = TRUE))
  expect_false(grepl("small deterministic grid of resolution", prose, fixed = TRUE))
  expect_false(grepl("small deterministic resolution grid", prose, fixed = TRUE))
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

test_that("GitHub docs keep implementation labels out of public method/backend API", {
  docs_files <- test_path("../../docs", c(
    "backend-capabilities.md",
    "implementation.md",
    "usage-api.md",
    "nn-methods.md"
  ))
  missing <- !file.exists(docs_files)
  if (any(missing)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(
    vapply(docs_files, function(path) paste(readLines(path, warn = FALSE), collapse = "\n"), character(1L)),
    collapse = "\n"
  )
  expect_true(grepl("not public `method` values", prose, fixed = TRUE))
  expect_false(grepl("legacy explicit `backend` calls", prose, fixed = TRUE))
  expect_false(grepl("legacy backend labels", prose, fixed = TRUE))
})

test_that("autotuning docs describe CUDA auto non-Euclidean routing", {
  docs_file <- test_path("../../docs/autotuning.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("CUDA grid", prose, fixed = TRUE))
  expect_true(grepl("Euclidean/cosine/correlation", prose, fixed = TRUE))
  expect_true(grepl("FAISS[[:space:]]+GPU[[:space:]]+Flat[[:space:]]+IP[[:space:]]+routes", prose))
  expect_true(grepl("cosine", prose, fixed = TRUE))
  expect_true(grepl("correlation", prose, fixed = TRUE))
  expect_true(grepl("inner-product", prose, fixed = TRUE))
  expect_true(grepl("cuVS-only runtimes", prose, fixed = TRUE))
  expect_true(grepl("non-grid[[:space:]]+non-Euclidean searches on CPU", prose))
})

test_that("autotuning docs distinguish historical probes from full NN metric benchmark", {
  docs_file <- test_path("../../docs/autotuning.md")
  if (!file.exists(docs_file)) {
    skip("GitHub documentation files are not available in this installed-package test context.")
  }

  prose <- paste(readLines(docs_file, warn = FALSE), collapse = " ")
  expect_true(grepl("original tuning pass", prose, fixed = TRUE))
  expect_true(grepl("all four public metrics", prose, fixed = TRUE))
  expect_true(grepl("k grid 5, 10, 15, 50, and 100", prose, fixed = TRUE))
})
