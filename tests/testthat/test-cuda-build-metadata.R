test_that("CUDA version summaries omit unavailable components", {
    expect_identical(
        faissR:::cuda_version_summary("13.2", "13.2", "595.58"),
        "compiled CUDA 13.2, runtime 13.2, driver API 595.58"
    )
    expect_identical(
        faissR:::cuda_version_summary("12.8", NA_character_, "unknown"),
        "compiled CUDA 12.8"
    )
    expect_true(is.na(faissR:::cuda_version_summary(
        NA_character_,
        "unknown",
        NA_character_
    )))
})

test_that("native CUDA metadata records compatibility versions", {
    source_file <- test_path("../../src/nn_cuda_kernels.cpp")
    if (!file.exists(source_file)) {
        skip("CUDA source is unavailable in the installed-package context.")
    }
    source_lines <- readLines(source_file, warn = FALSE)
    source <- paste(source_lines, collapse = "\n")
    expect_match(source, "compiled_toolkit", fixed = TRUE)
    expect_match(source, "runtime_version", fixed = TRUE)
    expect_match(source, "driver_version", fixed = TRUE)
    expect_match(source, "CUDART_VERSION", fixed = TRUE)
    expect_match(source, "#include <math_constants.h>", fixed = TRUE)
    expect_false(any(grepl(
        "^__device__ (?!__forceinline__)",
        source_lines,
        perl = TRUE
    )))
})

test_that("FAISS version metadata tolerates older headers", {
    source_file <- test_path("../../src/nn_faiss_impl.cpp")
    if (!file.exists(source_file)) {
        skip("FAISS source is unavailable in the installed-package context.")
    }
    source <- paste(readLines(source_file, warn = FALSE), collapse = "\n")
    expect_match(source, "defined(FAISS_VERSION_MAJOR)", fixed = TRUE)
    expect_match(source, "const std::string version = \"unknown\"", fixed = TRUE)
})

test_that("FAISS GPU dispatch uses FAISS's own cuVS capability", {
    source_file <- test_path("../../src/nn_faiss_impl.cpp")
    if (!file.exists(source_file)) {
        skip("FAISS source is unavailable in the installed-package context.")
    }
    source <- paste(readLines(source_file, warn = FALSE), collapse = "\n")
    expect_match(source, "args.use_cuvs", fixed = TRUE)
    expect_false(grepl(
        "FAISSR_HAS_CUVS[[:space:]]+args\\.use_cuvs[[:space:]]*=[[:space:]]*true",
        source
    ))
})

test_that("CUDA package checks prefer the system compiler toolchain", {
    runner <- test_path(
        "../../.github/package-check/linux/run-cuda.sh"
    )
    if (!file.exists(runner)) {
        skip("CUDA package-check runner is unavailable.")
    }
    runner_text <- paste(readLines(runner, warn = FALSE), collapse = "\n")
    expect_match(
        runner_text,
        "PATH=${CHECK_TOOL_BIN}$R_BIN:/usr/local/bin:/usr/bin:/bin:$CUDA_ROOT/bin",
        fixed = TRUE
    )
    expect_match(
        runner_text,
        "LD_LIBRARY_PATH=${CHECK_TOOL_LIB}$CUDA_ROOT/lib:$CUDA_ROOT/lib64",
        fixed = TRUE
    )
})

test_that("Debian CUDA images configure FAISS from a clean build tree", {
    definition <- test_path(
        "../../.github/package-check/linux/cuda-debian-faiss.def.in"
    )
    if (!file.exists(definition)) {
        skip("Debian CUDA image definition is unavailable.")
    }
    definition_text <- paste(
        readLines(definition, warn = FALSE),
        collapse = "\n"
    )
    expect_match(definition_text, "rm -rf /tmp/faiss-build", fixed = TRUE)
    expect_match(
        definition_text,
        "-DFAISS_ENABLE_PYTHON:BOOL=OFF",
        fixed = TRUE
    )
    expect_match(
        definition_text,
        "-DFAISS_ENABLE_EXTRAS:BOOL=OFF",
        fixed = TRUE
    )
})
