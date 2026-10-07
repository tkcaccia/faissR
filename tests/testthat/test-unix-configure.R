test_that("strict FAISS mode rejects a diagnostic WebAssembly build", {
    configure <- test_path("../../configure")
    if (!file.exists(configure)) {
        skip("configure is unavailable in the installed-package context.")
    }
    if (!nzchar(Sys.which("sh"))) {
        skip("A POSIX shell is required for this source-tree configure test.")
    }

    root <- tempfile("faissR-configure-unix-")
    dir.create(root)
    file.copy(configure, file.path(root, "configure"))

    old <- setwd(root)
    on.exit(setwd(old), add = TRUE)
    output <- suppressWarnings(system2(
        "sh",
        c("configure", "--host=wasm32-unknown-emscripten"),
        stdout = TRUE,
        stderr = TRUE,
        env = "FAISSR_REQUIRE_FAISS=1"
    ))

    expect_equal(attr(output, "status"), 1L)
    expect_true(any(grepl(
        "FAISSR_REQUIRE_FAISS=1, but FAISS is unavailable",
        output,
        fixed = TRUE
    )))
})

test_that("Unix FAISS builds link R numerical and Fortran libraries", {
    configure <- test_path("../../configure")
    if (!file.exists(configure)) {
        skip("configure is unavailable in the installed-package context.")
    }

    configure_source <- readLines(configure, warn = FALSE)
    configure_text <- paste(configure_source, collapse = "\n")

    expect_true(grepl(
        "LAPACK_LIBS.*BLAS_LIBS.*FLIBS",
        configure_text
    ))
    expect_true(grepl(
        "libs \\$faiss_numerical_libs.*LAPACK_LIBS.*BLAS_LIBS.*FLIBS",
        configure_text
    ))
    expect_match(configure_text, "void ssyrk_()", fixed = TRUE)
    expect_match(configure_text, "void sgeqrf_()", fixed = TRUE)
    expect_match(configure_text, "LD_BIND_NOW=1", fixed = TRUE)
    expect_match(configure_text, "local = TRUE, now = TRUE", fixed = TRUE)
})

test_that("Unix configure uses system-library conventions without installation", {
    configure <- test_path("../../configure")
    if (!file.exists(configure)) {
        skip("configure is unavailable in the installed-package context.")
    }

    configure_text <- paste(readLines(configure, warn = FALSE), collapse = "\n")
    expect_match(configure_text, 'PKG_CONFIG_NAME="faiss"', fixed = TRUE)
    expect_match(configure_text, 'PKG_DEB_NAME="libfaiss-dev', fixed = TRUE)
    expect_match(configure_text, 'PKG_RPM_NAME="faiss-devel', fixed = TRUE)
    expect_match(configure_text, "INCLUDE_DIR", fixed = TRUE)
    expect_match(configure_text, "LIB_DIR", fixed = TRUE)
    expect_match(configure_text, '/opt/R/$macos_arch', fixed = TRUE)
    expect_false(grepl("FAISSR_AUTO_INSTALL_FAISS", configure_text, fixed = TRUE))
    expect_false(grepl('brew_cmd" install', configure_text, fixed = TRUE))
})

test_that("Unix CUDA configuration probes the selected toolkit", {
    configure <- test_path("../../configure")
    if (!file.exists(configure)) {
        skip("configure is unavailable in the installed-package context.")
    }

    configure_text <- paste(readLines(configure, warn = FALSE), collapse = "\n")
    expect_match(configure_text, "nvcc\" --version", fixed = TRUE)
    expect_match(configure_text, "CUDA compiler/linker probe failed", fixed = TRUE)
    expect_match(
        configure_text,
        "FAISSR_REQUIRE_CUDA=1, but no usable CUDA device",
        fixed = TRUE
    )
    expect_match(
        configure_text,
        "FAISSR_SKIP_CUDA_RUNTIME_CHECK",
        fixed = TRUE
    )
    expect_match(
        configure_text,
        "FAISSR_REQUIRE_CUDA_RUNTIME is deprecated",
        fixed = TRUE
    )
    expect_match(configure_text, "FAISSR_CUDA_PTX_ARCH", fixed = TRUE)
    expect_match(configure_text, "FAISSR_CUDA_ARCH:-auto", fixed = TRUE)
    expect_match(configure_text, "Architecture detection probe", fixed = TRUE)
    expect_match(configure_text, "--list-gpu-code", fixed = TRUE)
    expect_match(configure_text, "--list-gpu-arch", fixed = TRUE)
    expect_match(
        configure_text,
        "Selected native architectures",
        fixed = TRUE
    )
    expect_match(configure_text, "code=compute_$cuda_ptx_arch", fixed = TRUE)
    expect_match(configure_text, "all: \\$(SHLIB)", fixed = TRUE)
    expect_false(grepl(".DEFAULT_GOAL", configure_text, fixed = TRUE))
    expect_match(configure_text, "cudaDriverGetVersion", fixed = TRUE)
    expect_match(configure_text, "cudaRuntimeGetVersion", fixed = TRUE)
    expect_match(configure_text, "cudaGetDeviceProperties", fixed = TRUE)
    expect_match(configure_text, "capability=%d.%d", fixed = TRUE)
    expect_match(
        configure_text,
        '"$cuda_home"/targets/*/include',
        fixed = TRUE
    )
    expect_match(
        configure_text,
        '"$cuda_home"/targets/*/lib',
        fixed = TRUE
    )
})

test_that("Unix CUDA architecture settings reject malformed values", {
    configure <- test_path("../../configure")
    if (!file.exists(configure)) {
        skip("configure is unavailable in the installed-package context.")
    }

    configure_text <- paste(readLines(configure, warn = FALSE), collapse = "\n")
    expect_match(
        configure_text,
        "FAISSR_CUDA_ARCH must be 'auto' or contain space-separated numeric",
        fixed = TRUE
    )
    expect_match(
        configure_text,
        "FAISSR_CUDA_PTX_ARCH must be 'auto', a numeric compute capability",
        fixed = TRUE
    )
})
