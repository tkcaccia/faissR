test_that("Windows diagnostic Makevars do not pass unsupported flang flags", {
    configure <- test_path("../../configure.win")
    if (!file.exists(configure)) {
        skip("configure.win is unavailable in the installed-package context.")
    }
    if (!nzchar(Sys.which("sh"))) {
        skip("A POSIX shell is required for this source-tree configure test.")
    }
    configure_source <- readLines(configure, warn = FALSE)
    expect_false(any(grepl(
        "Wno-unused-command-line-argument",
        configure_source,
        fixed = TRUE
    )))

    root <- tempfile("faissR-configure-win-")
    dir.create(file.path(root, "src"), recursive = TRUE)
    file.copy(configure, file.path(root, "configure.win"))
    file.copy(test_path("../../src/build-faiss.sh"),
        file.path(root, "src", "build-faiss.sh"))

    old <- setwd(root)
    on.exit(setwd(old), add = TRUE)
    withr::local_envvar(c(FAISS_HOME = "", CONDA_PREFIX = "",
        INCLUDE_DIR = "", LIB_DIR = "", FAISSR_REQUIRE_FAISS = "0",
        FAISSR_FAISS_SOURCE = "none", R_HOME = R.home()))
    status <- system2("sh", "configure.win", stdout = TRUE, stderr = TRUE)
    exit_status <- attr(status, "status")
    if (is.null(exit_status)) {
        exit_status <- 0L
    }
    expect_identical(exit_status, 0L)

    makevars <- readLines(file.path(root, "src", "Makevars.win"), warn = FALSE)
    expect_false(any(grepl("Wno-unused-command-line-argument", makevars, fixed = TRUE)))
    expect_false(any(grepl("^PKG_FFLAGS", makevars)))
    expect_true(any(grepl("FAISSR_WINDOWS_NO_FAISS", makevars, fixed = TRUE)))
    expect_true(any(grepl("FAISSR_NO_FORTRAN_NN", makevars, fixed = TRUE)))
    expect_false(any(grepl("nn_fortran[.]o", makevars)))
})

test_that("Windows FAISS linkage probes complete numerical libraries", {
    configure <- test_path("../../configure.win")
    if (!file.exists(configure)) {
        skip("configure.win is unavailable in the installed-package context.")
    }
    source <- paste(c(readLines(configure, warn = FALSE),
        readLines(test_path("../../src/build-faiss.sh"), warn = FALSE)),
        collapse = "\n")
    expect_match(source, "void ssyrk_()", fixed = TRUE)
    expect_match(source, "void sgeqrf_()", fixed = TRUE)
    expect_match(source, "faiss_probe_ext=dll", fixed = TRUE)
    expect_match(source, "local=TRUE, now=TRUE", fixed = TRUE)
    expect_match(source, "FAISSR_NUMERICAL_LIBS", fixed = TRUE)
    expect_match(source, "INCLUDE_DIR", fixed = TRUE)
    expect_match(source, "LIB_DIR", fixed = TRUE)
    expect_match(source, 'faiss_candidate="-llapack -lblas"', fixed = TRUE)
    expect_match(source, "faiss_numerical_libs.*LAPACK_LIBS.*BLAS_LIBS.*FLIBS")
})
