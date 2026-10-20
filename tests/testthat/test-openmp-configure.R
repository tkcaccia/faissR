openmp_configure_probe <- function(accept = c("compiler", "r", "none")) {
    accept <- match.arg(accept)
    helper <- test_path("../../src/build-faiss.sh")
    if (!file.exists(helper)) skip("Source-only configure test.")
    skip_if_not(nzchar(Sys.which("sh")))
    root <- tempfile("openmp-probe-")
    dir.create(root)
    fake_r <- file.path(root, "R")
    writeLines(c("#!/bin/sh", "case \"$*\" in",
        "  'CMD SHLIB conftest.cpp')",
        paste0("    case ", accept, " in"),
        "      r) exit 0 ;;",
        "      compiler) grep -q -- ' -fopenmp' Makevars && exit 0 ;;",
        "    esac",
        "    echo 'fixture: OpenMP unavailable' >&2; exit 1 ;;",
        "  *) exit 0 ;;", "esac"), fake_r)
    Sys.chmod(fake_r, "0755")
    script <- file.path(root, "probe.sh")
    writeLines(c("#!/bin/sh", "set -eu",
        paste(".", shQuote(normalizePath(helper))),
        paste0("faiss_r_bin=", shQuote(fake_r)),
        "faiss_windows=0", "faissr_configure_openmp",
        'printf "selected=<%s>\\n" "$faiss_openmp_flags"',
        'printf "compile=<%s>\\n" "$extra_cxxflags"'), script)
    old <- setwd(root)
    on.exit(setwd(old), add = TRUE)
    suppressWarnings(system2("sh", shQuote(script), stdout = TRUE, stderr = TRUE))
}

test_that("OpenMP detection retains usable R flags", {
    output <- openmp_configure_probe("r")
    expect_null(attr(output, "status"))
    expect_match(paste(output, collapse = "\n"), "selected=<>", fixed = TRUE)
})

test_that("OpenMP detection tests the compiler when R flags are absent", {
    output <- openmp_configure_probe("compiler")
    expect_null(attr(output, "status"))
    expect_match(paste(output, collapse = "\n"), "selected=<-fopenmp>", fixed = TRUE)
    expect_match(paste(output, collapse = "\n"), "compile=< -fopenmp>", fixed = TRUE)
})

test_that("OpenMP failure includes its diagnostics in installation output", {
    output <- openmp_configure_probe("none")
    expect_identical(attr(output, "status"), 1L)
    expect_match(paste(output, collapse = "\n"), "fixture: OpenMP unavailable", fixed = TRUE)
})
