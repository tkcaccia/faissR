source(file.path(".github", "package-check", "audit.R"))
source(file.path(".github", "package-check", "dependency-requirements.R"))
requirements <- tools:::.split_dependencies("base (>= 999.0), R (>= 4.0), tools")
stopifnot(identical(dependency_status(requirements)$satisfied, c(FALSE, TRUE, TRUE)))
work <- tempfile("audit-regression-")
dir.create(work)
log <- file.path(work, "install.log")
out <- file.path(work, "audit.csv")
writeLines("faiss/Index.cpp:10:3: warning: comparison [-Wsign-compare]", log)
stopifnot(audit_native_logs(log, out))
writeLines("faiss/Index.cpp:10:3: warning: uninitialized [-Wmaybe-uninitialized]", log)
stopifnot(!audit_native_logs(log, out))
writeLines("ld: warning: ignoring duplicate libraries: '-lemutls_w', '-lgfortran', '-lheapt_w', '-lquadmath'", log)
stopifnot(audit_native_logs(log, out))
writeLines("nn.cpp:10:3: warning: comparison [-Wsign-compare]", log)
stopifnot(!audit_native_logs(log, out))
writeLines("test.cpp:2: error: undefined symbol", log)
stopifnot(!audit_native_logs(log, out))
stopifnot(audit_check_notes(c("* DONE", "Status: OK"), out))
stopifnot(audit_check_notes(c("* checking CRAN incoming feasibility ... NOTE",
    "New submission", "* checking installation ... OK", "* DONE"), out))
stopifnot(!audit_check_notes(c("* checking CRAN incoming feasibility ... NOTE",
    "New submission", "Possibly misspelled words", "* DONE"), out))
stopifnot(!audit_check_notes(c("* checking CRAN incoming feasibility ... NOTE",
    "New submission", "Previously unknown incoming diagnostic", "* DONE"), out))
stopifnot(!audit_check_notes(c("* checking examples ... NOTE",
    "Examples with CPU or elapsed time > 5s", "* DONE"), out))
timing_notes <- c(
    "* checking examples ... [18s/2s] NOTE",
    "Examples with CPU time > 2.5 times elapsed time",
    "                 user system elapsed ratio",
    "nn              1.051 0.041   0.035   31.2",
    "* checking tests ... [409s/37s] NOTE",
    "Running R code in 'testthat.R' had CPU time 11.2 times elapsed time",
    "* checking re-building of vignette outputs ... [26s/6s] NOTE",
    "Re-building vignettes had CPU time 4.1 times elapsed time",
    "* DONE", "Status: 3 NOTEs")
stopifnot(!audit_check_notes(timing_notes, out))
stopifnot(nrow(read.csv(out)) == 3L)
runner_text <- readLines(file.path(".github", "package-check", "run.R"))
for (stage in c("EXAMPLE", "TEST", "VIGNETTE")) {
    variable <- paste0("_R_CHECK_", stage,
        "_TIMING_CPU_TO_ELAPSED_THRESHOLD_")
    stopifnot(any(grepl(paste0(variable, "` = \"2.5\""),
        runner_text, fixed = TRUE)))
}
hardening <- c("* checking compilation flags used ... NOTE",
    "Compilation used the following non-portable flag(s):",
    "  '-Wp,-D_FORTIFY_SOURCE=3'", "* DONE")
stopifnot(!audit_check_notes(hardening, out))
stopifnot(audit_check_notes(hardening, out, "CXX20FLAGS=-Wp,-D_FORTIFY_SOURCE=3"))
runner <- normalizePath(file.path(".github", "package-check", "run.R"))
command <- file.path(R.home("bin"), "Rscript")
usage <- suppressWarnings(system2(command, shQuote(runner),
    stdout = TRUE, stderr = TRUE))
stopifnot(any(grepl("Usage: run.R", usage, fixed = TRUE)))
fixture <- file.path(work, "fixture")
dir.create(fixture)
writeLines(c("Package: fixture", "Version: 0.1", "Depends: R (>= 4.0)"),
    file.path(fixture, "DESCRIPTION"))
archive <- file.path(work, "fixture.tar.gz")
old <- setwd(work)
utils::tar(archive, "fixture", compression = "gzip", tar = "internal")
setwd(old)
dependency_script <- normalizePath(file.path(".github", "package-check", "dependencies.R"))
result <- system2(file.path(R.home("bin"), "R"),
    c("--vanilla", "--slave", "-f", shQuote(dependency_script), "--args",
        shQuote(file.path(work, "library")), shQuote(archive), "all"),
    stdout = TRUE, stderr = TRUE)
stopifnot(is.null(attr(result, "status")))
cat("Harness audit regression tests: PASS\n")
