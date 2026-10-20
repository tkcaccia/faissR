# Install required dependency versions only into a dedicated lab library.
args <- commandArgs(TRUE)
stopifnot(length(args) %in% 1:3)
runner <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
if (!length(runner)) runner <- commandArgs()[match("-f", commandArgs()) + 1L]
stopifnot(length(runner) == 1L, !is.na(runner))
runner <- gsub("~+~", " ", runner, fixed = TRUE)
source(file.path(dirname(normalizePath(runner)), "dependency-requirements.R"))
dir.create(args[[1]], recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(args[[1]]), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))
requirements <- if (length(args) >= 2L) {
    archive_requirements(normalizePath(args[[2L]]),
        length(args) < 3L || identical(args[[3L]], "all"))
} else {
    tools:::.split_dependencies(paste(
        "gpuinfo (>= 0.2.0), Rcpp (>= 1.1.0), testthat, withr, float,",
        "Matrix, knitr, rmarkdown"))
}
status <- dependency_status(requirements)
if (tolower(Sys.getenv("PACKAGE_TEST_INCOMING_REMOTE", "false")) %in%
        c("true", "1", "yes")) {
    requirements <- c(requirements, tools:::.split_dependencies("curl"))
    status <- dependency_status(requirements)
}
missing <- unique(status$package[!status$satisfied & status$package != "R"])
if (length(missing)) install.packages(missing, lib = .libPaths()[1L])
status <- dependency_status(requirements)
# A newly released source version may precede its Windows/macOS binary.
missing <- unique(status$package[!status$satisfied & status$package != "R"])
if (length(missing)) {
    install.packages(missing, lib = .libPaths()[1L], type = "source")
}
status <- dependency_status(requirements)
print(status, row.names = FALSE)
stopifnot(all(status$satisfied))
print(sessionInfo())
