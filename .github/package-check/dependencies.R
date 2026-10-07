# Install only missing test dependencies into a dedicated lab library.
args <- commandArgs(TRUE)
stopifnot(length(args) %in% 1:3)
dir.create(args[[1]], recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(args[[1]]), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

archive_dependencies <- function(archive, include_suggests = TRUE) {
    entries <- utils::untar(archive, list = TRUE)
    description <- entries[grepl("^[^/]+/DESCRIPTION$", entries)]
    stopifnot(length(description) == 1L)
    scratch <- tempfile("package-description-")
    dir.create(scratch)
    on.exit(unlink(scratch, recursive = TRUE), add = TRUE)
    utils::untar(archive, files = description, exdir = scratch)
    metadata <- read.dcf(file.path(scratch, description))
    requested <- c("Depends", "Imports", "LinkingTo")
    if (include_suggests) requested <- c(requested, "Suggests")
    fields <- intersect(requested, colnames(metadata))
    values <- unlist(strsplit(metadata[1L, fields], ",", fixed = TRUE))
    packages <- trimws(sub("[[:space:]]*\\(.*$", "", values))
    setdiff(packages[nzchar(packages)], c("R", rownames(installed.packages(
        priority = c("base", "recommended")
    ))))
}

packages <- "Rcpp"
if (length(args) == 1L) {
    packages <- c(
        "Biobase", "BiocStyle", "Rcpp", "testthat", "withr", "float",
        "Matrix", "knitr", "rmarkdown"
    )
}
if (length(args) == 2L) {
    packages <- union(
        packages,
        archive_dependencies(normalizePath(args[[2L]]))
    )
}
if (length(args) == 3L) {
    mode <- match.arg(args[[3L]], c("required", "all"))
    packages <- union(
        packages,
        archive_dependencies(
            normalizePath(args[[2L]]),
            include_suggests = identical(mode, "all")
        )
    )
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", lib = .libPaths()[1L])
}
missing <- packages[
    !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (requireNamespace("Rcpp", quietly = TRUE) && packageVersion("Rcpp") < "1.1.0") {
    missing <- union(missing, "Rcpp")
}
if (length(missing)) {
    BiocManager::install(missing, lib = .libPaths()[1L], update = FALSE, ask = FALSE)
}
stopifnot(all(vapply(packages, requireNamespace, logical(1), quietly = TRUE)))
print(sessionInfo())
