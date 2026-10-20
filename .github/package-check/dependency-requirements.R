archive_requirements <- function(archive, include_suggests = TRUE) {
    entries <- utils::untar(archive, list = TRUE)
    description <- entries[grepl("^[^/]+/DESCRIPTION$", entries)]
    stopifnot(length(description) == 1L)
    scratch <- tempfile("package-description-")
    dir.create(scratch)
    on.exit(unlink(scratch, recursive = TRUE), add = TRUE)
    utils::untar(archive, files = description, exdir = scratch)
    metadata <- read.dcf(file.path(scratch, description))
    fields <- c("Depends", "Imports", "LinkingTo")
    if (include_suggests) fields <- c(fields, "Suggests")
    fields <- intersect(fields, colnames(metadata))
    unlist(lapply(metadata[1L, fields], tools:::.split_dependencies),
        recursive = FALSE, use.names = FALSE)
}

dependency_status <- function(requirements) {
    rows <- lapply(requirements, function(requirement) {
        name <- requirement$name
        version <- tryCatch(if (name == "R") getRversion() else packageVersion(name),
            error = function(e) NULL)
        constraint <- if (is.null(requirement$op)) "" else {
            paste(requirement$op, requirement$version)
        }
        satisfied <- !is.null(version) && (is.null(requirement$op) ||
            do.call(requirement$op, list(version, requirement$version)))
        data.frame(package = name, constraint = constraint,
            installed = if (is.null(version)) NA_character_ else as.character(version),
            satisfied = satisfied)
    })
    do.call(rbind, rows)
}
