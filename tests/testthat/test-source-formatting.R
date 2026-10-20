test_that("package sources fit within 80 columns", {
    root <- test_path("../../")
    directories <- file.path(root, c("R", "man", "vignettes"))
    if (!all(dir.exists(directories))) {
        skip("Package source files are unavailable in this test context.")
    }

    files <- c(
        file.path(root, c("DESCRIPTION", "NAMESPACE")),
        list.files(
            directories[[1L]],
            pattern = "[.]R$",
            full.names = TRUE
        ),
        list.files(
            directories[[2L]],
            pattern = "[.]Rd$",
            full.names = TRUE
        ),
        list.files(
            directories[[3L]],
            pattern = "[.](Rmd|R)$",
            full.names = TRUE
        )
    )
    files <- files[file.exists(files)]
    long_lines <- character()
    for (file in files) {
        lines <- readLines(file, warn = FALSE)
        widths <- nchar(lines, type = "width", allowNA = TRUE)
        indexes <- which(!is.na(widths) & widths > 80L)
        if (length(indexes)) {
            long_lines <- c(
                long_lines,
                paste0(
                    basename(file),
                    ":",
                    indexes,
                    " (",
                    widths[indexes],
                    " columns)"
                )
            )
        }
    }

    expect_true(
        length(long_lines) == 0L,
        info = paste(long_lines, collapse = "\n")
    )
})
