# Keep compiler diagnostics separate from the R CMD check status.
audit_native_logs <- function(paths, output) {
    rows <- lapply(paths[file.exists(paths)], function(path) {
        lines <- readLines(path, warn = FALSE)
        found <- grep("(^|[[:space:]])(fatal error|error|warning):", lines)
        if (!length(found)) return(NULL)
        text <- lines[found]
        # FAISS 1.11 mixes signed index types with STL size types. Retain these
        # upstream diagnostics explicitly; never exempt initialization or ABI warnings.
        known <- grepl("^(\\./)?faiss/.*\\[-Wsign-compare\\]$", text)
        duplicate_runtime <- grepl(
            "^ld: warning: ignoring duplicate libraries: '-lemutls_w', '-lgfortran', '-lheapt_w', '-lquadmath'$",
            text)
        classification <- ifelse(known, "upstream-signedness", "unreviewed")
        classification[duplicate_runtime] <- "R-Fortran-runtime-duplicates"
        data.frame(log = basename(path), line = found, diagnostic = text,
            classification = classification)
    })
    rows <- Filter(Negate(is.null), rows)
    result <- if (length(rows)) do.call(rbind, rows) else {
        data.frame(log = character(), line = integer(), diagnostic = character(),
            classification = character())
    }
    write.csv(result, output, row.names = FALSE)
    !any(result$classification == "unreviewed")
}

audit_check_notes <- function(lines, output, toolchain = character()) {
    starts <- grep("^\\* checking .* \\.\\.\\..*NOTE$", lines)
    boundaries <- c(grep("^\\* (checking|DONE)|^Status:", lines), length(lines) + 1L)
    notes <- lapply(starts, function(start) {
        end <- min(boundaries[boundaries > start]) - 1L
        block <- paste(lines[seq.int(start, end)], collapse = "\n")
        contents <- trimws(lines[seq.int(start + 1L, end)])
        initial <- grepl("CRAN incoming feasibility", lines[start]) &&
            grepl("New submission", block, fixed = TRUE) &&
            all(grepl("^(Maintainer:.*|New submission|)$", contents))
        flag <- "-Wp,-D_FORTIFY_SOURCE=3"
        cleaned <- gsub("[\"']", "", chartr("\u2018\u2019", "''", contents))
        hardening <- grepl("checking compilation flags used", lines[start]) &&
            any(grepl(flag, toolchain, fixed = TRUE)) &&
            all(cleaned %in% c("", flag,
                "Compilation used the following non-portable flag(s):")) &&
            flag %in% cleaned
        data.frame(check = lines[start], text = block,
            classification = if (initial) "initial-submission" else {
                if (hardening) "toolchain-hardening" else "unreviewed"
            })
    })
    notes <- if (length(notes)) do.call(rbind, notes) else {
        data.frame(check = character(), text = character(), classification = character())
    }
    write.csv(notes, output, row.names = FALSE)
    !any(notes$classification == "unreviewed")
}
