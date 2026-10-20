#!/usr/bin/env Rscript
# Add explicit small test budgets without deparsing or changing existing code.
check_only <- "--check" %in% commandArgs(TRUE)
targets <- c("nn", "knn", "candidate_knn", "fast_kmeans", "hnsw_tune",
    "nn_compute", "clustered_self_knn")
changed <- character()
for (file in list.files("tests/testthat", "[.]R$", full.names = TRUE)) {
    lines <- readLines(file, warn = FALSE)
    parsed <- getParseData(parse(text = lines, keep.source = TRUE))
    calls <- parsed[parsed$token == "SYMBOL_FUNCTION_CALL" &
        parsed$text %in% targets, ]
    edits <- list()
    offsets <- c(0L, cumsum(nchar(lines, type = "chars") + 1L))
    text <- paste0(paste(lines, collapse = "\n"), "\n")
    for (i in seq_len(nrow(calls))) {
        head <- parsed[as.character(calls$parent[[i]]), ]
        children <- parsed[parsed$parent == head$parent, ]
        if (any(children$token == "SYMBOL_SUB" &
                children$text == "n_threads")) next
        close <- children[children$token == "')'", ]
        arguments <- children[children$token == "expr" &
            children$id != head$id, ]
        stopifnot(nrow(close) == 1L, nrow(arguments) > 0L)
        last <- arguments[order(arguments$line2, arguments$col2), ]
        last <- last[nrow(last), ]
        position <- offsets[[last$line2]] + last$col2
        own_line <- grepl("^[ ]*\\)[,;]?[ ]*$", lines[[close$line1]])
        if (own_line) {
            edits[[length(edits) + 1L]] <- list(pos = position, text = ",")
            indent <- strrep(" ", close$col1 - 1L)
            argument_indent <- sub("^([ ]*).*", "\\1", lines[[last$line1]])
            extra <- substring(argument_indent, nchar(indent) + 1L)
            edits[[length(edits) + 1L]] <- list(
                pos = offsets[[close$line1]] + close$col1 - 1L,
                text = paste0(extra, "n_threads = 1L\n", indent))
        } else {
            edits[[length(edits) + 1L]] <- list(
                pos = position, text = ", n_threads = 1L")
        }
    }
    if (!length(edits)) next
    changed <- c(changed, file)
    if (check_only) next
    for (i in order(vapply(edits, `[[`, integer(1), "pos"), decreasing = TRUE)) {
        edit <- edits[[i]]
        text <- paste0(substr(text, 1L, edit$pos), edit$text,
            substring(text, edit$pos + 1L))
    }
    parse(text = text)
    writeChar(text, file, eos = NULL, useBytes = TRUE)
}
if (check_only && length(changed)) {
    stop("Missing explicit test thread budgets: ", paste(changed, collapse = ", "))
}
for (file in list.files("src", "[.](cpp|hpp|h)$", full.names = TRUE)) {
    text <- readLines(file, warn = FALSE)
    directives <- grep("#pragma omp parallel", text, value = TRUE)
    if (any(!grepl("num_threads(", directives, fixed = TRUE))) {
        stop("Native OpenMP directive has no explicit thread budget: ", file)
    }
}
cat(length(changed), if (check_only) "files need changes\n" else "test files updated\n")
