#!/usr/bin/env Rscript

# Rcpp emits wrapper signatures on one line. Reformat the generated wrappers
# after Rcpp::compileAttributes() so they satisfy package-check line-width and
# four-space indentation recommendations.

path <- "R/RcppExports.R"

if (!file.exists(path)) {
    stop("Run this script from the package root.", call. = FALSE)
}

source_lines <- readLines(path, warn = FALSE)
header_end <- match("", source_lines) - 1L

if (is.na(header_end) || header_end < 1L) {
    stop("The generated Rcpp header was not found.", call. = FALSE)
}

expressions <- parse(path, keep.source = FALSE)

deparse_one <- function(x) {
    paste(
        deparse(
            x,
            width.cutoff = 500L,
            backtick = TRUE,
            control = c("keepInteger", "keepNA")
        ),
        collapse = " "
    )
}

format_formal <- function(formals_list, index) {
    argument_name <- names(formals_list)[[index]]
    default_value <- deparse_one(formals_list[[index]])
    if (!nzchar(default_value)) {
        return(argument_name)
    }
    paste0(argument_name, " = ", default_value)
}

pack_tokens <- function(
    tokens,
    first_prefix,
    continuation_prefix,
    suffix
) {
    if (!length(tokens)) {
        return(paste0(first_prefix, suffix))
    }

    tokens <- paste0(
        tokens,
        c(rep(",", length(tokens) - 1L), "")
    )
    tokens[[length(tokens)]] <- paste0(tokens[[length(tokens)]], suffix)
    lines <- character()
    current <- first_prefix
    for (token in tokens) {
        separator <- if (identical(current, first_prefix)) "" else " "
        candidate <- paste0(current, separator, token)
        if (nchar(candidate, type = "width") <= 80L) {
            current <- candidate
        } else {
            lines <- c(lines, current)
            current <- paste0(continuation_prefix, token)
        }
    }
    lines <- c(lines, current)
    lines
}

format_wrapper <- function(expression) {
    valid_assignment <-
        is.call(expression) &&
        identical(expression[[1L]], as.name("<-")) &&
        is.call(expression[[3L]]) &&
        identical(expression[[3L]][[1L]], as.name("function"))
    if (!valid_assignment) {
        stop("Unexpected expression in R/RcppExports.R.", call. = FALSE)
    }

    function_name <- deparse_one(expression[[2L]])
    wrapper <- eval(expression[[3L]], envir = baseenv())
    wrapper_formals <- formals(wrapper)
    wrapper_body <- body(wrapper)
    if (identical(wrapper_body[[1L]], as.name("{"))) {
        if (length(wrapper_body) != 2L) {
            stop("Rcpp wrapper has an unexpected body.", call. = FALSE)
        }
        wrapper_body <- wrapper_body[[2L]]
    }
    if (!is.call(wrapper_body) ||
        !identical(wrapper_body[[1L]], as.name(".Call"))) {
        stop("Rcpp wrapper does not contain one .Call().", call. = FALSE)
    }

    formal_lines <- vapply(
        seq_along(wrapper_formals),
        function(index) format_formal(wrapper_formals, index),
        character(1L)
    )
    call_arguments <- as.list(wrapper_body)[-1L]
    call_lines <- vapply(call_arguments, deparse_one, character(1L))

    formal_block <- pack_tokens(
        formal_lines,
        paste0(function_name, " <- function("),
        "    ",
        ") {"
    )
    call_block <- pack_tokens(
        call_lines,
        "    .Call(",
        "        ",
        ")"
    )
    output <- c(formal_block, call_block, "}")

    output
}

formatted_wrappers <- lapply(expressions, format_wrapper)
formatted <- unlist(
    lapply(
        seq_along(formatted_wrappers),
        function(index) {
            separator <- if (index < length(formatted_wrappers)) "" else NULL
            c(formatted_wrappers[[index]], separator)
        }
    ),
    use.names = FALSE
)

output <- c(source_lines[seq_len(header_end)], "", formatted)
long_lines <- which(nchar(output, type = "width") > 80L)
if (length(long_lines)) {
    stop(
        "Formatted Rcpp exports still contain lines over 80 characters.",
        call. = FALSE
    )
}

writeLines(output, path, useBytes = TRUE)
