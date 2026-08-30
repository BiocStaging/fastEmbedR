#!/usr/bin/env Rscript

max_width <- 80L

leading_spaces <- function(line) {
    match <- regexpr("^ *", line)
    attr(match, "match.length")
}

normalize_indent <- function(line) {
    indent <- leading_spaces(line)
    if (indent == 0L || indent %% 4L == 0L) {
        return(line)
    }
    target <- 4L * ceiling(indent / 4L)
    paste0(strrep(" ", target), substring(line, indent + 1L))
}

wrap_rd_line <- function(line) {
    if (nchar(line, type = "width") <= max_width) {
        return(line)
    }
    indent <- leading_spaces(line)
    content <- substring(line, indent + 1L)
    continuation <- if (startsWith(content, "\\item")) {
        indent + 4L
    } else {
        indent
    }
    wrapped <- strwrap(
        content,
        width = max_width,
        indent = indent,
        exdent = continuation,
        simplify = FALSE
    )
    if (length(wrapped)) wrapped else line
}

format_rd_file <- function(path) {
    lines <- readLines(path, warn = FALSE)
    lines <- unlist(lapply(lines, wrap_rd_line), use.names = FALSE)
    lines <- vapply(lines, normalize_indent, character(1L), USE.NAMES = FALSE)
    writeLines(lines, path, useBytes = TRUE)
}

paths <- list.files("man", "[.]Rd$", full.names = TRUE)
invisible(lapply(paths, format_rd_file))
