args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
utils::untar(args[[1L]], exdir = args[[2L]], tar = "internal")
stopifnot(file.exists(file.path(args[[2L]], args[[3L]])))
