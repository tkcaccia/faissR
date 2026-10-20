args <- commandArgs(TRUE)
stopifnot(length(args) >= 1L)
.libPaths(c(args, .libPaths()))
library(faissR)
stopifnot(
    normalizePath(find.package("faissR")) ==
        normalizePath(file.path(args[[1L]], "faissR")),
    faiss_available()
)
x <- matrix(seq_len(240L), ncol = 6L)
z <- nn(x, k = 3L, method = "exact", backend = "cpu")
stopifnot(identical(dim(z$indices), c(40L, 3L)))
cat("WINDOWS RUNTIME WITHOUT RTOOLS PATH: PASS\n")
