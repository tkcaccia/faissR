lib <- Sys.getenv("PACKAGE_TEST_LIBRARY")
library(faissR, lib.loc = lib)
stopifnot(normalizePath(find.package("faissR")) ==
    normalizePath(file.path(lib, "faissR")))
cat("Package:", as.character(packageVersion("faissR")), "\n")
print(faissR::faiss_available())
print(faissR::backend_info())
print(faissR::nn_capabilities())
print(sessionInfo())
profile <- Sys.getenv("PACKAGE_TEST_PROFILE")
functional <- profile %in% c("functional", "cuda-native", "cuda-cuvs")
stopifnot(identical(faissR::faiss_available(), functional))
set.seed(104)
x <- matrix(rnorm(2048L * 16L), 2048L, 16L)
if (functional) {
    for (metric in c("euclidean", "cosine", "correlation")) {
        z <- nn(x, k = 5L, method = "flat", backend = "cpu",
            metric = metric, exclude_self = TRUE)
        stopifnot(identical(dim(z$indices), c(2048L, 5L)),
            all(is.finite(z$distances)),
            all(z$indices >= 1L & z$indices <= nrow(x)),
            !any(z$indices == row(z$indices)))
        y <- x
        if (metric == "correlation") y <- y - rowMeans(y)
        if (metric != "euclidean") y <- y / sqrt(rowSums(y * y))
        for (i in seq_len(5L)) {
            d <- sqrt(rowSums(sweep(y, 2L, y[i, ], "-")^2))
            d[i] <- Inf
            stopifnot(identical(as.integer(z$indices[i, ]),
                as.integer(order(d)[seq_len(5L)])))
        }
        cat(metric, "independent-reference smoke passed\n")
    }
} else {
    # A diagnostic install must not claim functional FAISS capability.
    cat("DIAGNOSTIC ONLY: FAISS unavailable; this is not a functional CPU pass.\n")
}
if (profile %in% c("cuda-native", "cuda-cuvs")) {
    stopifnot(isTRUE(faissR::cuda_available()))
    if (profile == "cuda-cuvs") {
        stopifnot(isTRUE(faissR::cuvs_available()))
    }
    gpu_x <- matrix(rnorm(512L * 16L), 512L, 16L)
    gpu_q <- matrix(rnorm(32L * 16L), 32L, 16L)
    gpu <- nn(
        gpu_x,
        points = gpu_q,
        k = 5L,
        method = "bruteforce",
        backend = "cuda",
        metric = "euclidean",
        exclude_self = FALSE
    )
    stopifnot(
        identical(dim(gpu$indices), c(32L, 5L)),
        all(is.finite(gpu$distances)),
        all(gpu$indices >= 1L & gpu$indices <= nrow(gpu_x))
    )
    route <- attr(gpu, "backend_used")
    if (profile == "cuda-cuvs") {
        stopifnot(identical(route, "cuda_cuvs_bruteforce"))
    } else {
        stopifnot(!isTRUE(faissR::cuvs_available()), identical(route, "cuda"))
    }
    cat("Verified CUDA route:", route, "\n")
    for (i in seq_len(5L)) {
        distance <- sqrt(rowSums(sweep(gpu_x, 2L, gpu_q[i, ], "-")^2))
        stopifnot(identical(
            as.integer(gpu$indices[i, ]),
            as.integer(order(distance)[seq_len(5L)])
        ))
    }
    cat(profile, "CUDA independent-reference smoke passed\n")
}
cat("SMOKE PASSED\n")
