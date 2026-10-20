local({
    withr::local_envvar(c(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
        MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1",
        VECLIB_MAXIMUM_THREADS = "1"))
    # BLAS has a separate worker pool from the search's n_threads argument.
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        previous <- RhpcBLASctl::blas_get_num_procs()
        on.exit(RhpcBLASctl::blas_set_num_threads(previous), add = TRUE)
        RhpcBLASctl::blas_set_num_threads(1L)
        previous_omp <- RhpcBLASctl::omp_get_max_threads()
        if (length(previous_omp) == 1L && is.finite(previous_omp) &&
                previous_omp > 0L) {
            on.exit(RhpcBLASctl::omp_set_num_threads(previous_omp), add = TRUE)
            RhpcBLASctl::omp_set_num_threads(1L)
        }
    }
    library(testthat)
    library(faissR)
    test_check("faissR")
})
