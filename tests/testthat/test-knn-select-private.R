test_that("native Metal selection respects IVF's index and width limits", {
    skip_if_not_installed("Rcpp")
    header <- system.file("include", "faissR_api_v1.h", package = "faissR")
    Rcpp::cppFunction(
        code = '
        SEXP select_metal_knn(SEXP spec) {
          return faissR_get_knn_select_v1()(spec);
        }',
        includes = paste0('#include "', header, '"')
    )
    choose <- function(rows, columns, k = 30L, query = rows,
                       self = TRUE, file = FALSE) {
        select_metal_knn(list(backend = "metal", metric = "euclidean",
            n_reference = rows, n_features = columns, n_query = query,
            k = k, target_recall = 0.95, self_query = self,
            file_backed = file, allow_ivf = TRUE))
    }
    for (file in c(FALSE, TRUE)) {
        expect_identical(choose(50001L, 1024L, file = file)$method, "ivf")
        wide <- choose(50001L, 1025L, file = file)
        expect_identical(wide$method, "exact")
        expect_identical(wide$reason, "metal_ivf_shape_limit")
        expect_identical(choose(8388607L, 512L, file = file)$method, "ivf")
        expect_identical(choose(8388608L, 512L, file = file)$method, "exact")
    }
    expect_identical(choose(33554432L, 64L, 128L)$method, "exact")
    expect_identical(choose(4096L, 512L, query = 8388608L,
        self = FALSE)$method, "exact")
})

test_that("wide CAGRA width-four presets honor the requested recall tier", {
  preset <- faissR:::cuvs_cagra_wide_params
  expect_identical(preset(0.99, TRUE),
    list(width = 4L, queue = 384L, builder = "nn_descent"))
  expect_identical(preset(0.95, TRUE),
    list(width = 4L, queue = 512L, builder = "ivf_pq"))
  expect_identical(preset(0.90, TRUE), preset(0.95, TRUE))
  expect_identical(preset(0.99, FALSE),
    list(width = 2L, queue = 512L, builder = "nn_descent"))
})
