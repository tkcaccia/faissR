print(sessionInfo())
print(.libPaths())
p <- installed.packages()
print(p[intersect(c("Rcpp", "gpuinfo", "testthat", "withr", "float",
    "Matrix", "knitr", "rmarkdown"), rownames(p)),
    c("Package", "Version", "Built")])
