source_mode_probe <- function(mode = "auto", args = character(), require = "0") {
    helper <- test_path("../../src/build-faiss.sh")
    if (!file.exists(helper)) {
        skip("Build helper is unavailable in the installed-package context.")
    }
    skip_if_not(nzchar(Sys.which("sh")))
    withr::local_envvar(c(FAISSR_FAISS_SOURCE = mode,
        FAISSR_REQUIRE_FAISS = require))
    script <- paste(".", shQuote(normalizePath(helper)),
        '; faissr_source_mode "$@"; printf "%s\\n" "$faiss_source_mode"')
    suppressWarnings(system2("sh", c("-c", shQuote(script), "probe", args),
        stdout = TRUE, stderr = TRUE))
}

test_that("FAISS source selection is explicit and deterministic", {
    expect_identical(source_mode_probe(), "auto")
    for (mode in c("system", "bundled", "none")) {
        expect_identical(source_mode_probe(mode), mode)
    }
    expect_identical(source_mode_probe("system", "--with-faiss=bundled"),
        "bundled")
})

test_that("invalid FAISS source modes fail before native compilation", {
    output <- source_mode_probe("invalid")
    expect_identical(attr(output, "status"), 1L)
    expect_match(paste(output, collapse = "\n"),
        "--with-faiss must be auto, system, bundled, or none.", fixed = TRUE)
    output <- source_mode_probe("none", require = "1")
    expect_identical(attr(output, "status"), 1L)
    expect_match(paste(output, collapse = "\n"),
        "conflicts with FAISSR_REQUIRE_FAISS=1", fixed = TRUE)
})
