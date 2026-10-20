#!/usr/bin/env Rscript
# Shared, isolated source-package check. No default R library is modified.
args <- commandArgs(TRUE)
runner_file <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
if (!length(runner_file)) stop("Run this harness with Rscript.")
if (!file.exists(runner_file)) {
    runner_file <- gsub("~+~", " ", runner_file, fixed = TRUE)
}
source(file.path(dirname(normalizePath(runner_file)), "audit.R"))
source(file.path(dirname(normalizePath(runner_file)), "dependency-requirements.R"))
if (length(args) < 3L) {
    stop(paste(
        "Usage: run.R SOURCE.tar.gz OUTPUT",
        "functional|diagnostic|cuda-native|cuda-cuvs [SMOKE.R]"
    ))
}
source_archive <- normalizePath(args[[1]], mustWork = TRUE)
out <- args[[2]]
profile <- match.arg(
    args[[3]],
    c("functional", "diagnostic", "cuda-native", "cuda-cuvs")
)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
out <- normalizePath(out, mustWork = TRUE)
if (file.exists(file.path(out, "status.csv"))) {
    stop("Output already contains a run; choose a new output directory.")
}
lib <- file.path(out, "library")
dir.create(lib, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
scratch <- tempfile("package-check-")
dir.create(scratch, showWarnings = FALSE)
if (grepl(" ", scratch, fixed = TRUE)) {
    stop("R CMD INSTALL requires a temporary directory without spaces.")
}
Sys.setenv(TMPDIR = scratch, TMP = scratch, TEMP = scratch)
r <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "R.exe" else "R")
smoke <- if (length(args) >= 4L) normalizePath(args[[4]], mustWork = TRUE) else ""
Sys.setenv(R_LIBS_USER = lib, OMP_NUM_THREADS = "2",
    OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
Sys.setenv(MAKEFLAGS = "-j2")
force_suggests <- tolower(Sys.getenv(
    "PACKAGE_TEST_FORCE_SUGGESTS",
    "false"
)) %in% c("1", "true", "yes")
bootstrap_dependencies <- tolower(Sys.getenv(
    "PACKAGE_TEST_BOOTSTRAP_DEPENDENCIES",
    "false"
)) %in% c("1", "true", "yes")
manual <- tolower(Sys.getenv("PACKAGE_TEST_MANUAL", "false")) %in%
    c("1", "true", "yes")
incoming <- tolower(Sys.getenv("PACKAGE_TEST_INCOMING_REMOTE", "false")) %in%
    c("1", "true", "yes")
# Incoming repository checks are network-dependent submission checks, not
# portability tests. Full submission checks can require every Suggests package;
# the default portability matrix records missing Suggests as NOTEs.
Sys.setenv(`_R_CHECK_CRAN_INCOMING_REMOTE_` = if (incoming) "true" else "false",
    `_R_CHECK_FORCE_SUGGESTS_` = if (force_suggests) "true" else "false")
# Keep the isolated dependency library visible to child R CMD processes, but
# always install the package under test separately.
library_path <- paste(.libPaths(), collapse = .Platform$path.sep)
renviron_user <- file.path(out, "test-Renviron")
writeLines(c(
    paste0("R_LIBS=", encodeString(library_path, quote = '"')),
    paste0("R_LIBS_USER=", encodeString(lib, quote = '"'))
), renviron_user)
Sys.setenv(
    R_ENVIRON_USER = renviron_user,
    R_LIBS = library_path
)
writeLines(c(capture.output(sessionInfo()), capture.output(Sys.info()),
    paste("source:", source_archive), paste("profile:", profile),
    paste("commit:", Sys.getenv("PACKAGE_TEST_COMMIT", "UNRECORDED")),
    paste("image:", Sys.getenv("PACKAGE_TEST_IMAGE", "native")),
    paste("library_paths:", library_path),
    paste("force_suggests:", force_suggests),
    paste("bootstrap_dependencies:", bootstrap_dependencies),
    paste("manual:", manual), paste("incoming_remote:", incoming),
    paste("faiss_source:", Sys.getenv("FAISSR_FAISS_SOURCE", "auto"))),
    file.path(out, "environment.txt"))
config <- system2(r, c("CMD", "config", "--all"), stdout = TRUE, stderr = TRUE)
writeLines(config, file.path(out, "R-config.txt"))
user_makevars <- Sys.getenv("R_MAKEVARS_USER")
if (nzchar(user_makevars) && file.exists(user_makevars)) {
    file.copy(user_makevars, file.path(out, "Makevars.user"))
}
native_commands <- c("nvcc", "nvidia-smi")
for (command in native_commands) {
    executable <- Sys.which(command)
    if (!nzchar(executable)) next
    command_args <- if (command == "nvcc") "--version" else {
        c("--query-gpu=name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader")
    }
    command_output <- tryCatch(
        system2(executable, command_args, stdout = TRUE, stderr = TRUE),
        error = conditionMessage
    )
    writeLines(
        c(paste("command:", executable, paste(command_args, collapse = " ")),
            command_output),
        file.path(out, paste0(command, ".txt"))
    )
}
status <- data.frame(stage = character(), exit_code = integer())
record <- function(stage, code) {
    status[nrow(status) + 1L, ] <<- list(stage, as.integer(code))
    write.csv(status, file.path(out, "status.csv"), row.names = FALSE)
}
run <- function(stage, argv) {
    log <- file.path(out, paste0(stage, ".log"))
    code <- system2(r, argv, stdout = log, stderr = log)
    record(stage, code)
    code
}
setwd(out)
dependency_script <- if (nzchar(smoke)) {
    file.path(dirname(smoke), "dependencies.R")
} else {
    ""
}
if (bootstrap_dependencies && nzchar(dependency_script) &&
        file.exists(dependency_script)) {
    code <- run("dependencies", c(
        "--vanilla", "--slave", "-f", shQuote(dependency_script),
        "--args", shQuote(lib), shQuote(source_archive),
        if (force_suggests) "all" else "required"
    ))
    if (code != 0L) quit(status = 1L)
}
requirements <- archive_requirements(source_archive, force_suggests)
if (incoming) requirements <- c(requirements, tools:::.split_dependencies("curl"))
dependency_table <- dependency_status(requirements)
write.csv(dependency_table, file.path(out, "dependencies.csv"), row.names = FALSE)
record("dependency_versions", as.integer(!all(dependency_table$satisfied)))
if (!all(dependency_table$satisfied)) quit(status = 1L)
# Retain configure output and the extracted source even if installation fails.
install_source <- file.path(out, "install-source")
dir.create(install_source)
utils::untar(source_archive, exdir = install_source)
roots <- list.dirs(install_source, recursive = FALSE, full.names = TRUE)
roots <- roots[file.exists(file.path(roots, "DESCRIPTION"))]
if (length(roots) != 1L) stop("Archive must contain exactly one source package.")
code <- run("install", c("CMD", "INSTALL", "--install-tests",
    paste0("--library=", shQuote(lib)), shQuote(roots)))
if (code != 0L) quit(status = 1L)
if (nzchar(smoke)) {
    Sys.setenv(PACKAGE_TEST_LIBRARY = lib, PACKAGE_TEST_PROFILE = profile)
    code <- run("smoke", c("--vanilla", "--slave", "-f", shQuote(smoke)))
    if (code != 0L) quit(status = 1L)
}
# --no-manual avoids requiring TeX on every host. A portability run without
# all Suggests checks the prebuilt vignette but does not rebuild it.
check_args <- c("CMD", "check", "--as-cran")
if (!manual) check_args <- c(check_args, "--no-manual")
if (!force_suggests) {
    check_args <- c(check_args, "--no-vignettes")
}
check_args <- c(
    check_args,
    paste0("--library=", shQuote(lib)),
    shQuote(source_archive)
)
code <- run("check", check_args)
checks <- list.files(out, "00check.log$", recursive = TRUE, full.names = TRUE)
if (length(checks) != 1L) {
    record("check_log", 1L)
    quit(status = 1L)
}
lines <- readLines(checks, warn = FALSE)
writeLines(tail(lines, 20L), file.path(out, "check-summary.txt"))
has_problem <- any(grepl("^Status:.*(ERROR|WARNING)", lines))
complete <- any(grepl("^\\* DONE", lines))
record("check_complete", as.integer(!complete))
record("check_errors_warnings", as.integer(has_problem))
install_logs <- c(file.path(out, "install.log"),
    list.files(out, "00install.out$", recursive = TRUE, full.names = TRUE))
native_ok <- audit_native_logs(install_logs, file.path(out, "compiler-diagnostics.csv"))
notes_ok <- audit_check_notes(lines, file.path(out, "check-notes.csv"), config)
record("compiler_diagnostics", as.integer(!native_ok))
record("check_notes_reviewed", as.integer(!notes_ok))
quit(status = as.integer(code != 0L || has_problem || !complete ||
    !native_ok || !notes_ok))
