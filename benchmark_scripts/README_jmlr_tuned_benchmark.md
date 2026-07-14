# JMLR Tuned Nearest-Neighbour Benchmark

This benchmark generates the evidence table used to compare tuned faissR nearest-neighbour methods with other R packages on the same datasets, metrics, and k values. It is designed for the HPC layout used in `/scratch/firenze/NN` and runs inside the Singularity image.

## Files

- `benchmark_jmlr_tuned_methods.R`: shared R benchmark driver.
- `run_hpc_jmlr_tuned_benchmark_cpu12.sh`: CPU SLURM launcher, 12 CPU tasks.
- `run_hpc_jmlr_tuned_benchmark_cuda.sh`: CUDA SLURM launcher, one L40S GPU.
- `make_hpc_float32_manifest.R`: helper used by the launchers to find the `*_float32.RData` datasets.

## Run On HPC

From `/scratch/firenze/NN`:

```bash
sbatch benchmark_scripts/run_hpc_jmlr_tuned_benchmark_cpu12.sh
sbatch benchmark_scripts/run_hpc_jmlr_tuned_benchmark_cuda.sh
```

The default run uses:

- datasets listed in the generated float32 manifest under `/scratch/firenze/NN/Data`;
- `k = 15,30,50,100`;
- metrics `euclidean`, `cosine`, `correlation`, and `inner_product`;
- target recall values `0.9,0.95,0.99`;
- two independent exact-reference seeds with 1,024 queries each;
- three measured repetitions per seed and method configuration;
- timeout `2000` seconds per dataset/method/metric/k/target combination;
- all tuned faissR methods for the selected backend;
- external R package methods when installed.

Useful overrides:

```bash
METRICS=euclidean,cosine,correlation,inner_product sbatch benchmark_scripts/run_hpc_jmlr_tuned_benchmark_cpu12.sh
DATASETS=MNIST,USPS METHODS=auto,hnsw,ivf TARGET_RECALLS=0.99 sbatch benchmark_scripts/run_hpc_jmlr_tuned_benchmark_cuda.sh
```

## Methods

faissR CPU methods:

`auto`, `exact`, `flat`, `bruteforce`, `grid`, `hnsw`, `ivf`, `ivfpq`, `ivfpq_fastscan`, `nndescent`, `nsg`, `vamana`.

faissR CUDA methods:

`auto`, `exact`, `flat`, `bruteforce`, `grid`, `hnsw`, `ivf`, `ivfpq`, `ivfpq_fastscan`, `nndescent`, `nsg`, `vamana`, `cagra`.

The CUDA launcher also benchmarks GPU-resident `nn_gpu()` routes for `auto`, `exact`, `flat`, and `bruteforce`. These rows keep the result on the GPU during timing and record any explicit host transfer separately as `host_copy_sec` for quality evaluation.

External R packages are attempted when available: `Rnanoflann`, `RANN`, `rnndescent`, `RcppAnnoy`, `BiocNeighbors`, `uwot`, `Rtsne`, `umap`, and `cuda.ml`.

## Outputs

Each run creates a timestamped output folder containing:

- `jmlr_tuned_benchmark_results.csv`: all rows.
- `jmlr_tuned_benchmark_failures.csv`: failures, timeouts, and non-standalone comparison rows.
- `jmlr_ranked_speed_recall.csv`: successful rows ranked by recall, rank agreement, distance error, time, and memory.
- `jmlr_repeated_run_summary.csv`: median time, timing IQR, recall across validation seeds, and robust target attainment.
- `jmlr_best_robust_by_dataset_backend_metric_k_target.csv`: fastest method only when every repeated validation run reaches the requested recall.
- `jmlr_best_by_dataset_backend_metric_k_target.csv`: best method per dataset/backend/metric/k/target recall.
- `jmlr_faissr_vs_external_speed.csv`: fastest faissR row versus fastest external-package row where both exist.
- `jmlr_method_backend_matrix.csv`: tested method matrix.
- `faissR_backend_info.csv` and `faissR_nn_capabilities_runtime.csv`: runtime capability records.
- `JMLR_BENCHMARK_README.md`: generated run-level methods and reviewer-response notes.

## Analyses Included For Reviewer Concerns

The benchmark directly records:

- speed and peak resident memory for every successful row;
- recall, median recall, minimum recall, rank correlation, and relative distance error against an exact reference subset;
- whether tuned methods met the requested recall target;
- the selected backend and resolved method reported by faissR;
- external-package comparisons under the same data, metric, and k settings;
- failures and unsupported combinations without silently falling back to another method;
- GPU-resident timing and explicit host-copy timing as separate quantities.

## Additional Publication Experiments

The real-data comparison should be accompanied by the controlled MIPS stress
benchmark:

```bash
sbatch benchmark_scripts/run_hpc_jmlr_mips_stress_cpu12.sh
sbatch benchmark_scripts/run_hpc_jmlr_mips_stress_cuda.sh
```

It varies dataset shape and vector-norm distributions (unit norm, log-normal,
and Pareto). This separates failures caused by raw-inner-product geometry from
failures caused by an implementation. It also includes 2D/3D unit-norm data
for the low-dimensional grid-method comparison.

For a JMLR MLOSS submission, report both cold end-to-end time and repeated-run
medians, the timing IQR, peak memory, target attainment on held-out reference
queries, unsupported/failure counts, and the resolved backend. External R
packages should be compared on metrics they genuinely expose; unsupported
metric rows are evidence, not silently substituted distances.

This makes it possible to answer whether faissR is faster than external R alternatives, whether automatic tuning reaches the stated recall target, and whether CUDA pipelines avoid unnecessary device-to-host transfers.
