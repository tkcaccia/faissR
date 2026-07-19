# JMLR MLOSS Publication Benchmark

This directory is separate from the earlier tuning files. Every Slurm file in
`cpu/`, `cuda/`, and `calibration/` tests exactly one method and is intended to
be submitted individually with `sbatch`.

## Resource headers

CPU files retain the established CPU resources: account `immunology`,
partition `ada`, one node, 12 tasks, and 48 hours. CUDA files retain account
`l40sfree`, partition `l40s`, one node, two tasks, one L40S GPU, and 48 hours.
Only the job name and log filename vary by method.

## Files

- `references/run_exact_references_cuda.sh`: preferred fast CUDA exact
  references, with an independent CPU FAISS Flat audit before saving.
- `references/run_exact_references_cpu12.sh`: slower CPU-only alternative.
- `calibration/cpu/`: one raw-inner-product tuning file per CPU method.
- `calibration/cuda/`: one raw-inner-product tuning file per CUDA method.
- `cpu/`: one held-out publication benchmark file per CPU method or external
  R-package method.
- `cuda/`: one held-out publication benchmark file per CUDA method.
- `common/`: shared R drivers required by the individual Slurm files.

## Run One By One

First submit the CUDA reference job and wait until it finishes:

```bash
cd /scratch/firenze/NN
sbatch benchmark_scripts/jmlr_mloss_publication/references/run_exact_references_cuda.sh
```

Use `run_exact_references_cpu12.sh` instead only when a CUDA node is not
available. Both scripts create the same reference filenames; the CUDA script
saves a result only after its CPU audit passes. The audit uses up to 64
queries and automatically reduces that count for very large datasets to keep
the independent CPU check near five billion distance operations.

Then submit inner-product calibration methods individually:

```bash
sbatch benchmark_scripts/jmlr_mloss_publication/calibration/cpu/run_tune_faissR_hnsw_cpu12_inner_product.sh
sbatch benchmark_scripts/jmlr_mloss_publication/calibration/cuda/run_tune_faissR_cagra_cuda_inner_product.sh
```

After using the calibration results to update `tuning = "auto"` and rebuilding
the Singularity image, submit publication methods individually:

```bash
sbatch benchmark_scripts/jmlr_mloss_publication/cpu/run_faissR_hnsw_cpu12.sh
sbatch benchmark_scripts/jmlr_mloss_publication/cpu/run_RANN_kd_cpu12.sh
sbatch benchmark_scripts/jmlr_mloss_publication/cuda/run_faissR_cagra_cuda.sh
sbatch benchmark_scripts/jmlr_mloss_publication/cuda/run_faissR_gpu_resident_exact_cuda.sh
```

All files in `cpu/` and `cuda/` are independent jobs. Submit each file once.
Do not combine CPU and CUDA methods in one Slurm job, and do not reuse
calibration output as held-out validation.

After the one-method jobs finish, aggregate CPU and CUDA evidence separately:

```bash
sbatch benchmark_scripts/jmlr_mloss_publication/analysis/run_aggregate_cpu12.sh
sbatch benchmark_scripts/jmlr_mloss_publication/analysis/run_aggregate_cuda.sh
```

The aggregator selects the newest run for each method and suite, requires two
validation seeds and three repetitions, and ranks a method only when every
measured run reaches the requested recall. It writes fastest and second-fastest
qualifying methods, exact baselines, `method = "auto"` versus the oracle method,
recall-compliance counts, failures, and successful route mismatches.

Run the systems ablations independently:

```bash
sbatch benchmark_scripts/jmlr_mloss_publication/ablations/run_systems_ablations_cpu12.sh
sbatch benchmark_scripts/jmlr_mloss_publication/ablations/run_systems_ablations_cuda.sh
```

These jobs compare float32 and double input, cold and warm fitted-index reuse,
compiled and R-side self-neighbour removal, and GPU-resident exact search with
an explicit device-to-host copy. They use COIL20, MNIST, and TabulaMuris at
`k = 30` to cover three different dataset shapes without duplicating the full
method grid.

Every one-method and systems-ablation launcher performs a package/backend
preflight inside the same Singularity invocation used for measurement. A stale
image or missing shared library therefore fails once with the original R load
diagnostic before method workers are launched. Systems-ablation worker output
is retained separately under `worker_logs/`.

The double-input ablation converts a `float::float32` dataset explicitly with
`float::dbl()`. Calling `as.matrix()` on a float object retains its S4 float32
class and is not a valid double-input control.

Reference files are saved in their dataset directories and reused by every
method. Synthetic data are generated only when their manifest is absent.
References and result rows include the source dataset MD5 fingerprint. A
reference with a missing or different fingerprint is rejected, even when its
matrix dimensions and `k` still match. For a changed TabulaMuris file, follow
`refresh_tabula_muris/README.md`; targeted replacement runs are selected per
dataset and do not displace newer evidence for unrelated datasets.

## Experimental Design

Each faissR publication file tests one method with `tuning = "auto"`,
`k = 15,30,50,100`, target recall 0.90/0.95/0.99, two held-out seeds, three
repetitions, and a 2,000-second timeout per combination. Real datasets use all
four metrics. faissR methods also run the controlled raw-inner-product norm
stress suite. Grid files run only the 2D/3D spatial suite. External packages
run only metrics supported by their public KNN interface.

CUDA NN-descent with raw inner product is an expected unsupported combination:
cuVS NN-descent builds a symmetric graph from one L2 dataset and cannot apply
the asymmetric maximum-inner-product transformation. The row must remain in
the failure evidence; it must not be replaced by another algorithm.

Rtsne and `umap::umap` are not included as standalone KNN methods because they
are embedding consumers rather than comparable KNN result providers.
