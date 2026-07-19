# TabulaMuris dataset refresh

Use this workflow whenever the contents of `TabulaMuris_float32.RData` change.
Results generated from an earlier file fingerprint are not interchangeable.

## 1. Regenerate exact references

```bash
cd /scratch/firenze/NN
sbatch benchmark_scripts/jmlr_mloss_publication/refresh_tabula_muris/run_references_cuda.sh
```

Wait for this job to finish. It regenerates all four metric references for
seeds 4, 20260706, and 20260807 with `resume = FALSE`. The CUDA result is saved
only after the independent CPU audit passes.

## 2. Rerun calibration

Submit every file under `calibration/cpu/` and `calibration/cuda/` with the
dataset restriction. Run them one by one:

```bash
export DATASETS=TabulaMuris
export METRICS=euclidean,cosine,correlation,inner_product
sbatch --export=ALL benchmark_scripts/jmlr_mloss_publication/calibration/cpu/run_tune_faissR_hnsw_cpu12_inner_product.sh
sbatch --export=ALL benchmark_scripts/jmlr_mloss_publication/calibration/cuda/run_tune_faissR_cagra_cuda_inner_product.sh
```

Apply the same exported values to every other method-specific calibration
file. Do not import the old and new TabulaMuris rows together until their
dataset fingerprints have been audited.

Review the new calibration recommendations before changing the compiled
`tuning = "auto"` tables. The refresh scripts do not modify package defaults
automatically. If the changed dataset alters a shape/k/target-recall decision,
update the selector from the fingerprinted calibration rows, rebuild the
Singularity image, and only then run the held-out method jobs below.

## 3. Rerun held-out methods

Submit every method file under `cpu/` and `cuda/` separately with the same
dataset restriction, for example:

```bash
sbatch --export=ALL,DATASETS=TabulaMuris benchmark_scripts/jmlr_mloss_publication/cpu/run_faissR_hnsw_cpu12.sh
sbatch --export=ALL,DATASETS=TabulaMuris benchmark_scripts/jmlr_mloss_publication/cuda/run_faissR_cagra_cuda.sh
```

The publication aggregator selects the newest run separately for each dataset,
so a targeted TabulaMuris run replaces only TabulaMuris evidence and preserves
the newest valid run for every other dataset.

## 4. Rerun systems ablations

```bash
sbatch --export=ALL,DATASETS=TabulaMuris benchmark_scripts/jmlr_mloss_publication/ablations/run_systems_ablations_cpu12.sh
sbatch --export=ALL,DATASETS=TabulaMuris benchmark_scripts/jmlr_mloss_publication/ablations/run_systems_ablations_cuda.sh
```

After all held-out method jobs finish, rerun both aggregation launchers.
The old TabulaMuris files may remain as an audit trail: their different or
missing `dataset_md5` prevents them from being merged with the refreshed rows.
