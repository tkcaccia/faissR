# Autotuning Notes

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**Autotuning** |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

These notes summarize empirical `nn()` tuning runs for k = 50, Euclidean/L2
search, raw unscaled data, and the package benchmark datasets. Important
benchmark artifacts include:

- `autotune_results.csv`: one row per dataset and method label.
- `autotune_method_summary.csv`: method-level speed/recall/failure summary.
- `autotune_recommendations_by_dataset.csv`: fastest method by recall target.
- `autotune_issues.csv`: low-recall, unavailable, or failed rows.

## Default Policy

Use these rules for `backend = "auto"` and for explicit backend recommendations:

- Prefer exact GPU search when the data fits and target recall is very high.
  `faiss_gpu_flat_l2` and `cuda_cuvs_bruteforce` were the most reliable
  high-recall CUDA paths [1-3,13-16].
- Prefer `faiss_hnsw` for CPU approximate self-KNN. In this benchmark it gave a
  better speed/accuracy balance than FAISS NN-Descent [4-5].
- Prefer `cpu_grid` for 2D/3D Euclidean simulated data. The grid backends are
  intentionally unavailable for higher-dimensional data.
- Treat IVFPQ backends as memory-pressure tools, not accuracy-first defaults.
  Product quantization is useful for compression, but it changes recall
  behaviour [6].
- Treat direct `cuda_cuvs_cagra` as experimental on high-dimensional raw data
  until pilot tuning proves adequate recall. `faiss_gpu_cagra` was reliable in
  the same MNIST test where direct cuVS CAGRA was not [13-15].

## Method-Specific Settings

| Method label | Role | Current tuning rule |
|---|---|---|
| `faiss_flat_exact` | CPU exact baseline | Use for exact CPU reference on small/medium data [1-2,16]; avoid as default for large high-dimensional self-search because MNIST/FashionMNIST timed out. |
| `faiss_gpu_flat_l2` | CUDA exact/high-recall | Preferred high-recall GPU default when FAISS GPU is available and data fits. |
| `cuda_cuvs_bruteforce` | CUDA exact/high-recall | Preferred exact cuVS path; consistently recall 1 in this benchmark and often fastest. |
| `faiss_hnsw_fast` | CPU speed tier | M = 24, efConstruction = 120, efSearch = 80; good speed, but recall can be below 0.999 on image/flow data [5]. |
| `faiss_hnsw_balanced` | CPU default tier | M = 32, efConstruction = 200, efSearch = 150; best general CPU balance and used as the default `faiss_hnsw` setting. |
| `faiss_hnsw_high` | CPU high-recall tier | M = 48, efConstruction = 240, efSearch = 220; use when recall target is about 0.999 or higher. |
| `rcpphnsw` | CPU fallback | Good fallback when FAISS is unavailable, but FAISS HNSW is preferred when FAISS is built. |
| `faiss_ivf_fast` | CPU IVF speed tier | nprobe = 4; too low-recall on many datasets, not a default accuracy path. |
| `faiss_ivf_balanced` | CPU IVF middle tier | Default `nprobe` now uses at least 16 probes; useful when HNSW is not desired. |
| `faiss_ivf_high` | CPU IVF high-recall tier | nprobe = 16 in the benchmark; often much better recall, but slower on image data. |
| `faiss_gpu_ivf_flat` | CUDA IVF-Flat | Useful but not consistently faster than exact GPU on these sample sizes; keep pilot/cache tuning enabled. |
| `cuda_cuvs_ivf_flat` | CUDA cuVS IVF-Flat | Fast on low-dimensional flow/simulated data at about 0.99-0.999 recall; not high-recall default. |
| `faiss_ivfpq_fast` | CPU memory-pressure tier | Low recall on many datasets; use only when memory reduction is the priority [6]. |
| `faiss_ivfpq_balanced` | CPU memory-pressure tier | Still low recall on image/low-dimensional flow datasets; explicit opt-in only. |
| `faiss_gpu_ivfpq` | CUDA memory-pressure tier | Fast but low recall in this benchmark; explicit opt-in only. |
| `cuda_cuvs_ivfpq` | CUDA memory-pressure tier | Better than FAISS GPU IVFPQ on some datasets but still not an accuracy-first default. |
| `faiss_nsg_fast` | CPU graph candidate | Can be accurate but failed on some datasets with fewer than k neighbours; use safer params or retry. |
| `faiss_nsg_balanced` | CPU graph candidate | Safer than fast but still had a failure; default params increased to r = 48 and search_l = 200. |
| `faiss_nndescent_fast` | CPU graph speed tier | Fast, but recall was usually lower than HNSW. |
| `faiss_nndescent_balanced` | CPU graph candidate | Defaults increased to graph_k = 100, iter = 20, search_l = 100 for k = 50; still not the CPU auto default. |
| `cuda_cuvs_nndescent` | CUDA graph speed tier | Fast and useful at around 0.99 recall on some datasets; failed on COIL20. |
| `faiss_gpu_cagra` | CUDA graph high-recall tier | Reliable high-recall CAGRA path through FAISS/cuVS integration [13-15]. |
| `cuda_cuvs_cagra` | Direct cuVS CAGRA | Guarded by pilot recall. Direct cuVS CAGRA had anomalously poor MNIST recall; do not trust without pilot success. |
| `cpu_grid` | Exact 2D/3D spatial path | Best for simulated 2D/3D Euclidean data; unavailable by design outside 2D/3D. |
| `cuda_grid` | CUDA 2D/3D spatial path | Correct for 2D/3D, but benchmark speed depends strongly on GPU model and transfer overhead. |


## ImageNet Probe

Additional ImageNet probes used a dataset object with `data` and `labels`
fields. The data table had 1,281,167 rows and 1,024 columns and occupied about
10 GB as double-precision R columns. Important probe artifacts include:

- `imagenet_probe_results.csv`: 10k and 50k self-KNN sample results with exact
  FAISS GPU Flat as the recall reference.
- `imagenet_full_query_results.csv`: full-reference query attempts against
  1,000 sampled queries.

On the 50k sample, the fastest exact/high-recall routes were:

| Method | Seconds | Recall vs FAISS GPU Flat |
|---|---:|---:|
| `faiss_gpu_flat_l2` | 1.208 | 1.000000 |
| `cuda_cuvs_bruteforce` | 1.747 | 0.999999 |
| `faiss_hnsw` | 31.692 | 0.999524 |
| `faiss_gpu_cagra` | 8.769 | 0.996410 |
| `cuda_cuvs_cagra` | 3.406 | 0.993652 |

Full-reference tests with 1,281,167 reference rows and 1,000 query rows were
attempted for `faiss_hnsw`, `faiss_ivf`, `faiss_gpu_cagra`, and
`cuda_cuvs_ivf_flat`. All four worker processes were killed with exit code 137.
The common failure mode was host-memory pressure from loading a 10 GB double
`data.table`, coercing it to a second contiguous double matrix, and then building
backend-side float/index buffers. On this machine, full ImageNet should be run
from a memory-efficient matrix/float32 representation or on a host with more RAM.


## Shape-Aware `backend` Plus `method = "auto"`

A follow-up auto-policy run tested the CPU-only and CUDA-only automatic
selectors on simulated shapes and benchmark dataset folders.

Policy summary:

- `backend = "cpu", method = "auto"`: exact CPU for small work; CPU grid for
  large 2D/3D self-KNN; FAISS IVF for million-row self-KNN where HNSW graph
  construction is too memory-heavy; FAISS HNSW for large high-dimensional CPU
  self-KNN.
- `backend = "cuda", method = "auto"`: CUDA grid for large 2D/3D self-KNN;
  FAISS GPU Flat for small and medium datasets where exact GPU search is fast;
  FAISS GPU CAGRA for very large self-KNN.

Observed examples from the run:

| Dataset | n x p | CPU auto selected | CPU seconds | CPU recall | CUDA auto selected | CUDA seconds | CUDA recall |
|---|---:|---|---:|---:|---|---:|---:|
| simulated2d | 20000 x 2 | `cpu_grid2d` | 0.782 | 0.999963 | `cuda_grid2d` | 0.697 | 0.999965 |
| COIL20 | 1440 x 16384 | `cpu` | 4.877 | 1.000000 | `faiss_gpu_flat_l2` | 1.914 | 1.000000 |
| FashionMNIST | 70000 x 784 | `faiss_hnsw` | 20.879 | 0.998682 | `faiss_gpu_flat_l2` | 6.455 | 1.000000 |
| FlowRepository | 5220347 x 32 | timeout | NA | NA | `faiss_gpu_cagra` | 118.268 | NA |
| flow18 | 1000021 x 11 | `faiss_ivf` | 35.165 | NA | `faiss_gpu_cagra` | 8.181 | NA |
| MNIST | 70000 x 784 | `faiss_hnsw` | 21.602 | 0.996334 | `faiss_gpu_flat_l2` | 6.197 | 1.000000 |
| TabulaMuris | 70118 x 50 | `faiss_hnsw` | 3.246 | 0.998619 | `faiss_gpu_flat_l2` | 2.314 | 1.000000 |
| ImageNet sample | 50000 x 1024 | `faiss_hnsw` | 93.956 | 0.999436 | `faiss_gpu_flat_l2` | 62.963 | 1.000000 |

The simulated random high-dimensional datasets exposed an important limitation:
FAISS HNSW is fast but may have low recall on noise-like high-dimensional data.
For MNIST, FAISS IVF with `nprobe = 64` reached about 0.99999 recall but took
about 365 seconds, so it is better treated as an explicit accuracy-first CPU
setting rather than the default balanced `backend = "cpu", method = "auto"`
route.

FlowRepository remains a CPU stress case. The full 5.2M x 32 matrix timed out
with `backend = "cpu", method = "auto"`; a follow-up probe with FAISS IVF and
`nprobe = 4` also failed to return in a practical interactive window. On the
same dataset, `backend = "cuda", method = "auto"` selected FAISS GPU CAGRA and
completed, so this shape is currently a GPU-first case rather than a reliable
CPU-auto default.

## Known Issues From The Run

- Direct cuVS CAGRA can produce very low recall on high-dimensional raw MNIST.
  The package now stops when pilot tuning cannot meet the target recall, instead
  of silently returning a poor result.
- FAISS NSG can return fewer neighbours than requested on some datasets. Keep
  safer defaults and consider adding a retry path before using it as an auto
  default.
- cuVS NN-Descent failed on COIL20 with a CUDA invalid-argument error. It should
  remain explicit or secondary until more robust guards are added.
- IVFPQ methods are often fast or memory-efficient, but recall was frequently
  poor. They should be documented as compressed-memory methods.

## Reproducibility

The run used isolated worker processes with a fixed timeout per
method/dataset row. Failures and timeouts were recorded and did not stop the
benchmark matrix. CPU methods used a fixed OpenMP/BLAS thread count.
