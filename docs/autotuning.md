# Autotuning Notes

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
**Autotuning** |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

These notes summarize empirical `nn()` tuning probes and how they inform the
current shape-aware defaults. The original tuning pass used k = 50,
Euclidean/L2 search, raw unscaled data, and the package benchmark datasets; the
current NN metric benchmark extends that work to all four public metrics and
the k grid 5, 10, 15, 50, and 100. Important benchmark artifacts include:

- `autotune_results.csv`: one row per dataset and resolved implementation
  label.
- `autotune_method_summary.csv`: method-level speed/recall/failure summary.
- `autotune_recommendations_by_dataset.csv`: fastest method by recall target.
- `autotune_issues.csv`: low-recall, unavailable, or failed rows.

## Default Policy

Use these rules for `backend = "auto"` and for explicit backend
recommendations. Public calls should use canonical method names such as
`"exact"`, `"flat"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"nndescent"`, or
`"cagra"`; labels such as `faiss_hnsw` or `cuda_cuvs_cagra` are resolved
implementation routes recorded in benchmark output, not separate public
`method` values.

Both parts of the automatic policy are compiled C++ rules. The method/backend
route is selected by `nn_auto_select_backend_cpp()`, and deterministic
`tuning = "auto"` parameters are selected by C++ helpers such as
`nn_tune_faiss_hnsw_cpp()`, `nn_tune_faiss_ivf_cpp()`,
`nn_tune_cuvs_cagra_cpp()`, `nn_tune_native_nsg_cpp()`, and
`nn_tune_vamana_cpp()`. The R front end reads `options(faissR.*)`, normalizes
public arguments, and passes those values into C++; it does not maintain a
separate R implementation of the shape/k/metric policy. Returned tuning
metadata includes `tuning_source = "cpp"` for deterministic approximate-method
parameter rules.

`fast_kmeans()` follows the same compiled-policy contract. Its automatic
`max_iter`, `n_init`, and `tol` values are selected by
`kmeans_auto_params_cpp()`, and its `backend = "auto"` CUDA/CPU gate is
selected by `kmeans_auto_select_backend_cpp()`, using the policy object from
`kmeans_auto_backend_policy_cpp()`. The R layer reads documented threshold
options and runtime availability flags, then forwards them into C++; it does
not maintain a separate R implementation of the k-means shape policy or final
CPU/CUDA selection. Returned k-means metadata records `tuning_source = "cpp"`
for the parameter rule, backend policy, and final selection.

For `graph_cluster(n_clusters = ...)`, faissR builds the KNN graph once and
then evaluates a deterministic resolution grid. The candidate center and grid
width are computed by `graph_resolution_candidates_cpp()`, with wider grids for
small graphs and narrower grids for large graphs. Target-count results record
`resolution_selection$tuning_source = "cpp"` so benchmark summaries can verify
that the no-pilot graph-shape rule came from compiled policy.

- Prefer `method = "flat"`/`"exact"` on CUDA when the data fits and target
  recall is very high. The resolved routes `faiss_gpu_flat_l2` and
  `cuda_cuvs_bruteforce` were the most reliable high-recall CUDA paths
  [1-3,13-16].
- For exact CUDA self-KNN routes, `method = "auto"` chooses between FAISS GPU
  Flat and direct cuVS brute force by shape. Compact high-dimensional shapes
  such as COIL20 and USPS prefer `cuda_cuvs_bruteforce`; larger exact
  image-scale shapes such as FashionMNIST prefer `faiss_gpu_flat_l2` when FAISS
  GPU is available. `cuda_cuvs_bruteforce` remains the explicit cuVS exact
  method and the fallback exact CUDA route when FAISS GPU Flat is unavailable.
- Prefer `method = "hnsw"` for CPU approximate self-KNN. In this benchmark its
  FAISS HNSW implementation route gave a better speed/accuracy balance than
  NN-Descent [4-5].
- Prefer `method = "grid"` for 2D/3D Euclidean, cosine, or correlation
  simulated data. The grid
  backends are intentionally unavailable for higher-dimensional data.
- Treat IVFPQ backends as memory-pressure tools, not accuracy-first defaults.
  Product quantization is useful for compression, but it changes recall
  behaviour [6].
- Treat CUDA CAGRA as two separable decisions: the provider
  (`faiss_gpu_cagra` through FAISS GPU/cuVS integration, or
  `cuda_cuvs_cagra` through the direct RAPIDS cuVS C API) and, for direct cuVS
  only, the graph-build algorithm. The public call remains
  `method = "cagra"`; provider/build choices are recorded in result metadata.
  Direct cuVS CAGRA should be benchmarked with measured recall on
  high-dimensional raw data before being used as an accuracy-first default
  [3,13-15].

## Method-Specific Settings

| Public method | Resolved implementation route | Role | Current tuning rule |
|---|---|---|---|
| `exact` / `flat` | `faiss_flat_exact`, `faiss_flat_l2` | CPU exact baseline | Use for exact CPU reference on small/medium data [1-2,16]; avoid as default for large high-dimensional self-search because MNIST/FashionMNIST timed out. |
| `exact` / `flat` | `faiss_gpu_flat_l2` | CUDA exact/high-recall | Explicit FAISS GPU Flat route when requested and available. |
| `bruteforce` | `cuda_cuvs_bruteforce` | CUDA exact/high-recall | Preferred explicit cuVS exact path; consistently recall 1 in this benchmark and often fastest on compact high-dimensional self-KNN. Also selected by CUDA `method = "auto"` for compact exact self-KNN when cuVS is available, and used as the fallback exact CUDA route when FAISS GPU Flat is unavailable. |
| `hnsw` | `faiss_hnsw` large low-dimensional Euclidean tiers | CPU target-recall speed tiers | For `n >= 50000`, `p <= 64`, `target_recall` selects separate 0.90, 0.95, or 0.99 M/ef tiers. The 0.99 defaults keep the prior high-recall flow-cytometry settings; the lower targets reduce graph/search effort for speed and record the selected rule in approximation metadata [5]. |
| `hnsw` | `faiss_hnsw` large high-dimensional Euclidean tiers | CPU target-recall speed tiers | For `n >= 50000`, `p >= 256`, `target_recall` selects separate 0.90, 0.95, or 0.99 M/ef tiers. The 0.90 small-k tier was tightened from the first MNIST/Fashion smoke run to `M = 8`, `efConstruction = 30`, and `efSearch = max(k, 15)`, which kept sampled recall above 0.90 while avoiding the slower 0.95 tier [5]. |
| `hnsw` | `faiss_hnsw` small-k metric tier | CPU metric-aware tier | M = 32, efConstruction = 160, efSearch = max(120, 4k); used for cosine, correlation, and inner-product `k <= 10` jobs so normalized metric searches keep more graph-search breadth without paying the full high-recall cost [5]. |
| `hnsw` | `faiss_hnsw` balanced tier | CPU default tier | M = 32, efConstruction = 200, efSearch = max(150, 3k); default deterministic shape/metric rule for general CPU HNSW. |
| `hnsw` | `faiss_hnsw` high-recall tier | CPU high-recall tier | M = 48, efConstruction = 240, efSearch = max(220, 3k); used for large-k high-dimensional searches and high-dimensional non-Euclidean searches where normalized IP/correlation routes need extra graph-search breadth. |
| `hnsw` | `hnsw`/hnswlib fallback | CPU fallback | Good fallback when FAISS is unavailable, but FAISS HNSW is preferred when FAISS is built. |
| `ivf` | `faiss_ivf` speed tier | CPU IVF speed tier | nprobe = 4; too low-recall on many datasets, not a default accuracy path. |
| `ivf` | `faiss_ivf` balanced tier | CPU IVF middle tier | Default `nprobe` now uses at least 16 probes; cosine, correlation, and raw inner-product routes use a deterministic metric-aware probe increase and record `tuning_metric`/`tuning_metric_aware`. Useful when HNSW is not desired. |
| `ivf` | `faiss_ivf` high-recall tier | CPU IVF high-recall tier | Larger `k` and million-row shapes increase probe breadth through deterministic `n`/`k` rules; non-Euclidean metrics add the metric-aware probe tier. This is often much better recall, but slower on image data. |
| `ivf` | `faiss_gpu_ivf_flat` | CUDA IVF-Flat | Useful but not consistently faster than exact GPU on these sample sizes. Deterministic `tuning = "auto"` is metric-aware; explicit `tuning = "cache"` or `"pilot"` currently runs only for Euclidean IVF because the pilot reference/candidates are raw-L2. |
| `ivf` | `cuda_cuvs_ivf_flat` | CUDA cuVS IVF-Flat | Direct benchmark route for Euclidean/L2 plus transformed cosine, correlation, and raw inner product. Fast on low-dimensional flow/simulated data at about 0.99-0.999 recall; not high-recall default. |
| `ivfpq` | `faiss_ivfpq` speed/balanced tiers | CPU memory-pressure tier | Low recall on many datasets; use only when memory reduction is the priority. Requires at least 624 training rows for the CPU FAISS route; auto tuning uses 4-bit PQ for 624-9,983 rows and 8-bit PQ above that unless manually overridden [6]. |
| `ivfpq` | `faiss_gpu_ivfpq` | CUDA memory-pressure tier | Fast but low recall in this benchmark; explicit opt-in only. |
| `ivfpq` | `cuda_cuvs_ivfpq` | CUDA memory-pressure tier | Direct benchmark route for Euclidean/L2 plus transformed cosine, correlation, and raw inner product. It uses the same deterministic small-training rule as CPU PQ: below 9,984 training rows, auto tuning requests 4-bit PQ unless the user manually sets `cuvs_ivfpq_pq_bits`/`ivfpq_pq_bits`. Better than FAISS GPU IVFPQ on some datasets but still not an accuracy-first default. |
| `nsg` | `cpu_nsg` speed/balanced tiers | CPU graph candidate | Native faissR NSG-style route for all public metrics; avoids linked-FAISS NSG aborts in public calls. Large high-dimensional CPU inputs use deterministic HNSW seeding before NSG/MRNG-style pruning so explicit NSG no longer starts with an all-pairs exact seed on MNIST/FashionMNIST-scale matrices. |
| `vamana` | `cpu_vamana` speed/balanced tiers | CPU graph candidate | Native DiskANN/Vamana-style robust-pruned candidate graph; large high-dimensional CPU inputs use deterministic HNSW seeding before robust pruning, while smaller inputs keep exact seeding. |
| `nndescent` | `cpu_nndescent` speed/balanced tiers | CPU graph speed tier | Native faissR NN-descent route; useful as an explicit Euclidean, normalized cosine/correlation, or raw inner-product graph-search candidate, but recall was usually lower than HNSW. |
| `nndescent` | `cuda_cuvs_nndescent` | CUDA graph speed tier | Fast and useful at around 0.99 recall on some datasets; failed on COIL20. |
| `nndescent` | `cuda_native_nndescent` | CUDA raw inner-product tier | Native CUDA candidate-refinement route used by public `backend = "cuda", method = "nndescent", metric = "inner_product"` because direct cuVS NN-descent does not expose raw IP. |
| `cagra` | `faiss_gpu_cagra` | CUDA graph high-recall tier | FAISS `GpuIndexCagra` path using the FAISS GPU/cuVS integration when the linked FAISS build exposes it. This provider is selected by `cagra_implementation = "faiss_gpu"` and remains the default for most shapes when both providers are available [13-15]. |
| `cagra` | `cuda_cuvs_cagra` with `ivf_pq` build | Direct cuVS CAGRA default large-shape builder | Direct RAPIDS cuVS CAGRA using the cuVS IVF-PQ graph builder. This can be fast, but high-dimensional compact matrices can request very large temporary workspace, so the auto build rule avoids it for COIL20-like shapes [3]. |
| `cagra` | `cuda_cuvs_cagra` with `iterative_cagra_search` build | Direct cuVS compact high-dimensional builder | Direct RAPIDS cuVS CAGRA using iterative CAGRA graph construction. This is the default direct-cuVS build for compact high-dimensional self-KNN (`n <= 5000`, `p >= 1024`, `k <= 100`) because it avoids the IVF-PQ workspace spike observed on COIL20 while keeping the method as CAGRA. |
| `cagra` | `cuda_cuvs_cagra` with `nn_descent` build | Direct cuVS experimental builder | Direct RAPIDS cuVS CAGRA using cuVS NN-descent graph construction. This failed on COIL20 in the current runtime with a CUDA invalid-argument error and should remain an explicit benchmark setting rather than an auto default. |
| `grid` | `cpu_grid` | Exact 2D/3D spatial path | Best for simulated 2D/3D Euclidean/cosine/correlation data; unavailable by design outside 2D/3D. |
| `grid` | `cuda_grid` | CUDA 2D/3D spatial path | Correct for 2D/3D, but benchmark speed depends strongly on GPU model and transfer overhead. |

Public calls use `method = "cagra"` for both CUDA CAGRA providers. The provider
is selected by `cagra_implementation = NULL`, `"auto"`, `"faiss_gpu"`, or
`"cuvs"`; `NULL` uses `options(faissR.cagra_implementation = "auto")`. The
default `"auto"` is deterministic and shape-aware: compact high-dimensional
self-KNN uses direct cuVS CAGRA when both providers are available, while other
shapes keep FAISS GPU CAGRA as the default when it is available. Forcing
`"cuvs"` is useful for isolated cuVS benchmarks, while forcing `"faiss_gpu"`
isolates the FAISS GPU/cuVS integration path. Preflight availability checks
respect this forced provider for supported metrics, and returned approximation
metadata records `cagra_provider` plus `cagra_provider_option`.

Direct RAPIDS cuVS CAGRA has a second choice, `cagra_build_algo`. This is not a
fallback to a different public method; it is the graph-construction algorithm
inside cuVS CAGRA. `cagra_build_algo = "auto"` currently applies this no-pilot
rule:

| Direct-cuVS CAGRA shape | Automatic build algorithm | Reason |
|---|---|---|
| self-KNN, `n <= 5000`, `p >= 1024`, `k <= 100` | `iterative_cagra_search` | Compact high-dimensional matrices can make the IVF-PQ builder request excessive temporary workspace; the iterative builder was validated on COIL20. |
| self-KNN with compact-tuning flag from the CAGRA parameter rule | `iterative_cagra_search` | Keeps small compact graph builds on the lower-workspace direct CAGRA path. |
| non-self queries or other shapes | `ivf_pq` | cuVS IVF-PQ graph construction remains the general direct-cuVS CAGRA builder. |

cuVS HNSW is different from direct CAGRA search: faissR first builds a CAGRA
graph and then converts it to a cuVS HNSW index. For `method = "hnsw"` with
`backend = "cuda"`, automatic tuning therefore uses
`iterative_cagra_search` as the CAGRA seed builder even on larger shapes. On
MNIST70k (`70000 x 784`, `k = 50`) the IVF-PQ seed builder was fast but produced
near-zero sampled recall for HNSW, while `iterative_cagra_search` restored high
sampled recall. The graph/search effort now comes from the requested
`target_recall` tier: 0.90 is the fastest, 0.95 is the middle tier, and 0.99 is
the default high-recall tier where feasible. Large high-dimensional data
(`n >= 50000`, `p >= 256`) uses progressively wider graph/search settings as
the target rises. Medium low-dimensional data (`50000 <= n < 500000`,
`p <= 64`) uses narrower graph degrees for 0.90/0.95 and the prior graph-96
tier for 0.99. Very large low-dimensional data keeps graph degree 48 for
`k <= 15`; for 5M-row-class inputs with larger `k`, faissR applies a runtime
guard for the 0.99 target rather than the graph-64/96 high-recall tier because
FlowRepository k = 50 timed out at 600 seconds on Chiamaka with both graph
64/intermediate 128/ef 200 and graph 96/intermediate 192/ef 250. Users can raise
`options(faissR.cuvs_graph_degree = ..., faissR.cuvs_intermediate_graph_degree = ..., faissR.cuvs_hnsw_ef = ...)`
for stricter recall, or request `cagra_build_algo = "ivf_pq"` explicitly for
experiments, but IVF-PQ is not the automatic HNSW seed builder.

The all-dataset HNSW target-recall validation is run with:

```bash
benchmark_scripts/run_benchmark_hnsw_target_recall_chiamaka.sh
```

It uses the float32 manifest, explicit `backend = "cpu"` and `"cuda"`,
`method = "hnsw"`, Euclidean distance, `k = 10, 15, 50, 100`,
`target_recall = 0.9, 0.95, 0.99`, 4 CPU threads, and a 600-second timeout per
row. Result rows record the requested target, actual target, HNSW parameters,
sampled recall, speed, and CPU/CUDA backend separately.

The run is accepted only when the full matrix is present: every requested
dataset, both explicit backends, all four `k` values, and all three target
recall tiers. The summarizer writes:

- `hnsw_target_recall_completeness.csv`: one expected row per
  dataset/backend/`k`/target combination, with missing, duplicate, target-met,
  below-target, and failed/timeout status.
- `hnsw_target_recall_missing_rows.csv`: combinations that were not produced by
  the benchmark. A finished launcher calls the summarizer with
  `--require_complete=TRUE`, so an interrupted or partial sweep is not reported
  as complete.
- `hnsw_target_recall_below_target.csv`: successful rows whose sampled recall
  did not meet the requested tier. These rows are the primary input for
  tightening the C++ shape/`k` policy.
- `hnsw_target_recall_recommendations.csv`: for each
  dataset/backend/`k`/target combination, the fastest successful row that met
  the target, or the highest-recall successful row when the current tier missed.
  These recommendation rows are used to decide whether the C++ defaults should
  move to a wider or narrower HNSW setting.

If a long run is interrupted, resume only the missing combinations with:

```bash
OUT_DIR=/path/to/faissR_HNSW_TARGET_RECALL_FLOAT32_YYYYMMDD_HHMMSS \
  benchmark_scripts/resume_benchmark_hnsw_target_recall_chiamaka.sh
```

The resume launcher re-runs the summarizer, reads
`hnsw_target_recall_missing_rows.csv`, evaluates only those combinations, and
then requires the final completeness audit to pass.

The explicit `"nn_descent"` builder is available for experiments, but it is not
selected automatically because the COIL20 diagnostic failed inside the cuVS
NN-descent CAGRA build with `cudaErrorInvalidValue`.

Approximate routes now attach deterministic no-pilot tuning metadata to
`attr(result, "approximation")`. IVF, IVFPQ/PQ, NSG, NN-descent, CAGRA, and
HNSW report `tuning_policy`, `tuning_rule`, and shape flags where relevant;
IVF also records `tuning_metric` and `tuning_metric_aware`, and IVFPQ/PQ
compression fields use `pq_tuning_*` names. These fields let benchmark tables
compare parameter tiers by dataset shape, `k`, and metric without running extra
tuning inside ordinary `nn()` calls. Deterministic auto-tuned methods report
`tuning_source = "cpp"` so downstream benchmark code can verify that the rule
came from the compiled policy layer rather than from an ad hoc R branch.

## ImageNet Probe

Additional ImageNet probes used a dataset object with `data` and `labels`
fields. The data table had 1,281,167 rows and 1,024 columns and occupied about
10 GB as double-precision R columns. Important probe artifacts include:

- `imagenet_probe_results.csv`: 10k and 50k self-KNN sample results with exact
  FAISS GPU Flat as the recall reference.
- `imagenet_full_query_results.csv`: full-reference query attempts against
  1,000 sampled queries.

On the 50k sample, the fastest exact/high-recall implementation routes were:

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
backend-side float/index buffers. Full ImageNet-style benchmarks should
therefore use a memory-efficient
matrix/float32 representation or run on a host with enough RAM for the source
data, converted matrix, and backend-side buffers.


## Shape-Aware `backend` Plus `method = "auto"`

A follow-up auto-policy run tested the CPU-only and CUDA-only automatic
selectors on simulated shapes and benchmark dataset folders. The automatic
route policy is implemented in C++ by `nn_auto_select_backend_cpp()`. The R
front end supplies normalized arguments, runtime availability flags
(`faiss_available`, `faiss_gpu_available`, `cuvs_available`, `cuda_available`),
and option thresholds, but the selected backend and auto-selection metadata are
produced by the compiled selector. The metadata policy string is
`cpp_static_shape_k_metric_selector`.

The same C++ policy layer now owns the deterministic tuning rules used after a
route is selected. For example, CPU HNSW tiering, IVF `nlist`/`nprobe`, PQ
bit-width selection for small training sets, cuVS CAGRA graph degree and build
widths, cuVS HNSW-from-CAGRA parameters, native NSG/Vamana candidate graph
sizes, and native CUDA NN-descent iteration widths all return
`tuning_source = "cpp"`. Pilot/cache tuning, where explicitly requested,
remains opt-in and separate from `tuning = "auto"`.

Policy summary:

- `backend = "cpu", method = "auto"`: exact CPU for small work; CPU grid for
  large Euclidean/cosine/correlation 2D/3D self-KNN; FAISS IVF for million-row
  Euclidean self-KNN
  where HNSW graph construction is too memory-heavy; FAISS HNSW for large
  high-dimensional CPU self-KNN, including cosine, correlation, and
  inner-product HNSW when FAISS is available; FAISS Flat exact search for larger
  non-Euclidean query or exact workloads; RcppHNSW/hnswlib remains the fallback
  for large non-Euclidean self-KNN when FAISS is unavailable; native CPU
  NN-descent is the final large self-KNN fallback when neither FAISS nor
  RcppHNSW is available.
  On the benchmark `k` grid, large high-dimensional CPU self-search uses
  graph-search routes across `k = 5, 10, 15, 50, and 100` when FAISS HNSW,
  RcppHNSW/hnswlib, or native CPU NN-descent is available; small `k` alone is
  not enough reason to run exact CPU on MNIST/FashionMNIST-scale workloads.
  Non-self non-Euclidean `k = 5` can still use exact FAISS Flat, and explicit
  CPU HNSW uses deterministic `n`, `p`, `k`, and `metric` tiers without
  running a pilot benchmark.
  Flat metric routes when FAISS is available.
- `backend = "cuda", method = "auto"`: CUDA grid for large 2D/3D
  Euclidean/cosine/correlation self-KNN; FAISS GPU Flat for small and medium
  Euclidean or non-Euclidean datasets where exact GPU search is fast; FAISS GPU
  CAGRA for very large Euclidean self-KNN when available; and FAISS GPU or
  direct cuVS CAGRA for very large cosine/correlation/inner-product self-KNN.
  Raw inner-product CAGRA uses the maximum-inner-product-to-L2 graph-search
  transform and returns shifted inner-product distances. On cuVS-only runtimes,
  `backend = "auto"` can select direct cuVS CAGRA for large
  non-Euclidean self-search; smaller non-grid non-Euclidean searches stay on
  CPU unless FAISS GPU Flat is available.

Observed examples from the run:

| Dataset | n x p | CPU auto selected | CPU seconds | CPU recall | CUDA auto selected | CUDA seconds | CUDA recall |
|---|---:|---|---:|---:|---|---:|---:|
| simulated2d | 20000 x 2 | `method = "grid"` (`cpu_grid2d`) | 0.782 | 0.999963 | `method = "grid"` (`cuda_grid2d`) | 0.697 | 0.999965 |
| COIL20 | 1440 x 16384 | `method = "exact"` (`cpu`) | 4.877 | 1.000000 | `method = "flat"` (`faiss_gpu_flat_l2`) | 1.914 | 1.000000 |
| FashionMNIST | 70000 x 784 | `method = "hnsw"` (`faiss_hnsw`) | 20.879 | 0.998682 | `method = "flat"` (`faiss_gpu_flat_l2`) | 6.455 | 1.000000 |
| FlowRepository | 5220347 x 32 | timeout | NA | NA | `method = "cagra"` (`faiss_gpu_cagra`) | 118.268 | NA |
| flow18 | 1000021 x 11 | `method = "ivf"` (`faiss_ivf`) | 35.165 | NA | `method = "cagra"` (`faiss_gpu_cagra`) | 8.181 | NA |
| MNIST | 70000 x 784 | `method = "hnsw"` (`faiss_hnsw`) | 21.602 | 0.996334 | `method = "flat"` (`faiss_gpu_flat_l2`) | 6.197 | 1.000000 |
| TabulaMuris | 70118 x 50 | `method = "hnsw"` (`faiss_hnsw`) | 3.246 | 0.998619 | `method = "flat"` (`faiss_gpu_flat_l2`) | 2.314 | 1.000000 |
| ImageNet sample | 50000 x 1024 | `method = "hnsw"` (`faiss_hnsw`) | 93.956 | 0.999436 | `method = "flat"` (`faiss_gpu_flat_l2`) | 62.963 | 1.000000 |

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
  Default calls now use deterministic no-pilot parameters, so benchmark reports
  must include measured recall. Explicit pilot/cache tuning stops when it cannot
  meet the target recall instead of silently returning a poor result.
- cuVS HNSW should not inherit the direct-CAGRA IVF-PQ auto builder blindly.
  The HNSW conversion depends on a high-quality seed graph; MNIST70k diagnostics
  selected `iterative_cagra_search` as the default HNSW seed builder after IVF-PQ
  returned near-zero recall. CUDA HNSW graph/search effort is selected by
  `target_recall = 0.9`, `0.95`, or `0.99` and records both requested and
  selected target metadata.
- Direct cuVS CAGRA has provider-internal build modes with different memory and
  robustness profiles. In the focused CUDA diagnostic, COIL20 (`1440 x 16384`, `k = 50`,
  Euclidean) completed with FAISS GPU CAGRA, direct cuVS CAGRA `ivf_pq`, and
  direct cuVS CAGRA `iterative_cagra_search`; the `ivf_pq` builder emitted a
  temporary workspace warning around 45 GB, while `iterative_cagra_search`
  completed fastest in that focused diagnostic. The direct cuVS `nn_descent`
  builder failed with `cudaErrorInvalidValue`, so it is explicit-only.
- Public CPU NSG now uses the native faissR NSG-style route for all metrics.
  Large high-dimensional CPU NSG/Vamana use deterministic HNSW seeding before
  method-specific pruning/refinement to avoid all-pairs exact seed timeouts.
  Keep reporting recall before considering either route as a broad auto
  default.
- cuVS NN-Descent failed on COIL20 with a CUDA invalid-argument error. It should
  remain explicit or secondary until more robust guards are added.
- IVFPQ methods are often fast or memory-efficient, but recall was frequently
  poor. They should be documented as compressed-memory methods; CPU IVFPQ rows
  below 624 training rows are expected skips, and CPU/direct-cuVS IVFPQ auto
  tuning uses 4-bit PQ for small training sets to avoid undertrained 8-bit
  codebooks where the backend supports 4-bit PQ. FAISS GPU IVFPQ remains an
  explicit 8-bit GPU route because FAISS' GPU IVFPQ implementation requires
  8-bit codes.

## Reproducibility

The run used isolated worker processes with a fixed timeout per
method/dataset row. Failures and timeouts were recorded and did not stop the
benchmark matrix. CPU methods used a fixed OpenMP/BLAS thread count.
