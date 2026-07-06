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
Euclidean/L2 search, raw unscaled data, and the package benchmark datasets. The
broad NN metric benchmark extends that work to all four public metrics and the
k grid 5, 10, 15, 50, and 100. The dedicated HPC method-tuning sweeps are more
focused: they use explicit CPU or CUDA backends, float32 datasets, Euclidean
distance, target recall tiers of 0.90, 0.95, and 0.99, and the k grid 15, 30,
50, and 100. Important benchmark artifacts include:

- `autotune_results.csv`: one row per dataset and resolved implementation
  label.
- `autotune_method_summary.csv`: method-level speed/recall/failure summary.
- `autotune_recommendations_by_dataset.csv`: fastest method by recall target.
- `autotune_issues.csv`: low-recall, unavailable, or failed rows.

## How The HPC Tuning Sweeps Become Defaults

The HPC method-tuning scripts are not general timing demos. They are designed
to create the evidence used to write deterministic C++ tuning rules for each
method. Each run answers this question: for a fixed method, backend, dataset
shape, `k`, and recall target, what is the fastest parameter setting that still
meets the requested nearest-neighbour recall?

The tuning workflow is:

1. Build one exact reference per dataset folder. The reference job computes a
   high-quality Euclidean `k = 100` self-neighbour table and saves it beside the
   dataset. Later tuning jobs crop that table to `k = 15`, `30`, `50`, or `100`
   instead of recomputing exact neighbours for every method and parameter
   setting. This makes recall comparisons consistent across methods and keeps
   the expensive exact calculation out of the inner tuning loop.
2. Load the float32 dataset manifest. The tuning jobs read `*_float32.RData`
   files and keep the benchmark input in 32-bit form when the method supports
   it. That measures the path we want faissR to use in practice: float32 input,
   C++/CUDA processing, and minimal R-side conversion.
3. Run one method and one backend at a time. CPU and CUDA are benchmarked in
   separate launchers, and `backend = "auto"` is not used. This separation is
   essential because a CPU HNSW setting and a CUDA CAGRA setting can both reach
   recall 0.99 while having completely different bottlenecks, memory traffic,
   and parameter meanings.
4. Evaluate a structured parameter grid. Each candidate row stores the method
   parameters, backend provider, requested output type, thread count, batch
   size, and shape metadata (`n`, `p`, `k`). The grids deliberately include
   fast low-effort settings, balanced settings, and high-recall rescue settings
   so the benchmark can identify both the fastest acceptable point and the
   settings that are still insufficient.
5. Keep every failure row. Timeouts, unavailable backends, CUDA launch errors,
   memory errors, and below-target recall are not discarded. They are guardrails
   for the automatic policy: a method that is fastest on successful rows should
   not become a default for a shape where the same route repeatedly fails.
6. Select recommendations by target recall. For each
   dataset/backend/method/`k`/target-recall group, the recommendation table
   chooses the fastest successful candidate with recall at or above the target.
   If no candidate reaches the target, the table keeps the highest-recall
   successful candidate and marks the row as below target. This prevents a
   missing 0.99 setting from being mistaken for a valid high-recall default.
7. Aggregate by dataset shape. The shape recommendation files group datasets by
   size and dimension, then summarize which parameter tiers worked for similar
   shapes. These files are the bridge between individual benchmark datasets and
   C++ defaults that can generalize to new user matrices. The generated shape
   table is
   `benchmark_scripts/euclidean_shape_tuning_defaults_from_uploaded_results.csv`.
8. Encode conservative deterministic rules in C++. The final package defaults
   are embedded in `src/nn_hpc_tuning_tables.hpp` and consumed by compiled
   helpers such as `nn_tune_faiss_hnsw_cpp()`,
   `nn_tune_faiss_ivf_cpp()`, `nn_tune_cpu_nndescent_cpp()`,
   `nn_tune_cuvs_cagra_cpp()`, `nn_tune_native_nsg_cpp()`, and
   `nn_tune_vamana_cpp()`. Public calls with `tuning = "auto"` use those rules
   without running a pilot benchmark. The default target recall is 0.99; users
   can request 0.90 or 0.95 when they want a faster approximate setting.

The k grid is intentionally `15, 30, 50, 100`. `k = 15` is the common
embedding/graph-neighbour size, `k = 30` tests a denser graph without jumping
straight to a large neighbourhood, and `k = 50`/`100` stress high-degree
search, candidate storage, GPU memory traffic, and self-neighbour filtering.
The exact reference is saved at `k = 100`, so every smaller k value is a
consistent crop of the same reference calculation.

The current all-metric rerun uses the same method-specific launchers with
`METRICS=euclidean,cosine,correlation,inner_product`. The scripts reject
legacy metric aliases and typo compatibility labels, so use only the canonical
metric names when submitting work. The exact-reference precompute writes one
reference per metric in each dataset directory, while the tuning runner groups recommendations by
dataset, backend, method, metric, `k`, and target recall. The Euclidean
recommendation table from the uploaded results is consolidated in
`benchmark_scripts/euclidean_tuning_settings_from_uploaded_results.csv`; the
shape-level table used for compiled defaults is
`benchmark_scripts/euclidean_shape_tuning_defaults_from_uploaded_results.csv`.
Those shape defaults are embedded in `src/nn_hpc_tuning_tables.hpp`, which is
the generated bridge between the uploaded HPC results and C++ `tuning = "auto"`
rules. The all-metric rerun also reads
`benchmark_scripts/previous_tuning_timeouts.csv` so candidates that timed out
in the Euclidean sweeps are recorded as `skipped_previous_timeout` instead of
being resubmitted for cosine, correlation, and inner-product runs. CUDA
NN-descent has metric-specific wrappers for Euclidean and cosine; cosine uses
row-normalized float32 data before direct cuVS NN-descent. Its first compiled
`tuning = "auto"` table is seeded from the CUDA Euclidean NN-descent sweep and
is marked validation-pending until the CUDA cosine sweep is rerun.

For scheduler use, the package also includes generated metric-specific Slurm
wrappers. Their names append the metric to the base launcher, for example
`run_hpc_bruteforce_tuning_cpu12_euclidean.sh`,
`run_hpc_bruteforce_tuning_cpu12_cosine.sh`,
`run_hpc_bruteforce_tuning_cpu12_correlation.sh`, and
`run_hpc_bruteforce_tuning_cpu12_inner_product.sh`. CUDA wrappers follow the
same pattern with `_cuda_<metric>.sh`. Each wrapper exports one `METRICS` value,
sets a metric-specific default `OUT_DIR`, and then executes the base
method/backend launcher. These wrappers are generated by
`benchmark_scripts/generate_metric_specific_launchers.R` so new method launchers
can be expanded consistently.

The target recall tiers have different roles:

- `target_recall = 0.90`: speed-first approximate settings. These are useful
  when downstream embedding or graph construction tolerates a small neighbour
  error and the matrix is large enough that exact search is too expensive.
- `target_recall = 0.95`: balanced settings. These often become the practical
  recommendation when 0.99 requires a large increase in graph degree, search
  breadth, `nprobe`, or refinement cost.
- `target_recall = 0.99`: accuracy-first approximate settings and the default
  policy. This tier is used for the public `tuning = "auto"` default unless a
  method is exact by construction.

Method-specific interpretation of the tuning files:

- `exact`, `flat`, and `bruteforce` do not trade recall for approximation
  parameters. Their tuning files measure provider choice, CPU threads, batch
  size, float output, fitted-index reuse, and GPU resource reuse. These methods
  should reach recall 1.0. CPU `method = "exact"` now uses compiled C++
  selectors for the FAISS Flat L2 Euclidean route and the normalized FAISS Flat
  cosine route. The selector is keyed by metric, shape
  group, `k` bucket, and requested target recall so result metadata records the
  same target tier as approximate methods, but
  `exact_recall_by_construction = TRUE` and `expected_recall_at_k = 1.0` make
  clear that the target is satisfied by the exhaustive search itself. The
  selected row controls the FAISS query batch size and, where implemented for
  the route, fitted Flat-index reuse. Cosine rows labelled
  `best_available_partial_shape_datasets` record
  `tuning_benchmark_target_met = FALSE` to show that the benchmark coverage was
  partial even though the method itself is exact. Failures usually indicate
  memory, timeout, library, or data-layout issues rather than a
  recall/parameter tradeoff.
- `hnsw` uses the sweeps to choose `M`, `efConstruction`, and `efSearch` by
  backend, shape, `k`, and target recall. CPU HNSW is usually limited by index
  construction plus graph-search breadth. CUDA HNSW uses the cuVS wrapper route
  and is reported separately from pure CAGRA because its bottlenecks are not the
  same as FAISS CPU HNSW.
- `ivf` uses the sweeps to tune `nlist` and `nprobe`. Lower `nprobe` is faster
  but may miss clusters; larger `nprobe` improves recall but can approach Flat
  search cost. CPU Euclidean, cosine, correlation, and raw inner-product
  `method = "ivf"` use compiled shape tables from
  `faissR_IVF_TUNING_CPU12_euclidean_20260630_161409`,
  `faissR_IVF_TUNING_CPU12_cosine_20260701_090337`,
  `faissR_IVF_TUNING_CPU12_correlation_20260701_090337`, and
  `faissR_IVF_TUNING_CPU12_inner_product_20260701_090337`. The metric-specific
  source tables are summarized in
  `benchmark_scripts/cosine_ivf_shape_tuning_defaults_from_uploaded_results.csv`,
  `benchmark_scripts/correlation_ivf_shape_tuning_defaults_from_uploaded_results.csv`,
  and
  `benchmark_scripts/inner_product_ivf_shape_tuning_defaults_from_uploaded_results.csv`.
  Rows that reached the target across every dataset in the shape group record
  `tuning_benchmark_target_met = TRUE`; rows labelled
  `best_available_partial_shape_datasets` or `best_recall_below_target` record
  `FALSE` and should be read as the best measured IVF setting, not as a
  guaranteed recall tier.
- `ivfpq` uses the sweeps to tune `nlist`, `nprobe`, `pq_m`, and `pq_nbits`.
  CPU Euclidean, cosine, correlation, and raw inner-product
  `method = "ivfpq"` use compiled tables generated from
  `faissR_IVFPQ_TUNING_CPU12_euclidean_20260630_161409`,
  `faissR_IVFPQ_TUNING_CPU12_cosine_20260701_090337`,
  `faissR_IVFPQ_TUNING_CPU12_correlation_20260701_090337`, and
  `faissR_IVFPQ_TUNING_CPU12_inner_product_20260701_090337`. The promoted
  metric-specific source tables are summarized in
  `benchmark_scripts/cosine_ivfpq_shape_tuning_defaults_from_uploaded_results.csv`,
  `benchmark_scripts/correlation_ivfpq_shape_tuning_defaults_from_uploaded_results.csv`,
  and
  `benchmark_scripts/inner_product_ivfpq_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA Euclidean, correlation, and raw-inner-product `method = "ivfpq"` use
  FAISS GPU IVF-PQ shape/k/target rows. Euclidean and correlation come from
  `faissR_IVFPQ_TUNING_CUDA_euclidean_20260701_194051` and
  `faissR_IVFPQ_TUNING_CUDA_correlation_20260703_095008`, summarized in
  `benchmark_scripts/cuda_ivfpq_euclidean_shape_tuning_defaults_from_uploaded_results.csv`
  and
  `benchmark_scripts/cuda_ivfpq_correlation_shape_tuning_defaults_from_uploaded_results.csv`;
  raw inner product uses the validation-pending seed table
  `benchmark_scripts/cuda_ivfpq_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`
  until `run_hpc_ivfpq_tuning_cuda_inner_product.sh` replaces it.
  IVFPQ is interpreted primarily as a memory-compression method: a best
  available or target-not-reached row with poor recall must not become an
  accuracy default.
- `ivfpq_fastscan` uses the sweeps to tune FastScan-specific compressed-code
  settings. CPU runs evaluate FAISS `IndexIVFPQFastScan` choices such as
  `nlist`, `nprobe`, `pq_m`, block size, and refinement factor. CPU Euclidean
  `method = "ivfpq_fastscan"` uses the compiled
  `hpc_ivfpq_fastscan_cpu12_euclidean_20260630_161409` table with separate
  shape buckets for small high-dimensional data, small lower-dimensional data,
  large high-dimensional data, and very large low-dimensional data. Rows from
  COIL20, USPS, FashionMNIST, and FlowRepository are marked as target hits only
  when the measured recall reached the requested tier; FashionMNIST k=15 at
  0.99 recall, FlowRepository 0.99 rows, and the unmeasured large-low-dimensional
  k=100 fallback record `tuning_benchmark_target_met = FALSE`. CPU raw
  inner-product FastScan uses FAISS FastScan IP; CPU cosine FastScan uses row
  L2 normalization followed by FastScan L2 search; CPU correlation FastScan
  subtracts each row mean, L2-normalizes rows, and then uses the same FastScan
  L2 route. The compiled seed policies are summarized in
  `benchmark_scripts/cosine_ivfpq_fastscan_shape_tuning_defaults_from_uploaded_results.csv`
  and
  `benchmark_scripts/correlation_ivfpq_fastscan_shape_tuning_defaults_from_uploaded_results.csv`,
  and
  `benchmark_scripts/inner_product_ivfpq_fastscan_shape_tuning_defaults_from_uploaded_results.csv`.
  The uploaded cosine, correlation, and inner-product sweeps failed before
  backend execution due to previous metric guards, so these rows deliberately
  record `tuning_benchmark_target_met = FALSE` until the corrected sweeps are
  rerun.
  CUDA runs evaluate cuVS IVF-PQ choices such as `nlist`, `nprobe`,
  byte-aligned 4-bit `pq_dim`, and query batch size. CUDA Euclidean searches
  the original float32 rows. CUDA cosine row-normalizes float32 input before
  cuVS L2 search; CUDA correlation row-centers and row-normalizes float32 input
  before cuVS L2 search, then converts normalized Euclidean distances back to
  correlation distance.
- `cagra` uses the sweeps to tune provider and graph-search parameters:
  FAISS GPU CAGRA versus direct cuVS CAGRA, direct-cuVS build algorithm,
  graph degree, intermediate graph degree, search width, `itopk_size`, and
  batch size. The result tells the package when CAGRA is a fast high-recall GPU
  route and when a specific build algorithm causes workspace or recall issues.
  CUDA Euclidean uses the measured CAGRA table from
  `faissR_CAGRA_TUNING_CUDA_20260628_054710`, summarized in
  `benchmark_scripts/cuda_cagra_euclidean_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA cosine is implemented as row-normalized float32 Euclidean CAGRA search
  and currently uses
  `benchmark_scripts/cuda_cagra_cosine_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the Euclidean sweep because the uploaded cosine sweep stopped at
  an old float32 metric guard before reaching FAISS/cuVS. These cosine defaults
  report `tuning_benchmark_target_met = FALSE` until the corrected cosine sweep
  is rerun. CUDA correlation is implemented as row-centered row-normalized
  float32 Euclidean CAGRA search and currently uses
  `benchmark_scripts/cuda_cagra_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the same Euclidean sweep until the dedicated correlation sweep is
  rerun. CUDA raw inner product uses a maximum-inner-product-to-L2
  extra-dimension transform before CAGRA search and currently uses
  `benchmark_scripts/cuda_cagra_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the Euclidean sweep until
  `run_hpc_cagra_tuning_cuda_inner_product.sh` replaces it with measured rows.
- `nndescent` uses the sweeps to tune candidate pool size, iteration count,
  maximum candidate breadth, and random-projection seed count. CPU Euclidean,
  cosine, correlation, and raw inner-product `method = "nndescent"` use compiled tables generated from
  `faissR_NNDESCENT_TUNING_CPU12_euclidean_20260630_161409`,
  `faissR_NNDESCENT_TUNING_CPU12_cosine_20260701_090337`,
  `faissR_NNDESCENT_TUNING_CPU12_correlation_20260701_090337`, and
  `faissR_NNDESCENT_TUNING_CPU12_inner_product_20260701_090337`; the metric
  shape source tables are
  `benchmark_scripts/cosine_nndescent_shape_tuning_defaults_from_uploaded_results.csv`,
  `benchmark_scripts/correlation_nndescent_shape_tuning_defaults_from_uploaded_results.csv`,
  and
  `benchmark_scripts/inner_product_nndescent_shape_tuning_defaults_from_uploaded_results.csv`.
  These tables cover small-n, medium low-dimensional, large low-dimensional,
  and large high-dimensional shapes for `k` buckets 15, 30, 50, and 100 and
  target recall tiers 0.90, 0.95, and 0.99. Cosine and correlation are
  implemented as normalized Euclidean graph search after row normalization
  or row centering plus normalization, while raw inner product ranks larger
  dot products with faissR's shifted smaller-is-better distance convention.
  Each CPU metric keeps its own measured tuning
  table so normalized L2 searches do not accidentally reuse Euclidean recall
  evidence. CUDA Euclidean uses
  `benchmark_scripts/cuda_nndescent_euclidean_shape_tuning_defaults_from_uploaded_results.csv`;
  CUDA cosine uses
  `benchmark_scripts/cuda_nndescent_cosine_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  and CUDA correlation uses
  `benchmark_scripts/cuda_nndescent_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
  seeded from the CUDA Euclidean sweep because the prior CUDA cosine route was
  blocked before reaching the cuVS backend and no measured CUDA correlation
  sweep has been uploaded yet. CUDA cosine and correlation rows report
  `tuning_benchmark_target_met = FALSE` until the corrected metric-specific
  sweeps are rerun.
  Rows labelled `best_available_all_shape_datasets`
  return `tuning_benchmark_target_met = FALSE`, because NN-descent did not
  always reach the requested recall tier across every dataset in that shape
  group.
  CPU and CUDA are interpreted separately because the CUDA route depends on
  cuVS kernels and can expose GPU-launch or shared-memory limits not present in
  the CPU mathematics.
- `nsg` and `vamana` use the sweeps to tune seed-neighbour count, graph degree,
  pruning/search breadth, and Vamana `alpha`. CPU Euclidean, cosine, correlation, and raw inner-product
  `method = "nsg"` use compiled tables generated from
  `faissR_NSG_TUNING_CPU12_euclidean_20260630_161409`,
  `faissR_NSG_TUNING_CPU12_cosine_20260701_090337`,
  `faissR_NSG_TUNING_CPU12_correlation_20260701_090337`, and
  `faissR_NSG_TUNING_CPU12_inner_product_20260701_090337`; the metric shape
  source tables are `benchmark_scripts/cosine_nsg_shape_tuning_defaults_from_uploaded_results.csv`,
  `benchmark_scripts/correlation_nsg_shape_tuning_defaults_from_uploaded_results.csv`,
  and `benchmark_scripts/inner_product_nsg_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA Euclidean and cosine `method = "nsg"` use the measured GPU shape tables
  from `faissR_NSG_TUNING_CUDA_euclidean_20260702_013830` and
  `faissR_NSG_TUNING_CUDA_cosine_20260702_211910`; their source tables are
  `benchmark_scripts/cuda_nsg_euclidean_shape_tuning_defaults_from_uploaded_results.csv`
  and
  `benchmark_scripts/cuda_nsg_cosine_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA correlation uses the same centered-normalized NSG route and currently
  seeds `r`/`graph_k` from the measured CUDA cosine table in
  `benchmark_scripts/cuda_nsg_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv`.
  CUDA raw inner product uses the native CUDA NSG shifted dot-product route and
  currently seeds `r`/`graph_k` from the same measured CUDA cosine table in
  `benchmark_scripts/cuda_nsg_inner_product_shape_tuning_defaults_from_seeded_cosine_results.csv`.
  Both validation-pending CUDA NSG metric tables report
  `tuning_benchmark_target_met = FALSE` until their dedicated sweeps replace
  them.
  The tables select NSG `r` and seed/candidate graph width `graph_k` by
  small-n, medium low-dimensional, large low-dimensional, and large
  high-dimensional shape groups, `k`, and target recall. Cosine is implemented
  as row-normalized Euclidean NSG refinement and correlation as row-centered,
  normalized Euclidean NSG refinement; raw inner product keeps the package-wide
  shifted dot-product ordering and now has its own metric-specific tuning table.
  CPU Euclidean, cosine, correlation, and raw inner-product `method = "vamana"` use compiled
  tables generated from `faissR_VAMANA_TUNING_CPU12_euclidean_20260630_161409`,
  `faissR_VAMANA_TUNING_CPU12_cosine_20260701_090337`,
  `faissR_VAMANA_TUNING_CPU12_correlation_20260701_090337`, and
  `faissR_VAMANA_TUNING_CPU12_inner_product_20260701_090337`; the metric shape
  source tables are `benchmark_scripts/cosine_vamana_shape_tuning_defaults_from_uploaded_results.csv`,
  `benchmark_scripts/correlation_vamana_shape_tuning_defaults_from_uploaded_results.csv`,
  and `benchmark_scripts/inner_product_vamana_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA Euclidean and cosine `method = "vamana"` use measured GPU shape tables
  from `faissR_VAMANA_TUNING_CUDA_euclidean_20260702_042943` and
  `faissR_VAMANA_TUNING_CUDA_cosine_20260702_232209`; their source tables are
  `benchmark_scripts/cuda_vamana_euclidean_shape_tuning_defaults_from_uploaded_results.csv`
  and
  `benchmark_scripts/cuda_vamana_cosine_shape_tuning_defaults_from_uploaded_results.csv`.
  CUDA correlation uses the same centered-normalized Vamana route and currently
  seeds `r`/`search_l`/`alpha` from the measured CUDA cosine table in
  `benchmark_scripts/cuda_vamana_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv`.
  CUDA raw inner product uses the native CUDA Vamana shifted dot-product route
  and currently seeds `r`/`search_l`/`alpha` from the same measured CUDA cosine
  table in
  `benchmark_scripts/cuda_vamana_inner_product_shape_tuning_defaults_from_seeded_cosine_results.csv`.
  Both validation-pending CUDA Vamana metric tables report
  `tuning_benchmark_target_met = FALSE` until their dedicated sweeps replace
  them.
  The tables select Vamana `r`, search breadth `search_l`, and robust-pruning
  `alpha` over the same shape/k/target-recall grid. Cosine is implemented as
  row-normalized Euclidean Vamana refinement and correlation as row-centered,
  normalized Euclidean Vamana refinement; raw inner product keeps shifted
  dot-product ordering, with metric-specific tuning tables.
  Rows labelled `best_available_partial_shape_datasets` return
  `tuning_benchmark_target_met = FALSE`, because not every dataset in that
  large low-dimensional shape completed with the same candidate. These methods
  can be fast only if the candidate graph is built with enough neighbours for
  the requested recall and not so many that construction dominates the timing.
- `grid` is a specialized exact route for two- and three-dimensional data. It
  is not part of the general high-dimensional tuning policy.

The rule for updating package defaults is conservative. A new C++ default
should be added only after the CPU and CUDA recommendation files agree that the
setting is the fastest successful choice for the relevant shape and target, or
after failures show that a previously selected route should be avoided for that
shape. The C++ policy should record the selected target, rule name, and
parameters in result metadata so benchmark tables can verify which rule was
used.

## Default Policy

Use these rules for `backend = "auto"` and for explicit backend
recommendations. Public calls should use canonical method names such as
`"exact"`, `"flat"`, `"hnsw"`, `"ivf"`, `"ivfpq"`, `"ivfpq_fastscan"`,
`"nndescent"`, or `"cagra"`; labels such as `faiss_hnsw`,
`faiss_ivfpq_fastscan`, `cuda_cuvs_ivfpq_fastscan`, or `cuda_cuvs_cagra` are resolved
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
- For Euclidean CUDA self-KNN, `method = "auto"` uses the compiled
  shape/k/target-recall selector derived from the tuning sweeps. It chooses
  IVF-Flat for large low-dimensional data, keeps exact Flat/brute force for
  measured small, medium, and high-dimensional accuracy-first shapes, and uses
  IVF for very large high-dimensional data only at lower target-recall tiers.
  This avoids treating IVF as a generic "large data" rule when exact FAISS GPU
  Flat was faster on MNIST/FashionMNIST-like and ImageNet-like high-dimensional
  rows.
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
  For CUDA cosine, correlation, and raw inner product, CAGRA now follows the
  same transformed float32 graph-search conventions as other graph methods:
  cosine normalizes rows, correlation row-centers before normalization, and raw
  inner product uses the maximum-inner-product-to-L2 transform. The current
  compiled shape/k/target policies are validation-pending and seeded from the
  measured Euclidean table.

## Method-Specific Settings

| Public method | Resolved implementation route | Role | Current tuning rule |
|---|---|---|---|
| `exact` | `faiss_flat_l2` / `faiss_flat_cosine` / `faiss_flat_correlation` / `faiss_flat_ip` | CPU exact baseline | CPU `method = "exact"` resolves to FAISS Flat L2 for Euclidean, normalized FAISS Flat cosine for cosine, centered/normalized FAISS Flat correlation for correlation, and FAISS Flat IP for raw inner product. `tuning = "auto"` uses compiled `hpc_cpu_exact_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_EXACT_TUNING_CPU12_euclidean_20260630_161409`, `faissR_EXACT_TUNING_CPU12_cosine_20260630_161539`, `faissR_EXACT_TUNING_CPU12_correlation_20260701_090337`, and `faissR_EXACT_TUNING_CPU12_inner_product_20260630_161530`: shape groups are small-n, medium low-dimensional, large low-dimensional, and large high-dimensional; k buckets are 15, 30, 50, and 100; target recall rows 0.90/0.95/0.99 record exact recall by construction and tune only batch/cache metadata. Correlation and inner-product rows with partial shape coverage or below-target benchmark rows report `tuning_benchmark_target_met = FALSE`. |
| `flat` | `faiss_flat_l2` / `faiss_flat_cosine` / `faiss_flat_correlation` / `faiss_flat_ip` | CPU FAISS Flat baseline | CPU `method = "flat"` uses compiled metric-aware policies for Euclidean FAISS Flat L2, normalized FAISS Flat cosine, centered/normalized FAISS Flat correlation, and raw FAISS Flat IP from `faissR_FLAT_TUNING_CPU12_euclidean_20260630_161409`, `faissR_FLAT_TUNING_CPU12_cosine_20260701_015607`, `faissR_FLAT_TUNING_CPU12_correlation_20260701_090337`, and `faissR_FLAT_TUNING_CPU12_inner_product_20260630_161530`. The shape groups, k buckets, and recall tiers match the exact table, but the selected query batch size and fitted-index reuse flags are stored separately in `attr(result, "flat_tuning")`; recall remains exact by construction. Correlation and inner-product rows with partial shape coverage or below-target benchmark rows report `tuning_benchmark_target_met = FALSE`. |
| `bruteforce` | `faiss_flat_l2` / `faiss_flat_cosine` / `faiss_flat_correlation` / `faiss_flat_ip` | CPU exhaustive baseline | CPU `method = "bruteforce"` resolves to FAISS Flat L2 for Euclidean, normalized FAISS Flat cosine for cosine, centered/normalized FAISS Flat correlation for correlation, and FAISS Flat IP for raw inner product. `tuning = "auto"` uses compiled `hpc_cpu_bruteforce_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_BRUTEFORCE_TUNING_CPU12_euclidean_20260630_161409`, `faissR_BRUTEFORCE_TUNING_CPU12_cosine_20260630_161535`, `faissR_BRUTEFORCE_TUNING_CPU12_correlation_20260701_090337`, and `faissR_BRUTEFORCE_TUNING_CPU12_inner_product_20260630_161530`; selected batch/cache settings are stored in `attr(result, "bruteforce_tuning")`, and recall is exact by construction. Rows with partial dataset coverage or below-target benchmark rows report `tuning_benchmark_target_met = FALSE`; this includes the inner-product large-shape rows whose uploaded run did not finish every large dataset. |
| `exact` / `flat` | `faiss_gpu_flat_l2` / `faiss_gpu_flat_cosine` / `faiss_gpu_flat_correlation` / `faiss_gpu_flat_ip` | CUDA exact/high-recall | Explicit FAISS GPU Flat route when requested and available. CUDA `method = "exact"` uses compiled Euclidean, cosine, correlation, and raw-inner-product shape/k/target query-batch policies. Euclidean/cosine/correlation rows come from `faissR_EXACT_TUNING_CUDA_euclidean_20260701_014100`, `faissR_EXACT_TUNING_CUDA_cosine_20260702_110455`, and `faissR_EXACT_TUNING_CUDA_correlation_20260703_023519`; inner-product rows currently use `benchmark_scripts/cuda_exact_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`, seeded from measured CUDA exact Euclidean batch/resource choices and marked validation-pending with `tuning_benchmark_target_met = FALSE` until the dedicated inner-product sweep replaces them. CUDA `method = "flat"` uses the corresponding Flat sweeps for Euclidean/cosine/correlation plus `benchmark_scripts/cuda_flat_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv` for raw inner product until `run_hpc_flat_tuning_cuda_inner_product.sh` replaces the seed. Exact/Flat CUDA correlation is searched as centered, L2-normalized float32 vectors through FAISS GPU Flat, while raw inner product uses FAISS GPU Flat IP. Exact/Flat recall is exact by construction while benchmark provenance remains visible in tuning metadata. |
| `bruteforce` | `cuda_cuvs_bruteforce` | CUDA exact/high-recall | Preferred explicit cuVS exact path; consistently recall 1 in this benchmark and often fastest on compact high-dimensional self-KNN. CUDA Euclidean/cosine/correlation/inner-product `tuning = "auto"` selects compiled query-batch/resource rows by shape, `k`, and target recall; correlation rows use `benchmark_scripts/cuda_bruteforce_correlation_shape_tuning_defaults_from_proxy_results.csv`, and raw inner-product rows use `benchmark_scripts/cuda_bruteforce_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`, both seeded from measured Euclidean cuVS brute-force rows until corrected metric-specific sweeps replace them. Also selected by CUDA `method = "auto"` for compact exact self-KNN when cuVS is available, and used as the fallback exact CUDA route when FAISS GPU Flat is unavailable. |
| `hnsw` | `faiss_hnsw` Euclidean/cosine/correlation/inner-product CPU12 tiers | CPU target-recall speed tiers | CPU `method = "hnsw"` uses compiled tables from `faissR_HNSW_TUNING_CPU12_euclidean_20260630_161409`, `faissR_HNSW_TUNING_CPU12_cosine_20260701_082849`, `faissR_HNSW_TUNING_CPU12_correlation_20260701_090337`, and `faissR_HNSW_TUNING_CPU12_inner_product_20260701_090337`. The tables have separate settings for small-n, medium low-dimensional, large low-dimensional, and large high-dimensional shapes; `k` buckets are 15, 30, 50, and 100; `target_recall` selects 0.90, 0.95, or 0.99 `M`/`efConstruction`/`efSearch` tiers. Rows that reached the requested target on every dataset in the shape group report `tuning_benchmark_target_met = TRUE`; best-available or below-target rows, including many raw inner-product large-shape and 0.99 rows, report `FALSE` [5]. |
| `hnsw` | `faiss_hnsw` balanced tier | CPU default tier | M = 32, efConstruction = 200, efSearch = max(150, 3k); default deterministic shape/metric rule for general CPU HNSW. |
| `hnsw` | `faiss_hnsw` high-recall tier | CPU high-recall tier | M = 48, efConstruction = 240, efSearch = max(220, 3k); used for large-k high-dimensional searches and high-dimensional non-Euclidean searches where normalized IP/correlation routes need extra graph-search breadth. |
| `ivf` | `faiss_ivf` Euclidean/cosine/correlation/inner-product CPU12 tiers | CPU IVF target-recall tiers | CPU Euclidean, cosine, correlation, and raw inner-product `method = "ivf"` use compiled `hpc_cpu_ivf_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_IVF_TUNING_CPU12_euclidean_20260630_161409`, `faissR_IVF_TUNING_CPU12_cosine_20260701_090337`, `faissR_IVF_TUNING_CPU12_correlation_20260701_090337`, and `faissR_IVF_TUNING_CPU12_inner_product_20260701_090337`. The tables store `nlist`/`nprobe` for small-n, medium low-dimensional, large low-dimensional, and large high-dimensional shapes; `k` buckets are 15, 30, 50, and 100; `target_recall` selects 0.90, 0.95, or 0.99 probe tiers. Rows marked `fastest_meeting_target_*` reached the requested target for every dataset represented in that shape row. Rows marked `best_available_partial_shape_datasets` or `best_recall_below_target` are deliberately labelled as best available and return `tuning_benchmark_target_met = FALSE`, including raw inner-product rows where ImageNet had no successful CPU IVF IP rows and large low-dimensional datasets did not reach the requested target. |
| `ivf` | `faiss_gpu_ivf_flat` | CUDA IVF-Flat | `tuning = "auto"` uses compiled shape/k/target-recall policies from CUDA IVF HPC sweeps for Euclidean, cosine, correlation, and raw inner-product search: compact high-dimensional, medium image-like, large low-dimensional flow-like, and large high-dimensional ImageNet-like matrices get different `nlist`/`nprobe` tiers. Cosine uses row-normalized float32 IVF search and the metric-specific table from `faissR_IVF_TUNING_CUDA_cosine_20260702_192200`; correlation uses centered row-normalized float32 IVF search and `benchmark_scripts/cuda_ivf_correlation_shape_tuning_defaults_from_uploaded_results.csv`, derived from `faissR_IVF_TUNING_CUDA_correlation_20260703_133655`; raw inner product uses FAISS GPU IVF IP with `benchmark_scripts/cuda_ivf_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`, seeded from measured CUDA IVF Euclidean rows until `run_hpc_ivf_tuning_cuda_inner_product.sh` replaces it. Seeded rows return `tuning_benchmark_target_met = FALSE`. Explicit `tuning = "cache"` or `"pilot"` still runs only for Euclidean IVF because the pilot reference/candidates are raw-L2. |
| `ivf` | `cuda_cuvs_ivf_flat` | CUDA cuVS IVF-Flat | Direct benchmark route for Euclidean/L2 plus transformed cosine, correlation, and raw inner product. It uses the same compiled CUDA IVF `nlist`/`nprobe` policy as FAISS GPU IVF, including the validation-pending raw-inner-product seed table, with manual overrides through `options(faissR.cuda_ivf_nlist = ..., faissR.cuda_ivf_nprobe = ...)` or provider-specific options. |
| `ivfpq` | `faiss_ivfpq` Euclidean/cosine/correlation/inner-product CPU12 tiers | CPU memory-pressure tier | CPU Euclidean, cosine, correlation, and raw inner-product `method = "ivfpq"` use compiled `hpc_cpu_ivfpq_<metric>_<shape>_k<bucket>_recall<target>` policies from `hpc_ivfpq_cpu12_euclidean_shape_defaults_20260630_161409`, `faissR_IVFPQ_TUNING_CPU12_cosine_20260701_090337`, `faissR_IVFPQ_TUNING_CPU12_correlation_20260701_090337`, and `faissR_IVFPQ_TUNING_CPU12_inner_product_20260701_090337`. The tables store `nlist`, `nprobe`, `pq_m`, and `pq_nbits`; `k` buckets are 15, 30, 50, and 100; `target_recall` selects 0.90, 0.95, or 0.99 rows. Rows marked `target_not_reached_best_available_*` or `best_available_partial_shape_datasets_*` return `tuning_benchmark_target_met = FALSE`; most raw inner-product IVFPQ rows are best-available rather than verified target-recall rows. Use IVFPQ when compression/memory pressure matters more than accuracy. |
| `ivfpq` | `faiss_gpu_ivfpq` | CUDA memory-pressure tier | CUDA Euclidean, correlation, and raw-inner-product `tuning = "auto"` use compiled shape/k/target policies for FAISS GPU IVF-PQ `nlist`, `nprobe`, `pq_m`, and `pq_nbits`. Euclidean rows come from `faissR_IVFPQ_TUNING_CUDA_euclidean_20260701_194051`; correlation rows come from `faissR_IVFPQ_TUNING_CUDA_correlation_20260703_095008` and use centered row-normalized float32 search before FAISS GPU IVFPQ; raw inner product uses FAISS GPU IVFPQ IP with `benchmark_scripts/cuda_ivfpq_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`, seeded from measured CUDA Euclidean rows until `run_hpc_ivfpq_tuning_cuda_inner_product.sh` replaces it. Seeded rows and below-target/partial rows record `tuning_benchmark_target_met = FALSE`. IVFPQ is fast and memory-compressed but low recall in several shape groups; explicit opt-in only. |
| `ivfpq` | `cuda_cuvs_ivfpq` | CUDA memory-pressure tier | Direct benchmark route for Euclidean/L2 plus transformed cosine, correlation, and raw inner product. It uses the same deterministic small-training rule as CPU PQ: below 9,984 training rows, auto tuning requests 4-bit PQ unless the user manually sets `cuvs_ivfpq_pq_bits`/`ivfpq_pq_bits`. Better than FAISS GPU IVFPQ on some datasets but still not an accuracy-first default. |
| `ivfpq_fastscan` | `faiss_ivfpq_fastscan` | CPU IVFPQ FastScan compressed-code tier | CPU Euclidean, cosine, correlation, and raw inner-product `tuning = "auto"` use compiled `hpc_cpu_ivfpq_fastscan_<metric>_<shape>_k<bucket>_recall<target>` policies. Euclidean uses `faissR_IVFPQ_FASTSCAN_TUNING_CPU12_euclidean_20260630_161409`; cosine, correlation, and inner product currently use `cosine_ivfpq_fastscan_shape_tuning_defaults_from_uploaded_results.csv`, `correlation_ivfpq_fastscan_shape_tuning_defaults_from_uploaded_results.csv`, and `inner_product_ivfpq_fastscan_shape_tuning_defaults_from_uploaded_results.csv`, seeded from the Euclidean table because the uploaded metric-specific runs were stopped by old metric guards. The table stores `nlist`, `nprobe`, `pq_m`, fixed 4-bit PQ, `refine_factor`, and FastScan block size, with separate small high-dimensional and small lower-dimensional buckets. Seeded non-Euclidean rows return `tuning_benchmark_target_met = FALSE` until the corrected sweeps are rerun. Requires a FAISS build exposing `IndexIVFPQFastScan` [6,34]. |
| `ivfpq_fastscan` | `cuda_cuvs_ivfpq_fastscan` | CUDA IVFPQ FastScan compressed-code tier | Uses direct cuVS IVF-PQ with 4-bit compressed codes for Euclidean, cosine, correlation, and raw-inner-product search. Cosine row-normalizes the input to float32; correlation row-centers and row-normalizes the input to float32; raw inner product applies the maximum-inner-product-to-L2 extra-dimension transform; transformed metrics search with cuVS L2 and convert distances back to the public distance. Compatible raw `nn()` calls reuse the fitted cuVS IVF-PQ index, dataset device buffer, compressed codes, and cuVS resources through a bounded session cache; self-query uses the fitted dataset device buffer directly, and repeated separate-query calls can reuse one cached query device buffer. The C++ tuner and cuVS wrapper repair invalid 4-bit `pq_dim` values to satisfy byte-aligned packed codes, then record requested versus actual PQ settings. CUDA cosine `tuning = "auto"` uses `benchmark_scripts/cuda_ivfpq_fastscan_cosine_shape_tuning_defaults_from_seeded_euclidean_results.csv`; CUDA correlation uses `benchmark_scripts/cuda_ivfpq_fastscan_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`; CUDA raw inner product uses `benchmark_scripts/cuda_ivfpq_fastscan_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`. These are seeded from `faissR_IVFPQ_FASTSCAN_TUNING_CUDA_euclidean_20260701_100837` where metric-specific runs have not yet replaced the seed. Seeded rows report `tuning_benchmark_target_met = FALSE` until the corrected metric-specific sweeps are rerun. It is kept separate from FAISS GPU IVFPQ and does not fall back to CPU FastScan when CUDA/cuVS is unavailable [3,6,34]. |
| `nsg` | `cpu_nsg` Euclidean/cosine/correlation/inner-product CPU12 tiers; `cuda_nsg` Euclidean/cosine measured CUDA tiers plus validation-pending CUDA correlation/inner-product tiers | Native graph candidate | CPU Euclidean, cosine, correlation, and raw inner-product `tuning = "auto"` use compiled `hpc_cpu_nsg_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_NSG_TUNING_CPU12_euclidean_20260630_161409`, `faissR_NSG_TUNING_CPU12_cosine_20260701_090337`, `faissR_NSG_TUNING_CPU12_correlation_20260701_090337`, and `faissR_NSG_TUNING_CPU12_inner_product_20260701_090337`. CUDA Euclidean and cosine `tuning = "auto"` use measured `hpc_cuda_nsg_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_NSG_TUNING_CUDA_euclidean_20260702_013830` and `faissR_NSG_TUNING_CUDA_cosine_20260702_211910`; CUDA correlation and raw inner product use `hpc_cuda_nsg_<metric>_<shape>_k<bucket>_recall<target>` policies seeded from the measured CUDA cosine rows in `benchmark_scripts/cuda_nsg_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv` and `benchmark_scripts/cuda_nsg_inner_product_shape_tuning_defaults_from_seeded_cosine_results.csv`, respectively, and return `tuning_benchmark_target_met = FALSE` until the dedicated metric sweeps are rerun. The tables store NSG pruning degree `r` and seed/candidate graph width `graph_k` for 0.90, 0.95, and 0.99 target recall tiers. Native faissR NSG avoids linked-FAISS NSG aborts in public calls; large high-dimensional CPU inputs use deterministic HNSW seeding before NSG/MRNG-style pruning, while CUDA keeps the native CUDA row-candidate refinement path. Cosine uses row-normalized Euclidean refinement, correlation uses row-centered normalized Euclidean refinement, and raw inner product uses shifted dot-product ordering, with metric-specific tuning tables. Rows marked `best_available_all_shape_datasets`, `best_available_partial_shape_datasets`, or seeded validation-pending return `tuning_benchmark_target_met = FALSE`, including large high-dimensional raw inner-product rows that did not reach the requested target and large-low-dimensional rows where FlowRepository did not complete trusted rows. |
| `vamana` | `cpu_vamana` Euclidean/cosine/correlation/inner-product CPU12 tiers; `cuda_vamana` Euclidean/cosine measured CUDA tiers plus validation-pending CUDA correlation/inner-product tiers | Native graph candidate | CPU Euclidean, cosine, correlation, and raw inner-product `tuning = "auto"` use compiled `hpc_cpu_vamana_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_VAMANA_TUNING_CPU12_euclidean_20260630_161409`, `faissR_VAMANA_TUNING_CPU12_cosine_20260701_090337`, `faissR_VAMANA_TUNING_CPU12_correlation_20260701_090337`, and `faissR_VAMANA_TUNING_CPU12_inner_product_20260701_090337`. CUDA Euclidean and cosine `tuning = "auto"` use measured `hpc_cuda_vamana_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_VAMANA_TUNING_CUDA_euclidean_20260702_042943` and `faissR_VAMANA_TUNING_CUDA_cosine_20260702_232209`; CUDA correlation and raw inner product use `hpc_cuda_vamana_<metric>_<shape>_k<bucket>_recall<target>` policies seeded from the measured CUDA cosine rows in `benchmark_scripts/cuda_vamana_correlation_shape_tuning_defaults_from_seeded_cosine_results.csv` and `benchmark_scripts/cuda_vamana_inner_product_shape_tuning_defaults_from_seeded_cosine_results.csv`, respectively, and return `tuning_benchmark_target_met = FALSE` until the dedicated metric sweeps are rerun. The tables store Vamana graph degree `r`, search breadth `search_l`, and robust-pruning `alpha` for 0.90, 0.95, and 0.99 target recall tiers. Native faissR Vamana builds a DiskANN/Vamana-style robust-pruned candidate graph; large high-dimensional CPU inputs use deterministic HNSW seeding before robust pruning, while CUDA keeps the native CUDA row-candidate refinement path. Cosine uses row-normalized Euclidean refinement, correlation uses row-centered normalized Euclidean refinement, and raw inner product uses shifted dot-product ordering, with metric-specific tuning tables. Rows marked `best_available_all_shape_datasets`, `best_available_partial_shape_datasets`, or seeded validation-pending return `tuning_benchmark_target_met = FALSE`, including large high-dimensional raw inner-product rows that did not reach the requested target, large-low-dimensional CPU rows where FlowRepository did not complete trusted rows, and CUDA cosine k=100 rows whose best available recall was just below the requested target. |
| `nndescent` | `cpu_nndescent` Euclidean/cosine/correlation/inner-product CPU12 tiers | CPU graph speed tier | CPU Euclidean, cosine, correlation, and raw inner-product `tuning = "auto"` use compiled `hpc_cpu_nndescent_<metric>_<shape>_k<bucket>_recall<target>` policies from `faissR_NNDESCENT_TUNING_CPU12_euclidean_20260630_161409`, `faissR_NNDESCENT_TUNING_CPU12_cosine_20260701_090337`, `faissR_NNDESCENT_TUNING_CPU12_correlation_20260701_090337`, and `faissR_NNDESCENT_TUNING_CPU12_inner_product_20260701_090337`. The tables store candidate pool size, iteration count, maximum candidate breadth, and random-projection count for 0.90, 0.95, and 0.99 target recall tiers. Cosine uses row-normalized Euclidean graph search, correlation uses row-centered normalized Euclidean graph search, and raw inner product ranks larger dot products with shifted smaller-is-better distances. Rows marked `best_available_all_shape_datasets` are deliberately exposed as best available and return `tuning_benchmark_target_met = FALSE`; NN-descent is therefore an explicit speed/graph candidate rather than the accuracy-first CPU default. |
| `nndescent` | `cuda_cuvs_nndescent` | CUDA graph speed tier | Euclidean `tuning = "auto"` uses compiled shape/k/target rows from `faissR_NNDESCENT_TUNING_CUDA_20260630_173056`. Cosine uses row-normalized float32 input before direct cuVS NN-descent and currently uses `benchmark_scripts/cuda_nndescent_cosine_shape_tuning_defaults_from_seeded_euclidean_results.csv`; correlation row-centers and row-normalizes float32 input before direct cuVS NN-descent and currently uses `benchmark_scripts/cuda_nndescent_correlation_shape_tuning_defaults_from_seeded_euclidean_results.csv`. Both non-Euclidean CUDA policies are seeded from the Euclidean sweep with `tuning_benchmark_target_met = FALSE` until corrected metric-specific HPC sweeps are rerun. On affected cuVS builds, high-dimensional FP32 inputs such as COIL20 can fail with `cudaErrorInvalidValue` because cuVS NN-descent does not opt into the required dynamic shared-memory launch limit; a local upstream-style cuVS patch fixed COIL20 and MNIST70k on the test machine. |
| `cagra` | `faiss_gpu_cagra` | CUDA graph high-recall tier | FAISS `GpuIndexCagra` path using the FAISS GPU/cuVS integration when the linked FAISS build exposes it. This provider is selected by `cagra_implementation = "faiss_gpu"` and remains the default for most shapes when both providers are available [13-15]. |
| `cagra` | `cuda_cuvs_cagra` with `ivf_pq` build | Direct cuVS CAGRA default large-shape builder | Direct RAPIDS cuVS CAGRA using the cuVS IVF-PQ graph builder. This can be fast, but high-dimensional compact matrices can request very large temporary workspace, so the auto build rule avoids it for COIL20-like shapes [3]. |
| `cagra` | `cuda_cuvs_cagra` with `iterative_cagra_search` build | Direct cuVS compact high-dimensional builder | Direct RAPIDS cuVS CAGRA using iterative CAGRA graph construction. This is the default direct-cuVS build for compact high-dimensional self-KNN (`n <= 5000`, `p >= 1024`, `k <= 100`) because it avoids the IVF-PQ workspace spike observed on COIL20 while keeping the method as CAGRA. |
| `cagra` | `cuda_cuvs_cagra` with `nn_descent` build | Direct cuVS experimental builder | Direct RAPIDS cuVS CAGRA using cuVS NN-descent graph construction. This inherits the cuVS NN-descent dynamic shared-memory launch issue on affected builds for high-dimensional FP32 inputs, so it should remain an explicit benchmark setting until the linked cuVS contains the upstream fix. |
| `grid` | `cpu_grid` | Exact 2D/3D spatial path | Best for simulated 2D/3D Euclidean/cosine/correlation data; unavailable by design outside 2D/3D. |
| `grid` | `cuda_grid` | CUDA 2D/3D spatial path | Correct for 2D/3D, but benchmark speed depends strongly on GPU model and transfer overhead. |

CUDA IVF auto tuning is selected by `nn_tune_cuda_ivf_cpp()`. The current
policy was derived from float32 CUDA IVF HPC recommendations for `k = 15, 30,
50, 100` and `target_recall = 0.90, 0.95, 0.99`: Euclidean rows come from
`faissR_IVF_TUNING_CUDA_euclidean_20260702_001853`, cosine rows come from
`faissR_IVF_TUNING_CUDA_cosine_20260702_192200`, correlation rows come from
`faissR_IVF_TUNING_CUDA_correlation_20260703_133655`, and raw inner-product
rows are seeded from the measured Euclidean table in
`benchmark_scripts/cuda_ivf_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`
until `run_hpc_ivf_tuning_cuda_inner_product.sh` replaces them. It uses `n`,
`p`, `k`, metric, and target recall rather than dataset names:

- Small compact high-dimensional matrices (`n < 5000`, `p >= 1024`) use few
  coarse lists and increase probes mostly for larger `k`.
- Small high-dimensional matrices (`n < 20000`, `p >= 128`) use more lists for
  USPS-like shapes; if the 0.99 benchmark did not reach 0.99, metadata records
  `tuning_benchmark_target_met = FALSE`.
- Medium high-dimensional matrices (`20000 <= n < 200000`, `p >= 256`) use
  MNIST/Fashion-like tiers, widening probes and sometimes `nlist` at
  `target_recall = 0.99`.
- Large low-dimensional matrices (`n >= 500000`, `p <= 64`) use flow-data
  tiers with modest probes for 0.90/0.95 and wider probes for 0.99.
- Large high-dimensional matrices (`n >= 500000`, `p >= 256`) use ImageNet-like
  tiers with much wider probes at 0.99, especially for `k = 100`.

Result metadata records `tuning_cuda_shape_group`, `tuning_k_bucket`,
`tuning_target_recall_code`, `tuning_benchmark_basis`, and the actual
`nlist`/`nprobe`. Euclidean and cosine rows loaded from the HPC tables do not
receive the older conservative metric probe increase. Rows marked
`best_available_partial_shape_datasets` record `tuning_benchmark_target_met =
FALSE`, so partial shape coverage is visible in result metadata. Raw
inner-product seeded rows also report `tuning_benchmark_target_met = FALSE`
until metric-specific CUDA benchmark results replace the seed.

For HPC tuning of `cuda_cuvs_ivfpq_fastscan`, sweep `nlist`, `nprobe`, and
byte-aligned 4-bit `pq_dim` together. Smaller `pq_dim` and smaller `nprobe`
usually improve speed but reduce recall; `nlist` shifts work between training,
coarse assignment, and list scanning. The CUDA wrapper exposes these as
`IVFPQ_FASTSCAN_NLIST_MULTS`, `IVFPQ_FASTSCAN_NPROBE_MULTS`, and
`IVFPQ_FASTSCAN_PQ_DIMS`; metric-specific wrappers such as
`run_hpc_ivfpq_fastscan_tuning_cuda_correlation.sh` and
`run_hpc_ivfpq_fastscan_tuning_cuda_inner_product.sh` run a single metric without
changing the base Slurm header. The R driver records the aligned `pq_dim` actually
tested in the candidate grid.

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

For `metric = "cosine"`, faissR normalizes rows to float32, runs CAGRA with
Euclidean distance, and converts returned normalized Euclidean distances back
to cosine distance. The compiled cosine defaults are marked validation-pending
because they are seeded from the measured Euclidean sweep until the fixed
cosine tuning job is rerun.
For `metric = "inner_product"`, faissR applies the maximum-inner-product-to-L2
extra-dimension transform, runs CAGRA on the transformed float32 data, and
converts distances back to shifted raw-inner-product distances. The compiled
raw-IP defaults are marked validation-pending because they are seeded from the
measured Euclidean sweep until the dedicated inner-product tuning job is rerun.

| Direct-cuVS CAGRA shape | Automatic build algorithm | Reason |
|---|---|---|
| self-KNN, `n <= 5000`, `p >= 1024`, `k <= 100` | `iterative_cagra_search` | Compact high-dimensional matrices can make the IVF-PQ builder request excessive temporary workspace; the iterative builder was validated on COIL20. |
| self-KNN with compact-tuning flag from the CAGRA parameter rule | `iterative_cagra_search` | Keeps small compact graph builds on the lower-workspace direct CAGRA path. |
| non-self queries or other shapes | `ivf_pq` | cuVS IVF-PQ graph construction remains the general direct-cuVS CAGRA builder. |

CUDA HNSW tuning is active for the public request
`backend = "cuda", method = "hnsw"`. RAPIDS cuVS HNSW builds a CUDA CAGRA seed
graph and converts it with `cuvsHnswFromCagraWithDataset`; search is performed
through the cuVS HNSW wrapper over host-compatible tensors. faissR records the
concrete implementation label, for example `cuda_cuvs_hnsw`, only as resolved
backend metadata. It also records
`cuda_hnsw_design = "cuvs_hnsw_from_cagra_cpu_hierarchy"` and
`cuda_hnsw_pure_gpu = FALSE`. Use CUDA `method = "cagra"` for a pure GPU
graph-search baseline, and use CUDA `method = "hnsw"` when the benchmark should
include the cuVS HNSW wrapper route.

CUDA HNSW `tuning = "auto"` is metric-aware for Euclidean, cosine, correlation,
and raw inner product. Euclidean uses the compiled table derived from
`faissR_HNSW_TUNING_CUDA_euclidean_20260701_083355`; cosine uses the separate
compiled table derived from `faissR_HNSW_TUNING_CUDA_cosine_20260702_123021`;
correlation uses
`benchmark_scripts/cuda_hnsw_correlation_shape_tuning_defaults_from_uploaded_results.csv`,
derived from `faissR_HNSW_TUNING_CUDA_correlation_20260703_070901`; raw inner
product currently uses
`benchmark_scripts/cuda_hnsw_inner_product_shape_tuning_defaults_from_seeded_euclidean_results.csv`,
seeded from the measured Euclidean CUDA HNSW table until
`run_hpc_hnsw_tuning_cuda_inner_product.sh` replaces it with measured IP rows.
Cosine is implemented as row-normalized float32 Euclidean graph search, and
correlation as centered row-normalized float32 Euclidean graph search; raw
inner product uses a maximum-inner-product-to-L2 transform. Returned distances
are converted back to the public metric. Each table is keyed by shape
group, `k = 15, 30, 50, 100`, and target recall `0.9`, `0.95`, or `0.99`; rows
that did not meet the requested target across all datasets in the shape group
expose `tuning_benchmark_target_met = FALSE`.

All-dataset HNSW tuning can be run from a package checkout with the exact
reference precompute job and the HNSW tuning launchers:

```bash
benchmark_scripts/run_hpc_precompute_exact_references_cpu12.sh
benchmark_scripts/run_hpc_hnsw_tuning_cpu12.sh
benchmark_scripts/run_hpc_hnsw_tuning_cuda.sh
```

The precompute job writes exact reference files into the dataset folders. The
HNSW tuning jobs then create a float32 manifest from the configured data
directory with `make_hpc_float32_manifest.R` and run
`benchmark_hnsw_tuning_from_reference.R`. They use explicit `backend = "cpu"`
or `backend = "cuda"`, `method = "hnsw"`, metric-specific wrappers such as
`run_hpc_hnsw_tuning_cuda_euclidean.sh` and
`run_hpc_hnsw_tuning_cuda_cosine.sh`, `k = 15, 30, 50, 100`,
`target_recall = 0.9, 0.95, 0.99`, and a
2000-second timeout per candidate. `backend = "auto"` is not used. Result rows
record the requested target, actual target, HNSW parameters, sampled recall,
speed, memory, output type, and backend metadata. If a dataset has no completed
`*_float32.RData` file, the manifest marks it as missing and the result table
records `status = "missing_dataset"` instead of misclassifying the algorithm.

Key outputs are:

- `float32_dataset_manifest.csv`: dataset paths, dimensions, labels flag, and
  readiness status.
- `hnsw_tuning_candidate_grid.csv`: CPU FAISS HNSW candidate parameters
  or CUDA cuVS HNSW candidate parameters evaluated for each `k`.
- `hnsw_tuning_results.csv`: raw speed, memory, sampled recall, and resolved
  backend metadata for every candidate.
- `hnsw_tuning_recommendations.csv`: fastest successful candidate per
  dataset/backend/`k`/target-recall threshold.
- `hnsw_tuning_shape_candidates.csv` and
  `hnsw_tuning_shape_recommendations.csv`: aggregated evidence for converting
  benchmark results into shape-aware C++ defaults.
- `hnsw_tuning_report.md`: compact Markdown summary for review.

If running directly from a package checkout, the launchers set
`FAISSR_SOURCE_DIR` automatically so child R processes use the source tree. On
systems using an already installed package, make sure the installed faissR
version includes the current `target_recall` HNSW API before submitting the
jobs.

The explicit `"nn_descent"` direct-cuVS CAGRA builder is available for
experiments, but it is not selected automatically because validation runs have
shown CUDA invalid-argument failures on compact high-dimensional inputs.

Approximate routes now attach deterministic no-pilot tuning metadata to
`attr(result, "approximation")`. IVF, IVFPQ/PQ, NSG, NN-descent, CAGRA, and
HNSW report `tuning_policy`, `tuning_rule`, and shape flags where relevant;
IVF also records `tuning_metric` and `tuning_metric_aware`, and IVFPQ/PQ
compression fields use `pq_tuning_*` names. These fields let benchmark tables
compare parameter tiers by dataset shape, `k`, and metric without running extra
tuning inside ordinary `nn()` calls. Deterministic auto-tuned methods report
`tuning_source = "cpp"` so downstream benchmark code can verify that the rule
came from the compiled policy layer rather than from an ad hoc R branch.

## Large High-Dimensional Probe

Additional large high-dimensional probes used a dataset object with `data` and
`labels` fields. One tested feature table had 1,281,167 rows and 1,024 columns
and occupied about 10 GB as double-precision R columns. Important probe
artifacts include:

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
`data.table`, coercing it to a second contiguous double matrix, and then
building backend-side float/index buffers. Full-reference benchmarks at this
scale should therefore use a memory-efficient matrix/float32 representation or
run on a host with enough RAM for the source data, converted matrix, and
backend-side buffers.


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
route is selected. The generated header `src/nn_hpc_tuning_tables.hpp` provides
Euclidean shape/k/target-recall defaults for CPU/CUDA IVF, CPU/CUDA IVFPQ,
CPU NN-descent, CUDA CAGRA, CPU/CUDA NSG, and CPU/CUDA Vamana, plus
validation-pending CUDA CAGRA cosine/correlation seed policies. Existing HNSW
tables remain compiled separately. Together these rules cover the approximate
methods where recall-target parameters exist; exact, Flat, and brute-force
routes instead use provider, batching, float32, and reuse settings from the
benchmark scripts. IVF `nlist`/`nprobe`, PQ bit-width and width selection, cuVS
CAGRA graph degree and build widths, CPU NN-descent candidate breadth, and
native NSG/Vamana candidate graph sizes all return `tuning_source = "cpp"`.
Pilot/cache tuning, where explicitly requested, remains opt-in and separate
from `tuning = "auto"`.

Policy summary:

- `backend = "cpu", method = "auto"`: exact CPU for small work; CPU grid for
  large Euclidean/cosine/correlation 2D/3D self-KNN; FAISS IVF for million-row
  Euclidean self-KNN
  where HNSW graph construction is too memory-heavy; FAISS HNSW for large
  high-dimensional CPU self-KNN, including cosine, correlation, and
  inner-product HNSW; FAISS Flat exact search for larger non-Euclidean query or
  exact workloads; native CPU NSG-style refinement for selected larger
  non-Euclidean self-KNN cases; and native CPU NN-descent for other large
  self-KNN cases.
  On the benchmark `k` grid, large high-dimensional CPU self-search uses
  graph-search routes across `k = 5, 10, 15, 50, and 100`; small `k` alone is
  not enough reason to run exact CPU on MNIST/FashionMNIST-scale workloads.
  Non-self non-Euclidean `k = 5` can still use exact FAISS Flat, and explicit
  CPU HNSW uses deterministic `n`, `p`, `k`, and `metric` tiers without
  running a pilot benchmark.
  Flat metric routes use FAISS.
- `backend = "cuda", method = "auto"`: CUDA grid for large 2D/3D
  Euclidean/cosine/correlation self-KNN. For Euclidean non-grid self-KNN, the
  selector chooses Flat/brute force or IVF-Flat from the compiled
  shape/k/target-recall policy. COIL20-like compact very-high-dimensional
  matrices, MNIST/Fashion-like image matrices, flow-like low-dimensional
  million-row matrices, and ImageNet-like large high-dimensional matrices use
  IVF when the selected target is supported by the tuning evidence. Tiny
  matrices, query searches, very small `k`, and below-target IVF cases stay on
  Flat/brute force. In cuVS-only runtimes, CUDA auto non-Euclidean capability
  rows are reported as shape-dependent instead of promising a route for every
  metric/method pair. The smaller non-grid non-Euclidean searches keep the existing
  exact FAISS GPU Flat path when available, while larger self-KNN routes can use
  transformed FAISS GPU/direct cuVS CAGRA or another validated graph-search path
  when available.

Historical examples from the earlier auto-policy run:

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
same dataset, the current `backend = "cuda", method = "auto"` policy selects
CUDA IVF-Flat for Euclidean self-KNN when the requested target-recall tier is
supported by the CUDA IVF tuning evidence, so this shape is a GPU-first case
rather than a reliable CPU-auto default.

## Known Issues From The Run

- Direct cuVS CAGRA can produce very low recall on high-dimensional raw MNIST.
  Default calls now use deterministic no-pilot parameters, so benchmark reports
  must include measured recall. Explicit pilot/cache tuning stops when it cannot
  meet the target recall instead of silently returning a poor result.
- cuVS HNSW is exposed as CUDA `method = "hnsw"`, but it is labelled as a
  CAGRA-to-HNSW wrapper in metadata. The C API converts CAGRA to an HNSW
  wrapper and searches host-compatible tensors, so benchmark reports should not
  compare it as if it were the same kind of pure GPU path as CAGRA.
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
