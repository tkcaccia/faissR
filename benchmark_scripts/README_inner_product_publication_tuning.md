# Inner-Product Publication Tuning

This sweep recalibrates raw maximum-inner-product search (MIPS) for target
recall 0.90, 0.95, and 0.99. It must be run with the current faissR source or
an image containing the current package because older images did not expose
all MIPS-to-L2 CUDA routes.

## Why the previous sweep missed recall targets

Raw inner product is sensitive to vector-norm variation. Settings that work
for Euclidean or cosine search may inspect too few coarse lists or too little
of a graph. Earlier CUDA IVF grids also requested `nprobe > 2048`, which FAISS
GPU rejects. Older package builds did not support inner product for several
CUDA routes. The new grid:

- caps CUDA IVF `nlist` and `nprobe` at 2,048;
- includes full-probe IVF anchors, so a valid high-recall recovery point is
  measured rather than assumed;
- widens PQ code dimensions and CPU FastScan refinement;
- adds high-breadth HNSW, CAGRA, NN-descent, NSG, and Vamana candidates;
- uses float32 datasets but double output for stable quality evaluation;
- ignores previous timeout skip lists, because those rows came from a
  different implementation/grid;
- evaluates 1,024 exact-reference queries for calibration.

## Files to copy

Copy the complete `benchmark_scripts` directory. In particular, the run needs
the common R driver, exact-reference driver, method wrappers, the new CUDA
NN-descent inner-product wrapper, and the submission driver in this directory.

## Submit

From `/scratch/firenze/NN`:

```bash
bash benchmark_scripts/submit_hpc_inner_product_publication_tuning.sh
```

The submission driver first creates the exact CPU inner-product references and
then submits every CPU and CUDA method with an `afterok` dependency. Do not run
it with `sbatch`; it calls `sbatch` itself.

For publication, use these rows only for calibration. The JMLR benchmark uses
different validation seeds and repeated timings to test the final
`tuning = "auto"` policy without selecting parameters on the validation rows.
