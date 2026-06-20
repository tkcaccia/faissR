# References

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Backends](backend-capabilities.md) |
**References**

References are listed in AACR journal style.

1. Johnson J, Douze M, Jegou H. Billion-scale similarity search with GPUs. IEEE Trans Big Data 2021;7:535-47.
2. Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazaré PE, et al. The FAISS library. arXiv 2024. Available from: https://github.com/facebookresearch/faiss.
3. RAPIDS Development Team. RAPIDS cuVS: GPU-accelerated vector search and clustering [software]. Available from: https://github.com/rapidsai/cuvs.
4. Dong W, Moses C, Li K. Efficient k-nearest neighbor graph construction for generic similarity measures. In: Proceedings of the 20th International Conference on World Wide Web; 2011. p. 577-86.
5. Malkov YA, Yashunin DA. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Trans Pattern Anal Mach Intell 2020;42:824-36.
6. Jégou H, Douze M, Schmid C. Product quantization for nearest neighbor search. IEEE Trans Pattern Anal Mach Intell 2011;33:117-28.
7. Lloyd SP. Least squares quantization in PCM. IEEE Trans Inf Theory 1982;28:129-37.
8. MacQueen J. Some methods for classification and analysis of multivariate observations. In: Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability; 1967. p. 281-97.
9. Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks. J Stat Mech 2008;2008:P10008.
10. Pons P, Latapy M. Computing communities in large networks using random walks. J Graph Algorithms Appl 2006;10:191-218.
11. Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 2019;9:5233.
12. RAPIDS Development Team. RAPIDS cuGraph: GPU graph analytics [software]. Available from: https://github.com/rapidsai/cugraph.
13. NVIDIA Developer Blog. Enhancing GPU-accelerated vector search in FAISS with NVIDIA cuVS. Available from: https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/.
14. Meta Engineering. Accelerating GPU indexes in FAISS with NVIDIA cuVS. Available from: https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/.
15. FAISS Project. GPU Faiss with cuVS. Available from: https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS.
16. FAISS Project. C++ API documentation. Available from: https://faiss.ai/cpp_api/classlist.html.

## Software Acknowledgements

`faissR` links to external FAISS and optional RAPIDS cuVS/cuGraph installations
rather than vendoring those libraries. HNSW, NN-descent, IVF, product
quantization, flat search, k-means, Louvain, Leiden, and random-walk clustering
are acknowledged as algorithmic and software foundations where the compiled
backend exposes them.

Native CPU graph clustering is faissR C++/OpenMP code inspired by the Louvain,
Leiden, walktrap/random-walk, and multicore graph-clustering literature. CUDA
Louvain and Leiden use RAPIDS libcugraph when available. FAISS GPU CAGRA and
FAISS GPU IVF routes follow the FAISS GPU/cuVS integration documented by FAISS,
NVIDIA, and Meta. Direct cuVS routes call RAPIDS cuVS C/C++ libraries. The
package does not use a Python/cuGraph bridge.
