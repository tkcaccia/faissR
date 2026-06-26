# References

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
**References**

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
17. Sahu S. GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting. arXiv:2312.13936.
18. Sahu S. Heuristic-based Dynamic Leiden Algorithm for Efficient Tracking of Communities on Evolving Graphs. arXiv:2410.15451.
19. Kapralov M, Lattanzi S, Nouri N, Tardos J. Efficient and Local Parallel Random Walks. arXiv:2112.00655.
20. Yianilos PN. Data structures and algorithms for nearest neighbor search in general metric spaces. In: Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms; 1993. p. 311-21.
21. Fu C, Xiang C, Wang C, Cai D. Fast approximate nearest neighbor search with the navigating spreading-out graph. Proc VLDB Endow 2019;12:461-74.
22. RAPIDS Development Team. cuVS HNSW C API documentation. Available from: https://docs.rapids.ai/api/cuvs/stable/c_api/neighbors_hnsw_c/.
23. Kim J. CUHNSW: CUDA implementation of Hierarchical Navigable Small World Graph algorithm [software, Apache-2.0]. Available from: https://github.com/js1010/cuhnsw.
24. Subramanya SJ, Devvrit, Kadekodi R, Krishaswamy R, Simhadri HV. DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node. NeurIPS 2019.
25. Groh F, Ruppert L, Wieschollek P, Lensch HPA. GGNN: Graph-based GPU Nearest Neighbor Search. IEEE Trans Big Data 2023.
26. Zhao W, Tan S, Li P. SONG: Approximate Nearest Neighbor Search on GPU. ICDE 2020.
27. Venkatasubba K, Khan S, Singh S, Simhadri HV, Vedurada J. BANG: Billion-Scale Approximate Nearest Neighbour Search Using a Single GPU. IEEE Trans Big Data 2025.
28. Gui Y, Li Z, Li Q, et al. PilotANN: Memory-Bounded GPU Acceleration for Vector Search. arXiv:2503.21206.
29. Fu C, Xiang C, Wang C, Cai D. NSG: Navigating Spreading-out Graph for Approximate Nearest Neighbor Search [software, MIT]. Available from: https://github.com/ZJULearning/nsg.
30. FAISS Project. Installing Faiss. Available from: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md.
31. NVIDIA. CUDA Installation Guide for Linux. Available from: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.
32. NVIDIA. CUDA Installation Guide for Microsoft Windows. Available from: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/.
33. RAPIDS Development Team. RAPIDS Installation Guide. Available from: https://docs.rapids.ai/install/.
34. FAISS Project. Fast accumulation of PQ and AQ codes (FastScan). Available from: https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan).

## Software Acknowledgements

`faissR` links to external FAISS and optional RAPIDS cuVS/cuGraph installations
rather than vendoring those libraries [1-3,12-16]. HNSW, NN-descent, IVF,
product quantization, flat search, k-means, Louvain, Leiden, and random-walk
clustering are acknowledged as algorithmic and software foundations where the
compiled backend exposes them [1-11,16,20-28].

Native CPU graph clustering is faissR C++/OpenMP code inspired by the Louvain,
Leiden, walktrap/random-walk, and multicore graph-clustering literature
[9-11,17-19]. CUDA Louvain and Leiden use RAPIDS libcugraph when available [12].
FAISS GPU CAGRA and FAISS GPU IVF routes follow the FAISS GPU/cuVS integration
documented by FAISS, NVIDIA, and Meta [13-15]. Direct cuVS routes call RAPIDS
cuVS C/C++ libraries [3]. RAPIDS cuVS HNSW is cited because its C API documents
the CAGRA-to-hnswlib wrapper behavior that faissR intentionally does not expose
as CUDA HNSW [22]. CUHNSW is acknowledged as related Apache-2.0 CUDA HNSW prior
software, but no CUHNSW source code is vendored or copied into faissR [23]. The
IVFPQ FastScan `method = "ivfpq_fastscan"` route uses
FAISS FastScan on CPU and direct RAPIDS cuVS 4-bit IVF-PQ on CUDA [3,6,34]. The native Vamana route is inspired by DiskANN/Vamana
robust-pruned graph construction [24] and uses faissR-owned candidate
refinement code; GGNN, SONG, BANG, and PilotANN are acknowledged as related GPU
ANN systems and design references, but their source code is not vendored or
copied into faissR [25-28]. The native CUDA NSG-style route is inspired by the
NSG paper and official MIT-licensed NSG software, but no ZJULearning/nsg source
code is vendored or copied into faissR [21,29]. The package does not use a
Python/cuGraph bridge.
