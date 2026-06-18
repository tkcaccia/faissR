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

## Software Acknowledgements

`faissR` links to external FAISS and optional RAPIDS cuVS installations rather
than vendoring those libraries. HNSW, NN-descent, IVF, product quantization,
flat search, and k-means implementations are acknowledged as algorithmic and
software foundations where the compiled backend exposes them.
