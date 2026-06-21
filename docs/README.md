# faissR Documentation

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[Autotuning](autotuning.md) |
[API](usage-api.md) |
[NN Methods](nn-methods.md) |
[Backends](backend-capabilities.md) |
[References](references.md)

`faissR` contains the vector-search, graph, k-means, and kNN-model layer used
by direct R workflows and by the `fastEmbedR` ecosystem. FAISS is mandatory
[1-2,16]; CUDA, RAPIDS cuVS, and RAPIDS libcugraph are optional compiled
backends [3,12-15].

## Pages

- [Installation](installation.md): CRAN/source-build expectations, mandatory
  FAISS, and optional CUDA/RAPIDS libraries.
- [Implementation](implementation.md): dense KNN, sparse/candidate KNN, native
  graphs, graph clustering, k-means, kNN prediction, and backend rules.
- [Examples](examples.md): compact examples based on iris.
- [Benchmarks](benchmarks.md): benchmark design for speed, recall, memory, and
  downstream embedding checks.
- [Autotuning](autotuning.md): empirical backend roles, defaults, and guardrails.
- [API](usage-api.md): function and argument reference.
- [NN Methods](nn-methods.md): detailed descriptions and literature for every
  `nn()` method.
- [Backends](backend-capabilities.md): supported CPU and CUDA paths.
- [References](references.md): references, software acknowledgements, and
  implementation inspirations.
