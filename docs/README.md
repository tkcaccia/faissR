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

`faissR` contains vector-search, k-means, and kNN-model tools for direct R
workflows and downstream embedding pipelines. FAISS is mandatory
[1-2,16]; CUDA and RAPIDS cuVS are optional compiled
backends [3,13-15].

## Pages

- [Installation](installation.md): CRAN/source-build expectations, mandatory
  FAISS, and optional CUDA/RAPIDS libraries.
- [Implementation](implementation.md): dense KNN, candidate KNN, k-means, kNN
  prediction, and backend rules.
- [Examples](examples.md): compact examples based on iris.
- [Benchmarks](benchmarks.md): benchmark design for speed, recall, memory, and
  downstream embedding checks.
- [Autotuning](autotuning.md): empirical backend roles, target-recall tuning
  workflow, defaults, and guardrails.
- [API](usage-api.md): function and argument reference.
- [NN Methods](nn-methods.md): detailed descriptions and literature for every
  `nn()` method, including the `nn_capabilities()` method/backend/metric
  support table.
- [Backends](backend-capabilities.md): supported CPU and CUDA paths.
- [cuVS NN-descent issue report](cuvs-nndescent-shared-memory-issue.md):
  copy-ready upstream report for the high-dimensional FP32 dynamic
  shared-memory launch issue observed in cuVS NN-descent.
- [References](references.md): references, software acknowledgements, and
  implementation inspirations.
