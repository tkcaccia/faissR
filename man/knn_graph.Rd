% Generated manually for the current development branch.
% Please keep in sync with R/graph.R until roxygen is regenerated.
\name{knn_graph}
\alias{knn_graph}
\title{Build a native nearest-neighbour graph}
\usage{
knn_graph(
  data,
  knn = NULL,
  k = 50L,
  backend = c("auto", "cpu", "cuda"),
  nn_method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf", "ivfpq",
    "vamana", "nsg", "nndescent", "ivfpq_fastscan", "cagra"),
  method = NULL,
  metric = c("euclidean", "cosine", "correlation", "inner_product"),
  tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
  target_recall = 0.99,
  cagra_implementation = NULL,
  cagra_build_algo = NULL,
  weight = c("auto", "snn", "adaptive", "distance", "binary"),
  mutual = FALSE,
  prune = 0,
  n_threads = NULL
)
}
\arguments{
\item{data}{Numeric matrix/data frame, a KNN object returned by \code{nn()}, or an embedding object with a matrix \code{layout}.}
\item{knn}{Optional precomputed KNN object returned by \code{nn()}. If supplied, \code{data} is ignored for neighbour search.}
\item{k}{Number of non-self neighbours used in the graph.}
\item{backend}{Device backend passed to \code{nn(..., exclude_self = TRUE)} when \code{knn} is not supplied: \code{"auto"}, \code{"cpu"}, or \code{"cuda"}.}
\item{nn_method}{Nearest-neighbour method passed to \code{nn(..., exclude_self = TRUE)} when \code{knn} is not supplied. The default \code{"auto"} uses the shape-aware selector for the chosen backend.}
\item{method}{Alias for \code{nn_method}, matching the public method argument used by \code{nn()} and \code{knn()}. If both \code{method} and \code{nn_method} are supplied they must resolve to the same public nearest-neighbour method label.}
\item{metric}{Distance metric passed to \code{nn(..., exclude_self = TRUE)} when \code{knn} is not supplied: \code{"euclidean"}, \code{"cosine"}, \code{"correlation"}, or \code{"inner_product"}. Legacy metric aliases such as \code{"l2"}, \code{"cor"}, \code{"pearson"}, and \code{"ip"} are rejected. Correlation is centered cosine similarity, not raw inner product. Inner-product graph construction ranks neighbours by larger raw dot product while reusing faissR's shifted smaller-is-better distance convention from \code{nn()}.}
\item{tuning}{Tuning policy passed to \code{nn(..., exclude_self = TRUE)} when \code{knn} is not supplied.}
\item{target_recall}{Speed/recall tier passed to \code{nn(..., exclude_self = TRUE)} when KNN is computed here. Use \code{0.9}, \code{0.95}, or \code{0.99}; CUDA \code{method = "auto"} uses it for Flat-vs-IVF selection, CUDA IVF uses it for probing defaults, and HNSW uses it for graph-search tiers.}
\item{cagra_implementation}{CUDA CAGRA provider passed to \code{nn(..., exclude_self = TRUE)} when KNN is computed here. \code{NULL} uses the global option; \code{"auto"} uses the same deterministic shape-aware provider rule as \code{nn()}, while \code{"faiss_gpu"} or \code{"cuvs"} force one provider.}
\item{cagra_build_algo}{Direct RAPIDS cuVS CAGRA graph-build algorithm passed to \code{nn(..., exclude_self = TRUE)} when KNN is computed here. \code{NULL} uses the global \code{faissR.cuvs_cagra_build_algo} option. This setting applies to direct cuVS CAGRA and accepts \code{"auto"}, \code{"ivf_pq"}, \code{"nn_descent"}, or \code{"iterative_cagra_search"}.}
\item{weight}{Graph weighting. \code{"auto"} uses SNN/Jaccard weights for input space and distance weights for embedding space. \code{"snn"} builds shared-nearest-neighbour Jaccard weights. \code{"adaptive"} uses a local radius kernel. \code{"distance"} uses \code{1 / (1 + distance)}. \code{"binary"} gives every edge weight 1.}
\item{mutual}{If \code{TRUE}, keep only reciprocal nearest-neighbour edges.}
\item{prune}{Drop edges with weight less than or equal to this value.}
\item{n_threads}{CPU threads passed to \code{nn()} when KNN is computed here.}
}
\value{
A native \code{faissR_graph} edge-list object. The \code{faissR_graph}
attribute stores graph-construction metadata: graph size, weighting,
nearest-neighbour method, metric, tuning policy, requested/resolved KNN
backends, and compact-relevant KNN result metadata such as \code{nn_approximation}, \code{nn_faiss},
\code{nn_cuvs}, \code{nn_spatial_index}, \code{nn_auto_selection},
\code{nn_metric_transform}, and \code{nn_distance_transform}. For
precomputed KNN input, \code{nn_backend} prefers the KNN object's resolved
backend when available, so benchmark metadata records concrete FAISS/cuVS
routes rather than only the public requested backend.
}
\description{
\code{knn_graph()} turns a data matrix, a precomputed \code{nn()} result, or an embedding object with a matrix \code{layout} into a weighted nearest-neighbour graph. It returns a native \code{faissR_graph} edge-list object and does not require igraph.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
g <- knn_graph(x, k = 15, backend = "cpu")
cl <- graph_cluster(g, method = "louvain", backend = "cpu", n_clusters = 3)
table(cl$membership)
}
