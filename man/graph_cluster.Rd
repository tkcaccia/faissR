% Generated manually for the current development branch.
% Please keep in sync with R/graph.R until roxygen is regenerated.
\name{graph_cluster}
\alias{graph_cluster}
\alias{print.faissR_graph_cluster}
\title{Cluster a nearest-neighbour graph without igraph}
\usage{
graph_cluster(
  graph,
  method = c("random_walking", "louvain", "leiden"),
  backend = c("auto", "cpu", "cuda"),
  k = 50L,
  graph_backend = "auto",
  graph_method = c("auto", "exact", "flat", "bruteforce", "grid", "hnsw", "ivf",
    "ivfpq", "vamana", "nsg", "nndescent", "cagra"),
  metric = c("euclidean", "cosine", "correlation", "inner_product"),
  tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
  cagra_implementation = NULL,
  weight = c("auto", "snn", "adaptive", "distance", "binary"),
  mutual = FALSE,
  prune = 0,
  n_threads = NULL,
  n_runs = 1L,
  resolution = 1,
  n_clusters = NULL,
  objective_function = c("modularity", "CPM"),
  n_iterations = 10L,
  steps = 4L,
  seed = NULL,
  ...
)
}
\arguments{
\item{graph}{A numeric matrix/data frame, a KNN object returned by \code{nn()}, or an embedding object with a matrix \code{layout}.}
\item{method}{One of \code{"random_walking"}, \code{"louvain"}, or \code{"leiden"}. \code{"random_walking"} uses a native random-walk label-propagation pass inspired by walktrap/random-walk clustering. \code{"louvain"} uses native modularity local moving. \code{"leiden"} adds a native refinement pass that splits disconnected communities after local moving. See References.}
\item{backend}{Community-detection backend. \code{"auto"} uses CUDA when
libcugraph is available for Louvain/Leiden and CPU otherwise; auto keeps
\code{"random_walking"} on CPU. \code{"cpu"} uses native faissR C++/OpenMP code.
\code{"cuda"} uses native RAPIDS libcugraph for Louvain and Leiden when
libcugraph was detected at build time; random-walking is CPU-only.}
\item{k}{Number of neighbours when \code{graph} is not already a KNN object.}
\item{graph_backend}{Backend passed to \code{nn_without_self()} for neighbour search.}
\item{graph_method}{Nearest-neighbour method passed to \code{nn_without_self()} when \code{graph} is a matrix or embedding.}
\item{metric}{Distance metric passed to \code{nn_without_self()} when \code{graph} is a matrix or embedding. Aliases such as \code{"l2"}, \code{"cor"}/\code{"pearson"}, and \code{"ip"} are accepted and stored as canonical metric labels. Correlation is centered cosine similarity, not raw inner product. Inner-product graph construction ranks neighbours by larger raw dot product while reusing faissR's shifted smaller-is-better distance convention from \code{nn()}.}
\item{tuning}{Tuning policy passed to \code{nn_without_self()} when \code{graph} is a matrix or embedding.}
\item{cagra_implementation}{CUDA CAGRA provider passed to \code{nn_without_self()} when graph KNN is computed here. \code{NULL} uses the global option; \code{"auto"}, \code{"faiss_gpu"}, or \code{"cuvs"} select the CAGRA provider for CUDA \code{graph_method = "cagra"} and CUDA-auto CAGRA routes.}
\item{weight}{KNN graph weighting used when a graph must be built.}
\item{mutual}{If \code{TRUE}, keep only reciprocal nearest-neighbour edges when a graph must be built from data or a KNN object.}
\item{prune}{Drop graph edges with weight less than or equal to this value when a graph must be built from data or a KNN object.}
\item{n_threads}{CPU threads. Used by KNN construction and native CPU clustering.}
\item{n_runs}{Number of independent native clustering runs. The result with the largest modularity is returned. Values greater than one can use several CPU cores.}
\item{resolution}{Modularity resolution for Louvain/Leiden-style scoring.
Use a positive number for direct resolution control. When \code{n_clusters}
is supplied, \code{resolution = NULL} lets faissR seed the bounded
deterministic grid from the no-pilot graph-shape heuristic instead of a user
resolution.}
\item{n_clusters}{Optional target number of communities for Louvain/Leiden.
If supplied, faissR evaluates a bounded deterministic resolution grid on the
already-built graph and keeps the result whose community count is closest to
the target. The grid is centered from the requested \code{resolution}, or
from an automatic seed when \code{resolution = NULL}, and when graph size is
known uses a no-pilot shape heuristic based on
\code{n_clusters / sqrt(n_vertices)}. The search width is shape-aware: small
graphs use a wider deterministic grid, while large graphs use fewer candidates
to reduce repeated Louvain/Leiden passes. This is a convenience target, not a
hard guarantee. If \code{n_clusters} is supplied and \code{method} is omitted,
faissR uses \code{"louvain"} as the target-count clustering method. The target
must be a positive integer and cannot exceed the number of graph vertices.}
\item{objective_function}{Reserved for Leiden-compatible APIs; currently recorded in parameters.}
\item{n_iterations}{Native clustering iteration count.}
\item{steps}{Random-walk propagation depth.}
\item{seed}{Optional seed for reproducible repeated CPU runs.}
\item{...}{Reserved for future backend options.}
}
\description{
\code{graph_cluster()} runs native faissR community detection on a KNN graph. The graph can be supplied as a precomputed \code{nn()} result or built from a matrix/data frame using any faissR nearest-neighbour backend, including FAISS and cuVS KNN backends. The CPU clustering backend is implemented in C++ and uses OpenMP when available; it does not depend on igraph. CUDA graph clustering uses native RAPIDS libcugraph for Louvain and Leiden when faissR was built against libcugraph. Random-walking currently remains CPU-only. CUDA graph clustering never calls Python and never silently falls back to CPU.
}
\details{
\code{n_clusters} is supplied directly to \code{graph_cluster()}. When a target
count is available for Louvain or Leiden, faissR builds or reuses the graph once,
evaluates a bounded deterministic resolution grid centered by the requested
\code{resolution} and graph shape. The grid width is shape-aware so large graph
target searches use fewer deterministic candidates than small graph searches.
faissR records the selected resolution, center, and search table in the returned
object. If \code{n_clusters} is supplied and
\code{method} is omitted, faissR uses \code{"louvain"} as the target-count
clustering method. Passing \code{n_clusters} to explicit
\code{method = "random_walking"} still errors. The target must be a positive
integer and cannot exceed the number of graph vertices.

Method descriptions:
\itemize{
  \item \code{"random_walking"}: native CPU random-walk label-propagation/community method inspired by walktrap-style random-walk clustering and local parallel random-walk literature. CUDA random-walking is not enabled yet.
  \item \code{"louvain"}: native CPU Louvain modularity local-moving implementation, with optional RAPIDS libcugraph CUDA execution when faissR is built with cuGraph.
  \item \code{"leiden"}: native CPU Leiden-style local moving plus refinement to split disconnected communities, with optional RAPIDS libcugraph CUDA execution when available.
}
}
\value{
A \code{faissR_graph_cluster} list with membership, modularity, method,
backend, graph edge list, parameters, and source acknowledgements.
\code{backend} records the clustering implementation that actually ran, while
\code{parameters$requested_backend} and \code{parameters$resolved_backend}
record the public backend request and the device policy after resolving
\code{"auto"}. When \code{graph_cluster()} builds a graph internally or
receives a \code{faissR_graph},
\code{parameters$graph_backend}, \code{parameters$graph_requested_backend},
and \code{parameters$graph_resolved_backend} record the concrete KNN
implementation, public graph backend request, and resolved KNN backend.
\code{parameters$n_vertices} and \code{parameters$n_edges}
record the clustered graph size for benchmark summaries.
\code{parameters$nn_metric_transform} and
\code{parameters$nn_distance_transform} preserve normalized metric conversion
metadata from the KNN route that built the graph.
\code{parameters$nn_approximation}, \code{parameters$nn_faiss},
\code{parameters$nn_cuvs}, \code{parameters$nn_spatial_index}, and
\code{parameters$nn_auto_selection}
preserve compact KNN route metadata when the graph is built internally. When a target
community count is used, \code{target_n_clusters},
\code{selected_resolution}, \code{target_gap}, \code{resolution_selection},
and \code{resolution_search} record the requested target, selected resolution,
final community-count gap, deterministic selection rule, and full resolution
search table. \code{parameters$resolution_source} is \code{"default"},
\code{"user"}, or \code{"target_auto"} depending on how the resolution-grid
seed was chosen.
}
\references{
Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment. 2008;2008(10):P10008.

Pons P, Latapy M. Computing communities in large networks using random walks. Journal of Graph Algorithms and Applications. 2006;10(2):191-218.

Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports. 2019;9:5233.

Sahu S. GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting. arXiv:2312.13936. Source repository: \url{https://github.com/puzzlef/leiden-communities-openmp}.

Sahu S. Heuristic-based Dynamic Leiden Algorithm for Efficient Tracking of Communities on Evolving Graphs. arXiv:2410.15451. Source repository: \url{https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic}.

Kapralov M, Lattanzi S, Nouri N, Tardos J. Efficient and Local Parallel Random Walks. arXiv:2112.00655.

CPU implementations are native faissR C++/OpenMP code. CUDA Louvain and Leiden use RAPIDS libcugraph when available at build time; cuGraph also provides GPU random-walk primitives that may support a future native random-walking adapter. See \url{https://github.com/rapidsai/cugraph} and \url{https://docs.rapids.ai/api/cugraph/}.
}
\examples{
set.seed(1)
x <- rbind(matrix(rnorm(80, -2), ncol = 4), matrix(rnorm(80, 2), ncol = 4))
cl <- graph_cluster(x, method = "louvain", backend = "cpu", k = 8, graph_backend = "cpu")
table(cl$membership)
}
