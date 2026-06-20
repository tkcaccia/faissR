% Generated manually for the current development branch.
% Please keep in sync with R/graph.R until roxygen is regenerated.
\name{graph_cluster}
\alias{graph_cluster}
\alias{print.faissR_graph_cluster}
\title{Cluster a nearest-neighbour graph}
\usage{
graph_cluster(
  graph,
  method = c("random_walking", "louvain", "leiden"),
  backend = c("cpu", "cuda"),
  k = 50L,
  graph_backend = "auto",
  weight = c("auto", "snn", "adaptive", "distance", "binary"),
  mutual = FALSE,
  prune = 0,
  n_threads = NULL,
  n_runs = 1L,
  resolution = 1,
  objective_function = c("modularity", "CPM"),
  n_iterations = 2L,
  steps = 4L,
  seed = NULL,
  ...
)
}
\arguments{
\item{graph}{An \code{igraph} graph, a matrix/data frame accepted by \code{knn_graph()}, a KNN object returned by \code{nn()}, or an embedding object accepted by \code{knn_graph()}.}
\item{method}{One of \code{"random_walking"}, \code{"louvain"}, or \code{"leiden"}. \code{"random_walking"} uses igraph's walktrap/random-walk community method.}
\item{backend}{Community-detection backend. \code{"cpu"} uses \code{igraph}. \code{"cuda"} is reserved for RAPIDS cuGraph and currently fails clearly unless a future build links a CUDA graph backend.}
\item{k}{Number of neighbours when \code{graph} is not already an \code{igraph} graph.}
\item{graph_backend}{Backend passed to \code{knn_graph()} for neighbour search.}
\item{weight}{Weighting passed to \code{knn_graph()} when a graph must be built.}
\item{mutual, prune}{Graph-construction options passed to \code{knn_graph()}.}
\item{n_threads}{CPU threads. Used by \code{knn_graph()} and by repeated CPU runs when \code{n_runs > 1} on Unix-like platforms.}
\item{n_runs}{Number of repeated CPU community-detection runs. The result with the largest modularity is returned. Values greater than one can use several CPU cores.}
\item{resolution}{Modularity resolution for Louvain/Leiden.}
\item{objective_function}{Leiden objective, \code{"modularity"} or \code{"CPM"}.}
\item{n_iterations}{Leiden iteration count.}
\item{steps}{Walk length used by igraph's walktrap implementation.}
\item{seed}{Optional seed for reproducible repeated CPU runs.}
\item{...}{Reserved for future backend options.}
}
\description{
\code{graph_cluster()} runs community detection on an \code{igraph} graph or first builds one with \code{knn_graph()}. The nearest-neighbour graph can therefore use FAISS/cuVS backends, while CPU community detection uses \code{igraph} algorithms. The CUDA backend is reserved for RAPIDS cuGraph Louvain, Leiden, and random-walk implementations and fails clearly when faissR was not built with a cuGraph binding.
}
\value{
A \code{faissR_graph_cluster} list with membership, modularity, method, backend, graph, and the backend-specific community object.
}
\references{
Blondel VD, Guillaume JL, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment. 2008;2008(10):P10008.

Pons P, Latapy M. Computing communities in large networks using random walks. Journal of Graph Algorithms and Applications. 2006;10(2):191-218.

Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports. 2019;9:5233.

Sahu S. GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting. arXiv:2312.13936. Source repository: \url{https://github.com/puzzlef/leiden-communities-openmp}.

Sahu S. Heuristic-based Dynamic Leiden Algorithm for Efficient Tracking of Communities on Evolving Graphs. arXiv:2410.15451. Source repository: \url{https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic}.

Kapralov M, Lattanzi S, Nouri N, Tardos J. Efficient and Local Parallel Random Walks. arXiv:2112.00655.

CPU implementations are provided through the igraph R package. The CUDA route is designed for a future RAPIDS cuGraph binding; cuGraph provides GPU Louvain, Leiden, and random-walk algorithms in its CUDA/C++/Python stack. See \url{https://github.com/rapidsai/cugraph} and \url{https://docs.rapids.ai/api/cugraph/}.
}
\examples{
if (requireNamespace("igraph", quietly = TRUE)) {
  set.seed(1)
  x <- rbind(matrix(rnorm(80, -2), ncol = 4), matrix(rnorm(80, 2), ncol = 4))
  g <- knn_graph(x, k = 8, backend = "cpu", weight = "snn")
  cl <- graph_cluster(g, method = "louvain", backend = "cpu")
  table(cl$membership)
}
}
