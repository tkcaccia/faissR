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
  nn_method = c("auto", "exact", "flat", "bruteforce", "grid", "vptree", "sparse", "hnsw",
    "ivf", "ivfpq", "nsg", "nndescent", "cagra"),
  method = NULL,
  metric = c("euclidean", "cosine", "correlation", "inner_product"),
  tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
  weight = c("auto", "snn", "adaptive", "distance", "binary"),
  mutual = FALSE,
  prune = 0,
  n_clusters = NULL,
  n_threads = NULL
)
}
\arguments{
\item{data}{Numeric matrix/data frame, a KNN object returned by \code{nn()}, or an embedding object with a matrix \code{layout}.}
\item{knn}{Optional precomputed KNN object returned by \code{nn()}. If supplied, \code{data} is ignored for neighbour search.}
\item{k}{Number of non-self neighbours used in the graph.}
\item{backend}{Device backend passed to \code{nn_without_self()} when \code{knn} is not supplied: \code{"auto"}, \code{"cpu"}, or \code{"cuda"}.}
\item{nn_method}{Nearest-neighbour method passed to \code{nn_without_self()} when \code{knn} is not supplied. The default \code{"auto"} uses the shape-aware selector for the chosen backend.}
\item{method}{Alias for \code{nn_method}, matching the public method argument used by \code{nn()} and \code{knn()}. If both \code{method} and \code{nn_method} are supplied they must resolve to the same public nearest-neighbour method label.}
\item{metric}{Distance metric passed to \code{nn_without_self()} when \code{knn} is not supplied. Aliases such as \code{"l2"}, \code{"cor"}/\code{"pearson"}, and \code{"ip"} are accepted and stored as canonical metric labels.}
\item{tuning}{Tuning policy passed to \code{nn_without_self()} when \code{knn} is not supplied.}
\item{weight}{Graph weighting. \code{"auto"} uses SNN/Jaccard weights for input space and distance weights for embedding space. \code{"snn"} builds shared-nearest-neighbour Jaccard weights. \code{"adaptive"} uses a local radius kernel. \code{"distance"} uses \code{1 / (1 + distance)}. \code{"binary"} gives every edge weight 1.}
\item{mutual}{If \code{TRUE}, keep only reciprocal nearest-neighbour edges.}
\item{prune}{Drop edges with weight less than or equal to this value.}
\item{n_clusters}{Optional target number of communities to store with the
graph. When this graph is later passed to \code{graph_cluster()} with
\code{method = "louvain"} or \code{"leiden"} and no explicit \code{n_clusters},
the stored target is used instead of relying only on \code{resolution}. Stored
targets are ignored by \code{method = "random_walking"}; explicitly passing
\code{n_clusters} to random-walking still errors. The target must be a
positive integer and cannot exceed the number of graph vertices.}
\item{n_threads}{CPU threads passed to \code{nn()} when KNN is computed here.}
}
\value{
A native \code{faissR_graph} edge-list object. The \code{faissR_graph}
attribute stores graph-construction metadata: graph size, weighting,
nearest-neighbour method, metric, tuning policy, optional
\code{target_n_clusters}, and requested/resolved KNN backends. For precomputed
KNN input, \code{nn_backend} prefers the KNN object's resolved backend when
available, so benchmark metadata records concrete FAISS/cuVS routes rather
than only the public requested backend.
}
\description{
\code{knn_graph()} turns a data matrix, a precomputed \code{nn()} result, or an embedding object with a matrix \code{layout} into a weighted nearest-neighbour graph. It returns a native \code{faissR_graph} edge-list object and does not require igraph.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
g <- knn_graph(x, k = 15, backend = "cpu", n_clusters = 3)
cl <- graph_cluster(g, method = "louvain", backend = "cpu")
table(cl$membership)
}
