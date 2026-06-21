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
  weight = c("auto", "snn", "adaptive", "distance", "binary"),
  mutual = FALSE,
  prune = 0,
  n_clusters = NULL,
  n_threads = NULL
)
}
\arguments{
\item{data}{Numeric matrix/data frame, a KNN object returned by \code{nn()}, or a \code{fastEmbedR_embedding} object.}
\item{knn}{Optional precomputed KNN object returned by \code{nn()}. If supplied, \code{data} is ignored for neighbour search.}
\item{k}{Number of non-self neighbours used in the graph.}
\item{backend}{Device backend passed to \code{nn_without_self()} when \code{knn} is not supplied: \code{"auto"}, \code{"cpu"}, or \code{"cuda"}.}
\item{weight}{Graph weighting. \code{"auto"} uses SNN/Jaccard weights for input space and distance weights for embedding space. \code{"snn"} builds shared-nearest-neighbour Jaccard weights. \code{"adaptive"} uses a local radius kernel. \code{"distance"} uses \code{1 / (1 + distance)}. \code{"binary"} gives every edge weight 1.}
\item{mutual}{If \code{TRUE}, keep only reciprocal nearest-neighbour edges.}
\item{prune}{Drop edges with weight less than or equal to this value.}
\item{n_clusters}{Optional target number of communities to store with the
graph. When this graph is later passed to \code{graph_cluster()} with
\code{method = "louvain"} or \code{"leiden"} and no explicit \code{n_clusters},
the stored target is used instead of relying only on \code{resolution}.}
\item{n_threads}{CPU threads passed to \code{nn()} when KNN is computed here.}
}
\value{
A native \code{faissR_graph} edge-list object.
}
\description{
\code{knn_graph()} turns a data matrix, a precomputed \code{nn()} result, or a \code{fastEmbedR_embedding} object into a weighted nearest-neighbour graph. It returns a native \code{faissR_graph} edge-list object and does not require igraph.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
g <- knn_graph(x, k = 15, backend = "cpu", n_clusters = 3)
cl <- graph_cluster(g, method = "louvain", backend = "cpu")
table(cl$membership)
}
