\name{nn_capabilities}
\alias{nn_capabilities}
\title{Nearest-neighbour method capabilities}
\usage{
nn_capabilities(runtime = FALSE)
}
\arguments{
\item{runtime}{Logical; when \code{FALSE} (the default), report support by
design without checking the current compiled/runtime libraries. When
\code{TRUE}, add \code{resolved_backend}, \code{runtime_available},
\code{runtime_reason}, and \code{runtime_notes} columns for the current
installation. \code{runtime_reason} uses stable labels such as
\code{"available"}, \code{"unsupported_combination"}, \code{"missing_faiss"},
\code{"missing_faiss_gpu"}, \code{"missing_cuda"}, and \code{"missing_cuvs"}
for benchmark preflight tables.}
}
\value{
A data frame with one row per public \code{method}, \code{backend}
(\code{"auto"}, \code{"cpu"}, or \code{"cuda"}), and \code{metric}
combination. Columns include \code{supported}, \code{exact},
\code{implementation}, and \code{notes}. If \code{runtime = TRUE}, runtime
availability columns are appended.
}
\description{
\code{nn_capabilities()} returns the public method/backend/metric support table
used by the nearest-neighbour API. It separates combinations that are supported
by design from combinations that should be treated as expected skips in
benchmarks.
}
\details{
faissR exposes four public nearest-neighbour metrics: \code{"euclidean"},
\code{"cosine"}, \code{"correlation"}, and \code{"inner_product"}. Correlation
is centered cosine similarity, whereas inner product is the raw dot product.
For \code{metric = "inner_product"}, neighbours are ranked by larger raw dot
product, but returned \code{distances} keep faissR's smaller-is-better
convention by shifting each query row so the best returned dot product has
distance \code{0}.

The capability table reports metric support only when the selected route
computes neighbours under that metric. FAISS Flat, IVF-Flat, IVFPQ, and HNSW
support cosine and correlation by using normalized inner-product searches.
CAGRA and NNDescent support cosine and correlation by using normalized
Euclidean graph search. Native CPU NNDescent also supports raw inner-product
self-KNN through faissR's shifted dot-product distance convention. FAISS NSG
remains Euclidean-only because linked FAISS graph builders can abort during
non-Euclidean graph construction. Raw inner-product search is not exposed for
NSG, CUDA NNDescent, or CAGRA.

Use \code{runtime = TRUE} for benchmark preflight checks that must distinguish
a valid method/metric combination from a route that is unavailable in the
current build, such as FAISS GPU rows on a CPU-only installation.
}
\examples{
caps <- nn_capabilities()
subset(caps, method == "flat" & supported)
runtime_caps <- nn_capabilities(runtime = TRUE)
}
