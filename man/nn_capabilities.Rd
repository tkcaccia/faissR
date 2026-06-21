\name{nn_capabilities}
\alias{nn_capabilities}
\title{Nearest-neighbour method capabilities}
\usage{
nn_capabilities()
}
\value{
A data frame with one row per public \code{method}, \code{backend}
(\code{"auto"}, \code{"cpu"}, or \code{"cuda"}), and \code{metric}
combination. Columns include \code{supported}, \code{exact},
\code{implementation}, and \code{notes}.
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

The capability table reports metric support only when the selected route
computes neighbours under that metric. FAISS Flat, IVF-Flat, IVFPQ, and HNSW
support cosine and correlation by using normalized inner-product searches.
CAGRA and NNDescent support cosine and correlation by using normalized
Euclidean graph search. FAISS NSG remains Euclidean-only because linked FAISS
graph builders can abort during non-Euclidean graph construction. Raw
inner-product search is not exposed for NSG, NNDescent, or CAGRA.
}
\examples{
caps <- nn_capabilities()
subset(caps, method == "flat" & supported)
}
