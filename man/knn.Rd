% Generated manually for the current development branch.
% Please keep in sync with R/knn_model.R until roxygen is regenerated.
\name{knn}
\alias{knn}
\title{Fit or apply a k-nearest-neighbour classifier or regressor}
\usage{
knn(
  Xtrain,
  Ytrain,
  Xtest = NULL,
  backend = c("auto", "cpu", "cuda"),
  method = c("auto", "exact", "flat", "bruteforce", "grid", "vptree", "sparse", "HNSW",
    "IVF", "IVFPQ", "NSG", "NNDescent", "CAGRA"),
  metric = c("euclidean", "cosine", "correlation", "inner_product"),
  tuning = c("auto", "cache", "pilot", "fixed", "off", "none"),
  task = c("auto", "classification", "regression"),
  k = 15L,
  n_threads = NULL,
  vote = c("majority", "weighted"),
  type = c("response", "prob"),
  ...
)
}
\arguments{
\item{Xtrain}{Numeric training matrix with observations in rows.}
\item{Ytrain}{Training labels or numeric response.}
\item{Xtest}{Optional numeric query matrix. If supplied, \code{knn()} returns predictions for \code{Xtest}; otherwise it returns a fitted model.}
\item{backend}{Device backend passed to \code{nn()}: \code{"auto"}, \code{"cpu"}, or
\code{"cuda"}. \code{"auto"} follows \code{nn()} backend/method/metric
resolution, using CUDA only for validated CUDA combinations and CPU otherwise.}
\item{method}{Nearest-neighbour algorithm selector passed to \code{nn()}. \code{"auto"} chooses the most appropriate method for the selected backend. See \code{?nn} for descriptions and references for \code{"exact"}, \code{"flat"}, \code{"bruteforce"}, \code{"grid"}, \code{"vptree"}, \code{"sparse"}, \code{"HNSW"}, \code{"IVF"}, \code{"IVFPQ"}, \code{"NSG"}, \code{"NNDescent"}, and \code{"CAGRA"}.}
\item{metric}{Distance metric passed to \code{nn()}: \code{"euclidean"},
\code{"cosine"}, \code{"correlation"}, or \code{"inner_product"}. See
\code{nn()} for metric/backend support details, including metric-aware CPU
HNSW routing.}
\item{tuning}{Tuning policy passed to \code{nn()}. \code{"auto"} uses the tuned default for the resolved method.}
\item{task}{\code{"auto"}, \code{"classification"}, or \code{"regression"}. \code{"auto"} treats numeric responses as regression and other response types as classification.}
\item{k}{Default number of neighbours used by \code{predict()} and by immediate predictions when \code{Xtest} is supplied.}
\item{n_threads}{CPU threads passed to \code{nn()} for CPU backends.}
\item{vote}{\code{"majority"} or \code{"weighted"} for immediate predictions.}
\item{type}{\code{"response"} for class/regression predictions or \code{"prob"} for class probability matrices from classification models.}
\item{...}{Reserved for future options.}
}
\value{
If \code{Xtest} is not supplied, a fitted \code{faissR_knn_model} object. If \code{Xtest} is supplied, a factor for classification, a numeric vector for regression, or a numeric class-probability matrix when \code{type = "prob"}.
}
\description{
\code{knn()} is the high-level supervised kNN API. With \code{Xtrain} and \code{Ytrain} only, it returns a reusable model. With \code{Xtest}, it immediately predicts the query rows by fitting the model and calling \code{predict()} internally. The low-level nearest-neighbour search API remains \code{nn()}.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
model <- knn(x, iris$Species, backend = "cpu", k = 5)
head(predict(model, x, k = 5))
head(predict(model, x, k = 5, type = "prob"))

pred <- knn(x, iris$Species, x, backend = "cpu", k = 5)
head(pred)
}
