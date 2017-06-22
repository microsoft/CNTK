metrics <- reticulate::import("cntk.metrics")

#' @export
classification_error <- metrics$classification_error
