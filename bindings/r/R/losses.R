losses <- reticulate::import("cntk.losses")

#' @export
squared_error <- losses$squared_error

#' @export
cross_entropy_with_softmax <- losses$cross_entropy_with_softmax
