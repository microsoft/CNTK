tensor <- reticulate::import("cntk.tensor")

#' @export
ArrayMixin <- tensor$ArrayMixin

#' @export
TensorOpsMixin <- tensor$TensorOpsMixin
