core <- reticulate::import("cntk.core")

#' @export
NDArrayView <- core$NDArrayView

#' @export
Value <- core$Value

#' @export
asarray <- core$asarray

#' @export
asvalue <- core$asvalue

#' @export
user_function <- core$user_function
