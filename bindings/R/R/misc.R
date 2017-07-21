#' @param root_func
#'
#' @param filter
#' @param converter
#'
#' @export
convert_graph <- function(root_func, filter, converter) {
	cntk$misc$converter$convert(
		root_func,
		filter,
		converter
	)
}

#' @param cudnn_model
#'
#' @export
convert_optimized_rnnstack <- function(cudnn_model) {
	cntk$rnnstack$convert_optimized_rnnstack(cudnn_model)
}

#' @param ...
#'
#' @export
dict <- function(...) {
	reticulate::dict(...)
}
