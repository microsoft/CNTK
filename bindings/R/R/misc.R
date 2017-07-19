#' @export
convert_graph <- function(root_func, filter, converter) {
	cntk$misc$converter$convert(
		root_func,
		filter,
		converter
	)
}

#' @export
convert_optimized_rnnstack <- function(cudnn_model) {
	cntk$rnnstack$convert_optimized_rnnstack(cudnn_model)
}

#' @export
dict <- function(...) {
	reticulate::dict(...)
}
