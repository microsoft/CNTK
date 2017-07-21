ops <- reticulate::import("cntk.ops")
np <- reticulate::import("numpy")

#' @export
input_variable <- function(shape, needs_gradient = FALSE, is_sparse = FALSE,
						   dynamic_axes = c(Axis$default_batch_axis()),
						   name = '') {
	ops$input_variable(shape, np$float32, needs_gradient, is_sparse,
					   dynamic_axes, name)
}

#' @export
minus <- ops$minus

#' @export
constant <- function(value = NULL, shape = NULL, name = '') {
	ops$constant(
		value = value,
		shape = to_int(shape),
		dtype = np$float32,
		name = name
	)
}

#' @export
sigmoid <- ops$sigmoid

#' @export
relu <- ops$relu

#' @export
element_times <- ops$element_times

#' @export
func_eval <- function(func, arguments = NULL, outputs = NULL, device = NULL,
					  as_matrix = TRUE) {
	func$eval(
		arguments = arguments,
		outputs = outputs,
		device = device,
		as_numpy = as_matrix
	)
}
