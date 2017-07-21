#' @param value
#'
#' @param shape
#' @param dtype
#' @param device
#' @param name
#'
#' @export
Constant <- function(value = NULL, shape = NULL, dtype = 'float32',
					 device = NULL, name = '') {
	cntk$variables$Constant(
		value = value,
		shape = shape,
		dtype = type_map(dtype),
		device = device,
		name = name
	)
}

#' @param value
#'
#' @param shape
#' @param dtype
#' @param device
#' @param name
#'
#' @export
Parameter <- function(value = NULL, shape = NULL, dtype = 'float32',
					  device = NULL, name = '') {
	cntk$variables$Parameter(
		value = value,
		shape = shape,
		dtype = type_map(dtype),
		device = device,
		name = name
	)
}

#' @param ...
#'
#' @export
Record <- function(...) {
	cntk$variables$Record(...)
}

#' @param record
#'
#' @param ...
#'
#' @export
updated_record_with <- function(record, ...) {
	record$updated_with(record, ...)
}

#' @param shape
#'
#' @param dtype
#' @param needs_gradient
#' @param is_sparse
#' @param dynamic_axes
#' @param name
#'
#' @export
Variable <- function(shape = NULL, dtype = 'auto', needs_gradient = FALSE,
					 is_sparse = FALSE,
					 dynamic_axes = rep(c(get_default_batch_axis()), 2),
					 name = '') {
	cntk$variables$Variable(
		shape = shape,
		dtype = type_map(dtype),
		needs_gradient = needs_gradient,
		is_sparse = is_sparse,
		dynamic_axes = dynamic_axes,
		name = name
	)
}

#' @export
VariableMixin <- function() {
	cntk$variables$VariableMixin()
}
