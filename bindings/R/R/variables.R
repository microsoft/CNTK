#' Constant
#'
#' A constant value. It can be a scalar, vector, matrix, or tensor of floating
#' point numbers that cannot be modified.  A Constant is a Variable and
#' therefore inherits all its methods.
#'
#' ****** Properties: ******
#'
#' value
#'
#' @param value
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param device - instance of DeviceDescriptor
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

#' Parameter
#'
#' A trainable parameter. It can be a scalar, vector, matrix, or tensor of
#' floating point numbers that can be modified by a training procedure.
#'
#' ****** Properties: ******
#'
#' value
#'
#' @param value
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param device - instance of DeviceDescriptor
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

#' Record
#'
#' Easy construction of a record (=immutable singleton class) from keyword
#' arguments.
#'
#' ****** Associated Functions: ******
#'
#' updated_record_with
#'
#' @param ... named arguments to turn into record numbers
#'
#' @export
Record <- function(...) {
	cntk$variables$Record(...)
}

#' Updated Record With
#'
#' Create a new Record from an existing one with members modified or added.
#'
#' @param record - the Record instance to be updated
#' @param ... named arguments to turn into record numbers
#'
#' @export
updated_record_with <- function(record, ...) {
	record$updated_with(record, ...)
}

#' Variable
#'
#' Denotes a symbolic entity corresponding to the inputs and outputs of a
#' Function.
#'
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
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

#' Variable Mixin
#'
#' Standard properties for Variable and its derived classes Parameter and
#' Constant.
#'
#' ****** Properties: ******
#'
#' dtype
#'
#' dynamic_axes
#'
#' is_constant
#'
#' is_input
#'
#' is_output
#'
#' is_parameter
#'
#' is_placeholder
#'
#' is_sparse
#'
#' name
#'
#' needs_gradient
#'
#' owner
#'
#' shape
#'
#' uid
#'
#' @export
VariableMixin <- function() {
	cntk$variables$VariableMixin()
}
