#' CNTKAxis
#'
#' An axis object describes the axis of a variable and is used for specifying
#' the axes parameters of certain functions such as reductions. Besides the
#' static axes corresponding to each of the axes of the variable’s shape,
#' variables of kind ‘input’ and any ‘output’ variables dependent on an ‘input’
#' variable also have two additional dynamic axes whose dimensions are known
#' only when the variable is bound to actual data during compute time (viz.
#' sequence axis and batch axis denoting the axis along which multiple
#' sequences are batched).
#'
#' Axis parameters can also be negative, which allows to refere axis starting
#' from the last axis. Please be aware that Axis objects work in a column-major
#' wise, as opposed to any other function in the library.
#'
#' ****** Properties: ******
#'
#' is_ordered
#'
#' is_static_axis
#'
#' name
#'
#' ****** Associated Functions: ******
#'
#' get_all_axes()
#'
#' get_all_static_axes()
#'
#' get_default_batch_axis()
#'
#' create_new_leading_axis()
#'
#' new_unique_dynamic_axis(name)
#'
#' get_static_axis_index(axis, checked = TRUE)
#'
#' unknown_dynamic_axes()
#'
#' @export
CNTKAxis <- function(...) {
	cntk$axis$Axis(...)
}

#' get_all_axes
#'
#' Axis object representing all the axes–static and dynamic–of an operand.
#'
#' @export
get_all_axes <- function() {
	cntk$axis$Axis$all_axes()
}

#' get_all_static_axes
#'
#' Axis object representing all the static axes of an operand.
#'
#' @export
get_all_static_axes <- function() {
	cntk$axis$Axis$all_static_axes()
}

#' get_default_batch_axis
#'
#' Returns an Axis object representing the batch axis
#'
#' @export
get_default_batch_axis <- function() {
	cntk$axis$Axis$default_batch_axis()
}

#' get_default_dynamic_axis
#'
#' Returns an Axis object representing the default dynamic axis
#'
#' @export
get_default_dynamic_axis <- function() {
	cntk$axis$Axis$default_dynamic_axis()
}

#' create_new_leading_axis
#'
#' Creates an Axis object representing a new leading static axis.
#'
#' @export
create_new_leading_axis <- function() {
	cntk$axis$Axis$new_leading_axis()
}

#' get_static_axis_index
#'
#' Returns the integer with which the static axis is defined. For example, 0 =
#' first axis, 1 = second axis, etc.
#'
#' @export
get_static_axis_index <- function(ax, checked = TRUE) {
	ax$static_axis_index(checked = checked)
}

#' unknown_dynamic_axes
#'
#' Unknown dynamic axes
#'
#' @export
unknown_dynamic_axes <- function() {
	cntk$axis$Axis$unknown_dynamic_axes()
}
