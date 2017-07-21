#' @export
CNTKAxis <- function(...) {
	cntk$axis$Axis(...)
}

#' @export
get_all_axes <- function() {
	cntk$axis$Axis$all_axes()
}

#' @export
get_static_all_axes <- function() {
	cntk$axis$Axis$all_static_axes()
}

#' @export
get_default_batch_axis <- function() {
	cntk$axis$Axis$default_batch_axis()
}

#' @export
get_default_dynamic_batch_axis <- function() {
	cntk$axis$Axis$default_dynamic_batch_axis()
}

#' @export
create_new_leading_axis <- function() {
	cntk$axis$Axis$new_leading_axis()
}

#' @export
get_static_axis_index <- function(ax, checked = TRUE) {
	ax$static_axis_index(checked = checked)
}

#' @export
unknown_dynamic_axes <- function() {
	cntk$axis$Axis$unknown_dynamic_axes()
}
