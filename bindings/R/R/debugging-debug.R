#' @param model
#'
#' @param in_stream
#' @param out
#' @param exit_func
#'
#' @export
debug_model <- function(model, in_stream = sys$stdin, out = sys$stdout,
						exit_func = sys$exit) {
	cntk$debugging$debug$debug_model(
		model,
		in_stream = in_stream,
		out = out,
		exit_func = exit_func
	)
}

#' @param root_op
#'
#' @param filename
#'
#' @export
save_as_legacy_model <- function(root_op, filename) {
	cntk$debugging$debug$save_as_legacy_model(root_op, filename)
}

#' @param enable
#'
#' @export
set_checked_mode <- function(enable) {
	cntk$debugging$debug$set_checked_mode(enable)
}

#' @param level
#'
#' @export
set_computation_network_trace_lebel <- function(level) {
	cntk$debugging$debug$set_computation_network_trace_lebel(to_int(level))
}
