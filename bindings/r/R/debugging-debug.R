debug <- reticulate::import("cntk.debugging.debug")
sys <- reticulate::import("sys")

#' @export
debug_model <- function(model, in_stream = sys$stdin, out = sys$stdout,
						exit_func = sys$exit) {
	debug$debug_model(
		model,
		in_stream = in_stream,
		out = out,
		exit_func = exit_func
	)
}

#' @export
save_as_legacy_model <- function(root_op, filename) {
	debug$save_as_legacy_model(root_op, filename)
}

#' @export
set_checked_mode <- function(enable) {
	debug$set_checked_mode(enable)
}

#' @export
set_computation_network_trace_lebel <- function(level) {
	debug$set_computation_network_trace_lebel(to_int(level))
}
