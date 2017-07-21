logging <- reticulate::import("cntk.logging")

#' @param level
#'
#' @export
TraceLevel <- function(level) {
	reticulate::py_get_attr(cntk$logging$TraceLevel, level)
}

#' @export
get_logging_trace_level <- function() {
	cntk$logging$get_trace_level()
}

#' @param value
#'
#' @export
set_logging_trace_level <- function(value) {
	cntk$logging$set_trace_level(value)
}
