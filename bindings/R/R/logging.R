#' Trace Level
#'
#' @param level
#'
#' @export
TraceLevel <- function(level) {
	reticulate::py_get_attr(cntk$logging$TraceLevel, level)
}

#' Get Logging Trace Level
#'
#' @export
get_logging_trace_level <- function() {
	cntk$logging$get_trace_level()
}

#' Set Logging Trace Level
#'
#' @param value
#'
#' @export
set_logging_trace_level <- function(value) {
	cntk$logging$set_trace_level(value)
}
