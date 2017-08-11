#' Default Options
#'
#' @param ...
#'
#' @export
default_options <- function(...) {
	cntk$default_options$default_options(...)
}

#' Default Options For
#'
#' @param functions
#' @param ...
#'
#' @export
default_options_for <- function(functions, ...) {
	cntk$default_options$default_options_for(functions, ...)
}

#' Default Override Or
#'
#' @param value
#'
#' @export
default_override_or <- function(value) {
	cntk$default_options$default_override_or(value)
}

#' Get Default Override
#'
#' @param function_or_class
#' @param ...
#'
#' @export
get_default_override <- function(function_or_class, ...) {
	cntk$default_options$get_default_override(function_or_class, ...)
}

#' Get Global Option
#'
#' @param key
#' @param default_value
#'
#' @export
get_global_option <- function(key, default_value) {
	cntk$default_options$get_global_option(key, default_value)
}

#' Is Default Override
#'
#' @param value
#'
#' @export
is_default_override <- function(value) {
	cntk$default_options$is_default_override(value)
}

#' Set Global Option
#'
#' @param key
#' @param value
#'
#' @export
set_global_option <- function(key, value) {
	cntk$default_options$set_global_option(key, value)
}
