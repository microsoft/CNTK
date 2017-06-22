default_opts <- reticulate::import("cntk.default_options")

#' @export
default_options <- default_opts$default_options

#' @export
default_options_for <- default_opts$default_options_for

#' @export
default_override_or <- default_opts$default_override_or

#' @export
get_default_override <- default_opts$get_default_override

#' @export
get_global_option <- default_opts$get_global_option

#' @export
is_default_override <- default_opts$is_default_override

#' @export
set_global_option <- default_opts$set_global_option
