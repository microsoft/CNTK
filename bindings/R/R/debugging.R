debugging <- reticulate::import("cntk.debugging")

#' @param root
#'
#' @param tag
#'
#' @export
dump_function <- function(root, tag = NULL) {
	cntk$debugging$dump_function(
		root,
		tag = tag
	)
}

#' @param root
#'
#' @param tag
#'
#' @export
dump_signature <- function(root, tag = NULL) {
	cntk$debugging$dump_signature(
		root,
		tag = tag
	)
}
