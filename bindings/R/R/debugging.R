#' Dump Function
#'
#' @param root
#' @param tag
#'
#' @export
dump_function <- function(root, tag = NULL) {
	cntk$debugging$dump_function(
		root,
		tag = tag
	)
}

#' Dump Signature
#'
#' @param root
#' @param tag
#'
#' @export
dump_signature <- function(root, tag = NULL) {
	cntk$debugging$dump_signature(
		root,
		tag = tag
	)
}
