debugging <- reticulate::import("cntk.debugging")

#' @export
dump_function <- function(root, tag = NULL) {
	cntk$debugging$dump_function(
		root,
		tag = tag
	)
}

#' @export
dump_signature <- function(root, tag = NULL) {
	cntk$debugging$dump_signature(
		root,
		tag = tag
	)
}
