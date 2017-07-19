#' @export
ArrayMixin <- function() {
	cntk$tensor$ArrayMixin()
}

#' @export
arraymixin_as_array <- function(array) {
	array$asarray()
}

#' @export
TensorOpsMixin <- function() {
	cntk$tensor$TensorOpsMixin()
}
