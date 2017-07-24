#' Array Mixin
#'
#' @export
ArrayMixin <- function() {
	cntk$tensor$ArrayMixin()
}

#' ArrayMixin As Array
#'
#' @param arraymixin - ArrayMixin instance
#'
#' @export
arraymixin_as_array <- function(arraymixin) {
	arraymixin$asarray()
}

#' TensorOpsMixin
#'
#' @export
TensorOpsMixin <- function() {
	cntk$tensor$TensorOpsMixin()
}
