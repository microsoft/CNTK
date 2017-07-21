#' @export
For <- function(range, constructor, name = '') {
	cntk$layers$For(range, constructor, name = name)
}

#' @export
ResNetBlock <- function(f, name = '') {
	cntk$layers$ResNetBlock(f, name = name)
}

#' Higher-Order Wrapper for Layer Definitions
#'
#' Layer factory function to create a composite that applies a sequence of layers (or any functions) onto an input.
#' Sequential `([F, G, H])(x)` means the same as `H(G(F(x)))`.
#'
#' @param ... list of layer functions to apply in sequence
#' @return 	A function that accepts one argument and applies the given functions one after another.
#' @export
Sequential <- function(...) {
	cntk$layers$Sequential(c(...))
}

#' @export
SequentialClique <- function(..., name="") {
	cntk$layers$SequentialClique(c(...), name = name)
}
