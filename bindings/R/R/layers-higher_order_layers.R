#' CNTK Function For Loop Construct
#'
#' @param range
#' @param constructor
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
For <- function(range, constructor, name = '') {
	cntk$layers$For(range, constructor, name = name)
}

#' ResNet Block
#'
#' @param f
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
ResNetBlock <- function(f, name = '') {
	cntk$layers$ResNetBlock(f, name = name)
}

#' Sequential Higher-Order Wrapper for Layer Definitions
#'
#' Layer factory function to create a composite that applies a sequence of
#' layers (or any functions) onto an input.
#' Sequential `([F, G, H])(x)` means the same as `H(G(F(x)))`.
#'
#' @param ... list of layer functions to apply in sequence
#' @return 	A function that accepts one argument and applies the given
#' functions one after another.
#' @export
Sequential <- function(...) {
	cntk$layers$Sequential(c(...))
}

#' Sequential Clique
#'
#' @param ...
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
SequentialClique <- function(..., name="") {
	cntk$layers$SequentialClique(c(...), name = name)
}
