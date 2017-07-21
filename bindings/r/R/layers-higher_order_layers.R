layers <- reticulate::import("cntk.layers")

#' @export
For <- function(range, constructor, name = '') {
	layers$For(range, constructor, name = name)
}

#' @export
ResNetBlock <- function(f, name = '') {
	layers$ResNetBlock(f, name = name)
}

#' @export
Sequential <- function(...) {
	layers$Sequential(c(...))
}

#' @export
SequentialClique <- function(..., name="") {
	layers$SequentialClique(c(...), name = name)
}
