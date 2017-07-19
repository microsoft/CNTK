#' @export
For <- function(range, constructor, name = '') {
	cntk$layers$For(range, constructor, name = name)
}

#' @export
ResNetBlock <- function(f, name = '') {
	cntk$layers$ResNetBlock(f, name = name)
}

#' @export
Sequential <- function(...) {
	cntk$layers$Sequential(c(...))
}

#' @export
SequentialClique <- function(..., name="") {
	cntk$layers$SequentialClique(c(...), name = name)
}
