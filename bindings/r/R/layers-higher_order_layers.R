layers <- reticulate::import("cntk.layers")

#' @export
For <- layers$For

#' @export
ResNetBlock <- layers$ResNetBlock

#' @export
Sequential <- function(...) {
	layers$Sequential(c(...))
}

#' @export
SequentialClique <- function(..., name="") {
	layers$SequentialClique(c(...), name = name)
}
