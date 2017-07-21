io <- reticulate::import("cntk.io.transforms")

#' @export
transform.color <- io$color

#' @export
transform.crop <- io$crop

#' @export
transform.mean <- io$mean

#' @export
transform.scale <- function(width, height, channels, interpolations, scale_mode,
							pad_value) {
	io$scale(
		to_int(width),
		to_int(height),
		to_int(channels),
		interpolations,
		scale_mode,
		to_int(pad_value)
	)
}
