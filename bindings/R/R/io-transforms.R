#' @param brightness_radius
#'
#' @param contrast_radius
#' @param saturation_radius
#'
#' @export
transform_color <- function(brightness_radius = 0, contrast_radius = 0,
							saturation_radius = 0) {
	cntk$io$transforms$color(
		brightness_radius = brightness_radius,
		contrast_radius = contrast_radius,
		saturation_radius = saturation_radius
	)
}

#' @param crop_type
#'
#' @param crop_size
#' @param side_ratio
#' @param area_ratio
#' @param aspect_ratio
#' @param jitter_type
#'
#' @export
transform_crop <- function(crop_type = 'center', crop_size = 0, side_ratio = 0,
						   area_ratio = 0, aspect_ratio = 1,
						   jitter_type = 'none') {
	cntk$io$transform$color(
		crop_type = crop_type,
		crop_size = to_int(crop_size),
		side_ratio = side_ratio,
		area_ratio = area_ratio,
		aspect_ratio = aspect_ratio,
		jitter_type = jitter_type
	)
}

#' @param filename
#'
#' @export
transform_mean <- function(filename) {
	cntk$io$transforms$mean(filename)
}

#' @param width
#'
#' @param height
#' @param channels
#' @param interpolations
#' @param scale_mode
#' @param pad_value
#'
#' @export
transform_scale <- function(width, height, channels, interpolations, scale_mode,
							pad_value) {
	cntk$io$transforms$scale(
		to_int(width),
		to_int(height),
		to_int(channels),
		interpolations,
		scale_mode,
		to_int(pad_value)
	)
}
