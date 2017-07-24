#' Bilinear Initializer
#'
#' @param kernel_width
#' @param kernel_height
#'
#' @export
init_bilinear <- function(kernel_width, kernel_height) {
	cntk$initializer$bilinear(
		to_int(kernel_width),
		to_int(kernel_height)
	)
}

#' Glorot Normal Initializer
#'
#' @param scale
#' @param output_rank
#' @param filter_rank
#' @param seed
#'
#' @export
init_glorot_normal <- function(scale = 1, output_rank = 2147483647,
						  filter_rank = 2147483647, seed = NULL) {
	cntk$initializer$glorot_normal(
		scale = scale,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank),
		seed = to_int(seed)
	)
}

#' Glorot Uniform Initializer
#'
#' @param scale
#' @param output_rank
#' @param filter_rank
#' @param seed
#'
#' @export
init_glorot_uniform <- function(scale = 1, output_rank = 2147483647,
						  filter_rank = 2147483647, seed = NULL) {
	cntk$initializer$glorot_uniform(
		scale = scale,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank),
		seed = to_int(seed)
	)
}

#' He Normal Initializer
#'
#' @param scale
#' @param output_rank
#' @param filter_rank
#' @param seed
#'
#' @export
init_he_normal <- function(scale = 1, output_rank = 2147483647,
						  filter_rank = 2147483647, seed = NULL) {
	cntk$initializer$he_normal(
		scale = scale,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank),
		seed = to_int(seed)
	)
}

#' He Uniform Initializer
#'
#' @param scale
#' @param output_rank
#' @param filter_rank
#' @param seed
#'
#' @export
init_he_uniform <- function(scale = 1, output_rank = 2147483647,
						  filter_rank = 2147483647, seed = NULL) {
	cntk$initializer$he_uniform(
		scale = scale,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank),
		seed = to_int(seed)
	)
}

#' Initializer With Rank
#'
#' override output_rank and filter_rank specification in a random initializer
#' constructed without an explicit output_rank and filter_rank specification
#'
#' @param init - Initializer to be overriden
#' @param output_rank
#' @param filter_rank
#'
#' @export
init_with_rank <- function(initializer, output_rank = NULL,
								  filter_rank = NULL) {
	cntk$initializer$initializer_with_rank(
		initializer,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank)
	)
}

#' Normal Initializer
#'
#' @param scale
#' @param output_rank
#' @param filter_rank
#' @param seed
#'
#' @export
init_normal <- function(scale = 1, output_rank = 2147483647,
						  filter_rank = 2147483647, seed = NULL) {
	cntk$initializer$init_normal(
		scale = scale,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank),
		seed = to_int(seed)
	)
}

#' Truncated Normal Initializer
#'
#' @param stdev
#' @param seed
#'
#' @export
init_truncated_normal <- function(stdev, seed = NULL) {
	cntk$initializer$truncated_normal(
		stdev,
		seed = to_int(seed)
	)
}

#' Uniform Initializer
#'
#' @param scale
#' @param seed
#'
#' @export
init_uniform <- function(scale, seed = NULL) {
	cntk$initializer$uniform(
		scale,
		seed = to_int(seed)
	)
}

#' Xavier Initializer
#'
#' @param scale
#' @param output_rank
#' @param filter_rank
#' @param seed
#'
#' @export
init_xavier <- function(scale = 1, output_rank = 2147483647,
						  filter_rank = 2147483647, seed = NULL) {
	cntk$initializer$init_xavier(
		scale = scale,
		output_rank = to_int(output_rank),
		filter_rank = to_int(filter_rank),
		seed = to_int(seed)
	)
}
