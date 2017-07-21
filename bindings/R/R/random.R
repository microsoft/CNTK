#' @export
AUTO_SELECT_SEED <- 184467440L

#' @param shape
#'
#' @param dtype
#' @param mean
#' @param seed
#' @param name
#'
#' @export
rand_bernoulli <- function(shape, dtype = default_override_or(np$float32),
						   mean = 0.5, seed = AUTO_SELECT_SEED, name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$bernoulli(
			to_int(shape),
			dtype = type_map(dtype),
			mean = mean,
			name = name
		))
	}
	cntk$random$bernoulli(
		to_int(shape),
		dtype = type_map(dtype),
		mean = mean,
		seed = to_int(seed),
		name = name
	)
}

#' @param x
#'
#' @param mean
#' @param seed
#' @param name
#'
#' @export
rand_bernoulli_like <- function(x, mean = 0.5, seed = AUTO_SELECT_SEED,
								name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$bernoulli_like(
			x,
			mean = mean,
			name = name
		))
	}
	cntk$random$bernoulli_like(
		x,
		mean = mean,
		seed = to_int(seed),
		name = name
	)
}

#' @param shape
#'
#' @param dtype
#' @param loc
#' @param scale
#' @param seed
#' @param name
#'
#' @export
rand_gumbel <- function(shape, dtype = default_override_or(np$float32), loc = 0,
						scale = 1, seed = AUTO_SELECT_SEED, name = name) {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$gumbel(
			to_int(shape),
			dtype = type_map(dtype),
			loc = loc,
			scale = scale,
			name = name
		))
	}
	cntk$random$gumbel(
		to_int(shape),
		dtype = type_map(dtype),
		loc = loc,
		scale = scale,
		seed = to_int(seed),
		name = name
	)
}

#' @param x
#'
#' @param mean
#' @param scale
#' @param seed
#' @param name
#'
#' @export
rand_gumbel_like <- function(x, mean = 0, scale = 1, seed = AUTO_SELECT_SEED,
							 name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$gumbel_like(
			x,
			mean = mean,
			scale = scale,
			name = name
		))
	}
	cntk$random$gumbel_like(
		x,
		mean = mean,
		scale = scale,
		seed = to_int(seed),
		name = name
	)
}

#' @param shape
#'
#' @param dtype
#' @param mean
#' @param scale
#' @param seed
#' @param name
#'
#' @export
rand_normal <- function(shape, dtype = default_override_or(np$float32),
						mean = 0, scale = 1, seed = AUTO_SELECT_SEED,
						name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$normal(
			to_int(shape),
			dtype = type_map(dtype),
			mean = mean,
			scale = scale,
			name = name
		))
	}
	cntk$random$normal(
		to_int(shape),
		dtype = type_map(dtype),
		mean = mean,
		scale = scale,
		seed = to_int(seed),
		name = name
	)
}

#' @param x
#'
#' @param mean
#' @param scale
#' @param seed
#' @param name
#'
#' @export
rand_normal_like <- function(x, mean = 0, scale = 1, seed = AUTO_SELECT_SEED,
							 name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$normal_like(
			x,
			mean = mean,
			scale = scale,
			name = name
		))
	}
	cntk$random$normal_like(
		x,
		mean = mean,
		scale = scale,
		seed = to_int(seed),
		name = name
	)
}

#' @param shape
#'
#' @param dtype
#' @param low
#' @param high
#' @param seed
#' @param name
#'
#' @export
rand_uniform <- function(shape, dtype = default_override_or(np$float32),
						 low = 0, high = 1, seed = AUTO_SELECT_SEED,
						 name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$uniform(
			to_int(shape),
			dtype = type_map(dtype),
			mean = mean,
			scale = scale,
			name = name
		))
	}
	cntk$random$uniform(
		to_int(shape),
		dtype = type_map(dtype),
		mean = mean,
		scale = scale,
		seed = to_int(seed),
		name = name
	)
}

#' @param x
#'
#' @param low
#' @param high
#' @param seed
#' @param name
#'
#' @export
rand_uniform_like <- function(x, low = 0, high = 1, seed = AUTO_SELECT_SEED,
							 name = '') {
	if (seed == AUTO_SELECT_SEED) {
		return(cntk$random$uniform_like(
			x,
			mean = mean,
			scale = scale,
			name = name
		))
	}
	cntk$random$uniform_like(
		x,
		mean = mean,
		scale = scale,
		seed = to_int(seed),
		name = name
	)
}
