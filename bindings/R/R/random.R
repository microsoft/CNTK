#' @export
AUTO_SELECT_SEED <- 184467440L

#' Random Bernoulli Distribution
#'
#' Generates samples from the Bernoulli distribution with success probability
#' mean.
#'
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param mean - success probability
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Bernoulli Like
#'
#' Generates samples from the Bernoulli distribution with success probability
#' mean.
#'
#' @param x - CNTK variable from which to copy the shape, dtype, and dynamic axes
#' @param mean - success probability
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Gumbel Distribution
#'
#' @param shape - list of ints representing tensor shape
#'
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param loc
#' @param scale - scale of the distribution
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Gumbel Like Distribution
#'
#' @param x - CNTK variable from which to copy the shape, dtype, and dynamic axes
#' @param mean - success probability
#' @param scale - scale of the distribution
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Normal Distribution
#'
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param mean - success probability
#' @param scale - scale of the distribution
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Normal Like Distribution
#'
#' @param x - CNTK variable from which to copy the shape, dtype, and dynamic axes
#' @param mean - success probability
#' @param scale - scale of the distribution
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Uniform Distribution
#'
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param low - bottom of distribution range
#' @param high - top of distribution range
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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

#' Random Uniform Like Distribution
#'
#' @param x - CNTK variable from which to copy the shape, dtype, and dynamic axes
#' @param low - bottom of distribution range
#' @param high - top of distribution range
#' @param seed - pseudo random number generator seed
#' @param name - name of the Function instance in the network
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
