learners <- reticulate::import("cntk.learners")

#' @export
learning_rate_schedule <- learners$learning_rate_schedule

#' @export
momentum_as_time_constant_schedule <- function(momentum, epoch_size = NULL) {
	learners$momentum_as_time_constant_schedule(
		momentum,
		epoch_size = to_int(epoch_size)
	)
}

#' @export
momentum_sgd <- learners$momentum_sgd

#' @export
UnitType <- learners$UnitType

#' @export
sgd <- learners$sgd
