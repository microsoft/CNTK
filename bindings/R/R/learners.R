#' Learner
#'
#' Abstraction for learning a subset of parameters of a learnable function
#' using first order gradient values. For example momentum, AdaGrad, RMSProp,
#' etc. are different types of learners with their own algorithms for learning
#' parameter values using first order gradients. To instantiate a concrete
#' learner, use the factory methods in this module.
#'
#' @param parameters list of parameter associated with this learner
#' @param learningRateSchedule get_learning_rate(learner)
#' @example \dontrun{
#' reset_learning_rate(learner, learning_rate)
#' update_learner(learner, gradient_values, training_sample_count)
#' }
#'
#' @export
Learner <- function(parameters, learningRateSchedule) {
	cntk$learners$Learner(
		parameters,
		learningRateSchedule
	)
}

#' @param learner
#'
#' @export
get_learning_rate <- function(learner) {
	learner$get_learning_rate()
}

#' @param learner
#'
#' @param learning_rate
#'
#' @export
reset_learning_rate <- function(learner, learning_rate) {
	learner$reset_learning_rate(learning_rate)
}

#' @param learner
#'
#' @param gradient_values
#' @param training_sample_count
#'
#' @export
update_learner <- function(learner, gradient_values, training_sample_count) {
	learner$update_learner(gradient_values, training_sample_count)
}


#' @param value
#'
#' @export
UnitType <- function(value) {
	reticulate::py_get_attr(cntk$learners$UnitType, value)
}

#' UserLearner
#'
#' Base class of all user-defined learners. To implement your own learning
#' algorithm, derive from this class and override the \code{update()}.
#'
#' Certain optimizers (such as AdaGrad) require additional storage. This can be
#' allocated and initialized during construction.
#'
#' @param parameters
#' @param lr_schedule
#' @param as_matrix
#'
#' @seealso \code{\link{update_user_learner}}
#'
#' @export
UserLearner <- function(parameters, lr_schedule, as_matrix = TRUE) {
	cntk$learners$UserLearner(
		parameters,
		lr_schedule,
		as_numpy = as_matrix
	)
}

#' @param learner
#'
#' @param gradient_values
#' @param training_sample_count
#' @param sweep_end
#'
#' @export
update_user_learner <- function(learner, gradient_values, training_sample_count,
								sweep_end) {
	learner$update_user_learner(
		gradient_values,
		training_sample_count,
		sweep_end
	)
}

#' @param parameters
#'
#' @param lr
#' @param rho
#' @param epsilon
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#' @param use_mean_gradient
#'
#' @export
learner_adadelta <- function(parameters, lr, rho, epsilon,
						     l1_regularization_weight = 0,
						     l2_regularization_weight = 0,
						     gaussian_noise_injection_std_dev = 0,
						     gradient_clipping_threshold_per_sample = 0,
						     gradient_clipping_with_truncation = TRUE,
						     use_mean_gradient = FALSE) {
	cntk$learners$adadelta(
		parameters,
		lr,
		rho,
		epsilon,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation,
		use_mean_gradient = use_mean_gradient
	)
}

#' @param parameters
#'
#' @param lr
#' @param need_ave_multiplier
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#'
#' @export
learner_adagrad <- function(parameters, lr, need_ave_multiplier = TRUE,
						    l1_regularization_weight = 0,
						    l2_regularization_weight = 0,
						    gaussian_noise_injection_std_dev = 0,
						    gradient_clipping_threshold_per_sample = np$inf,
						    gradient_clipping_with_truncation = TRUE) {
	cntk$learners$adagrad(
		parameters,
		lr,
		need_ave_multiplier = need_ave_multiplier,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation
	)
}

#' @param parameters
#'
#' @param lr
#' @param momentum
#' @param unit_gain
#' @param variance_momentum
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#' @param epsilon
#' @param adamax
#'
#' @export
learner_adam <- function(parameters, lr, momentum,
						 unit_gain = cntk$default_unit_gain_value(),
						 variance_momentum = cntk$ops$momentum_as_time_constant_schedule(720000),
					     l1_regularization_weight = 0,
					     l2_regularization_weight = 0,
					     gaussian_noise_injection_std_dev = 0,
					     gradient_clipping_threshold_per_sample = np$inf,
					     gradient_clipping_with_truncation = TRUE,
						 epsilon = 1e-8, adamax = FALSE) {
	cntk$learners$adam(
		parameters,
		lr,
		momentum,
		unit_gain = unit_gain,
		variance_momentum = variance_momentum,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation,
		epsilon = epsilon,
		adamax = adamax
	)
}

#' @param parameters
#'
#' @param lr
#' @param momentum
#' @param unit_gain
#' @param variance_momentum
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#'
#' @export
learner_fsadagrad <- function(parameters, lr, momentum,
						 unit_gain = cntk$default_unit_gain_value(),
						 variance_momentum = cntk$ops$momentum_as_time_constant_schedule(720000),
					     l1_regularization_weight = 0,
					     l2_regularization_weight = 0,
					     gaussian_noise_injection_std_dev = 0,
					     gradient_clipping_threshold_per_sample = np$inf,
					     gradient_clipping_with_truncation = TRUE) {
	cntk$learners$fsadagrad(
		parameters,
		lr,
		momentum,
		unit_gain = unit_gain,
		variance_momentum = variance_momentum,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation
	)
}

#' @param lr
#'
#' @param unit
#' @param epoch_size
#'
#' @export
learning_rate_schedule <- function(lr, unit, epoch_size = NULL) {
	cntk$learners$learning_rate_schedule(
		lr,
		unit,
		epoch_size = to_int(epoch_size)
	)
}

#' @param momentum
#'
#' @param epoch_size
#'
#' @export
momentum_as_time_constant_schedule <- function(momentum, epoch_size = NULL) {
	cntk$learners$momentum_as_time_constant_schedule(
		momentum,
		epoch_size = to_int(epoch_size)
	)
}

#' @param momentum
#'
#' @param epoch_size
#'
#' @export
momentum_schedule <- function(momentum, epoch_size = NULL) {
	cntk$learners$momentum_schedule(
		momentum,
		epoch_size = to_int(epoch_size)
	)
}

#' Creates a Momentum SGD learner instance to learn the parameters.
#'
#' @param parameters list of network parameters to tune.
#' @param lr output of \code{learning_rate_schedule}
#' @param momentum output of \code{momentum_schdule} or \code{momentum_as_time_constant_schedule}
#' @param unit_gain logical whether to interpret momentum as a unit-gain filter
#' @param l1_regularization_weight double of l1 regularization
#' @param l2_regularization_weight double of l2 regularization
#' @param gaussian_noise_injection_std_dev double of noise injection
#' @param gradient_clipping_threshold_per_sample double of gradient clipping threshold per sample
#' @param gradient_clipping_with_truncation logical for gradient clipping with truncation
#' @param use_mean_gradient logical use averaged gradient as input to learner.
#'
#' @references \url{https://www.cntk.ai/pythondocs/cntk.learners.html#cntk.learners.momentum_sgd}
#' @export
learner_momentum_sgd <- function(parameters, lr, momentum,
                                 unit_gain = cntk$default_unit_gain_value(),
                                 l1_regularization_weight = 0,
                                 l2_regularization_weight = 0,
                                 gaussian_noise_injection_std_dev = 0,
                                 gradient_clipping_threshold_per_sample = np$inf,
                                 gradient_clipping_with_truncation = TRUE,
                                 use_mean_gradient = FALSE)
{
	cntk$learners$momentum_sgd(
		parameters,
		lr,
		momentum,
		unit_gain = unit_gain,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation,
		use_mean_gradient = use_mean_gradient
	)
}

#' @param parameters
#'
#' @param lr
#' @param momentum
#' @param unit_gain
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#'
#' @export
learner_nesterov <- function(parameters, lr, momentum,
							 unit_gain = cntk$default_unit_gain_value(),
							 l1_regularization_weight = 0,
							 l2_regularization_weight = 0,
							 gaussian_noise_injection_std_dev = 0,
							 gradient_clipping_threshold_per_sample = np$inf,
							 gradient_clipping_with_truncation = TRUE) {
	cntk$learners$nesterov(
		parameters,
		lr,
		momentum,
		unit_gain = unit_gain,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation
	)
}

#' @param parameters
#'
#' @param lr
#' @param gamma
#' @param inc
#' @param dec
#' @param max
#' @param min
#' @param need_ave_multiplier
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#'
#' @export
learner_rmsprop <- function(parameters, lr, gamma, inc, dec, max, min,
							need_ave_multiplier = TRUE,
							l1_regularization_weight = 0,
							l2_regularization_weight = 0,
							gaussian_noise_injection_std_dev = 0,
							gradient_clipping_threshold_per_sample = np$inf,
							gradient_clipping_with_truncation = TRUE) {
	cntk$learners$rmsprop(
		parameters,
		lr,
		gamma,
		inc,
		dec,
		max,
		min,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation
	)
}

#' @param parameters
#'
#' @param lr
#' @param l1_regularization_weight
#' @param l2_regularization_weight
#' @param gaussian_noise_injection_std_dev
#' @param gradient_clipping_threshold_per_sample
#' @param gradient_clipping_with_truncation
#'
#' @export
learner_sgd <- function(parameters, lr,
						l1_regularization_weight = 0,
						l2_regularization_weight = 0,
						gaussian_noise_injection_std_dev = 0,
						gradient_clipping_threshold_per_sample = np$inf,
						gradient_clipping_with_truncation = TRUE) {
	cntk$learners$sgd(
		parameters,
		lr,
		l1_regularization_weight = l1_regularization_weight,
		l2_regularization_weight = l2_regularization_weight,
		gaussian_noise_injection_std_dev = gaussian_noise_injection_std_dev,
		gradient_clipping_threshold_per_sample = gradient_clipping_threshold_per_sample,
		gradient_clipping_with_truncation = gradient_clipping_with_truncation
	)
}

#' @param schedule
#'
#' @param unit
#' @param epoch_size
#'
#' @export
training_parameter_schedule <- function(schedule, unit, epoch_size = NULL) {
	cntk$learners$training_parameter_schedule(
		schedule,
		unit,
		epoch_size = to_int(epoch_size)
	)
}

#' @param update_func
#'
#' @param parameters
#'
#' @export
universal_learner <- function(update_func, parameters) {
	cntk$learners$universal(update_func, parameters)
}
