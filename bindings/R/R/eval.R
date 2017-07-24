#' Evaluator
#'
#' Class for evaluation of minibatches against the specified evaluation
#' function.
#'
#' ****** Attributes: ******
#'
#' evaluation_function
#'
#' ****** Associated Functions: ******
#'
#' summarize_test_progress(evaluator)
#'
#' eval_test_minibatch(evaluator, arguments, device = NULL)
#'
#' @param eval_function
#' @param progress_writers
#'
#' @export
Evaluator <- function(eval_function, progress_writers = NULL) {
	cntk$eval$evaluator$Evaluator(
		eval_function,
		progress_writers = progress_writers
	)
}

#' Summarize Evaluator Test Progress
#'
#' @param evaluator
#'
#' @export
summarize_test_progress <- function(evaluator) {
	cntk$eval$evaluator$summarize_test_progress()
}

#' Test Evaluator Minibatch
#'
#' @param evaluator
#'
#' @param arguments
#' @param device - instance of DeviceDescriptor
#'
#' @export
eval_test_minibatch <- function(evaluator, arguments, device = NULL) {
	cntk$eval$evaluator$test_minibatch(arguments, device = device)
}
