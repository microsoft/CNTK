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
#' @param eval_function (Function) evaluation function
#' @param progress_writers (list) progress writers to track training progress
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
#' @param evaluator the Evaluator instance from which to get progress
#'
#' @export
summarize_test_progress <- function(evaluator) {
	cntk$eval$evaluator$summarize_test_progress()
}

#' Test Evaluator Minibatch
#'
#' @param evaluator the Evaluator instance on which to perform the operation
#' @param arguments named list of input variable names to input data or if node
#' has a unique input, arguments is mapped to this input. For nodes with more
#' than one input, only a named list is allowed.
#' @param device - instance of DeviceDescriptor
#'
#' @export
eval_test_minibatch <- function(evaluator, arguments, device = NULL) {
	cntk$eval$evaluator$test_minibatch(arguments, device = device)
}
