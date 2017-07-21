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

#' @param evaluator
#'
#' @export
summarize_test_progress <- function(evaluator) {
	evaluator$summarize_test_progress()
}

#' @param evaluator
#'
#' @param arguments
#' @param device
#'
#' @export
eval_test_minibatch <- function(evaluator, arguments, device = NULL) {
	evaluator(arguments, device = device)
}
