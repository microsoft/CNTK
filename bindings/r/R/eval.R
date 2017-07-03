eval <- reticulate::import("cntk.eval.evaluator")

#' @export
Evaluator <- eval$Evaluator

#' @export
get_evaluation_function <- function(evaluator) {
	evaluator$evaluation_function
}

#' @export
summarize_test_progress <- function(evaluator) {
	evaluator$summarize_test_progress()
}

#' @export
eval_test_minibatch <- function(evaluator, arguments, device = NULL) {
	evaluator(arguments, device = device)
}
