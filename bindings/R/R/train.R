#' @param model
#'
#' @param criterion
#' @param parameter_learners
#' @param progress_writers
#'
#' @export
Trainer <- function(model, criterion, parameter_learners,
					progress_writers = NULL) {
	cntk$train$Trainer(
		model,
		criterion,
		parameter_learners,
		progress_writers = progress_writers
	)
}

#' @param trainer
#'
#' @param filename
#'
#' @export
restore_trainer_from_checkpoint <- function(trainer, filename) {
	trainer$restore_from_checkpoint(filename)
}

#' @param trainer
#'
#' @param filename
#'
#' @export
save_trainer_checkpoint  <- function(trainer, filename) {
	trainer$save_checkpoint(filename)
}

#' @param trainer
#'
#' @export
summarize_test_progress <- function(trainer) {
	trainer$summarize_test_progress()
}

#' @param trainer
#'
#' @export
summarize_training_progress <- function(trainer) {
	trainer$summarize_training_progress()
}

#' @param trainer
#'
#' @param arguments
#' @param device
#'
#' @export
test_minibatch <- function(trainer, arguments, device = NULL) {
	trainer$test_minibatch(
		arguments,
		device = device
	)
}

#' @param trainer
#'
#' @param data
#' @param outputs
#' @param device
#'
#' @export
train_minibatch <- function(trainer, data, outputs = NULL, device = NULL) {
	trainer$train_minibatch(data, outputs, device)
}
