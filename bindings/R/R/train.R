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

#' @export
restore_trainer_from_checkpoint <- function(trainer, filename) {
	trainer$restore_from_checkpoint(filename)
}

#' @export
save_trainer_checkpoint  <- function(trainer, filename) {
	trainer$save_checkpoint(filename)
}

#' @export
summarize_test_progress <- function(trainer) {
	trainer$summarize_test_progress()
}

#' @export
summarize_training_progress <- function(trainer) {
	trainer$summarize_training_progress()
}

#' @export
test_minibatch <- function(trainer, arguments, device = NULL) {
	trainer$test_minibatch(
		arguments,
		device = device
	)
}

#' @export
train_minibatch <- function(trainer, data, outputs = NULL, device = NULL) {
	trainer$train_minibatch(data, outputs, device)
}
