#' Trainer
#'
#' The instance of the class should be created by using training_session()
#' function.
#'
#' A training session trains a model using the specified trainer and configs.
#' Different aspects of training such as data sources, checkpointing, cross
#' validation, progress printing can be configured using the corresponding
#' config classes.
#'
#' ****** Properties: ******
#'
#' evaluation_function - The evaluation function that the trainer is using.
#'
#' loss_function - The loss function that the trainer is using.
#'
#' model - The model that the trainer is training.
#'
#' parameter_learners - The parameter learners that the trainer is using.
#'
#' previous_minibatch_evaluation_average - The average evaluation criterion
#' value per sample for the last minibatch trained
#'
#' previous_minibatch_loss_average - The average training loss per sample for
#' the last minibatch trained
#'
#' previous_minibatch_sample_count - The number of samples in the last
#' minibatch trained with
#'
#' total_number_of_samples_seen - The number of samples seen globally between
#' all workers from the beginning of training.
#'
#' ****** Associated Functions: ******
#'
#' restore_trainer_from_checkpoint
#'
#' save_trainer_checkpoint
#'
#' summarize_training_progress
#'
#' summarize_test_progress
#'
#' test_minibatch
#'
#' train_minibatch
#'
#' @param model - root node of the Function to train
#' @param criterion (list of Function or Variable) - Function with one or two
#' outputs, representing loss and, if given, evaluation metric (in this order)
#' @param parameter_learners (list) – list of learners
#' @param progress_writers (progress writer or list of them) – optionally, list
#' of progress writers to automatically track training progress.
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

#' Restore Trainer From Checkpoint
#'
#' @param trainer the Trainer instance on which to perform the operation
#' @param filename
#'
#' @export
restore_trainer_from_checkpoint <- function(trainer, filename) {
	trainer$restore_from_checkpoint(filename)
}

#' Save Trainer Checkpoint
#'
#' @param trainer the Trainer instance on which to perform the operation
#' @param filename
#'
#' @export
save_trainer_checkpoint  <- function(trainer, filename) {
	trainer$save_checkpoint(filename)
}

#' Summarize Test Progress
#'
#' @param trainer the Trainer instance on which to perform the operation
#'
#' @export
summarize_test_progress <- function(trainer) {
	trainer$summarize_test_progress()
}

#' Summarize Training Progress
#'
#' @param trainer the Trainer instance on which to perform the operation
#'
#' @export
summarize_training_progress <- function(trainer) {
	trainer$summarize_training_progress()
}

#' Test Minibatch
#'
#' @param trainer the Trainer instance on which to perform the operation
#' @param arguments
#' @param device - instance of DeviceDescriptor
#'
#' @export
test_minibatch <- function(trainer, arguments, device = NULL) {
	trainer$test_minibatch(
		arguments,
		device = device
	)
}

#' Train Minibatch
#'
#' @param trainer the Trainer instance on which to perform the operation
#' @param data
#' @param outputs
#' @param device - instance of DeviceDescriptor
#'
#' @export
train_minibatch <- function(trainer, data, outputs = NULL, device = NULL) {
	trainer$train_minibatch(data, outputs, device)
}
