#' @param filename
#'
#' @param frequency
#' @param restore
#' @param preserve_all
#'
#' @export
CheckpointConfig <- function(filename, frequency = NULL, restore = TRUE,
							 preserve_all = FALSE) {
	cntk$train$training_session$CheckpointConfig(
		filename,
		frequency = to_int(frequency),
		restore = restore,
		preserve_all = preserve_all
	)
}

#' @param minibatch_source
#'
#' @param frequency
#' @param minibatch_size
#' @param callback
#' @param max_samples
#' @param model_inputs_to_streams
#' @param criterion
#' @param source
#' @param mb_size
#'
#' @export
CrossValidationConfig <- function(minibatch_source = NULL, frequency = NULL,
								  minibatch_size = 32, callback = NULL,
								  max_samples = NULL,
								  model_inputs_to_streams = NULL,
								  criterion = NULL, source = NULL,
								  mb_size = NULL) {
	cntk$train$training_session$CrossValidationConfig(
		minibatch_source = minibatch_source,
		frequency = to_int(frequency),
		minibatch_size = to_int(minibatch_source),
		callback = callback,
		max_samples = to_int(max_samples),
		model_inputs_to_streams = model_inputs_to_streams,
		criterion = criterion,
		mb_size = to_int(mb_size)
	)
}

#' @param minibatch_source
#'
#' @param minibatch_size
#' @param model_inputs_to_streams
#' @param criterion
#' @param source
#' @param mb_size
#'
#' @export
TestConfig <- function(minibatch_source = NULL, minibatch_size = 32,
					   model_inputs_to_streams = NULL, criterion = NULL,
					   source = NULL, mb_size = NULL) {
	cntk$train$training_session$TestConfig(
		minibatch_source = minibatch_source,
		minibatch_size = to_int(minibatch_size),
		model_inputs_to_streams = model_inputs_to_streams,
		criterion = criterion,
		source = source,
		mb_size = to_int(mb_size)
	)
}

#' @param trainer
#'
#' @param mb_source
#' @param mb_size
#' @param model_inputs_to_streams
#' @param max_samples
#' @param progress_frequency
#' @param checkpoint_config
#' @param cv_config
#' @param test_config
#'
#' @export
TrainingSession <- function(trainer, mb_source, mb_size,
							model_inputs_to_streams, max_samples,
							progress_frequency, checkpoint_config, cv_config,
							test_config) {
	cntk$train$training_session$TrainingSession(
		trainer,
		mb_source,
		to_int(mb_size),
		model_inputs_to_streams,
		to_int(max_samples),
		to_int(progress_frequency),
		checkpoint_config,
		cv_config,
		test_config
	)
}

#' @param sess
#'
#' @param index
#' @param average_error
#' @param num_samples
#' @param num_minibatches
#'
#' @export
on_train_cross_validation_end <- function(sess, index, average_error,
										  num_samples, num_minibatches) {
	sess$on_cross_validation_end(
		to_int(index),
		average_error,
		to_int(num_samples),
		to_int(num_minibatches)
	)
}

#' @param sess
#'
#' @param device
#'
#' @export
train_on_session <- function(sess, device = NULL) {
	sess$train(device = device)
}

#' @param schedule
#'
#' @param epoch_size
#'
#' @export
sess_minibatch_size_schedule <- function(schedule, epoch_size = 1) {
	cntk$train$training_session$minibatch_size_schedule(
		to_int(schedule),
		epoch_size = to_int(epoch_size)
	)
}

#' @param trainer
#'
#' @param mb_source
#' @param mb_size
#' @param model_inputs_to_streams
#' @param progress_frequency
#' @param max_samples
#' @param checkpoint_config
#' @param cv_config
#' @param test_config
#'
#' @export
sess_training_session <- function(trainer, mb_source, mb_size,
								  model_inputs_to_streams,
								  progress_frequency = NULL, max_samples = NULL,
								  checkpoint_config = NULL, cv_config = NULL,
								  test_config = NULL) {
	cntk$train$training_session$training_session(
		trainer,
		mb_source,
		mb_size,
		model_inputs_to_streams,
		progress_frequency = to_int(progress_frequency),
		max_samples = to_int(max_samples),
		checkpoint_config = checkpoint_config,
		cv_config = cv_config,
		test_config = test_config
	)
}
