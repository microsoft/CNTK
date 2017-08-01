#' New Checkpoint Config
#'
#' @param filename (str): checkpoint file name.
#' @param frequency (int): checkpointing period (number samples between checkpoints). If `NULL`, no checkpointing takes place.
#' If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
#' @param restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
#' @param preserve_all (bool): saves all checkpoints, using ``filename`` as prefix and checkpoint index as a suffix.
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

#' New Cross Validation Configuration
#'
#' A cross validation configuration for the training session.
#'
#' @param minibatch_source (MinibatchSource): minibatch source used for cross validation
#' @param frequency (int): frequency in samples for cross validation
#' If NULL or ``sys.maxsize``, a single cross validation is performed at the end of training.
#' @param minibatch_size(int or minibatch_size_schedule, defaults to 32): minibatch schedule for cross validation
#' @param callback (func (index, average_error, cv_num_samples, cv_num_minibatches)): Callback that will
#' be called with frequency which can implement custom cross validation logic,
#' returns FALSE if training should be stopped.
#' @param max_samples (int, default NULL): number of samples to perform
#' cross-validation on. If NULL, all samples are taken.
#' @param model_inputs_to_streams (dict): mapping between input variables and input streams
#' If NULL, the mapping provided to the training session constructor is used.
#' Don't specify this if `minibatch_source` is a tuple of numpy/scipy arrays.
#' @param criterion (): criterion function): criterion function.
#' Must be specified if `minibatch_source` is a tuple of numpy/scipy arrays.
#' @param source (MinibatchSource): DEPRECATED, use minibatch_source instead
#'
#' @export
CrossValidationConfig <- function(minibatch_source = NULL, frequency = NULL,
								  minibatch_size = 32, callback = NULL,
								  max_samples = NULL,
								  model_inputs_to_streams = NULL,
								  criterion = NULL, source = NULL) {
	cntk$train$training_session$CrossValidationConfig(
		minibatch_source = minibatch_source,
		frequency = to_int(frequency),
		minibatch_size = to_int(minibatch_source),
		callback = callback,
		max_samples = to_int(max_samples),
		model_inputs_to_streams = model_inputs_to_streams,
		criterion = criterion
	)
}

#' New Test Configuration
#'
#' A test configuration for the training session.
#'
#' @param minibatch_source (MinibatchSource): minibatch source used for cross validation
#' @param minibatch_size(int or minibatch_size_schedule, defaults to 32): minibatch schedule for cross validation
#' @param model_inputs_to_streams (dict): mapping between input variables and input streams
#' If NULL, the mapping provided to the training session constructor is used.
#' Don't specify this if `minibatch_source` is a tuple of numpy/scipy arrays.
#' @param criterion (): criterion function): criterion function.
#' Must be specified if `minibatch_source` is a tuple of numpy/scipy arrays.
#' @param source (MinibatchSource): DEPRECATED, use minibatch_source instead
#' @param mb_size(int or minibatch_size_schedule, defaults to 32): DEPRECATED, use minibatch_size instead
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

#' New Training Configuration
#'
#' The instance of the class should be created by using training_session()
#' function.  A training session trains a model using the specified trainer and
#' configs. Different aspects of training such as data sources, checkpointing,
#' cross validation, progress printing can be configured using the
#' corresponding config classes.
#'
#' ****** Associated Functions: ******
#'
#' on_train_cross_validation_end
#'
#' train_on_session
#'
#' @param trainer (Trainer): trainer
#' @param mb_source (MinibatchSource): minibatch source used for training
#' @param mb_size (minibatch_size_schedule or int): minibatch size schedule for training
#' @param model_inputs_to_streams (dict): mapping between input variables and input streams
#' @param max_samples (int): maximum number of samples used for training
#' @param progress_frequency (int): frequency in samples for aggregated progress printing
#' @param checkpoint_config (CheckpointConfig): checkpoint configuration
#' @param cv_config (CrossValidationConfig): cross validation configuration
#' @param test_config (TestConfig): test configuration
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

#' On Training Cross Validation End
#'
#' Callback that gets executed at the end of cross validation.
#'
#' @param sess Session instance on which to perform the operation
#' @param index (int): index of the current callback.
#' @param average_error (float): average error for the cross validation
#' @param num_samples (int): number of samples in cross validation
#' @param num_minibatches (int): number of minibatch in cross validation
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

#' Train On TrainingSession
#'
#' @param sess Session instance on which to perform the operation
#' @param device instance of DeviceDescriptor
#'
#' @export
train_on_session <- function(sess, device = NULL) {
	sess$train(device = device)
}

#' Create a Minibatch Size Schedule
#'
#' Creates a minibatch size schedule.
#'
#' @param schedule (int or list): if integer, this minibatch size will be used for the whole training.
#' In case of list of integers, the elements are used as the values for ``epoch_size`` samples.
#' If list contains pair, the second element is used as a value for (``epoch_size`` x first element) samples
#' @param epoch_size (int): number of samples as a scheduling unit.
#'
#' @export
sess_minibatch_size_schedule <- function(schedule, epoch_size = 1) {
	cntk$train$training_session$minibatch_size_schedule(
		to_int(schedule),
		epoch_size = to_int(epoch_size)
	)
}

#' Create Training Session Object
#'
#' A factory function to create a training session object.
#'
#' @param trainer (Trainer): trainer
#' @param mb_source (MinibatchSource): minibatch source used for training
#' @param mb_size (minibatch_size_schedule): minibatch schedule for training
#' @param model_inputs_to_streams (dict): mapping between input variables and input streams
#' @param progress_frequency (int): frequency in samples for aggregated progress printing
#' @param max_samples (int): maximum number of samples used for training
#' @param checkpoint_config (~CheckpointConfig): checkpoint configuration
#' @param cv_config (~CrossValidationConfig): cross validation configuration
#' @param test_config (~TestConfig): test configuration
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
