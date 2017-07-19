train <- reticulate::import("cntk.train.distributed")

#' @export
Communicator <- function(...) {
	cntk$train$distributed$Communicator(...)
}

#' @export
comm_barrier <- function(communicator) {
	communicator$barrier()
}

#' @export
comm_current_worker <- function(communicator) {
	communicator$current_worker()
}

#' @export
comm_finalize <- function() {
	cntk$train$distributed$Communicator$finalize()
}

#' @export
comm_is_main <- function(communicator) {
	communicator$is_main()
}

#' @export
comm_num_workers <- function() {
	cntk$train$distributed$Communicator$num_workers()
}

#' @export
comm_rank <- function() {
	cntk$train$distributed$Communicator$rank()
}

#' @export
comm_workers <- function(communicator) {
	communicator$workers()
}

#' @export
DistributedLearner <- function(...) {
	cntk$train$distributed$DistributedLearner(...)
}

#' @export
get_communicator <- function(distributed_learner) {
	distributed_learner$communicator()
}

#' @export
WorkerDescriptor <- function() {
	cntk$train$distributed$WorkerDescriptor()
}

#' @export
block_momentum_distributed_learner <- function(learner, block_size,
											   block_momentum_as_time_constant = NULL,
											   use_nestrov_momentum = TRUE,
											   reset_sgd_momentum_after_aggregation = TRUE,
											   block_learning_rate = 1,
											   distributed_after = 0) {
	cntk$train$distributed$block_momentum_distributed_learner(
		learner,
		to_int(block_size),
		block_momentum_as_time_constant = block_momentum_as_time_constant,
		use_nestrov_momentum = use_nestrov_momentum,
		reset_sgd_momentum_after_aggregation = reset_sgd_momentum_after_aggregation,
		block_learning_rate = block_learning_rate,
		distributed_after = to_int(distributed_after)
	)
}

#' @export
data_parallel_distributed_learner <- function(learner, distributed_after = 0,
											  num_quantization_bits = 32,
											  async_parameter_update = FALSE) {
	cntk$train$distributed$data_parallel_distributed_learner(
		learner,
		distributed_after = to_int(distributed_after),
		num_quantization_bits = to_int(num_quantization_bits),
		use_async_parameter_update = async_parameter_update
	)
}

#' @export
mpi_communicator <- function() {
	cntk$train$distributed$mpi_communicator()
}
