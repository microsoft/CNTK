#' New Communicator
#'
#' A communicator interface exposing communication primitives that serve as
#' building blocks for distributed training.
#'
#' @param ... - constructor args
#'
#' @export
Communicator <- function(...) {
	cntk$train$distributed$Communicator(...)
}

#' Communicator Barrier
#'
#' @param communicator - communicator instance on which to call operation
#'
#' @export
comm_barrier <- function(communicator) {
	communicator$barrier()
}

#' Communicator Current Worker
#'
#' @param communicator - communicator instance on which to call operation
#'
#' @export
comm_current_worker <- function(communicator) {
	communicator$current_worker()
}

#' Communicator Finalize
#'
#' @export
comm_finalize <- function() {
	cntk$train$distributed$Communicator$finalize()
}

#' Communicator Is On Main Node
#'
#' @param communicator - communicator instance on which to call operation
#'
#' @export
comm_is_main <- function(communicator) {
	communicator$is_main()
}

#' Number of Communicator Nodes
#'
#' @export
comm_num_workers <- function() {
	cntk$train$distributed$Communicator$num_workers()
}

#' Communicator Rank
#'
#' @export
comm_rank <- function() {
	cntk$train$distributed$Communicator$rank()
}

#' Communicator Workers
#'
#' @param communicator - communicator instance on which to call operation
#'
#' @export
comm_workers <- function(communicator) {
	communicator$workers()
}

#' Distributed Learner
#'
#' A distributed learner that handles data like gradients/momentums across multiple MPI workers
#'
#' ****** Properties: ******
#'
#' total_number_of_samples_seen
#'
#' @param ...
#'
#' @export
DistributedLearner <- function(...) {
	cntk$train$distributed$DistributedLearner(...)
}

#' Get Distributed Communicator
#'
#' @param distributed_learner
#'
#' @export
get_communicator <- function(distributed_learner) {
	distributed_learner$communicator()
}

#' New Distributed Worker Descriptor
#'
#' Distributed worker descriptor, returned by Communicator instance.
#'
#' ****** Properties: ******
#'
#' global_rank
#'
#' host_id
#'
#' @export
WorkerDescriptor <- function() {
	cntk$train$distributed$WorkerDescriptor()
}

#' Block Momentum Distributed Learner
#'
#' @param learner
#' @param block_size
#' @param block_momentum_as_time_constant
#' @param use_nestrov_momentum
#' @param reset_sgd_momentum_after_aggregation
#' @param block_learning_rate
#' @param distributed_after
#'
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

#' New Data Parallel Distributed Learner
#'
#' @param learner
#' @param distributed_after
#' @param num_quantization_bits
#' @param async_parameter_update
#'
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

#' New MPI Communicator
#'
#' @export
mpi_communicator <- function() {
	cntk$train$distributed$mpi_communicator()
}
