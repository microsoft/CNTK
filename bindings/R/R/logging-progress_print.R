#' @export
ProgressPrinter <- function(freq = NULL, first = 0, tag = '',
							log_to_file = NULL, rank = NULL,
							gen_heartbeat = FALSE, num_epochs = NULL,
							test_freq = NULL, test_first = 0,
							metric_is_pct = TRUE, distributed_freq = NULL,
							distributed_first = 0) {
	cntk$logging$ProgressPrinter(
		freq = to_int(freq),
		first = to_int(first),
		tag = tag,
		log_to_file = log_to_file,
		rank = to_int(rank),
		gen_heartbeat = gen_heartbeat,
		num_epochs = to_int(num_epochs),
		test_freq = to_int(test_freq),
		test_first = to_int(test_first),
		metric_is_pct = metric_is_pct,
		distributed_freq = to_int(distributed_freq),
		distributed_first = to_int(distributed_first)
	)
}

#' @export
printer_end_progress_print <- function(printer, msg = '') {
	printer$end_progress_print(msg = msg)
}

#' @export
printer_log <- function(printer, message) {
	printer$log(message)
}

#' @export
printer_on_training_update_end <- function(printer) {
	printer$on_training_update_end()
}

#' @export
printer_on_write_distributed_sync_update <- function(printer, samples,
													 updates,
													 aggregate_metric) {
	printer$on_write_distributed_sync_update(
		samples,
		updates,
		aggregate_metric
	)
}

#' any writer
#' @export
printer_on_write_test_summary <- function(printer, samples, updates,
										  summaries, aggregate_metric,
										  elapsed_milliseconds) {
	printer$on_write_test_summary(
		samples,
		updates,
		summaries,
		aggregate_metric,
		elapsed_milliseconds
	)
}

#' any writer
#' @export
printer_on_write_test_update <- function(printer, samples, updates,
										 aggregate_metric) {
	printer$on_write_test_update(
		samples,
		updates,
		aggregate_metric
	)
}

#' any writer
#' @export
printer_on_write_training_summary <- function(printer, samples, updates,
											  summaries, aggregate_loss,
											  aggregate_metric,
											  elapsed_milliseconds) {
	printer$on_write_training_summary(
		samples,
		updates,
		summaries,
		aggregate_loss,
		aggregate_metric,
		elapsed_milliseconds
	)
}

#' any writer
#' @export
printer_on_write_training_update <- function(printer, samples, updates,
											 aggregate_loss, aggregate_metric) {
	printer$on_write_training_update(
		samples,
		updates,
		aggregate_loss,
		aggregate_metric
	)
}

#' @export
printer_write <- function(printer, key, value) {
	printer$write(
		key,
		value
	)
}


#' @export
TensorBoardProgressWriter <- function(freq = NULL, log_dir = '.', rank = NULL,
									  model = NULL) {
	cntk$logging$progress_print$TensorBoardProgressWriter(
		freq = to_int(freq),
		log_dir = log_dir,
		rank = to_int(rank),
		model = model
	)
}

#' @export
tensorboard_close <- function(tensorboard_writer) {
	tensorboard_writer$close()
}

#' @export
tensorboard_flush <- function(tensorboard_writer) {
	tensorboard_writer$flush()
}

#' @export
tensorboard_write_value <- function(tensorboard_writer, name, value, step) {
	tensorboard_writer$write_value(
		name,
		value,
		to_int(step)
	)
}


#' @export
TrainingSummaryProgressCallback <- function(epoch_size, callback) {
	cntk$logging$TrainingSummaryProgressCallback(
		to_int(epoch_size),
		callback
	)
}


#' @export
log_number_of_parameters <- function(model, trace_level = 0) {
	cntk$logging$log_number_of_parameters(
		model,
		trace_level = trace_level
	)
}
