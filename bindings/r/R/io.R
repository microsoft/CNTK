io <- reticulate::import("cntk.io")

#' @export
CTFDeserializer <- io$CTFDeserializer

#' @export
MinibatchSource <- io$MinibatchSource

#' @export
next_minibatch <- function(minibatch_source, minibatch_size_in_samples,
						   input_map = NULL, device = NULL,
						   num_data_partitions = NULL, partition_index = NULL) {
	minibatch_source$next_minibatch(
		to_int(minibatch_size_in_samples),
		input_map = input_map,
		device = device,
		num_data_partitions = to_int(num_data_partitions),
		partition_index = to_int(partition_index)
	)
}

#' @export
StreamDefs <- io$StreamDefs

#' @export
StreamDef <- function(field = NULL, shape = NULL, is_sparse = FALSE,
					  transforms = NULL, context = NULL, scp = NULL, mlf = NULL,
					  broadcast = NULL, defines_mb_size = FALSE) {
	io$StreamDef(
		field = field,
		shape = to_int(shape),
		is_sparse = is_sparse,
		transforms = transforms,
		context = context,
		scp = scp,
		mlf = mlf,
		broadcast = broadcast,
		defines_mb_size = defines_mb_size
	)
}

#' @export
IO_INFINITELY_REPEAT <- io$INFINITELY_REPEAT
