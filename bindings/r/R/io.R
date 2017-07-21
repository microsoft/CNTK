io <- reticulate::import("cntk.io")

#' @export
Base64ImageDeserializer <- io$Base64ImageDeserializer

#' @export
CTFDeserializer <- io$CTFDeserializer

#' @export
HTKFeatureDeserializer <- io$HTKFeatureDeserializer

#' @export
HTKMLFDeserializer <- io$HTKMLFDeserializer

#' @export
IO_INFINITELY_REPEAT <- to_int(io$INFINITELY_REPEAT)

#' @export
IO_DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS <- to_int(io$DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS)

#' @export
ImageDeserializer <- io$ImageDeserializer

#' a bunch of properties
#' @export
MinibatchData <- function(value, num_sequences, num_samples, sweep_end) {
	io$MinibatchData(
		value,
		to_int(num_sequences),
		to_int(num_samples),
		sweep_end
	)
}

#' @export
md_as_sequences <- function(minibatch_data, variable = NULL) {
	minibatch_data$as_sequences(variable = variable)
}


#' @export
MinibatchSource <- function(deserializers, max_samples = IO_INFINITELY_REPEAT,
							max_sweeps = IO_INFINITELY_REPEAT,
							randomization_window_in_chunks = IO_DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
							randomization_window_in_samples = 0,
							randomization_seed = 0,
							trace_level = get_trace_level(),
							multithreaded_deserializer = NULL,
							frame_mode = FALSE, truncation_length = 0,
							randomize = TRUE) {
	io$MinibatchSource(
		deserializers,
		max_samples = max_samples,
		max_sweeps = max_sweeps,
		randomization_window_in_chunks = randomization_window_in_chunks,
		randomization_window_in_samples = randomization_window_in_samples,
		randomization_seed = randomization_seed,
		trace_level = trace_level,
		multithreaded_deserializer = multithreaded_deserializer,
		frame_mode = frame_mode,
		truncation_length = truncation_length,
		randomize = randomize
	)
}

#' @export
MinibatchSourceFromData <- function(data_streams,
									max_samples = 1844674407379551615) {
	io$MinibatchSourceFromData(
		data_streams,
		max_samples = to_int(max_samples)
	)
}

#' @export
UserMinibatchSource <- io$UserMinibatchSource

#' @export
get_minibatch_checkpoint_state <- function(mb_source) {
	mb_source$get_checkpoint_state()
}

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
restore_mb_from_checkpoint <- function(mb_source, checkpoint) {
	mb_source$restore_from_checkpoint(checkpoint)
}

# only 2/3
#' @export
mb_stream_info <- function(mb_source, name) {
	mb$source$stream_info(name)
}

#' @export
mb_stream_infos <- function(mb_source) {
	mb$source$stream_infos()
}


#' @export
StreamConfiguration <- function(name, dim, is_sparse = FALSE, stream_alias = '',
								defines_mb_size = FALSE) {
	io$StreamConfiguration(
		name,
		to_int(dim),
		is_sparse = is_sparse,
		stream_alias,
		defines_mb_size = defines_mb_size
	)
}



#' @export
StreamDefs <- Record

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
StreamInformation <- function(name, stream_id, storage_format, shape) {
	io$StreamInformation(
		name,
		to_int(stream_id),
		storage_format,
		np$float32,
		to_int(shape)
	)
}

#' @export
sequence_to_cntk_text_format <- function(seq_inx, alias_tensor_map) {
	io$sequence_to_cntk_text_format(seq_inx, alias_tensor_map)
}
