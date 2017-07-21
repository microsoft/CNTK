#' Base64ImageDeserializer
#'
#' Configures the image reader that reads base64 encoded images and
#' corresponding labels from a file.
#'
#' Form: `[sequenceId <tab>] <numerical label (0-based class id)> <tab> <base64
#' encoded image>`
#'
#' Similarly to the ImageDeserializer, the sequenceId prefix is optional and
#' can be omitted.
#'
#' @param filename (str): file name of the input file dataset that contains
#' images and corresponding labels
#' @export
Base64ImageDeserializer <- function(filename, streams) {
	cntk$io$Base64ImageDeserializer(filename, streams)
}

#' CTFDeserializer
#'
#' Configures the CNTK text-format reader that reads text-based files.
#'
#' Form: `[Sequence_Id] (Sample)+` where `Sample=|Input_Name (Value )*`
#'
#' @param filename A string containing the path to the data location
#' @param streams A python dictionary-like object that contains a mapping from stream names to StreamDef objects. Each StreamDef object configures an input stream.
#' @return NULL
#' @references See also \url{https://www.cntk.ai/pythondocs/cntk.io.html?highlight=ctfdeserializer#cntk.io.CTFDeserializer}
#' @export
CTFDeserializer <- function(filename, streams) {
	cntk$io$CTFDeserializer(filename, streams)
}

#' HTKFeatureDeserializer
#'
#' Configures the HTK feature reader that reads speech data from scp files.
#'
#' @export
HTKFeatureDeserializer <- function(streams) {
	cntk$io$HTKFeatureDeserializer(streams)
}

#' Base64ImageDeserializer
#'
#' Configures an HTK label reader that reads speech HTK format MLF (Master
#' Label File)
#'
#' @export
HTKMLFDeserializer <- function(label_mapping_file, streams,
							   phoneBoundaries = FALSE) {
	cntk$io$HTKMLFDeserializer(
		label_mapping_file = label_mapping_file,
		streams = streams,
		phoneBoundaries = phoneBoundaries
	)
}

#' IO_INFINITELY_REPEAT
#'
#' Constant used to specify a minibatch scheduling unit to equal the size of
#' the full data sweep.
#'
#' @export
IO_INFINITELY_REPEAT <- 18446744L

io <- reticulate::import("cntk.io")

#' @export
IO_DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS <- io$DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS

#' ImageDeserializer
#'
#' Configures the image reader that reads images and corresponding labels from
#' a file.
#'
#' Form: `<full path to image> <tab> <numerical label (0-based class id)>` or
#' 'sequenceId <tab> path <tab> label`
#'
#' @export
ImageDeserializer <- function(filename, streams) {
	cntk$io$ImageDeserializer(filename, streams)
}

#' MinibatchData
#'
#' Holds a minibatch of input data. This is never directly created, but only
#' returned by `MinibatchSource` instances.
#'
#' ****** Attributes: ******
#'
#' data
#'
#' end_of_sweep
#'
#' is_sparse
#'
#' mask
#'
#' num_samples
#'
#' num_sequences
#'
#' shape
#'
#' ****** Associated Functions: ******
#'
#' mb_as_sequences(variable = NULL)
#'
#' @export
MinibatchData <- function(value, num_sequences, num_samples, sweep_end) {
	cntk$io$MinibatchData(
		value,
		to_int(num_sequences),
		to_int(num_samples),
		sweep_end
	)
}

#' mb_as_sequences
#'
#' Convert the value of this minibatch instance to a sequence of NumPy arrays
#' that have their masked entries removed.
#'
#' @return 	a list of matrices if dense, otherwise a SciPy CSR array
#'
#' @export
mb_as_sequences <- function(minibatch_data, variable = NULL) {
	minibatch_data$as_sequences(variable = variable)
}


#' MinibatchSource
#'
#' Source of minibatch data
#'
#' ****** Attributes: ******
#'
#' current_position
#'
#' is_distributed
#'
#' streams
#'
#' ****** Associated Functions: ******
#'
#' get_minibatch_checkpoint_state(minibatch_source)
#'
#' next_minibatch(minibatch_source, minibatch_size_in_samples,
#' input_map = NULL, device = NULL, num_data_partitions = NULL,
#' partition_index = NULL)
#'
#' restore_mb_from_checkpoint(minibatch_source, checkpoint)
#'
#' mb_stream_info(minibatch_source, name)
#' mb_stream_infos(minibatch_source)
#'
#' @export
MinibatchSource <- function(deserializers, max_samples = IO_INFINITELY_REPEAT,
							max_sweeps = IO_INFINITELY_REPEAT,
							randomization_window_in_chunks = IO_DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
							randomization_window_in_samples = 0,
							randomization_seed = 0,
							trace_level = get_logging_trace_level(),
							multithreaded_deserializer = NULL,
							frame_mode = FALSE, truncation_length = 0,
							randomize = TRUE) {
	# R cannot store the proper io.INFINITELY_REPEAT value,
	# so this painful check allows the default python value to be used
	if (max_sweeps != IO_INFINITELY_REPEAT) {
		return(cntk$io$MinibatchSource(
			deserializers,
			max_sweeps = to_int(max_sweeps),
			randomization_window_in_chunks = to_int(randomization_window_in_chunks),
			randomization_window_in_samples = to_int(randomization_window_in_samples),
			randomization_seed = to_int(randomization_seed),
			trace_level = trace_level,
			multithreaded_deserializer = multithreaded_deserializer,
			frame_mode = frame_mode,
			truncation_length = to_int(truncation_length),
			randomize = randomize
		))
	} else if (max_samples != IO_INFINITELY_REPEAT) {
		return(cntk$io$MinibatchSource(
			deserializers,
			max_samples = to_int(max_samples),
			randomization_window_in_chunks = to_int(randomization_window_in_chunks),
			randomization_window_in_samples = to_int(randomization_window_in_samples),
			randomization_seed = to_int(randomization_seed),
			trace_level = trace_level,
			multithreaded_deserializer = multithreaded_deserializer,
			frame_mode = frame_mode,
			truncation_length = to_int(truncation_length),
			randomize = randomize
		))
	}
	cntk$io$MinibatchSource(
		deserializers,
		randomization_window_in_chunks = to_int(randomization_window_in_chunks),
		randomization_window_in_samples = to_int(randomization_window_in_samples),
		randomization_seed = to_int(randomization_seed),
		trace_level = trace_level,
		multithreaded_deserializer = multithreaded_deserializer,
		frame_mode = frame_mode,
		truncation_length = to_int(truncation_length),
		randomize = randomize
	)
}


#' MinibatchSourceFromData
#'
#' This wraps in-memory data as a CNTK MinibatchSource object (aka “reader”),
#' used to feed the data into a TrainingSession.
#'
#' Use this if your data is small enough to be loaded into RAM in its entirety,
#' and the data is already sufficiently randomized.
#'
#' While CNTK allows user code to iterate through minibatches by itself and
#' feed data minibatch by minibatch through `train_minibatch()`, the standard
#' way is to iterate through data using a MinibatchSource object. For example,
#' the high-level TrainingSession interface, which manages a full training
#' including checkpointing and cross validation, operates on this level.
#'
#' A MinibatchSource created as a MinibatchSourceFromData linearly iterates
#' through the data provided by the caller as numpy arrays or
#' scipy.sparse.csr_matrix objects, without randomization. The data is not
#' copied, so if you want to modify the data while being read through a
#' MinibatchSourceFromData, please pass a copy.
#'
#'
#' @seealso \code{get_minibatch_checkpoint_state} \code{next_minibatch} \code{restore_mb_from_checkpoint} \code{mb_stream_infos}
#'
#' @export
MinibatchSourceFromData <- function(data_streams,
									max_samples = 1844674407379551615) {
	# use default if not provided (R cannot store as int)
	if (max_samples == max_samples) {
		return(cntk$io$MinibatchSourceFromData(
			data_streams
		))
	}
	cntk$io$MinibatchSourceFromData(
		data_streams,
		max_samples = to_int(max_samples)
	)
}

#' UserMinibatchSource
#'
#' Base class of all user minibatch sources.
#'
#' ****** Associated Functions: ******
#'
#' get_minibatch_checkpoint_state
#'
#' usermb_next_minibatch
#'
#' restore_mb_from_checkpoint
#'
#' mb_stream_info
#'
#' mb_stream_infos
#'
#' @export
UserMinibatchSource <- function() {
	cntk$io$UserMinibatchSource()
}

#' get_minibatch_checkpoint_state
#'
#' Returns a dictionary describing the current state of the minibatch source.
#' Needs to be overwritten if the state of the minibatch source needs to be
#' stored to and later restored from the checkpoint.
#'
#' @export
get_minibatch_checkpoint_state <- function(mb_source) {
	mb_source$get_checkpoint_state()
}

#' next_minibatch
#'
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


#' usermb_next_minibatch
#'
#' Function to be implemented by the user
#'
#' @export
usermb_next_minibatch <- function(usermb_source, num_samples,
								  num_workers, worker_rank, device = NULL) {
	usermb_source$next_minibatch(
		to_int(num_samples),
		to_int(num_workers),
		to_int(worker_rank),
		device = device
	)
}

#' restore_mb_from_checkpoint
#'
#' Sets the state of the checkpoint.
#'
#' @export
restore_mb_from_checkpoint <- function(mb_source, checkpoint) {
	mb_source$restore_from_checkpoint(checkpoint)
}

#' mb_stream_info
#'
#' Gets the description of the stream with given name. Throws an exception if
#' there are none or multiple streams with this same name.
#'
#' @export
mb_stream_info <- function(mb_source, name) {
	mb_source$stream_info(name)
}

#' mb_stream_infos
#'
#' Function to be implemented by the user.
#'
#' @export
mb_stream_infos <- function(mb_source) {
	mb_source$stream_infos()
}


#' StreamConfiguration
#'
#' Configuration of a stream in a text format reader.
#'
#' @export
StreamConfiguration <- function(name, dim, is_sparse = FALSE, stream_alias = '',
								defines_mb_size = FALSE) {
	cntk$io$StreamConfiguration(
		name,
		to_int(dim),
		is_sparse = is_sparse,
		stream_alias,
		defines_mb_size = defines_mb_size
	)
}

#' StreamDefs
#'
#' Alias for Record
#'
#' @export
StreamDefs <- function(...) {
	cntk$variables$Record(...)
}

#' StreamDef
#'
#' Configuration of a stream for use with the builtin Deserializers. The
#' meanings of some configuration keys have a mild dependency on the exact
#' deserializer, and certain keys are meaningless for certain deserializers.
#'
#' @param field string defining the name of the stream
#' @param shape integer defining the dimensions of the stream
#' @param is_sparse logical for whether the data is sparse (FALSE by default)
#' @param transforms list of transforms to be applied to the Deserializer
#' @param context vector of length two defining whther reading in HTK data, (only supported by `HTKFeatureDeserializer`)
#' @param scp list of `scp` files for HTK data
#' @param mlf list `mlf` files for HTK data
#' @param broadcast logical for whether the streams should be broadcast to the whole sequence
#' @param defines_mb_size logical for whether this stream defines minibatch size
#' @return A StreamDef object containing the stream dictionary
#' @references \url{https://www.cntk.ai/pythondocs/cntk.io.html#cntk.io.StreamDef}
#' @export
StreamDef <- function(field = NULL, shape = NULL, is_sparse = FALSE,
					  transforms = NULL, context = NULL, scp = NULL, mlf = NULL,
					  broadcast = NULL, defines_mb_size = FALSE) {
	cntk$io$StreamDef(
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

#' StreamInformation
#'
#' Stream information container that is used to describe streams when
#' implementing custom minibatch source through UserMinibatchSource.
#'
#' @export
StreamInformation <- function(name, stream_id, storage_format, shape) {
	cntk$io$StreamInformation(
		name,
		to_int(stream_id),
		storage_format,
		np$float32,
		to_int(shape)
	)
}

#' sequence_to_cntk_text_format
#'
#' Converts a list of NumPy arrays representing tensors of inputs into a format
#' that is readable by CTFDeserializer.
#'
#' @export
sequence_to_cntk_text_format <- function(seq_inx, alias_tensor_map) {
	cntk$io$sequence_to_cntk_text_format(seq_inx, alias_tensor_map)
}
