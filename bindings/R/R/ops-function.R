#' CloneMethod
#'
#' Object descibing different ways how clone() works.
#'
#' ****** Attributes: ******
#'
#' clone = 'clone' - New learnable parameters are created and initialized with
#' the current values of the corresponding parameters of the Function being
#' cloned
#'
#' freeze = 'freeze' - Parameters are cloned and made immutable; i.e. Constants
#' in the new clone (e.g. for use as a fixed feature extractor)
#'
#' share = 'share' - Parameters are shared between the Function being cloned
#' and the new clone
#'
#' @param value
#'
#' @export
CloneMethod <- function(value) {
	reticulate::py_get_attr(cntk$ops$functions$CloneMethod, value)
}

#' CNTK Function
#'
#' Base class of all primitive tensor operators.
#'
#' If it has only one output, one can invoke Variable methods on it, which it
#' will relay to its only output.
#'
#' Function objects can also be constructed directly from a Python lambda, by
#' means of the @Function decorator. The Function‘s input signature is defined
#' by the lambda.
#'
#' ****** Attributes: ******
#'
#' arguments - List of all input variables of the Function that are not of type
#' Parameter or Constant
#'
#' attributes - List of the attributes of the function
#'
#' block_arguments_mapping - The mapping from the arguments of the composite
#' underlying this block function to the Variables that they are bound to in
#' the outer graph of Functions that this block Function is part of.
#'
#' block_root - The root of the Function graph underlying this block Function.
#' Throws an exception if this is not a block Function.
#'
#' constants - List of all Constant variables of this Function
#'
#' inputs - List of variables that are inputs of this function. Note that
#' ‘inputs’ here denotes all Variables that feed into this Function including
#' any Parameter/Constant Variables that are children of this Function.
#'
#' is_block - Boolean indicating if this Function is a block function which is
#' basically a composite encapsulated as an opaque block which appears as a
#' primitive during traversing the graph of Functions that this block is part
#' of.
#'
#' is_composite - Boolean indicating if this Function is a composite Function.
#' A composite Function is a Function that is composed of primitive Functions.
#'
#' is_primitive - Boolean indicating if this Function is a primitive Function.
#' A primitive Function is the lowest level building block for composite
#' Function graphs and is either a CNTK built-in operator, a composite Function
#' encapsulated as a Block or a user-defined Function
#'
#' name - Name of the Function
#'
#' op_name - Name of the operation that this Function performs
#'
#' output - The single output variable if there is only one, or raises an
#' exception.
#'
#' outputs - List consisting of all output variables of this Function.
#'
#' parameters - List of all parameter variables of this Function.
#'
#' placeholders - List of all placeholders variables of this Function.
#'
#' root_function - The primitive function at the root of the graph of functions
#' underlying this function.
#'
#' signature - Signature of a Function. This is the $arguments list without
#' placeholders that belong to an outer, not yet completed @Function def.
#'
#' type - Get type of a Function's output
#'
#' uid - The internally generated unique name of the Function
#'
#' ****** Associated Functions: ******
#'
#' func_backward
#'
#' func_clone
#'
#' func_eval
#'
#' func_find_all_with_name
#'
#' func_find_by_name
#'
#' func_forward
#'
#' func_grad
#'
#' func_load
#'
#' register_udf_deserialize_callback
#'
#' func_replace_placeholder
#'
#' func_replace_placeholders
#'
#' func_restore
#'
#' func_save
#'
#' func_set_attribute
#'
#' func_test
#'
#' func_train
#'
#' @param ...
#'
#' @export
Function <- function(...) {
	cntk$ops$functions$Function(...)
}

#' Propogate Function Backward
#'
#' Backpropagates supplied root_gradients for one or more of the output
#' variables of the Function, to calculate gradients with respect to variables.
#' Formally, multiplies the values of root_gradients by the Jacobian of the
#' Function and returns the subset of the output that corresponds to variables.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param state
#' @param root_gradients
#' @param variables
#' @param as_matrix - whether to return as an R matrix. Defualt TRUE. Otherwise
#' returns as Python CNTK value which avoids costly conversions
#'
#' @export
func_backward <- function(func, state, root_gradients, variables,
						  as_matrix = TRUE) {
	func$backward(
		state,
		root_gradients,
		variables,
		as_numpy = as_matrix
	)
}

#' Clone Function
#'
#' Clones the function. The parameters of the Function are either cloned,
#' shared or frozen as specified by the method argument and any variable
#' substitutions requested are applied in the cloned Function instance.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param method
#' @param substitutions
#'
#' @export
func_clone <- function(func, method, substitutions = NULL) {
	func$clone(
		method,
		substitutions = substitutions
	)
}

#' Evaluate Function
#'
#' Evaluate the Function’s outputs using the specified arguments as input.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param arguments
#' @param outputs
#' @param device - instance of DeviceDescriptor
#' @param as_matrix - whether to return as an R matrix. Defualt TRUE. Otherwise
#' returns as Python CNTK value which avoids costly conversions
#'
#' @export
func_eval <- function(func, arguments = NULL, outputs = NULL, device = NULL,
					  as_matrix = TRUE) {
	func$eval(
		arguments = arguments,
		outputs = outputs,
		device = device,
		as_numpy = as_matrix
	)
}

#' Find All Functions With Name
#'
#' Returns a list of primitive function with name in the graph starting from
#' this node. Throws an exception if name occurs multiple times. If you expect
#' only one function to be returned, use find_by_name().
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param name
#' @param depth
#'
#' @export
func_find_all_with_name <- function(func, name, depth = 0) {
	func$find_all_with_name(name, depth = to_int(depth))
}

#' Find Function By Name
#'
#' Returns a primitive function with name in the graph starting from this node.
#' Throws an exception if name occurs multiple times. If you expect multiple
#' functions to be returned, use find_all_with_name().
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param name
#' @param depth
#'
#' @export
func_find_by_name <- function(func, name, depth = 0) {
	func$find_by_name(name, depth = to_int(depth))
}

#' Compute Function Forward
#'
#' Computes the values of speficied variables in outputs, using values provided
#' in arguments that correspond to each input Variable of the function (i.e.
#' those that have is_input = True).
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param arguments
#' @param outputs
#' @param keep_for_backward
#' @param device - instance of DeviceDescriptor
#' @param as_matrix - whether to return as an R matrix. Defualt TRUE. Otherwise
#' returns as Python CNTK value which avoids costly conversions
#'
#' @export
func_forward <- function(func, arguments, outputs = NULL,
						 keep_for_backward = NULL, device = NULL,
						 as_matrix = TRUE) {
	func$forward(
		arguments,
		outputs = outputs,
		keep_for_backward = keep_for_backward,
		device = device,
		as_numpy = as_matrix
	)
}

#' Compute Function Gradient
#'
#' Computes the gradient of this Function at location at with respect to wrt.
#' The Function must have a single output.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param at - mapping of the Function’s arguments to values
#' @param wrt - list of Variables with respect to which the gradient will be
#' computed. If omitted, the gradients with respect to all arguments of this
#' Function that need gradient will be computed.
#' @param outputs
#' @param device - instance of DeviceDescriptor
#' @param as_matrix - whether to return as an R matrix. Defualt TRUE. Otherwise
#' returns as Python CNTK value which avoids costly conversions
#' @param grad_root - specify the root of gradients calculation. If not
#' specified, the output of this function will be used as gradient root.
#'
#' @export
func_grad <- function(func, at, wrt = NULL, outputs = NULL, device = NULL,
					  as_matrix = TRUE, grad_root = NULL) {
	func$grad(
		at,
		wrt = wrt,
		outputs = outputs,
		device = device,
		as_numpy = as_matrix,
		grad_root = grad_root
	)
}

#' Load Function Model
#'
#' Load the model, that has been saved using save().
#'
#' @param model
#' @param device - instance of DeviceDescriptor
#'
#' @export
func_load <- function(model, device = NULL) {
	cntk$ops$functions$Function$load(
		model,
		device = device
	)
}

#' Register UDF Deserialize Callback
#'
#' Register a callback function to be invoked when deserializing a user-
#' defined function with the corresponding op name.
#'
#' When loading a model, CNTK will try to automatically reconstruct any
#' (non-native) user-defined functions by invoking a static deserialize()
#' method of the corresponding UserFunction sub-class. This method allows to
#' override default UDF deserialization behavior by specifying a user- defined
#' function op name and the corresponding callback that should be invoked
#' instead of the deserialize method.
#'
#' @param op_name
#' @param callback
#'
#' @export
register_udf_deserialize_callback <- function(op_name, callback) {
	cntk$ops$functions$Function$register_udf_deserialize_callback(
		op_name,
		callback
	)
}

#' Replace Function Placeholder
#'
#' In-place replace the only placeholder in the function graph with the
#' specified substitution.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param substitution - `Variable` that will replace the placeholder
#'
#' @export
func_replace_placeholder <- function(func, substitution) {
	func$replace_placeholder(substitution)
}

#' Replace Function Placeholders
#'
#' In-place replace specified placeholders in the Function graph with the
#' specified replacements in the map.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param substitutions - dict mapping placeholders to variables
#'
#' @export
func_replace_placeholders <- function(func, substitutions) {
	func$replace_placeholders(substitutions)
}

#' Restore Function
#'
#' Restore model parameters from saved model file.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param filename
#'
#' @export
func_restore <- function(func, filename) {
	func$restore(filename)
}

#' Save Function
#'
#' Save this function graph into a model file using protobuf-based serialization.
#'
#' Use comm_is_main() to gate your call to save() in distributed environment.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param filename
#'
#' @export
func_save <- function(func, filename) {
	func$restore(filename)
}

#' Set Function Attribute
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param name
#' @param value
#'
#' @export
func_set_attribute <- function(func, name, value) {
	if (name == 'rngSeed') {
		value = to_int(value)
	}
	func$set_attribute(name, value)
}

#' Test Function Model
#'
#' Measures the performance of a model, given by its criterion function, in the
#' form of average metric value (or loss if model has only one output) on a set
#' of data.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param minibatch_source - minibatch source for the test data
#' @param minibatch_size (minibatch_size_schedule or int) – minibatch size for
#' evaluation
#' @param streams (list) - the streams of the minibatch_source in argument
#' order
#' @param model_inputs_to_streams (dict or named list) - mapping between input
#' variables and #' input streams
#' @param callbacks  (progress writer or list of them) – optionally, list of
#' progress writers from cntk.logging to automatically track training progress.
#'
#' @export
func_test <- function(func, minibatch_source, minibatch_size = 32,
					  streams = NULL, model_inputs_to_streams = NULL,
					  callbacks = NULL) {
	func$test(
		minibatch_source,
		minibatch_size = to_int(minibatch_size),
		streams = streams,
		model_inputs_to_streams = model_inputs_to_streams,
		callbacks = callbacks
	)
}

#' Train Function Model
#'
#' Trains a model, given by its criterion function, using the specified
#' training parameters and configs. Different aspects of training such as data
#' sources, checkpointing, cross validation, progress printing can be
#' configured using the corresponding config classes.
#'
#' The input data can be specified as a data reader (MinibatchSource) for large
#' corpora; or directly as numpy/scipy arrays if the data is so small that it
#' is feasible to keep it all in RAM.
#'
#' Data is processed in minibatches. The minibatch size defaults to 32, which
#' is a choice that commonly works well. However, for maximum efficiency, we
#' recommend to experiment with minibatch sizes and choose the largest that
#' converges well and does not exceed the GPU RAM. This is particularly
#' important for distributed training, where often, the minibatch size can be
#' increased throughout the training, which reduces data bandwidth and thus
#' speeds up parallel training.
#'
#' If input data is given through a data reader (as opposed to directly as a
#' numpy/scipy array), the user must also specify the epoch size. This is
#' because data readers are used for large corpora, and the traditional
#' definition of epoch size as number of samples in the corpus is not very
#' relevant. Instead, CNTK really means the number of samples between summary
#' actions, such as printing training progress, adjusting the learning rate,
#' and/or checkpointing the model.
#'
#' The function returns an object that contains these members: epoch_summaries
#' is a list that contains the progression of epoch loss (.loss) and metric
#' (.metric) values and the corresponding number of labels (.samples) that they
#' were averaged over. This is the same value that a progress printer would
#' print as epoch summaries. updates is a similar list with the more
#' fine-grained minibatch updates. If a TestConfig was specified, then
#' test_summary is the metric and sample count on the specified test set for
#' the final model.
#'
#' A number of callback mechanisms can optionally be specified as a list as
#' callbacks. CNTK has a fixed set of callback types, and only those types are
#' allowed in the callbacks list: An object of type ProgressWriter from
#' cntk.logging is used for progress logging; a CheckpointConfig configures the
#' checkpointing mechanism, which keeps copies of models at regular intervals
#' and allows to seamlessly restart from a last checkpoint; a TestConfig allows
#' to specify a test set that is evaluated at the end of the training; and a
#' CrossValidationConfig specifies a user callback that can be used to adjust
#' learning hyper-parameters or to denote to stop training, optionally based on
#' a separate cross-validation data set.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param minibatch_source (MinibatchSource or list of matrices) –
#' data source used for training. For large data, use a MinibatchSource. For
#' small data, pass a list of matrices. The number of streams/arrays
#' must match the number of arguments of self.
#' @param minibatch_size (int or minibatch_size_schedule, defaults to 32) –
#' minibatch size (or schedule) for training
#' @param streams (list) – (only if minibatch_source is a data reader)
#' the streams of the minibatch_source in argument order. Not to be given if
#' minibatch_source is specified as numpy/scipy arrays rather than a data
#' reader.
#' @param model_inputs_to_streams (dict) – alternative to streams, specifying
#' the mapping as a map from input variables to streams
#' @param parameter_learners (list) – list of learners
#' @param callbacks - list of callback objects, which can be of type
#' ProgressWriter (for logging), CheckpointConfig (for #' check-pointing),
#' TestConfig (for automatic final evaluation on a test set), #' and
#' CrossValidationConfig (for cross-validation based training control).
#' @param progress_frequency (int) – frequency in samples for aggregated
#' progress printing. Defaults to epoch_size if given, or None otherwise
#' @param max_epochs (int, defaults to 1) – maximum number of samples used for
#' training; requires epoch_size
#' @param epoch_size (int) – in CNTK, epoch size means the number of samples
#' between outputting summary information and/or checkpointing. This must be
#' specified unless the user directly passes numpy/scipy arrays for the
#' minibatch_source.
#' @param max_samples (int) – maximum number of samples used for training;
#' mutually exclusive with max_epochs
#'
#' @export
func_train <- function(func, minibatch_source, minibatch_size = 32,
					   streams = NULL, model_inputs_to_streams = NULL,
					   parameter_learners = c(), callbacks = c(),
					   progress_frequency = NULL, max_epochs = NULL,
					   epoch_size = NULL, max_samples = NULL) {
	func$train(
		minibatch_source,
		minibatch_size = to_int(minibatch_size),
		streams = streams,
		model_inputs_to_streams = model_inputs_to_streams,
		parameter_learners = parameter_learners,
		callbacks = callbacks,
		progress_frequency = to_int(progress_frequency),
		max_epochs = to_int(max_epochs),
		epoch_size = to_int(epoch_size),
		max_samples = to_int(max_samples)
	)
}


#' User Function
#'
#' Base class of all user extension functions.
#'
#' If it has only one output, one can invoke Variable methods on it, which it
#' will relay to its only output.
#'
#' See ?Function for more info
#'
#' ****** Special Functions: ******
#'
#' userfunc_clone
#'
#' userfunc_deserialize
#'
#' userfunc_infer_outputs
#'
#' userfunc_serialize
#'
#' @param inputs
#' @param as_matrix - whether to return as an R matrix. Defualt TRUE. Otherwise
#' returns as Python CNTK value which avoids costly conversions
#' @param name
#'
#' @export
UserFunction <- function(inputs, as_matrix = TRUE, name = '') {
	cntk$ops$functions$UserFunction(
		inputs,
		as_numpy = as_matrix,
		name = name
	)
}

#' Clone UserFunction
#'
#' Creates a clone of this user-defined function.
#'
#' It assumes that the constructor signature of the user’s implementation of
#' the user function takes the inputs as individual arguments followed by the
#' operator name. If the signature is different, then this method needs to be
#' overriden.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#' @param cloned_inputs
#'
#' @export
userfunc_clone <- function(func, cloned_inputs) {
	func$clone(cloned_inputs)
}

#' Deserialize UserFunction
#'
#' A stub deserialize method for illustration purposes. User-defined functions
#' need to provide their own implementation in order for CNTK to be able to
#' reconstruct them when loading a model.
#'
#' @param inputs
#' @param name
#' @param state
#'
#' @export
userfunc_deserialize <- function(inputs, name, state) {
	cntk$ops$functions$UserFunction$deserialize(
		inputs,
		name,
		state
	)
}

#' Infer Outputs of UserFunction
#'
#' Returns a list of all output variables this user-defined function outputs.
#' Output variables are created by output_variable().
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#'
#' @export
userfunc_infer_outputs <- function(func) {
	func$infer_outputs()
}

#' Serialize UserFunction
#'
#' Generates a dictionary that captures the state of this user-defined
#' function.  This method must be overridden, if a user function has any state
#' that needs to be preserved in the model dictionary.
#'
#' @param func - The CNTK `Function` instance on which to apply the operation
#'
#' @export
userfunc_serialize <- function(func) {
	func$serialize()
}


#' Create Native UserFunction
#'
#' Creates an instance of a user-defined Function previously registered using
#' the ‘register_native_user_function’ method.
#'
#' @param op_id
#' @param operands
#' @param attributes
#' @param user_function_instance_name
#'
#' @export
native_user_function <- function(op_id, operands, attributes = NULL,
								 user_function_instance_name = '') {
	cntk$ops$functions$native_user_function(
		op_id,
		operands,
		attributes = attributes,
		user_function_instance_name = user_function_instance_name
	)
}

#' Register Native UserFunction
#'
#' Registers a native user-defined Function that can be subsequently
#' instantiated using the ‘native_user_function’ method.
#'
#' @param op_id
#' @param module_name
#' @param factory_method_name
#'
#' @export
register_native_user_function <- function(op_id, module_name,
										  factory_method_name) {
	cntk$ops$functions$register_native_user_function(
		op_id,
		module_name,
		factory_method_name
	)
}
