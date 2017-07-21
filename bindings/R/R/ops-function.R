#' @export
CloneMethod <- function(value) {
	reticulate::py_get_attr(cntk$ops$functions$CloneMethod, value)
}

#' @export
Function <- function(...) {
	cntk$ops$functions$Function(...)
}

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

#' @export
func_clone <- function(func, method, substitutions = NULL) {
	func$clone(
		method,
		substitutions = substitutions
	)
}

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

#' @export
func_find_all_with_name <- function(func, name, depth = 0) {
	func$find_all_with_name(name, depth = to_int(depth))
}

#' @export
func_find_by_name <- function(func, name, depth = 0) {
	func$find_by_name(name, depth = to_int(depth))
}

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

#' @export
func_load <- function(model, device = NULL) {
	cntk$ops$functions$Function$load(
		model,
		device = device
	)
}

#' @export
register_udf_deserialize_callback <- function(op_name, callback) {
	cntk$ops$functions$Function$register_udf_deserialize_callback(
		op_name,
		callback
	)
}

#' @export
func_replace_placeholder <- function(func, substitution) {
	func$replace_placeholder(substitution)
}

#' @export
func_replace_placeholders <- function(func, substitutions) {
	func$replace_placeholders(substitutions)
}

#' @export
func_restore <- function(func, filename) {
	func$restore(filename)
}

#' @export
func_save <- function(func, filename) {
	func$restore(filename)
}

#' @export
func_set_attribute <- function(func, name, value) {
	if (name == 'rngSeed') {
		value = to_int(value)
	}
	func$set_attribute(name, value)
}

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


#' @export
UserFunction <- function(inputs, as_matrix = TRUE, name = '') {
	cntk$ops$functions$UserFunction(
		inputs,
		as_numpy = as_matrix,
		name = name
	)
}

#' @export
userfunc_clone <- function(func, cloned_inputs) {
	func$clone(cloned_inputs)
}

#' @export
userfunc_deserialize <- function(inputs, name, state) {
	cntk$ops$functions$UserFunction$deserialize(
		inputs,
		name,
		state
	)
}

#' @export
userfunc_infer_outputs <- function(func) {
	func$infer_outputs()
}

#' @export
userfunc_serialize <- function(func) {
	func$serialize()
}


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

#' @export
register_native_user_function <- function(op_id, module_name,
										  factory_method_name) {
	cntk$ops$functions$register_native_user_function(
		op_id,
		module_name,
		factory_method_name
	)
}
