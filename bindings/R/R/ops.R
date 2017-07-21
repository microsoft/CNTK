#' @export
IO_AVG_POOLING <- 1L

#' @export
IO_MAX_POOLING <- 0L

#' @export
IO_MAX_UNPOOLING <- 0L

#' @export
op_abs <- function(x, name = '') {
	cntk$ops$abs(x, name = name)
}

#' @export
op_alias <- function(x, name = '') {
	cntk$ops$alias(x, name = name)
}

#' @export
op_argmax <- function(x, axis = NULL, name = '') {
	cntk$ops$argmax(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' @export
op_argmin <- function(x, axis = NULL, name = '') {
	cntk$ops$argmin(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' @export
as_block <- function(composite, block_arguments_map, block_op_name,
					 block_instance_name = '') {
	cntk$ops$as_block(
		composite,
		block_arguments_map,
		block_op_name,
		block_instance_name = block_instance_name
	)
}

#' @export
as_composite <- function(root_function, name = '') {
	cntk$ops$as_composite(
		root_function,
		name = name
	)
}

#' @export
op_assign <- function(ref, input, name = '') {
	cntk$ops$assign(
		ref,
		input,
		name = name
	)
}

#' @export
op_associative_multi_arg <- function(f) {
	cntk$ops$associative_multi_arg(f)
}

#' @export
op_batch_normalization <- function(operand, scale, bias, running_mean,
								   running_inv_std, spatial,
								   normalization_time_constant = 5000,
								   blent_time_constant = 0, epsilon = 1e-05,
								   use_cudnn_engine = FALSE, name = '',
								   running_count = NULL) {
	cntk$ops$batch_normalization(
		operand,
		scale,
		bias,
		running_mean,
		running_inv_std,
		spatial,
		normalization_time_constant = normalization_time_constant,
		blent_time_constant = blent_time_constant,
		epsilon = epsilon,
		use_cudnn_engine = use_cudnn_engine,
		name = name,
		running_count = running_count
	)
}

#' @export
op_ceil <- function(arg, name = '') {
	cntk$ops$ceil(arg, name = name)
}

#' @export
op_clip <- function(x, min_value, max_value, name = '') {
	cntk$ops$clip(
		x,
		min_value,
		max_value,
		name = name
	)
}

#' @export
op_combine <- function(operands, name = '') {
	cntk$ops$combine(
		operands,
		name = name
	)
}

#' @export
op_constant <- function(value = NULL, shape = NULL, name = '') {
	cntk$ops$constant(
		value = value,
		shape = to_int(shape),
		dtype = np$float32,
		name = name
	)
}

#' @export
op_convolution <- function(convolution_map, operand, strides = c(1),
						   sharing = c(TRUE), auto_padding = c(TRUE),
						   max_temp_mem_size_in_samples = 0, name = '') {
	cntk$ops$convolution(
		convolution_map,
		operand,
		strides = to_int(strides),
		sharing = sharing,
		auto_padding = auto_padding,
		max_temp_mem_size_in_samples = max_temp_mem_size_in_samples,
		name = name
	)
}

#' @export
op_convolution_transpose <- function(convolution_map, operand, strides = c(1),
						             sharing = c(TRUE), auto_padding = c(TRUE),
									 output_shape = NULL,
									 max_temp_mem_size_in_samples = 0,
									 name = '') {
	cntk$ops$convolution_transpose(
		convolution_map,
		operand,
		strides = to_int(strides),
		sharing = sharing,
		auto_padding = auto_padding,
		output_shape = to_int(output_shape),
		max_temp_mem_size_in_samples = max_temp_mem_size_in_samples,
		name = name
	)
}

#' @export
op_cos <- function(x, name = '') {
	cntk$ops$cos(x, name = name)
}

#' @export
op_dropout <- function(x, dropout_rate = 0, seed = 4294967293, name = '') {
	cntk$ops$dropout(
		x,
		dropout_rate = dropout_rate,
		seed = to_int(seed),
		name = name
	)
}

#' @export
op_element_divide <- function(left, right, name = '') {
	cntk$ops$element_divide(
		left,
		right,
		name = name
	)
}

#' @export
op_element_max <- function(left, right, name = '') {
	cntk$ops$element_max(
		left,
		right,
		name = name
	)
}

#' @export
op_element_min <- function(left, right, name = '') {
	cntk$ops$element_min(
		left,
		right,
		name = name
	)
}

#' @export
op_element_select <- function(flag, value_if_true, value_if_false, name = '') {
	cntk$ops$element_select(
		flag,
		value_if_true,
		value_if_false,
		name = name
	)
}

#' @export
op_element_times <- function(left, right, name = '') {
	cntk$ops$element_times(
		left,
		right,
		name = name
	)
}

#' @export
op_elu <- function(left, right, name = '') {
	cntk$ops$elu(
		left,
		right,
		name = name
	)
}

#' @export
op_equal <- function(left, right, name = '') {
	cntk$ops$equal(
		left,
		right,
		name = name
	)
}

#' @export
op_exp <- function(x, name = '') {
	cntk$ops$exp(x, name = name)
}

#' @export
op_floor <- function(arg, name = '') {
	cntk$ops$floor(arg, name = name)
}

#' @export
op_forward_backward <- function(graph, features, blank_token_id,
								delay_constraint = -1, name = '') {
	cntk$ops$forward_backward(
		graph,
		features,
		blankTokenId = blank_token_id,
		delayConstraint = delay_constraint,
		name = name
	)
}

#' @export
op_gather <- function(reference, indices) {
	cntk$ops$gather(
		reference,
		indices
	)
}

#' @export
op_greater <- function(left, right, name = '') {
	cntk$ops$greater(
		left,
		right,
		name = name
	)
}


#' @export
op_greater_equal <- function(left, right, name = '') {
	cntk$ops$greater_equal(
		left,
		right,
		name = name
	)
}

#' @export
op_hardmax <- function(x, name = '') {
	cntk$ops$hardmax(x, name = name)
}



#' Create input for network
#'
#' It creates an input in the network: a place where data, such as features and labels, should be provided.
#'
#' @param shape integer vector for dimensions of input tensor
#' @param needs_gradient logical whether to conduct backprop on the tensor
#' @param is_sparse logical whether variable is sparse
#' @param dynamic_axes list of dynamic axis (only a single axis can be dynamic, i.e., either batch axis or time axis)
#' @return Variable \url{https://www.cntk.ai/pythondocs/cntk.variables.html#cntk.variables.Variable}
#' @references \url{https://www.cntk.ai/pythondocs/cntk.ops.html#cntk.ops.input_variable}
#' @export
op_input_variable <- function(shape, needs_gradient = FALSE, is_sparse = FALSE,
						   	  dynamic_axes = c(get_default_batch_axis()),
						   	  name = '') {
	cntk$ops$input_variable(
		to_int(shape),
		dtype = np$float32,
		needs_gradient = needs_gradient,
		is_sparse = is_sparse,
		dynamic_axes = dynamic_axes,
		name = name
	)
}

#' @export
op_labels_to_graph <- function(labels, name = '') {
	cntk$ops$labels_to_graph(labels, name = name)
}

#' @export
op_leaky_relu <- function(x, name = '') {
	cntk$ops$leaky_relu(x, name = name)
}

#' @export
op_less <- function(left, right, name = '') {
	cntk$ops$less(
		left,
		right,
		name = name
	)
}

#' @export
op_less_equal <- function(left, right, name = '') {
	cntk$ops$less_equal(
		left,
		right,
		name = name
	)
}

#' @export
op_log <- function(x, name = '') {
	cntk$ops$log(x, name = name)
}

#' @export
op_log_add_exp <- function(left, right, name = '') {
	cntk$ops$log_add_exp(
		left,
		right,
		name = name
	)
}

#' @export
op_minus <- function(left, right, name = '') {
	cntk$ops$minus(
		left,
		right,
		name = name
	)
}

#' @export
op_negate <- function(x, name = '') {
	cntk$ops$negate(x, name = name)
}

#' @export
op_not_equal <- function(left, right, name = '') {
	cntk$ops$not_equal(
		left,
		right,
		name = name
	)
}

#' @export
op_one_hot <- function(x, num_classes, sparse_output = FALSE, axis = -1,
					   name = '') {
	cntk$ops$one_hot(
		x,
		to_int(num_classes),
		sparse_output = sparse_output,
		axis = to_int(axis),
		name = name
	)
}

#' @export
op_optimized_rnnstack <- function(operand, weights, hidden_size, num_layers,
								  bidirectional = FALSE, recurrent_op = 'lstm',
								  name = '') {
	cntk$ops$optimized_rnnstack(
		operand,
		weights,
		to_int(hidden_size),
		to_int(num_layers),
		bidirectional = bidirectional,
		recurrent_op = recurrent_op,
		name = name
	)
}

#' @export
op_output_variable <- function(shape, dynamic_axes, needs_gradient = TRUE,
							   name = '') {
	cntk$ops$output_variable(
		to_int(shape),
		np$float32,
		dynamic_axes,
		needs_gradient = needs_gradient,
		name = name
	)
}

#' @export
op_param_relu <- function(alpha, x, name = '') {
	cntk$ops$param_relu(
		alpha,
		x,
		name = name
	)
}

#' @export
op_parameter <- function(shape = NULL, init = NULL, device = NULL, name = '') {
	cntk$ops$parameter(
		shape = to_int(shape),
		init = init,
		dtype = np$float32,
		device = device,
		name = name
	)
}

#' @export
op_dim_mean_variance_normalize <- function(operand, mean, inv_stddev,
										   name = '') {
	cntk$ops$per_dim_mean_variance_normalize(
		operand,
		mean,
		inv_stddev,
		name
	)
}

#' @export
op_placeholder <- function(shape = NULL, dynamic_axes = NULL, name = '') {
	cntk$ops$placeholder(
		shape = to_int(shape),
		dynamic_axes = to_int(dynamic_axes),
		name = name
	)
}

#' @export
op_plus <- function(left, right, name = '') {
	cntk$ops$plus(left, right, name = name)
}

#' @export
op_pooling <- function(operand, pooling_type, pooling_window_shape,
					   strides = c(1), auto_padding = c(FALSE),
					   ceil_out_dim = FALSE, include_pad = FALSE, name = '') {
	cntk$ops$pooling(
		operand,
		pooling_type = pooling_type,
		pooling_window_shape = to_int(pooling_window_shape),
		strides = to_int(strides),
		auto_padding = auto_padding,
		ceil_out_dim = ceil_out_dim,
		include_pad = include_pad,
		name = name
	)
}

#' @export
op_pow <- function(base, exponent, name = '') {
	cntk$ops$pow(
		base,
		exponent,
		name = name
	)
}

#' @export
op_random_sample <- function(weights, num_samples, allow_duplicates,
							 seed = 4294967293, name = '') {
	cntk$ops$random_sample(
		weights,
		to_int(num_samples),
		allow_duplicates,
		seed = to_int(seed),
		name = name
	)
}

#' @export
op_random_sample_inclusion_frequency <- function(weights, num_samples,
												 allow_duplicates,
												 seed = 4294967293, name = '') {
	cntk$ops$random_sample_inclusion_frequency(
		weights,
		to_int(num_samples),
		allow_duplicates,
		seed = to_int(seed),
		name = name
	)
}

#' @export
op_reciprocal <- function(x, name = '') {
	cntk$ops$reciprocal(x, name = name)
}

#' @export
op_reconcile_dynamic_axes <- function(x, dynamic_axes_as, name = '') {
	cntk$ops$reconcile_dynamic_axes(
		x,
		dynamic_axes_as,
		name = name
	)
}

#' @export
op_reduce_log_sum_exp <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_log_sum_exp(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' @export
op_reduce_max <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_max(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' @export
op_reduce_mean <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_mean(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' @export
op_reduce_min <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_min(
		x,
		axis = to_int(x),
		name = name
	)
}

#' @export
op_reduce_prod <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_prod(
		x,
		axis = to_int(x),
		name = name
	)
}

#' @export
op_reduce_sum <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_sum(
		x,
		axis = to_int(x),
		name = name
	)
}

#' @export
op_relu <- function(x, name = '') {
	cntk$ops$relu(x, name = name)
}

#' @export
op_reshape <- function(x, shape, begin_axis = NULL, end_axis = NULL,
					   name = name) {
	cntk$ops$reshape(
		x,
		to_int(shape),
		begin_axis = to_int(begin_axis),
		end_axis = to_int(end_axis),
		name = name
	)
}

#' @export
op_roipooling <- function(conv_feature_map, rois, roi_output_shape, name='') {
	cntk$ops$roipooling(
		conv_feature_map,
		rois,
		to_int(roi_output_shape),
		name = name
	)
}

#' @export
op_round <- function(arg, name = '') {
	cntk$ops$round(arg, name = name)
}

#' @export
op_sigmoid <- function(x, name = '') {
	cntk$ops$sigmoid(x, name = name)
}

#' @export
op_sin <- function(x, name = '') {
	cntk$ops$sin(x, name = name)
}

#' @export
op_slice <- function(x, axis, begin_index, end_index, name = '') {
	cntk$ops$slice(
		x,
		to_int(axis),
		to_int(begin_index),
		to_int(end_index),
		name = name
	)
}

#' @export
op_softmax <- function(x, axis = NULL, name = '') {
	cntk$ops$softmax(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' @export
op_softplus <- function(x, steepness = 1, name = '') {
	cntk$ops$softplus(
		x,
		steepness = steepness,
		name = name
	)
}

#' @export
op_splice <- function(..., axis) {
	cntk$ops$splice(
		c(...),
		axis = to_int(axis)
	)
}

#' @export
op_sqrt <- function(x, name = '') {
	cntk$ops$sqrt(x, name = name)
}

#' @export
op_square <- function(x, name = '') {
	cntk$ops$square(x, name = name)
}

#' @export
op_stop_gradient <- function(input, name = '') {
	cntk$ops$stop_gradient(input, name = name)
}


#' @export
op_swap_axes <- function(x, axis1 = 0, axis2 = 1, name = '') {
	cntk$ops$swap_axes(
		x,
		axis1 = to_int(axis1),
		axis2 = to_int(axis2),
		name = name
	)
}

#' @export
op_tanh <- function(x, name = '') {
	cntk$ops$tanh(x, name = name)
}

#' @export
op_times <- function(left, right, output_rank = 1, infer_input_rank_to_map = -1,
					 name = '') {
	cntk$ops$times(
		left,
		right,
		output_rank = to_int(output_rank),
		infer_input_rank_to_map = to_int(infer_input_rank_to_map),
		name = name
	)
}

#' @export
op_times_transpose <- function(left, right, name = '') {
	cntk$ops$times_transpose(
		left,
		right,
		name = name
	)
}

#' @export
op_to_sequence <- function(x, sequence_lengths = NULL,
						   sequence_axis_name_prefix = 'toSequence_',
						   name = '') {
	cntk$ops$to_sequence(
		x,
		sequence_lengths = to_int(sequence_lengths),
		sequence_axis_name_prefix = sequence_axis_name_prefix,
		name = name
	)
}

#' @export
op_to_sequence_like <- function(x, dynamic_axes_like, name = '') {
	cntk$ops$to_sequence_like(
		x,
		dynamic_axes_like,
		name = name
	)
}

# TODO: check out if perm arg should be cast to int
#' @export
op_transpose <- function(x, perm, name = '') {
	cntk$ops$transpose(
		x,
		perm,
		name = name
	)
}

#' @export
op_unpooling <- function(operand, pooling_input, unpooling_window_shape,
						 strides = c(1), auto_padding = c(FALSE), name = '') {
	cntk$ops$unpooling(
		operand,
		pooling_input = pooling_input,
		unpooling_type = IO_MAX_UNPOOLING, # currently only supported type
		unpooling_window_shape = to_int(unpooling_window_shape),
		strides = to_int(strides),
		auto_padding = auto_padding,
		name = name
	)
}
