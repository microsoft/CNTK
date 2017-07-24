#' @export
IO_AVG_POOLING <- 1L

#' @export
IO_MAX_POOLING <- 0L

#' @export
IO_MAX_UNPOOLING <- 0L

#' Absolute Value
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_abs <- function(x, name = '') {
	cntk$ops$abs(x, name = name)
}

#' Alias
#'
#' Create a new Function instance which just aliases the specified ‘x’
#' Function/Variable such that the ‘Output’ of the new ‘Function’ is same as
#' the ‘Output’ of the specified ‘x’ Function/Variable, and has the newly
#' specified name. The purpose of this operator is to create a new distinct
#' reference to a symbolic computation which is different from the original
#' Function/Variable that it aliases and can be used for e.g. to substitute a
#' specific instance of the aliased Function/Variable in the computation graph
#' instead of substituting all usages of the aliased Function/Variable.
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_alias <- function(x, name = '') {
	cntk$ops$alias(x, name = name)
}

#' Argmax Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_argmax <- function(x, axis = NULL, name = '') {
	cntk$ops$argmax(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' Argmin Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_argmin <- function(x, axis = NULL, name = '') {
	cntk$ops$argmin(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' As Block
#'
#' Create a new block Function instance which just encapsulates the specified
#' composite Function to create a new Function that appears to be a primitive.
#' All the arguments of the composite being encapsulated must be Placeholder
#' variables. The purpose of block Functions is to enable creation of
#' hierarchical Function graphs where details of implementing certain building
#' block operations can be encapsulated away such that the actual structure of
#' the block’s implementation is not inlined into the parent graph where the
#' block is used, and instead the block just appears as an opaque primitive.
#' Users still have the ability to peek at the underlying Function graph that
#' implements the actual block Function.
#'
#' @param composite
#' @param block_arguments_map
#' @param block_op_name
#' @param block_instance_name
#'
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

#' As Composite
#'
#' Creates a composite Function that has the specified root_function as its
#' root. The composite denotes a higher-level Function encapsulating the entire
#' graph of Functions underlying the specified rootFunction.
#'
#' @param root_function
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
as_composite <- function(root_function, name = '') {
	cntk$ops$as_composite(
		root_function,
		name = name
	)
}

#' Assign
#'
#' Assign the value in input to ref and return the new value, ref need to be
#' the same layout as input. Both ref and input can’t have dynamic axis and
#' broadcast isn’t supported for the assign operator. During forward pass, ref
#' will get the new value after the forward or backward pass finish, so that
#' any part of the graph that depend on ref will get the old value. To get the
#' new value, use the one returned by the assign node. The reason for that is
#' to make assign have a deterministic behavior.
#'
#' If not computing gradients, the ref will be assigned the new value after the
#' forward pass over the entire Function graph is complete; i.e. all uses of
#' ref in the forward pass will use the original (pre-assignment) value of ref.
#'
#' If computing gradients (training mode), the assignment to ref will happen
#' after completing both the forward and backward passes over the entire
#' Function graph.
#'
#' The ref must be a Parameter or Constant. If the same ref is used in multiple
#' assign operations, then the order in which the assignment happens is
#' non-deterministic and the final value can be either of the assignments
#' unless an order is established using a data dependence between the
#' assignments.
#'
#' @param ref
#' @param input
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_assign <- function(ref, input, name = '') {
	cntk$ops$assign(
		ref,
		input,
		name = name
	)
}

#' Associative Multi-Arg
#'
#' The output of this operation is the result of an operation (plus,
#' log_add_exp, element_times, element_max, element_min) of two or more input
#' tensors. Broadcasting is supported.
#'
#' @param f
#'
#' @export
op_associative_multi_arg <- function(f) {
	cntk$ops$associative_multi_arg(f)
}

#' Batch Normalization
#'
#' Normalizes layer outputs for every minibatch for each output (feature)
#' independently and applies affine transformation to preserve representation
#' of the layer.
#'
#' @param operand
#' @param scale
#' @param bias (bool) – whether to include bias
#' @param running_mean
#' @param running_inv_std
#' @param spatial
#' @param normalization_time_constant
#' @param blent_time_constant
#' @param epsilon (float, default 0.00001) - added to avoid division by 0
#' @param use_cudnn_engine
#' @param name (str) - the name of the Function instance in the network
#' @param running_count
#'
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

#' Ceiling
#'
#' The output of this operation is the element wise value rounded to the
#' smallest integer greater than or equal to the input.
#'
#' @param arg
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_ceil <- function(arg, name = '') {
	cntk$ops$ceil(arg, name = name)
}

#' Clip
#'
#' Computes a tensor with all of its values clipped to fall between min_value
#' and max_value, i.e. min(max(x, min_value), max_value).  The output tensor
#' has the same shape as x.
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param min_value
#' @param max_value
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_clip <- function(x, min_value, max_value, name = '') {
	cntk$ops$clip(
		x,
		min_value,
		max_value,
		name = name
	)
}

#' Combine
#'
#' Create a new Function instance which just combines the outputs of the
#' specified list of ‘operands’ Functions such that the ‘Outputs’ of the new
#' ‘Function’ are union of the ‘Outputs’ of each of the specified ‘operands’
#' Functions. E.g., when creating a classification model, typically the
#' CrossEntropy loss Function and the ClassificationError Function comprise the
#' two roots of the computation graph which can be combined to create a single
#' Function with 2 outputs; viz. CrossEntropy loss and ClassificationError
#' output.
#'
#' @param operands
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_combine <- function(operands, name = '') {
	cntk$ops$combine(
		operands,
		name = name
	)
}

#' It creates a constant tensor initialized from a numpy array
#'
#' @param value
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_constant <- function(value = NULL, shape = NULL, dtype = 'auto', name = '') {
	cntk$ops$constant(
		value = value,
		shape = to_int(shape),
		dtype = type_map(dtype),
		name = name
	)
}

#' Convolution
#'
#' Computes the convolution of convolution_map (typically a tensor of learnable
#' parameters) with operand (commonly an image or output of a previous
#' convolution/pooling operation). This operation is used in image and language
#' processing applications. It supports arbitrary dimensions, strides, sharing,
#' and padding.
#'
#' This function operates on input tensors with dimensions
#' [C×M1×M2×…×Mn][C×M1×M2×…×Mn]. This can be understood as a rank-n object,
#' where each entry consists of a CC-dimensional vector. For example, an RGB
#' image would have dimensions [3×W×H][3×W×H], i.e. a [W×H][W×H]-sized
#' structure, where each entry (pixel) consists of a 3-tuple.
#'
#' convolution convolves the input operand with a n+2n+2 rank tensor of
#' (typically learnable) filters called convolution_map of shape
#' [O×I×m1×m2×…×mn][O×I×m1×m2×…×mn] (typically mi≪Mimi≪Mi). The first
#' dimension, OO, is the nunber of convolution filters (i.e. the number of
#' channels in the output). The second dimension, II, must match the number of
#' channels in the input. The last n dimensions are the spatial extent of the
#' filter. I.e. for each output position, a vector of dimension OO is computed.
#' Hence, the total number of filter parameters is O×I×m1×m2×…×mn
#'
#' @param convolution_map
#' @param operand
#' @param strides (int or tuple of ints, defaults to 1) – stride of the
#' operation. Use a list of ints to specify a per-axis value.
#' @param sharing
#' @param auto_padding
#' @param max_temp_mem_size_in_samples
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Convolution Transpose
#'
#' Computes the transposed convolution of convolution_map (typically a tensor
#' of learnable parameters) with operand (commonly an image or output of a
#' previous convolution/pooling operation). This is also known as fractionally
#' strided convolutional layers, or, deconvolution. This operation is used in
#' image and language processing applications. It supports arbitrary
#' dimensions, strides, sharing, and padding.
#'
#' This function operates on input tensors with dimensions
#' [C×M1×M2×…×Mn][C×M1×M2×…×Mn]. This can be understood as a rank-n object,
#' where each entry consists of a CC-dimensional vector. For example, an RGB
#' image would have dimensions [3×W×H][3×W×H], i.e. a [W×H][W×H]-sized
#' structure, where each entry (pixel) consists of a 3-tuple.
#'
#' convolution_transpose convolves the input operand with a n+2n+2 rank tensor
#' of (typically learnable) filters called convolution_map of shape
#' [I×O×m1×m2×…×mn][I×O×m1×m2×…×mn] (typically mi≪Mimi≪Mi). The first
#' dimension, II, must match the number of channels in the input. The second
#' dimension, OO, is the number of convolution filters (i.e. the number of
#' channels in the output). The last n dimensions are the spatial extent of the
#' filter. I.e. for each output position, a vector of dimension OO is computed.
#' Hence, the total number of filter parameters is I×O×m1×m2×…×mn
#'
#' @param convolution_map
#' @param operand
#' @param strides (int or tuple of ints, defaults to 1) – stride of the
#' operation. Use a list of ints to specify a per-axis value.
#' @param sharing
#' @param auto_padding
#' @param output_shape - user expected output shape after convolution transpose.
#' @param max_temp_mem_size_in_samples
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Element-wise Cosine
#'
#' Computes the element-wise cosine of x: The output tensor has the same shape
#' as x.
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_cos <- function(x, name = '') {
	cntk$ops$cos(x, name = name)
}

#' Dropout
#'
#' Each element of the input is independently set to 0 with probabily
#' dropout_rate or to 1 / (1 - dropout_rate) times its original value (with
#' probability 1-dropout_rate). Dropout is a good way to reduce overfitting.
#'
#' This behavior only happens during training. During inference dropout is a
#' no-op. In the paper that introduced dropout it was suggested to scale the
#' weights during inference In CNTK’s implementation, because the values that
#' are not set to 0 are multiplied with (1 / (1 - dropout_rate)), this is not
#' necessary.
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param dropout_rate
#' @param seed
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_dropout <- function(x, dropout_rate = 0, seed = 4294967293, name = '') {
	cntk$ops$dropout(
		x,
		dropout_rate = dropout_rate,
		seed = to_int(seed),
		name = name
	)
}

#' Element-wise Division
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_element_divide <- function(left, right, name = '') {
	cntk$ops$element_divide(
		left,
		right,
		name = name
	)
}

#' Element Max
#'
#' @param left - left side tensor
#'
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_element_max <- function(left, right, name = '') {
	cntk$ops$element_max(
		left,
		right,
		name = name
	)
}

#' Element Min
#'
#' @param left - left side tensor
#'
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_element_min <- function(left, right, name = '') {
	cntk$ops$element_min(
		left,
		right,
		name = name
	)
}

#' Element Select
#'
#' @param flag
#'
#' @param value_if_true
#' @param value_if_false
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_element_select <- function(flag, value_if_true, value_if_false, name = '') {
	cntk$ops$element_select(
		flag,
		value_if_true,
		value_if_false,
		name = name
	)
}

#' Element Times
#'
#' @param left - left side tensor
#'
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_element_times <- function(left, right, name = '') {
	cntk$ops$element_times(
		left,
		right,
		name = name
	)
}

#' Elu
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_elu <- function(left, right, name = '') {
	cntk$ops$elu(
		left,
		right,
		name = name
	)
}

#' Equal Comparison
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_equal <- function(left, right, name = '') {
	cntk$ops$equal(
		left,
		right,
		name = name
	)
}

#' Element-wise Exponential
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#'
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_exp <- function(x, name = '') {
	cntk$ops$exp(x, name = name)
}

#' Floor
#'
#' @param arg
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_floor <- function(arg, name = '') {
	cntk$ops$floor(arg, name = name)
}

#' Forward-Backward
#'
#' @param graph
#'
#' @param features
#' @param blank_token_id
#' @param delay_constraint
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Gather
#'
#' @param reference
#'
#' @param indices
#'
#' @export
op_gather <- function(reference, indices) {
	cntk$ops$gather(
		reference,
		indices
	)
}

#' Element-wise Greater Comparison
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_greater <- function(left, right, name = '') {
	cntk$ops$greater(
		left,
		right,
		name = name
	)
}


#' Element-wise Greater Equal Comparison
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_greater_equal <- function(left, right, name = '') {
	cntk$ops$greater_equal(
		left,
		right,
		name = name
	)
}

#' Hardmax
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_hardmax <- function(x, name = '') {
	cntk$ops$hardmax(x, name = name)
}



#' Create input for network
#'
#' It creates an input in the network: a place where data, such as features and labels, should be provided.
#'
#' @param shape - list of ints representing tensor shape integer vector for dimensions of input tensor
#' @param needs_gradient logical whether to conduct backprop on the tensor
#' @param is_sparse logical whether variable is sparse
#' @param dynamic_axes list of dynamic axis (only a single axis can be dynamic, i.e., either batch axis or time axis)
#' @return Variable \url{https://www.cntk.ai/pythondocs/cntk.variables.html#cntk.variables.Variable}
#' @references \url{https://www.cntk.ai/pythondocs/cntk.ops.html#cntk.ops.input_variable}
#' @export
op_input_variable <- function(shape, dtype = 'float32', needs_gradient = FALSE,
							  is_sparse = FALSE,
							  dynamic_axes = c(get_default_batch_axis()),
							  name = '') {
	cntk$ops$input_variable(
		to_int(shape),
		dtype = type_map(dtype),
		needs_gradient = needs_gradient,
		is_sparse = is_sparse,
		dynamic_axes = dynamic_axes,
		name = name
	)
}

#' Labels To Graph
#'
#' @param labels
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_labels_to_graph <- function(labels, name = '') {
	cntk$ops$labels_to_graph(labels, name = name)
}

#' Leaky Relu
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_leaky_relu <- function(x, name = '') {
	cntk$ops$leaky_relu(x, name = name)
}

#' Element-wise Less Comparison
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_less <- function(left, right, name = '') {
	cntk$ops$less(
		left,
		right,
		name = name
	)
}

#' Element-wise Less Equal Comparison
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_less_equal <- function(left, right, name = '') {
	cntk$ops$less_equal(
		left,
		right,
		name = name
	)
}

#' Element-wise Natural Log
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_log <- function(x, name = '') {
	cntk$ops$log(x, name = name)
}

#' Log Add Exp
#'
#' Calculates the log of the sum of the exponentials of the two or more input
#' tensors. It supports broadcasting.
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_log_add_exp <- function(left, right, name = '') {
	cntk$ops$log_add_exp(
		left,
		right,
		name = name
	)
}

#' Minus
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_minus <- function(left, right, name = '') {
	cntk$ops$minus(
		left,
		right,
		name = name
	)
}

#' Element-wise Negation
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_negate <- function(x, name = '') {
	cntk$ops$negate(x, name = name)
}

#' Element-wise Not Equal Comparison
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_not_equal <- function(left, right, name = '') {
	cntk$ops$not_equal(
		left,
		right,
		name = name
	)
}

#' Create One-Hot Encoding
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param num_classes
#' @param sparse_output
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Optimized RNN Stack
#'
#' @param operand
#' @param weights
#' @param hidden_size
#' @param num_layers
#' @param bidirectional
#' @param recurrent_op
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Output Variable
#'
#' @param shape - list of ints representing tensor shape
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param dynamic_axes
#' @param needs_gradient
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_output_variable <- function(shape, dtype, dynamic_axes,
							   needs_gradient = TRUE, name = '') {
	cntk$ops$output_variable(
		to_int(shape),
		type_map(dtype),
		dynamic_axes,
		needs_gradient = needs_gradient,
		name = name
	)
}

#' Parametric ReLU
#'
#' @param alpha
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_param_relu <- function(alpha, x, name = '') {
	cntk$ops$param_relu(
		alpha,
		x,
		name = name
	)
}

#' Parameter
#'
#' Creates a parameter tensor
#'
#' @param shape - list of ints representing tensor shape
#' @param init (scalar or matrix or initializer, defaults to
#' init_glorot_uniform()) – initial value of weights W
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param device - instance of DeviceDescriptor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_parameter <- function(shape = NULL, init = NULL, dtype = "auto",
						 device = NULL, name = '') {
	cntk$ops$parameter(
		shape = to_int(shape),
		init = init,
		dtype = type_map(dtype),
		device = device,
		name = name
	)
}

#' Per-dimension Mean-variance Normalization
#'
#' @param operand
#' @param mean
#' @param inv_stddev
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Placeholder
#'
#' @param shape - list of ints representing tensor shape
#' @param dynamic_axes
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_placeholder <- function(shape = NULL, dynamic_axes = NULL, name = '') {
	cntk$ops$placeholder(
		shape = to_int(shape),
		dynamic_axes = to_int(dynamic_axes),
		name = name
	)
}

#' Addition of Two Tensors
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_plus <- function(left, right, name = '') {
	cntk$ops$plus(left, right, name = name)
}

#' Pooling
#'
#' @param operand
#' @param pooling_type
#' @param pooling_window_shape
#' @param strides (int or tuple of ints, defaults to 1) – stride of the
#' operation. Use a list of ints to specify a per-axis value.
#' @param auto_padding
#' @param ceil_out_dim
#' @param include_pad
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Power Computation
#'
#' @param base
#' @param exponent
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_pow <- function(base, exponent, name = '') {
	cntk$ops$pow(
		base,
		exponent,
		name = name
	)
}

#' Random Sample
#'
#' @param weights
#' @param num_samples
#' @param allow_duplicates
#' @param seed
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Random Sample Inclusion Frequency
#'
#' @param weights
#' @param num_samples
#' @param allow_duplicates
#' @param seed
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Element-wise Reciprocal
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reciprocal <- function(x, name = '') {
	cntk$ops$reciprocal(x, name = name)
}

#' Reconcile Dynamic Axes
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param dynamic_axes_as
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reconcile_dynamic_axes <- function(x, dynamic_axes_as, name = '') {
	cntk$ops$reconcile_dynamic_axes(
		x,
		dynamic_axes_as,
		name = name
	)
}

#' Reduce Max Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reduce_log_sum_exp <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_log_sum_exp(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' Reduce Max Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reduce_max <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_max(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' Reduce Mean Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reduce_mean <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_mean(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' Reduce Minimum Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reduce_min <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_min(
		x,
		axis = to_int(x),
		name = name
	)
}

#' Reduce Prod Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reduce_prod <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_prod(
		x,
		axis = to_int(x),
		name = name
	)
}

#' Reduce Sum Across Axis
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_reduce_sum <- function(x, axis = NULL, name = '') {
	cntk$ops$reduce_sum(
		x,
		axis = to_int(x),
		name = name
	)
}

#' Rectified Linear Units Operation
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#'
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_relu <- function(x, name = '') {
	cntk$ops$relu(x, name = name)
}

#' Reshape
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param shape - list of ints representing tensor shape
#' @param begin_axis - shape replacement begins at this axis
#' @param end_axis - shape replacement ends at this axis (non-inclusive)
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Region of Interest Pooling
#'
#' @param conv_feature_map
#' @param rois
#' @param roi_output_shape
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_roipooling <- function(conv_feature_map, rois, roi_output_shape, name='') {
	cntk$ops$roipooling(
		conv_feature_map,
		rois,
		to_int(roi_output_shape),
		name = name
	)
}

#' Element-wise Rounding
#'
#' @param arg
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_round <- function(arg, name = '') {
	cntk$ops$round(arg, name = name)
}

#' Element-wise Sigmoid
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_sigmoid <- function(x, name = '') {
	cntk$ops$sigmoid(x, name = name)
}

#' Element-wise Sine
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_sin <- function(x, name = '') {
	cntk$ops$sin(x, name = name)
}

#' Slice
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param begin_index - index where slicing starts
#' @param end_index - index where slicing ends
#' @param name (str) - the name of the Function instance in the network
#'
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

#' Softmax
#'
#' Computes the gradient of f(z)=log∑iexp(zi)f(z)=log⁡∑iexp⁡(zi) at z = x.
#' Concretely,
#'
#' softmax(x)=[exp(x1)∑iexp(xi)exp(x1)∑iexp(xi)…exp(x1)∑iexp(xi)]softmax(x)=[exp⁡(x1)∑iexp⁡(xi)exp⁡(x1)∑iexp⁡(xi)…exp⁡(x1)∑iexp⁡(xi)]
#'
#' with the understanding that the implementation can use equivalent formulas
#' for efficiency and numerical stability.
#'
#' The output is a vector of non-negative numbers that sum to 1 and can
#' therefore be interpreted as probabilities for mutually exclusive outcomes as
#' in the case of multiclass classification.
#'
#' If axis is given, the softmax will be computed along that axis.
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis - axis across which to perform operation
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_softmax <- function(x, axis = NULL, name = '') {
	cntk$ops$softmax(
		x,
		axis = to_int(axis),
		name = name
	)
}

#' Softplus
#'
#' Softplus operation. Computes the element-wise softplus of x:
#'
#' softplus(x)=log(1+exp(x))softplus(x)=log⁡(1+exp⁡(x))
#'
#' The optional steepness allows to make the knee sharper (steepness>1) or
#' softer, by computing softplus(x * steepness) / steepness. (For very large
#' steepness, this approaches a linear rectifier).
#'
#' The output tensor has the same shape as x.
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param steepness
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_softplus <- function(x, steepness = 1, name = '') {
	cntk$ops$softplus(
		x,
		steepness = steepness,
		name = name
	)
}

#' Concatenate Across Axis
#'
#' @param inputs - one or more input tensors
#' @param ... named list of axis_names
#'
#' @export
op_splice <- function(inputs, ...) {
	cntk$ops$splice(
		inputs,
		to_int(c(axis))
	)
}

#' Element-wise Square-root
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_sqrt <- function(x, name = '') {
	cntk$ops$sqrt(x, name = name)
}

#' Element-wise Square
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_square <- function(x, name = '') {
	cntk$ops$square(x, name = name)
}

#' Stop Gradient
#'
#' @param input
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_stop_gradient <- function(input, name = '') {
	cntk$ops$stop_gradient(input, name = name)
}


#' Swap Axes
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param axis1
#' @param axis2
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_swap_axes <- function(x, axis1 = 0, axis2 = 1, name = '') {
	cntk$ops$swap_axes(
		x,
		axis1 = to_int(axis1),
		axis2 = to_int(axis2),
		name = name
	)
}

#' Hyperbolic tan
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_tanh <- function(x, name = '') {
	cntk$ops$tanh(x, name = name)
}

#' Matrix Product
#'
#' @param left - left side tensor
#' @param right - right side tensor
#' @param output_rank
#' @param infer_input_rank_to_map
#' @param name (str) - the name of the Function instance in the network
#'
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

#' One Element Times Another Transposed Element
#'
#' @param left - left side tensor
#'
#' @param right - right side tensor
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_times_transpose <- function(left, right, name = '') {
	cntk$ops$times_transpose(
		left,
		right,
		name = name
	)
}

#' To Sequence
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param sequence_lengths
#' @param sequence_axis_name_prefix
#' @param name (str) - the name of the Function instance in the network
#'
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

#' To Sequence Like
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#'
#' @param dynamic_axes_like
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_to_sequence_like <- function(x, dynamic_axes_like, name = '') {
	cntk$ops$to_sequence_like(
		x,
		dynamic_axes_like,
		name = name
	)
}

#' Transpose
#'
#' @param x - matrix or CNTK Function that outputs a tensor
#' @param perm
#' @param name (str) - the name of the Function instance in the network
#'
#' @export
op_transpose <- function(x, perm, name = '') {
	cntk$ops$transpose(
		x,
		perm,
		name = name
	)
}

#' Unpooling
#'
#' @param operand
#'
#' @param pooling_input
#' @param unpooling_window_shape
#' @param strides (int or tuple of ints, defaults to 1) – stride of the
#' operation. Use a list of ints to specify a per-axis value.
#' @param auto_padding
#' @param name (str) - the name of the Function instance in the network
#'
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
