#' @export
activation_identity <- function(keep) {
    cntk$layers$identity(keep)
}

#' Activation Layer
#'
#' Layer factory function to create an activation layer. Activation functions
#' can be used directly in CNTK, so there is no difference between `y = relu(x)`
#' and `y = Activation(relu)(x)`. This layer is useful if one wants to configure
#' the activation function with default <- options, or when its invocation
#' should be named.
#'
#' @param activation (defaults to `activation_identity`) – function to apply
#' at the end, e.g. `relu`
#' @param name (str, defaults to '') – the name of the function instance in
#' the network
#' @return A function that accepts one argument and applies the operation to it
#'
#' @examples
#' ```
#' model <- dense(500) %>% activation(op_relu)()
#' # is the same as
#' model <- dense(500) %>% op_relu
#' # and also the same as
#' model <- dense(500, activation=op_relu)
#' ```
#' @export
Activation <- function(activation = activation_identity, name = '') {
	cntk$layers$Activation(
		activation = activation,
		name = name
	)
}

# activation <- function(activation = activation_identity, name = '') {
#     cntk$layers$Activation(activation = activation, name = name)
# }

#' Average Pooling Layer
#'
#' Like Convolution(), AveragePooling() processes items arranged on an
#' N-dimensional grid, such as an image. Typically, each item is a vector. For
#' each item, average-pooling computes the element-wise mean over a window
#' (“receptive field”) of items surrounding the item’s position on the grid.
#'
#' The size (spatial extent) of the receptive field is given by filter <-
#' shape. E.g. for 2D pooling, filter <- shape should be a tuple of two
#' integers, such as (5,5).
#'
#' @param filter_shape (int or tuple of ints) – shape (spatial extent) of the
#' receptive field, not including the input feature-map depth. E.g. (3,3) for a
#' 2D convolution.
#' @param strides (int or tuple of ints, defaults to 1) – stride (increment when
#' sliding over the input). Use a tuple to specify a per-axis value.
#' @param pad (bool or tuple of bools, defaults to False) – if False, then the
#' pooling #' operation will be shifted over the “valid” area of input, that
#' is, no value #' outside the area is used. If pad=True on the other hand,
#' pooling will be #' applied to all input positions, and positions outside the
#' valid region will #' be excluded from the averaging. Use a tuple to specify
#' a per-axis value.
#' @param name (str, defaults to '') – the name of the function instance in the
#' network
#' @return A function that accepts one argument and applies the average-pooling
#' operation to it
#'
#' @examples
#' f <- avg_pooling(c(3, 3), strides = 2)
#' h <- ops_input_variable(c(num_classes))
#'
#' @export
AveragePooling <- function(filter_shape, strides = 1, pad = FALSE, name = '') {
	cntk$layers$AveragePooling(
		filter_shape = filter_shape,
		strides = to_int(strides),
		pad = pad,
		name = name
	)
}

#' BatchNormalization
#'
#' Layer factory function to create an average-pooling layer.
#'
#' Like Convolution(), AveragePooling() processes items arranged on an
#' N-dimensional grid, such as an image. Typically, each item is a vector. For
#' each item, average-pooling computes the element-wise mean over a window
#' (“receptive field”) of items surrounding the item’s position on the grid.
#'
#' The size (spatial extent) of the receptive field is given by filter_shape.
#' E.g. for 2D pooling, filter_shape should be a tuple of two integers, such as
#' (5,5).
#'
#' @export
BatchNormalization <- function(map_rank = NULL, init_scale = 1,
							   normalization_time_constant = 5000,
							   blend_time_constant = 0, epsilon = 0.00001,
							   use_cntk_engine = FALSE, name = '') {
	cntk$layers$BatchNormalization(
		map_rank = to_int(map_rank),
		init_scale = init_scale,
		normalization_time_constant = to_int(normalization_time_constant),
		epsilon = epsilon,
		use_cntk_engine = use_cntk_engine,
		name = name
	)
}

#' Convolution
#'
#' Layer factory function to create a batch-normalization layer.
#'
#' Batch normalization applies this formula to every input element
#' (element-wise): y = (x - batch_mean) / (batch_stddev + epsilon) * scale +
#' bias where batch_mean and batch_stddev are estimated on the minibatch and
#' scale and bias are learned parameters.
#'
#' During operation, this layer also estimates an aggregate running mean and
#' standard deviation for use in inference.
#'
#' A BatchNormalization layer instance owns its learnable parameter tensors and
#' exposes them as attributes .scale and .bias. The aggregate estimates are
#' exposed as attributes aggregate_mean, aggregate_variance, and
#' aggregate_count.
#'
#' @export
Convolution <- function(filter_shape, num_filters = NULL, sequential = FALSE,
						activation = activation_identity,
						init = init_glorot_uniform(), pad = FALSE, strides = 1,
						sharing = TRUE, bias = TRUE, init_bias = 0,
						reduction_rank = 1, transpose_weight = FALSE,
						max_temp_mem_size_in_samples = 0,
						op_name = 'Convolution', name = '') {
	cntk$layers$Convolution(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		sequential = sequential,
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		sharing = sharing,
		bias = bias,
		init_bias = init_bias,
		reduction_rank = to_int(reduction_rank),
		transpose_weight = transpose_weight,
		max_temp_mem_size_in_samples = to_int(max_temp_mem_size_in_samples),
		name = name
	)
}

#' @export
Convolution1D <- function(filter_shape, num_filters = NULL,
						  activation = activation_identity,
						  init = init_glorot_uniform(), pad = FALSE,
						  strides = 1, bias = TRUE, init_bias = 0,
						  reduction_rank = 1, name = '') {
	cntk$layers$Convolution1D(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		bias = bias,
		init_bias = init_bias,
		reduction_rank = to_int(reduction_rank),
		name = name
	)
}

#' @export
Convolution2D <- function(filter_shape, num_filters = NULL,
						  activation = activation_identity,
						  init = init_glorot_uniform(),
						  pad = FALSE, strides = 1, bias = TRUE, init_bias = 0,
						  reduction_rank = 1, name = '') {
	cntk$layers$Convolution2D(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		bias = bias,
		init_bias = init_bias,
		reduction_rank = to_int(reduction_rank),
		name = name
	)
}

#' @export
Convolution3D <- function(filter_shape, num_filters = NULL,
						  activation = activation_identity,
						  init = init_glorot_uniform(), pad = FALSE,
						  strides = 1, bias = TRUE, init_bias = 0,
						  reduction_rank = 1, name = '') {
	cntk$layers$Convolution3D(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		bias = bias,
		init_bias = init_bias,
		reduction_rank = to_int(reduction_rank),
		name = name
	)
}

#' @export
ConvolutionTranspose <- function(filter_shape, num_filters = NULL,
							     activation = activation_identity,
							     init = init_glorot_uniform(), pad = FALSE,
								 strides = 1, sharing = TRUE, bias = TRUE,
								 init_bias = 0, output_shape = NULL,
								 max_temp_mem_size_in_samples = 0,
								 reduction_rank = 1, name = '') {
	cntk$layers$ConvolutionTranspose(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		sharing = sharing,
		bias = bias,
		init_bias = init_bias,
		output_shape = to_int(output_shape),
		reduction_rank = to_int(reduction_rank),
		max_temp_mem_size_in_samples = to_int(max_temp_mem_size_in_samples),
		name = name
	)
}

#' @export
ConvolutionTranspose1D <- function(filter_shape, num_filters = NULL,
							       activation = activation_identity,
							       init = init_glorot_uniform(), pad = FALSE,
								   strides = 1, bias = TRUE, init_bias = 0,
								   output_shape = NULL, name = '') {
	cntk$layers$ConvolutionTranspose1D(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		bias = bias,
		init_bias = init_bias,
		output_shape = to_int(output_shape),
		name = name
	)
}

#' @export
ConvolutionTranspose2D <- function(filter_shape, num_filters = NULL,
							       activation = activation_identity,
							       init = init_glorot_uniform(), pad = FALSE,
								   strides = 1, bias = TRUE, init_bias = 0,
								   output_shape = NULL, name = '') {
	cntk$layers$ConvolutionTranspose2D(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		bias = bias,
		init_bias = init_bias,
		output_shape = to_int(output_shape),
		name = name
	)
}

#' @export
ConvolutionTranspose3D <- function(filter_shape, num_filters = NULL,
							       activation = activation_identity,
							       init = init_glorot_uniform(), pad = FALSE,
								   strides = 1, bias = TRUE, init_bias = 0,
								   output_shape = NULL, name = '') {
	cntk$layers$ConvolutionTranspose3D(
		filter_shape = to_int(filter_shape),
		num_filters = to_int(num_filters),
		activation = activation,
		init = init,
		pad = pad,
		strides = to_int(strides),
		bias = bias,
		init_bias = init_bias,
		output_shape = to_int(output_shape),
		name = name
	)
}

#' @export
Dense <- function(shape, activation = activation_identity,
				  init = init_glorot_uniform(), input_rank = NULL,
				  map_rank = NULL, bias = TRUE, init_bias = 0, name = '') {
	cntk$layers$Dense(
		to_int(shape),
		activation = activation,
		init = init,
		input_rank = to_int(input_rank),
		map_rank = to_int(map_rank),
		bias = bias,
		init_bias = init_bias,
		name = name
	)
}

#' @export
Dropout <- function(dropout_rate = NULL, keep_prob = NULL, seed = NULL,
					name = '') {
	if (is.null(seed)) {
		return(cntk$layers$Dropout(
			dropout_rate = dropout_rate,
			keep_prob = keep_prob,
			name = name
		))
	}
	cntk$layers$Dropout(
		dropout_rate = dropout_rate,
		keep_prob = keep_prob,
		seed = to_int(seed),
		name = name
	)
}

#' @export
Embedding <- function(shape = NULL,
					  init = init_glorot_uniform(), weights = NULL, name = '') {
	cntk$layers$Embedding(
		shape = to_int(shape),
		init = init,
		weights = weights,
		name = name
	)
}

#' @export
GlobalAveragePooling <- function(name = '') {
	cntk$layers$GlobalAveragePooling(name = name)
}

#' @export
GlobalMaxPooling <- function(name = '') {
	cntk$layers$GlobalMaxPooling(name = name)
}

#' @export
Label <- function(name) {
	cntk$layers$Label()
}

#' @export
LayerNormalization <- function(initial_scale = 1, initial_bias = 0,
							   epsilon = 0.00001, name = '') {
	cntk$layers$LayerNormalization(
		initial_scale = initial_scale,
		initial_bias = initial_bias,
		epsilon = epsilon,
		name = name
	)
}

#' @export
MaxPooling <- function(filter_shape, strides = 1, pad = FALSE, name = '') {
	cntk$layers$MaxPooling(
		to_int(filter_shape),
		strides = to_int(strides),
		pad = pad,
		name = name
	)
}

#' @export
MaxUnpooling <- function(filter_shape, strides = 1, pad = FALSE, name = '') {
	cntk$layers$MaxUnpooling(
		to_int(filter_shape),
		strides = to_int(strides),
		pad = pad,
		name = name
	)
}
