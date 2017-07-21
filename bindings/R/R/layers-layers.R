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
#' Layer factory function to create a convolution layer.
#'
#' This implements a convolution operation over items arranged on an
#' N-dimensional grid, such as pixels in an image. Typically, each item is a
#' vector (e.g. pixel: R,G,B), and the result is, in turn, a vector. The
#' item-grid dimensions are referred to as the spatial dimensions (e.g.
#' dimensions of an image), while the vector dimension of the individual items
#' is often called feature-map depth.
#'
#' For each item, convolution gathers a window (“receptive field”) of items
#' surrounding the item’s position on the grid, and applies a little
#' fully-connected network to it (the same little network is applied to all
#' item positions). The size (spatial extent) of the receptive field is given
#' by filter_shape. E.g. to specify a 2D convolution, filter_shape should be a
#' tuple of two integers, such as (5,5); an example for a 3D convolution (e.g.
#' video or an MRI scan) would be filter_shape=(3,3,3); while for a 1D
#' convolution (e.g. audio or text), filter_shape has one element, such as (3,)
#' or just 3.
#'
#' The dimension of the input items (input feature-map depth) is not to be
#' specified. It is known from the input. The dimension of the output items
#' (output feature-map depth) generated for each item position is given by
#' num_filters.
#'
#' If the input is a sequence, the sequence elements are by default treated
#' independently. To convolve along the sequence dimension as well, pass
#' sequential=True. This is useful for variable-length inputs, such as video or
#' natural-language processing (word n-grams). Note, however, that convolution
#' does not support sparse inputs.
#'
#' Both input and output items can be scalars intead of vectors. For
#' scalar-valued input items, such as pixels on a black-and-white image, or
#' samples of an audio clip, specify reduction_rank=0. If the output items are
#' scalar, pass num_filters=() or None.
#'
#' A Convolution instance owns its weight parameter tensors W and b, and
#' exposes them as an attributes .W and .b. The weights will have the shape
#' (num_filters, input_feature_map_depth, *filter_shape)
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

#' Convolution1D
#'
#' Layer factory function to create a 1D convolution layer with optional
#' non-linearity. Same as Convolution() except that filter_shape is verified to
#' be 1-dimensional. See Convolution() for extensive documentation.
#'
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

#' Convoluion2D
#'
#' Layer factory function to create a 2D convolution layer with optional
#' non-linearity. Same as Convolution() except that filter_shape is verified to
#' be 2-dimensional. See Convolution() for extensive documentation.
#'
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

#' Convolution3D
#'
#' Layer factory function to create a 3D convolution layer with optional
#' non-linearity. Same as Convolution() except that filter_shape is verified to
#' be 3-dimensional. See Convolution() for extensive documentation.
#'
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

#' ConvolutionTranspose
#'
#' Layer factory function to create a convolution transpose layer.
#'
#' This implements a convolution_transpose operation over items arranged on an
#' N-dimensional grid, such as pixels in an image. Typically, each item is a
#' vector (e.g. pixel: R,G,B), and the result is, in turn, a vector. The
#' item-grid dimensions are referred to as the spatial dimensions (e.g.
#' dimensions of an image), while the vector dimensions of the individual items
#' are often called feature-map depth.
#'
#' Convolution transpose is also known as fractionally strided convolutional
#' layers, or, deconvolution. This operation is used in image and language
#' processing applications. It supports arbitrary dimensions, strides, and
#' padding.
#'
#' The forward and backward computation of convolution transpose is the inverse
#' of convolution. That is, during forward pass the input layer’s items are
#' spread into the output same as the backward spread of gradients in
#' convolution. The backward pass, on the other hand, performs a convolution
#' same as the forward pass of convolution.
#'
#' The size (spatial extent) of the receptive field for convolution transpose
#' is given by filter_shape. E.g. to specify a 2D convolution transpose,
#' filter_shape should be a tuple of two integers, such as (5,5); an example
#' for a 3D convolution transpose (e.g. video or an MRI scan) would be
#' filter_shape=(3,3,3); while for a 1D convolution transpose (e.g. audio or
#' text), filter_shape has one element, such as (3,).
#'
#' The dimension of the input items (feature-map depth) is not specified, but
#' known from the input. The dimension of the output items generated for each
#' item position is given by num_filters.
#'
#' A ConvolutionTranspose instance owns its weight parameter tensors W and b,
#' and exposes them as an attributes .W and .b. The weights will have the shape
#' (input_feature_map_depth, num_filters, *filter_shape).
#'
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

#' ConvolutionTranspose1D
#'
#' Layer factory function to create a 1D convolution transpose layer with
#' optional non-linearity. Same as ConvolutionTranspose() except that
#' filter_shape is verified to be 1-dimensional. See ConvolutionTranspose() for
#' extensive documentation.
#'
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

#' ConvolutionTranspose2D
#'
#' Layer factory function to create a 2D convolution transpose layer with
#' optional non-linearity. Same as ConvolutionTranspose() except that
#' filter_shape is verified to be 2-dimensional. See ConvolutionTranspose() for
#' extensive documentation.
#'
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

#' ConvolutionTranspose3D
#'
#' Layer factory function to create a 3D convolution transpose layer with
#' optional non-linearity. Same as ConvolutionTranspose() except that
#' filter_shape is verified to be 3-dimensional. See ConvolutionTranspose() for
#' extensive documentation.
#'
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

#' Dense
#'
#' Layer factory function to create an instance of a fully-connected linear
#' layer of the form activation(input @ W + b) with weights W and bias b, and
#' activation and b being optional. shape may describe a tensor as well.
#'
#' A Dense layer instance owns its parameter tensors W and b, and exposes them
#' as attributes .W and .b.
#'
#' The Dense layer can be applied to inputs that are tensors, not just vectors.
#' This is useful, e.g., at the top of a image-processing cascade, where after
#' many convolutions with padding and strides it is difficult to know the
#' precise dimensions. For this case, CNTK has an extended definition of matrix
#' product, in which the input tensor will be treated as if it had been
#' automatically flattened. The weight matrix will be a tensor that reflects
#' the “flattened” dimensions in its axes.
#'
#' This behavior can be modified by telling CNTK either the number of axes that
#' should not be projected (map_rank) or the rank of the input (input_rank). If
#' neither is specified, all input dimensions are projected, as in the example
#' above.
#'
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

#' Dropout
#'
#' Layer factory function to create a drop-out layer.
#'
#' The dropout rate can be specified as the probability of dropping a value
#' (dropout_rate). E.g. Dropout(0.3) means “drop 30% of the activation values.”
#' Alternatively, it can also be specified as the probability of keeping a
#' value (keep_prob).
#'
#' The dropout operation is only applied during training. During testing, this
#' is a no-op. To make sure that this leads to correct results, the dropout
#' operation in training multiplies the result by (1/(1-dropout_rate)).
#'
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

#' Embedding
#'
#' Layer factory function to create a embedding layer.
#'
#' An embedding is conceptually a lookup table. For every input token (e.g. a
#' word or any category label), the corresponding entry in in the lookup table
#' is returned.
#'
#' In CNTK, discrete items such as words are represented as one-hot vectors.
#' The table lookup is realized as a matrix product, with a matrix whose rows
#' are the embedding vectors. Note that multiplying a matrix from the left with
#' a one-hot vector is the same as copying out the row for which the input
#' vector is 1. CNTK has special optimizations to make this operation as
#' efficient as an actual table lookup if the input is sparse.
#'
#' The lookup table in this layer is learnable, unless a user-specified one is
#' supplied through the weights parameter. For example, to use an existing
#' embedding table from a file in numpy format, use this:
#'
#' Embedding(weights=np.load('PATH.npy')) To initialize a learnable lookup
#' table with a given numpy array that is to be used as the initial value, pass
#' that array to the init parameter (not weights).
#'
#' An Embedding instance owns its weight parameter tensor E, and exposes it as
#' an attribute .E.
#'
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

#' GlobalAveragePooling
#'
#' Layer factory function to create a global average-pooling layer.
#'
#' The global average-pooling operation computes the element-wise mean over all
#' items on an N-dimensional grid, such as an image.
#'
#' This operation is the same as applying reduce_mean() to all grid dimensions.
#'
#' @export
GlobalAveragePooling <- function(name = '') {
	cntk$layers$GlobalAveragePooling(name = name)
}

#' GlobalMaxPooling
#'
#' Layer factory function to create a global max-pooling layer.
#'
#' The global max-pooling operation computes the element-wise maximum over all
#' items on an N-dimensional grid, such as an image.
#'
#' This operation is the same as applying reduce_max() to all grid dimensions.
#'
#' @export
GlobalMaxPooling <- function(name = '') {
	cntk$layers$GlobalMaxPooling(name = name)
}

#' Label
#'
#' Layer factory function to create a dummy layer with a given name. This can be
#' used to access an intermediate value flowing through computation.
#'
#' @export
Label <- function(name) {
	cntk$layers$Label()
}

#' LayerNormalization
#'
#' Layer factory function to create a function that implements layer
#' normalization.
#'
#' Layer normalization applies this formula to every input element
#' (element-wise): y = (x - mean(x)) / (stddev(x) + epsilon) * scale + bias
#' where scale and bias are learned scalar parameters.
#'
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

#' MaxPooling
#'
#' Layer factory function to create a max-pooling layer.
#'
#' Like Convolution(), MaxPooling() processes items arranged on an
#' N-dimensional grid, such as an image. Typically, each item is a vector. For
#' each item, max-pooling computes the element-wise maximum over a window
#' (“receptive field”) of items surrounding the item’s position on the grid.
#'
#' The size (spatial extent) of the receptive field is given by filter_shape.
#' E.g. for 2D pooling, filter_shape should be a tuple of two integers, such as
#' (5,5).
#'
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
