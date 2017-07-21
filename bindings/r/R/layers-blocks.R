blocks <- reticulate::import("cntk.layers.blocks")

#' @export
ForwardDeclaration <- function(name = 'forward_declaration') {
	blocks$ForwardDeclaration(name = name)
}

#' @export
GRU <- function(shape, cell_shape = NULL, activation = acvitation_tanh,
				init = glorot_uniform(), init_bias = 0,
				enable_self_stabilization = FALSE, name = '') {
	blocks$GRU(
		to_int(shape),
		to_int(cell_shape),
		activation = activation,
		init = init,
		init_bias = init_bias,
		enable_self_stabilization = enable_self_stabilization,
		name = name
	)
}

#' @export
LSTM <- function(shape, cell_shape = NULL, activation = activation_tanh,
				 use_peepholes = FALSE, init = glorot_uniform(), init_bias = 0,
				 enable_self_stabilization = FALSLE, name = '') {
	blocks$LSTM(
		to_int(shape),
		cell_shape = to_int(cell_shape),
		activation = activation,
		use_peepholes = use_peepholes,
		init = glorot_uniform(),
		init_bias = init_bias,
		enable_self_stabilization = enable_self_stabilization,
		name = name
	)
}

#' @export
RRNStep <- function(shape, cell_shape = NULL, activation = activation_sigmoid,
					init = glorot_uniform(), init_bias = 0,
					enable_self_stabilization = FALSE, name = '') {
	blocks$RNNStep(
		to_int(shape),
		cell_shape = to_int(cell_shape),
		activation = activation,
		init = init,
		init_bias = init_bias,
		enable_self_stabilization = enable_self_stabilization,
		name = name
	)
}

#' @export
Stabilizer <- function(steepness = 4, enable_self_stabilization = TRUE,
					   name = '') {
	blocks$Stabilizer(
		steepness = steepness,
		enable_self_stabilization = enable_self_stabilization,
		name = name
	)
}

#' @export
UntestedBranchError <- blocks$UntestedBranchError
