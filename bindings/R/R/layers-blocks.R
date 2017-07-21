#' @export
ForwardDeclaration <- function(name = 'forward_declaration') {
	cntk$layers$blocks$ForwardDeclaration(name = name)
}

#' @export
GRU <- function(shape, cell_shape = NULL, activation = op_tanh,
				init = init_glorot_uniform(), init_bias = 0,
				enable_self_stabilization = FALSE, name = '') {
	cntk$layers$blocks$GRU(
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
LSTM <- function(shape, cell_shape = NULL, activation = op_tanh,
				 use_peepholes = FALSE, init = init_glorot_uniform(),
				 init_bias = 0, enable_self_stabilization = FALSE, name = '') {
	cntk$layers$blocks$LSTM(
		to_int(shape),
		cell_shape = to_int(cell_shape),
		activation = activation,
		use_peepholes = use_peepholes,
		init = init_glorot_uniform(),
		init_bias = init_bias,
		enable_self_stabilization = enable_self_stabilization,
		name = name
	)
}

#' @export
RRNStep <- function(shape, cell_shape = NULL, activation = op_sigmoid,
					init = init_glorot_uniform(), init_bias = 0,
					enable_self_stabilization = FALSE, name = '') {
	cntk$layers$blocks$RNNStep(
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
	cntk$layers$blocks$Stabilizer(
		steepness = steepness,
		enable_self_stabilization = enable_self_stabilization,
		name = name
	)
}

#' @export
UntestedBranchError <- function(name) {
	cntk$layers$blocks$UntestedBranchError(name)
}
