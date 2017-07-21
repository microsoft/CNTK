#' @export
NDArrayView <- function(shape, dtype, device = NULL) {
	cntk$core$NDArrayView(
		to_int(shape),
		type_map(dtype),
		device = device
	)
}

#' @export
arrayview_from_csr <- function(csr_array, device = NULL, read_only = FALSE,
							   borrow = FALSE, shape = NULL) {
	cntk$core$NDArrayView$from_csr(
		csr_array,
		device = device,
		read_only = read_only,
		borrow = FALSE,
		shape = to_int(shape)
	)
}

#' @export
arrayview_from_data <- function(data, device = NULL, read_only = FALSE,
								borrow = FALSE) {
	cntk$core$NDArrayView$from_data(
		data,
		device = device,
		read_only = read_only,
		borrow = borrow
	)
}

#' @export
arrayview_from_dense <- function(np_array, device = NULL, read_only = FALSE,
								 borrow = FALSE) {
	cntk$core$NDArrayView$from_dense(
		np_array,
		device = device,
		read_only = read_only,
		borrow = borrow
	)
}

#' @export
arrayview_slice_view <- function(ndarrayview, start_offset, extent,
								 read_only = TRUE) {
	ndarrayview$slice_view(
		to_int(start_offset),
		to_int(extent),
		read_only = read_only
	)
}

#' @export
Value <- function(batch, seq_starts = NULL, device = NULL) {
	cntk$core$Value(
		batch,
		seq_starts = seq_starts,
		device = device
	)
}

#' @export
value_as_sequences <- function(value, variable = NULL) {
	value$as_sequences(variable = variable)
}

#' @export
value_create <- function(var, data, seq_starts = NULL, device = NULL,
						 read_only = FALSE) {
	cntk$core$Value$create(
		var,
		data,
		seq_starts = seq_starts,
		device = device,
		read_only = read_only
	)
}

#' @export
value_one_hot <- function(batch, num_classes, dtype = 'auto', device = NULL) {
	cntk$core$Value$one_hot(
		to_int(batch),
		to_int(num_classes),
		dtype = type_map(dtype),
		device = device
	)
}

#' @export
value_asarray <- function(value, dtype = 'auto') {
	cntk$core$asarray(
		value,
		dtype = type_map(dtype)
	)
}

#' @export
asvalue <- function(variable, data_array) {
	cntk$core$asvalue(
		variable,
		data_array
	)
}

#' @export
user_function <- function(user_func) {
	cntk$core$user_function(user_func)
}
