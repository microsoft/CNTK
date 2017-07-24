#' NDArrayView
#'
#' Creates an empty dense internal data representation of a Value object. To create an NDArrayView from a NumPy array, use from_dense(). To create an NDArrayView from a sparse array, use from_csr().
#'
#' @param shape - list of ints representing tensor shape list - shape of the data
#' @param dtype - data type to be used ("float32", "float64", or "auto") "float32" or "float64" - data type
#' @param device - instance of DeviceDescriptor DeviceDescriptor - device this value should be put on
#'
#' @export
NDArrayView <- function(shape, dtype, device = NULL) {
	cntk$core$NDArrayView(
		to_int(shape),
		type_map(dtype),
		device = device
	)
}

#'
#'
#' @param csr_array
#'
#' @param device - instance of DeviceDescriptor
#' @param read_only
#' @param borrow
#' @param shape - list of ints representing tensor shape
#'
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

#'
#'
#' @param data
#'
#' @param device - instance of DeviceDescriptor
#' @param read_only
#' @param borrow
#'
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

#'
#'
#' @param np_array
#'
#' @param device - instance of DeviceDescriptor
#' @param read_only
#' @param borrow
#'
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

#'
#'
#' @param ndarrayview
#'
#' @param start_offset
#' @param extent
#' @param read_only
#'
#' @export
arrayview_slice_view <- function(ndarrayview, start_offset, extent,
								 read_only = TRUE) {
	ndarrayview$slice_view(
		to_int(start_offset),
		to_int(extent),
		read_only = read_only
	)
}

#'
#'
#' @param batch
#'
#' @param seq_starts
#' @param device - instance of DeviceDescriptor
#'
#' @export
Value <- function(batch, seq_starts = NULL, device = NULL) {
	cntk$core$Value(
		batch,
		seq_starts = seq_starts,
		device = device
	)
}

#'
#'
#' @param value
#'
#' @param variable
#'
#' @export
value_as_sequences <- function(value, variable = NULL) {
	value$as_sequences(variable = variable)
}

#'
#'
#' @param var
#'
#' @param data
#' @param seq_starts
#' @param device - instance of DeviceDescriptor
#' @param read_only
#'
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

#'
#'
#' @param batch
#'
#' @param num_classes
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#' @param device - instance of DeviceDescriptor
#'
#' @export
value_one_hot <- function(batch, num_classes, dtype = 'auto', device = NULL) {
	cntk$core$Value$one_hot(
		to_int(batch),
		to_int(num_classes),
		dtype = type_map(dtype),
		device = device
	)
}

#'
#'
#' @param value
#'
#' @param dtype - data type to be used ("float32", "float64", or "auto")
#'
#' @export
value_asarray <- function(value, dtype = 'auto') {
	cntk$core$asarray(
		value,
		dtype = type_map(dtype)
	)
}

#'
#'
#' @param variable
#'
#' @param data_array
#'
#' @export
asvalue <- function(variable, data_array) {
	cntk$core$asvalue(
		variable,
		data_array
	)
}

#'
#'
#' @param user_func
#'
#' @export
user_function <- function(user_func) {
	cntk$core$user_function(user_func)
}
