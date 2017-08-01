#' Create an NDArrayView Instance
#'
#' Creates an empty dense internal data representation of a Value object. To
#' create an NDArrayView from a NumPy array, use from_dense(). To create an
#' NDArrayView from a sparse array, use from_csr().
#'
#' @param shape list of ints representing tensor shape list shape of the
#' data
#' @param dtype data type to be used ("float32", "float64", or "auto")
#' "float32" or "float64" data type
#' @param device instance of DeviceDescriptor DeviceDescriptor device this
#' value should be put on
#'
#' @export
NDArrayView <- function(shape, dtype, device = NULL) {
	cntk$core$NDArrayView(
		to_int(shape),
		type_map(dtype),
		device = device
	)
}

#' ArrayView From Data
#'
#' Create an NDArrayView instance from a sparse matrix in CSR format.
#'
#' @param data matrix to be converted
#' @param device instance of DeviceDescriptor
#' @param read_only (bool) whether the data can be modified
#' @param borrow (bool) whether nd_array memory can be borrowed internally to
#' speed up data creation
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

#' ArrayView From Dense Data
#'
#' Create an NDArrayView instance from a matrix.
#'
#' @param matrix matrix
#' @param device instance of DeviceDescriptor
#' @param read_only (bool) whether the data can be modified
#' @param borrow (bool) whether nd_array memory can be borrowed internally to
#' speed up data creation
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

#' ArrayView Slice View
#'
#' Returns a sliced view of the instance.
#'
#' @param ndarrayview NDArrayView
#' @param start_offset (list) shape of the same rank as this Value instance
#' that denotes the start of the slicing
#' @param extent (list) shape of the right-aligned extent to keep
#' @param read_only (bool) whether the data can be modified
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

#' New Value Instance
#'
#' Internal representation of minibatch data.
#'
#' @param batch (matrix or Value) batch input for var
#' @param seq_starts (list of bools or NULL) if None, every sequence is treated
#' as a new sequence. Otherwise, it is interpreted as a list of Booleans that
#' tell whether a sequence is a new sequence (True) or a continuation of the
#' sequence in the same slot of the previous minibatch (False)
#' @param device instance of DeviceDescriptor
#'
#' @export
Value <- function(batch, seq_starts = NULL, device = NULL) {
	cntk$core$Value(
		batch,
		seq_starts = seq_starts,
		device = device
	)
}

#' Get Value As Sequence of Matrices
#'
#' Convert a Value to a sequence of NumPy arrays that have their masked entries removed.
#'
#' @param value the Value instance
#' @param variable the Variable
#'
#' @export
value_as_sequences <- function(value, variable = NULL) {
	value$as_sequences(variable = variable)
}

#' Create a Value Object
#'
#' Creates a Value object
#'
#' @param var (Variable) variable into which data is passed
#' @param data matrix denoting the full minibatch or one parameter or constant
#' @param seq_starts (list of bools or NULL) if None, every sequence is treated
#' as a new sequence. Otherwise, it is interpreted as a list of Booleans that
#' tell whether a sequence is a new sequence (True) or a continuation of the
#' sequence in the same slot of the previous minibatch (False)
#' @param device instance of DeviceDescriptor
#' @param read_only (bool) whether the data can be modified
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

#' Value as One Hot
#'
#' Converts batch into a Value object of dtype such that the integer data in
#' batch is interpreted as the indices representing one-hot vectors.
#'
#' @param batch (list of list of int) batch input data of indices
#' @param num_classes (int or list)  number of classes or shape of each sample
#' whose trailing axis is one_hot
#' @param dtype data type to be used ("float32", "float64", or "auto")
#' @param device instance of DeviceDescriptor
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

#' Value As Matrix
#'
#' Converts a Value object to a sequence of matrices (if dense) or CSR arrays
#' (if sparse).
#'
#' @param value Value instance to be converted
#' @param dtype data type to be used ("float32", "float64", or "auto")
#'
#' @export
value_as_matrix <- function(value, dtype = 'auto') {
	cntk$core$asarray(
		value,
		dtype = type_map(dtype)
	)
}

#' Value Form Matrix
#'
#' Converts a sequence of matrices or CSR arrays to a Value object
#'
#' @param variable variable
#' @param data_array matrix of data
#'
#' @export
asvalue <- function(variable, data_array) {
	cntk$core$asvalue(
		variable,
		data_array
	)
}

#' User Defined Function
#'
#' Wraps the passed Function to create a composite representing the composite
#' Function graph rooted at the passed root Function.
#'
#' @param user_func User Function
#'
#' @export
user_function <- function(user_func) {
	cntk$core$user_function(user_func)
}
