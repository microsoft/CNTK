#' Binary Cross Entropy Loss
#'
#' @param output
#' @param target
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
loss_binary_cross_entropy <- function(output, target, name = '') {
	cntk$losses$binary_cross_entropy(
		output,
		target,
		name = ''
	)
}

#' Cosine Distance Loss
#'
#' @param x
#'
#' @param y
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
loss_cosine_distance <- function(x, y, name = '') {
	cntk$losses$cosine_distance(
		x,
		y,
		name = name
	)
}

#' Cosine Distance NEgative Samples Loss
#'
#' @param x
#'
#' @param y
#' @param shift
#' @param num_negative_samples
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
loss_cosine_distance_negative_samples <- function(x, y, shift,
												  num_negative_samples,
												  name = '') {
	cntk$losses$cosine_distance_with_negative_samples(
		x,
		y,
		to_int(shift),
		num_negative_samples,
		name = name
	)
}


#' Cross Entropy Loss with Softmax for Multiclass Classification
#'
#' This operation computes the cross entropy between the \code{target_vector}
#' and the softmax of the \code{output_vector}.
#' The elements of \code{target_vector} have to be non-negative and should sum
#' to 1.
#' The \code{output_vector} can contain any values.
#' The function will internally compute the softmax of the \code{output_vector}.
#'
#' @param output_vector unscaled computed output values from the network
#' @param target_vector one-hot encoded vector of target values
#' @param axis integer (optional) for axis to compute cross-entropy
#' @param name string (optional) the name of the Function instance in the network
#'
#' @references \url{https://www.cntk.ai/pythondocs/cntk.losses.html#cntk.losses.cross_entropy_with_softmax}
#' @export
loss_cross_entropy_with_softmax <- function(output_vector, target_vector,
											axis = -1, name = '') {
	cntk$losses$cross_entropy_with_softmax(
		output_vector,
		target_vector,
		axis = to_int(axis),
		name = name
	)
}

#' Lambda Rank Loss
#'
#' @param output
#'
#' @param gain
#' @param group
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
loss_lambda_rank <- function(output, gain, group, name = '') {
	cntk$losses$lambda_rank(
		output,
		gain,
		group,
		name = name
	)
}

#' Squared Error Loss
#'
#' @param output
#'
#' @param target
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
loss_squared_error <- function(output, target, name = '') {
	cntk$losses$squared_error(
		output,
		target,
		name = name
	)
}

#' Weighted Binary Cross Entropy Loss
#'
#' @param output
#'
#' @param target
#' @param weight
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
loss_weighted_binary_cross_entropy <- function(output, target, weight,
											   name = '') {
	cntk$losses$weighted_binary_cross_entropy(
		output,
		target,
		weight,
		name = name
	)
}
