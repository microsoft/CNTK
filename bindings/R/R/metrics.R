#' Classification Error Between Target and Predicted
#'
#' This operation computes the classification error.
#' It finds the index of the highest value in the output_vector and compares it to the actual ground truth label (the index of the hot bit in the target vector).
#' The result is a scalar (i.e., one by one matrix). This is often used as an evaluation criterion.
#' It cannot be used as a training criterion though since the gradient is not defined for it.
#'
#' @param output_vector the output values of the network
#' @param target_vector one-hot encoding of target values
#' @param axis integer for axis along which the classification error is computed
#' @param topN integer
#' @param name string (optional) the name of the Function instance in the network
#'
#' @references \url{https://www.cntk.ai/pythondocs/cntk.metrics.html#cntk.metrics.classification_error}
#' @export
classification_error <- function(output_vector, target_vector, axis = -1,
								 topN = 1, name = '') {
	cntk$metrics$classification_error(
		output_vector,
		target_vector,
		axis = to_int(axis),
		topN = to_int(topN),
		name = name
	)
}

#' @export
edit_distance_error <- function(input_a, input_b, subPen = 1, delPen = 1,
								squashInputs = FALSE, tokensToIgnore = c(),
								name = '') {
	cntk$metrics$edit_distance_error(
		input_a,
		input_b,
		subPen = to_int(subPen),
		delPen = to_int(delPen),
		squashInputs = squashInputs,
		tokensToIgnore = tokensToIgnore,
		name = name
	)
}

#' @export
ndcg_at_1 <- function(output, gain, group, name = '') {
	cntk$metrics$ndcg_at_1(
		output,
		gain,
		group,
		name = name
	)
}
