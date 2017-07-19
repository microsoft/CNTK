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
