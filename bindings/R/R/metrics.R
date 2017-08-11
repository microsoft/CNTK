#' Classification Error Between Target and Predicted
#'
#' This operation computes the classification error.
#'
#' It finds the index of the highest value in the output_vector and compares it
#' to the actual ground truth label (the index of the hot bit in the target
#' vector).
#'
#' The result is a scalar (i.e., one by one matrix). This is often used as an
#' evaluation criterion.
#'
#' It cannot be used as a training criterion though since the gradient is not
#' defined for it.
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

#' Edit Distance Error
#'
#' Edit distance error evaluation node with the option of specifying penalty of
#' substitution, deletion and insertion, as well as squashing the input
#' sequences and ignoring certain samples. Using the classic DP algorithm as
#' described in https://en.wikipedia.org/wiki/Edit_distance, adjusted to take
#' into account the penalties.
#'
#' Each sequence in the inputs is expected to be a matrix. Prior to computation
#' of the edit distance, the operation extracts the indices of maximum element
#' in each column. For example, a sequence matrix 1 2 9 1 3 0 3 2 will be
#' represented as the vector of labels (indices) as [1, 0, 0, 1], on which edit
#' distance will be actually evaluated.
#'
#' The node allows to squash sequences of repeating labels and ignore certain
#' labels. For example, if squashInputs is true and tokensToIgnore contains
#' label ‘-‘ then given first input sequence as s1=”1-12-” and second as
#' s2=”-11–122” the edit distance will be computed against s1’ = “112” and s2’
#' = “112”.
#'
#' The returned error is computed as: EditDistance(s1,s2) * length(s1’) / length(s1)
#'
#' Just like ClassificationError and other evaluation nodes, when used as an
#' evaluation criterion, the SGD process will aggregate all values over an
#' epoch and report the average, i.e. the error rate. Primary objective of this
#' node is for error evaluation of CTC training, see formula (1) in
#' “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
#' with Recurrent Neural Networks”, ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
#'
#' @param input_a
#' @param input_b
#' @param subPen
#' @param delPen
#' @param squashInputs
#' @param tokensToIgnore
#' @param name
#'
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

#' NDCG at 1
#'
#' Groups samples according to group, sorts them within each group based on
#' output and computes the Normalized Discounted Cumulative Gain (NDCG) at 1
#' for each group. Concretely, the NDCG at 1 is:
#'
#' \(\mathrm{NDCG_1} = \frac{gain_{(1)}}{\max_i gain_i}\)
#'
#' where gain(1)gain(1) means the gain of the first ranked sample.
#'
#' Samples in the same group must appear in order of decreasing gain.
#'
#' It returns the average NDCG at 1 across all the groups in the minibatch
#' multiplied by 100 times the number of samples in the minibatch.
#'
#' This is a forward-only operation, there is no gradient for it.
#'
#' @param output
#' @param gain
#' @param group
#' @param name
#'
#' @export
ndcg_at_1 <- function(output, gain, group, name = '') {
	cntk$metrics$ndcg_at_1(
		output,
		gain,
		group,
		name = name
	)
}
