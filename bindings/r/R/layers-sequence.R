sequence <- reticulate::import('cntk.layers.sequence')

#' @export
Delay <- function(T = 1, initial_state = 0, name = '') {
	sequence$delay(
		T = T,
		initial_state = to_int(initial_state),
		name = name
	)
}

#' @export
Fold <- function(folder_function, go_backwards, initial_state = 0,
				 return_full_state = FALSE, name = '') {
	sequence$Fold(
		folder_function,
		go_backwards = go_backwards,
		initial_state = to_int(initial_state),
		return_full_state = return_full_state,
		name = name
	)
}

#' @export
PastValueWindow <- function(window_size, axis, go_backwards = FALSE, name='') {
	axis <- ifelse(class(axis) == 'numeric', as.integer(axis), axis)
	sequence$PastValueWindow(
		to_int(window_size),
		axis,
		go_backwards = go_backwards,
		name = name
	)
}
