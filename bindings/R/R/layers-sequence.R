#' @param initial_state
#'
#' @param name
#'
#' @export
Delay <- function(T = 1, initial_state = 0, name = '') {
	cntk$layers$sequence$delay(
		T = T,
		initial_state = to_int(initial_state),
		name = name
	)
}

#' @param folder_function
#'
#' @param go_backwards
#' @param initial_state
#' @param return_full_state
#' @param name
#'
#' @export
Fold <- function(folder_function, go_backwards, initial_state = 0,
				 return_full_state = FALSE, name = '') {
	cntk$layers$sequence$Fold(
		folder_function,
		go_backwards = go_backwards,
		initial_state = to_int(initial_state),
		return_full_state = return_full_state,
		name = name
	)
}

#' @param window_size
#'
#' @param axis
#' @param go_backwards
#' @param name
#'
#' @export
PastValueWindow <- function(window_size, axis, go_backwards = FALSE, name='') {
	cntk$layers$sequence$PastValueWindow(
		to_int(window_size),
		to_int(axis),
		go_backwards = go_backwards,
		name = name
	)
}

#' @param step_function
#'
#' @param go_backwards
#' @param initial_state
#' @param return_full_state
#' @param name
#'
#' @export
Recurrence <- function(step_function, go_backwards = FALSE, initial_state = 0,
					   return_full_state = FALSE, name = '') {
	cntk$layers$sequence$Recurrence(
		step_function,
		go_backwards = go_backwards,
		initial_state = initial_state,
		return_full_state = return_full_state,
		name = name
	)
}

#' @param step_function
#'
#' @param go_backwards
#' @param return_full_state
#' @param name
#'
#' @export
RecurrenceFrom <- function(step_function, go_backwards, return_full_state,
						   name = '') {
	cntk$layers$sequence$RecurrenceFrom(
		step_function,
		go_backwards = go_backwards,
		return_full_state = return_full_state,
		name = name
	)
}

#' @param generator_function
#'
#' @param until_predicate
#' @param length_increase
#' @param name
#'
#' @export
UnfoldFrom <- function(generator_function, until_predicate = NULL,
					   length_increase = 1, name = '') {
	cntk$layers$sequence$UnfoldFrom(
		generator_function,
		until_predicate = until_predicate,
		length_increase = length_increase,
		name = name
	)
}
