#' Graph - Depth First Search
#'
#' @param root
#' @param visitor
#' @param depth
#'
#' @export
graph_depth_first_search <- function(root, visitor, depth = 0) {
	cntk$logging$graph$depth_first_search(
		root,
		visitor,
		to_int(depth)
	)
}

#' Graph - Find All With Name
#'
#' @param node
#'
#' @param node_name
#' @param depth
#'
#' @export
graph_find_all_with_name <- function(node, node_name, depth = 0) {
	cntk$logging$graph$find_all_with_name(
		node,
		node_name,
		to_int(depth)
	)
}

#' Graph - Find By Name
#'
#' @param node
#'
#' @param node_name
#' @param depth
#'
#' @export
graph_find_by_name <- function(node, node_name, depth = 0) {
	cntk$logging$graph$find_by_name(
		node,
		node_name,
		to_int(depth)
	)
}

#'
#'
#' @param node
#'
#' @param depth
#'
#' @export
graph_get_node_outputs <- function(node, depth = 0) {
	cntk$logging$graph$get_node_outputs(
		node,
		depth = to_int(depth)
	)
}

#'
#'
#' @param root
#'
#' @param filename
#'
#' @export
graph_plot <- function(root, filename = NULL) {
	cntk$logging$graph$plot(
		root,
		filename = filename
	)
}
