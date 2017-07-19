#' @export
graph_depth_first_search <- function(root, visitor, depth = 0) {
	cntk$logging$graph$depth_first_search(
		root,
		visitor,
		to_int(depth)
	)
}

#' @export
graph_find_all_with_name <- function(node, node_name, depth = 0) {
	cntk$logging$graph$find_all_with_name(
		node,
		node_name,
		to_int(depth)
	)
}

#' @export
graph_find_by_name <- function(node, node_name, depth = 0) {
	cntk$logging$graph$find_by_name(
		node,
		node_name,
		to_int(depth)
	)
}

#' @export
graph_get_node_outputs <- function(node, depth = 0) {
	cntk$logging$graph$get_node_outputs(
		node,
		depth = to_int(depth)
	)
}

#' @export
graph_plot <- function(root, filename = NULL) {
	cntk$logging$graph$plot(
		root,
		filename = filename
	)
}
