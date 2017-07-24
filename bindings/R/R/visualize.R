#' Visualize Network Architecture
#'
#' Visualize the network architecture as a dataflow graph
#' Describes the operations and how they interact with each other
#'
#' @param model CNTK network
#'
#' @return PNG plot
#' @export
#'
visualize_network <- function(model) {
	png_file <- tempfile(fileext = ".png")
	cntk$logging$graph$plot(model, png_file)
	img <- png::readPNG(png_file)
	grid::grid.raster(img)
}
