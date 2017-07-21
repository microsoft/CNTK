py <- reticulate::import_builtins(convert = FALSE)

Record <- "banana"

#' @export
mapping <- py$dict

to_int = function(num) {
	if (is.null(num)) {
		return(NULL)
	}
	as.integer(num)
}
