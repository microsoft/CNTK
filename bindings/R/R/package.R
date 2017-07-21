
#' @import reticulate
NULL

cntk <- NULL
np <- NULL
sys <- NULL
type_map <- NULL

to_int <- function(num) {
	if (is.numeric(num)) {
		return(as.integer(num))
	}
	num
}

.onLoad <- function(libname, pkgname) {
	cntk <<- import('cntk', delay_load = TRUE)
	np <<- import('numpy', delay_load = TRUE)
	sys <<- import('sys', delay_load = TRUE)
	type_map <<- function(type) {
		if (is.character(type)) {
			return(list(
				'float32' = np$float32,
				'float64' = np$float64,
			    'auto' = NULL
			)[[type]])
		}
		type
	}
}
