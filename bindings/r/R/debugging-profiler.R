profiler <- reticulate::import("cntk.debugging.profiler")

#' @export
disable_profiler <- function() {
	profiler$disable_profiler()
}

#' @export
enable_profiler <- function() {
	profiler$enable_profiler()
}

#' @export
start_profiler <- function(dir = 'profiler', sync_gpu = TRUE,
						   reserve_mem = 33554432) {
	profiler$start_profiler(
		dir = dir,
		sync_gpu = sync_gpu,
		reserve_mem = to_int(reserve_mem)
	)
}

#' @export
stop_profiler <- function() {
	profiler$stop_profiler()
}
