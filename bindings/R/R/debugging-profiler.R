#' @export
disable_profiler <- function() {
	cntk$debugging$profiler$disable_profiler()
}

#' @export
enable_profiler <- function() {
	cntk$debugging$profiler$enable_profiler()
}

#' @export
start_profiler <- function(dir = 'profiler', sync_gpu = TRUE,
						   reserve_mem = 33554432) {
	cntk$debugging$profiler$start_profiler(
		dir = dir,
		sync_gpu = sync_gpu,
		reserve_mem = to_int(reserve_mem)
	)
}

#' @export
stop_profiler <- function() {
	cntk$debugging$profiler$stop_profiler()
}
