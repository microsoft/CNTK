#' Disable Profiler
#'
#' @export
disable_profiler <- function() {
	cntk$debugging$profiler$disable_profiler()
}

#' Enable Profiler
#'
#' @export
enable_profiler <- function() {
	cntk$debugging$profiler$enable_profiler()
}

#' Start Profiler
#'
#' @param dir
#' @param sync_gpu
#' @param reserve_mem
#'
#' @export
start_profiler <- function(dir = 'profiler', sync_gpu = TRUE,
						   reserve_mem = 33554432) {
	cntk$debugging$profiler$start_profiler(
		dir = dir,
		sync_gpu = sync_gpu,
		reserve_mem = to_int(reserve_mem)
	)
}

#' Stop Profiler
#'
#' @export
stop_profiler <- function() {
	cntk$debugging$profiler$stop_profiler()
}
