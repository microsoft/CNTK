#' @export
DeviceDescriptor <- function(...) {
	cntk$device$DeviceDescriptor(...)
}

#' @export
DeviceKind <- function() {
	cntk$device$DeviceKind()
}

#' @export
all_devices <- function() {
	cntk$device$all_devices()
}

#' @export
cpu_descriptor <- function() {
	cntk$cntk$device$cpu()
}

#' @export
get_gpu_properties <- function(device) {
	cntk$device$get_gpu_properties
}

#' @export
set_excluded_devices <- function(excluded_devices) {
	cntk$device$set_excluded_devices(excluded_devices)
}

#' @export
try_set_default_device <- function(new_default_device,
								   acquire_device_lock = FALSE) {
	cntk$device$try_set_default_device(
		new_default_device,
		acquire_device_lock = acquire_device_lock
	)
}

#' @export
use_default_device <- function() {
	cntk$device$use_default_device()
}
