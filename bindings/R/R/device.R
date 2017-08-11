#' Create Device Descriptor
#'
#' @param ...
#'
#' @export
DeviceDescriptor <- function(...) {
	cntk$device$DeviceDescriptor(...)
}

#' Create Device Kind
#'
#' @export
DeviceKind <- function() {
	cntk$device$DeviceKind()
}

#' All Devices
#'
#' @export
all_devices <- function() {
	cntk$device$all_devices()
}

#' CPU Descriptor
#'
#' @export
cpu_descriptor <- function() {
	cntk$cntk$device$cpu()
}

#' Get GPU Properties
#'
#' @param device - instance of DeviceDescriptor
#'
#' @export
get_gpu_properties <- function(device) {
	cntk$device$get_gpu_properties
}

#' Set Excluded Devices
#'
#' @param excluded_devices
#'
#' @export
set_excluded_devices <- function(excluded_devices) {
	cntk$device$set_excluded_devices(excluded_devices)
}

#' Try Set Default Device
#'
#' @param new_default_device
#' @param acquire_device_lock
#'
#' @export
try_set_default_device <- function(new_default_device,
								   acquire_device_lock = FALSE) {
	cntk$device$try_set_default_device(
		new_default_device,
		acquire_device_lock = acquire_device_lock
	)
}

#' Use Default Device
#'
#' @export
use_default_device <- function() {
	cntk$device$use_default_device()
}
