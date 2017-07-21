device <- reticulate::import("cntk.device")

#' @export
DeviceDescriptor <- device$DeviceDescriptor

#' @export
DeviceKind <- device$DeviceKind

#' @export
all_devices <- device$all_devices

#' @export
get_gpu_properties <- device$get_gpu_properties

#' @export
set_excluded_devices <- device$set_excluded_devices

#' @export
try_set_default_device <- device$try_set_default_device

#' @export
use_default_device <- device$use_default_device
