#' Convolution2D Arguments
#'
#' @export
Conv2DArgs <- function() {
	cntk$crosstalk$Conv2DArgs()
}

#' Conv2D Variable Attribute
#'
#' @export
Conv2DAttr <- function() {
	cntk$crosstalk$conv2DAttr()
}

#' Crosstalk Base Class
#'
#' @export
Crosstalk <- function() {
	cntk$crosstalk$Crosstalk()
}

#' Assign Crosstalk Value
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param name string (optional) the name of the Function instance in the network
#' @param value
#' @param load
#' @param load_name
#'
#' @export
ct_assign <- function(ct, name, value = NULL, load = FALSE, load_name = NULL) {
	ct$assign(
		name,
		value = value,
		load = load,
		load_name = load_name
	)
}

#' Compare Crosstalk Var
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param name string (optional) the name of the Function instance in the network
#' @param compare_name
#' @param rtol
#' @param atol
#' @param equal_nan
#'
#' @export
ct_compare <- function(ct, name, compare_name = NULL, rtol = 1e-05, atol=1e-08,
					  equal_nan = FALSE) {
	ct$compare(
		name,
		compare_name = compare_name,
		rtol = rtol,
		atol = atol,
		equal_nan = equal_nan
	)
}

#' Fetch Crosstalk Var
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param name string (optional) the name of the Function instance in the network
#' @param save
#'
#' @export
ct_fetch <- function(ct, name, save = FALSE) {
	ct$fetch(name, save = save)
}

#' Load Crosstalk Vars
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
ct_load <- function(ct, names) {
	ct$load(names)
}

#' Load Crosstalk Raw Value
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
ct_load_raw_value <- function(ct, name) {
	ct$load_raw_value(name)
}

#' Next Crosstalk Pass
#'
#' @param ct Crosstalk instance on which to perform the operation
#'
#' @export
ct_next_pass <- function(ct) {
	ct$next_pass()
}

#' Register Crosstalk Var Type Getter/Setters
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param var_type
#' @param setter
#' @param getter
#'
#' @export
ct_register_funcs <- function(ct, var_type, setter = NULL, getter = NULL) {
	ct$register_funcs(
		var_type,
		setter = setter,
		getter = getter
	)
}

#' Reset Crosstalk Vars
#'
#' @param ct Crosstalk instance on which to perform the operation
#'
#' @export
ct_reset <- function(ct) {
	ct$reset()
}

#' Save Crosstalk Vars
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
ct_save <- function(ct, names) {
	ct$save(names)
}

#' Save All Crosstalk Vars
#'
#' @param ct Crosstalk instance on which to perform the operation
#'
#' @export
ct_save_all <- function(ct) {
	ct$save_all()
}

#' Set Crosstalk Working Directory
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param dir
#'
#' @export
ct_set_workdir <- function(ct, dir) {
	ct$set_workdir(dir)
}

#' Watch Crosstalk Variable
#'
#' @param ct Crosstalk instance on which to perform the operation
#' @param var
#' @param name string (optional) the name of the Function instance in the network
#' @param var_type
#' @param attr
#'
#' @export
ct_watch <- function(ct, var, name, var_type = NULL, attr = NULL) {
	ct$watch(
		var,
		name,
		var_type = var_type,
		attr = attr
	)
}


#' EmbedAttr
#'
#' @export
EmbedAttr <- function() {
	cntk$crosstalk$EmbedAttr()
}

#' Variable Setter/Getter Functions
#'
#' @export
FuncInfo <- function() {
	cntk$crosstalk$FuncInfo()
}

#' RNN Variable Arguments
#'
#' @export
RnnArgs <- function() {
	cntk$crosstalk$RnnArgs()
}

#' RNN Variable Attributes
#'
#' @export
RnnAttr <- function() {
	cntk$crosstalk$RnnAttr()
}

#' Variable Information
#'
#' @export
VarInfo <- function() {
	cntk$crosstalk$VarInfo()
}

#' CNTK Implementation for Crosstalk
#'
#' @export
CNTKCrosstalk <- function() {
	cntk$crosstalk$crosstalk_cntk$CNTKCrosstalk
}

#' CNTKCrosstalk Var is Parameter
#'
#' @param cct
#' @param name string (optional) the name of the Function instance in the network
#'
#' @export
ct_is_param <- function(cct, name) {
	cct$is_param(name)
}

#' Load All CNTKCrosstalk Params From Files
#'
#' @param cct
#'
#' @export
ct_load_all_params <- function(cct) {
	cct$load_all_params()
}

#' Save All CNTKCrosstalk Params to Files
#'
#' @param cct
#'
#' @export
ct_save_all_params <- function(cct) {
	cct$save_all_params()
}

#' Set CNTKCrosstalk Mapped Data
#'
#' @param cct
#' @param data
#'
#' @export
ct_set_data <- function(cct, data) {
	cct$set_data(data)
}

#' Find Parameter in Function - CNTKCrosstalk
#'
#' @param func
#' @param name string (optional) the name of the Function instance in the network
#' @param shape - list of ints representing tensor shape
#' @param allow_not_found
#'
#' @export
find_func_param <- function(func, name = NULL, shape = NULL,
							allow_not_found = FALSE) {
	cntk$crosstalk$crosstalk_cntk$find_func_param(
		func,
		name = name,
		shape = to_int(shape),
		allow_not_found = allow_not_found
	)
}
