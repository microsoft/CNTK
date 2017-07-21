#' @export
Conv2DArgs <- function() {
	cntk$crosstalk$Conv2DArgs()
}

#' @export
Conv2DAttr <- function() {
	cntk$crosstalk$conv2DAttr()
}

#' @export
Crosstalk <- function() {
	cntk$crosstalk$Crosstalk()
}

#' @param ct
#'
#' @param name
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

#' @param ct
#'
#' @param name
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

#' @param ct
#'
#' @param name
#' @param save
#'
#' @export
ct_fetch <- function(ct, name, save = FALSE) {
	ct$fetch(name, save = save)
}

#' @param ct
#'
#' @param names
#'
#' @export
ct_load <- function(ct, names) {
	ct$load(names)
}

#' @param ct
#'
#' @param name
#'
#' @export
ct_load_raw_value <- function(ct, name) {
	ct$load_raw_value(name)
}

#' @param ct
#'
#' @export
ct_next_pass <- function(ct) {
	ct$next_pass()
}

#' @param ct
#'
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

#' @param ct
#'
#' @export
ct_reset <- function(ct) {
	ct$reset()
}

#' @param ct
#'
#' @param names
#'
#' @export
ct_save <- function(ct, names) {
	ct$save(names)
}

#' @param ct
#'
#' @export
ct_save_all <- function(ct) {
	ct$save_all()
}

#' @param ct
#'
#' @param dir
#'
#' @export
ct_set_workdir <- function(ct, dir) {
	ct$set_workdir(dir)
}

#' @param ct
#'
#' @param var
#' @param name
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


#' @export
EmbedAttr <- function() {
	cntk$crosstalk$EmbedAttr()
}

#' @export
FuncInfo <- function() {
	cntk$crosstalk$FuncInfo()
}

#' @export
RnnArgs <- function() {
	cntk$crosstalk$RnnArgs()
}

#' @export
RnnAttr <- function() {
	cntk$crosstalk$RnnAttr()
}

#' @export
VarInfo <- function() {
	cntk$crosstalk$VarInfo()
}

#' @export
CNTKCrosstalk <- function() {
	cntk$crosstalk$crosstalk_cntk$CNTKCrosstalk
}

#' @param cct
#'
#' @param name
#'
#' @export
ct_is_param <- function(cct, name) {
	cct$is_param(name)
}

#' @param cct
#'
#' @export
ct_load_all_params <- function(cct) {
	cct$load_all_params()
}

#' @param cct
#'
#' @export
ct_save_all_params <- function(cct) {
	cct$save_all_params()
}

#' @param cct
#'
#' @param data
#'
#' @export
ct_set_data <- function(cct, data) {
	cct$set_data(data)
}

#' @param func
#'
#' @param name
#' @param shape
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
