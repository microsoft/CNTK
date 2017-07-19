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

#' @export
ct_assign <- function(ct, name, value = NULL, load = FALSE, load_name = NULL) {
	ct$assign(
		name,
		value = value,
		load = load,
		load_name = load_name
	)
}

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

#' @export
ct_fetch <- function(ct, name, save = FALSE) {
	ct$fetch(name, save = save)
}

#' @export
ct_load <- function(ct, names) {
	ct$load(names)
}

#' @export
ct_load_raw_value <- function(ct, name) {
	ct$load_raw_value(name)
}

#' @export
ct_next_pass <- function(ct) {
	ct$next_pass()
}

#' @export
ct_register_funcs <- function(ct, var_type, setter = NULL, getter = NULL) {
	ct$register_funcs(
		var_type,
		setter = setter,
		getter = getter
	)
}

#' @export
ct_reset <- function(ct) {
	ct$reset()
}

#' @export
ct_save <- function(ct, names) {
	ct$save(names)
}

#' @export
ct_save_all <- function(ct) {
	ct$save_all()
}

#' @export
ct_set_workdir <- function(ct, dir) {
	ct$set_workdir(dir)
}

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

#' @export
ct_is_param <- function(cct, name) {
	cct$is_param(name)
}

#' @export
ct_load_all_params <- function(cct) {
	cct$load_all_params()
}

#' @export
ct_save_all_params <- function(cct) {
	cct$save_all_params()
}

#' @export
ct_set_data <- function(cct, data) {
	cct$set_data(data)
}

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
