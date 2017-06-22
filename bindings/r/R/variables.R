variables <- reticulate::import("cntk.variables")

#' @export
Constant <- variables$Constant

#' @export
Parameter <- variables$Parameter

#' @export
Record <- variables$Record

#' @export
Variable <- variables$Variable

#' @export
VariableMixin <- variables$VariableMixin
