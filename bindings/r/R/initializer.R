init <- reticulate::import("cntk.initializer")

#' @export
bilinear <- init$bilinear

#' @export
glorot_normal <- init$glorot_normal

#' @export
glorot_uniform <- init$glorot_uniform

#' @export
he_normal <- init$he_normal

#' @export
he_uniform <- init$he_uniform

#' @export
initializer_with_rank <- init$initializer_with_rank

#' @export
init_normal <- init$normal

#' @export
truncated_normal <- init$truncated_normal

#' @export
init_uniform <- init$uniform

#' @export
xavier <- init$xavier
