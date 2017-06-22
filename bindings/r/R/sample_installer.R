sample_installer <- reticulate::import("cntk.sample_installer")

#' @export
default_sample_dir <- sample_installer$default_sample_dir

#' @export
default_sample_url <- sample_installer$default_sample_url

#' @export
install_samples <- sample_installer$install_samples

#' @export
module_is_unreleased <- sample_installer$module_is_unreleased
