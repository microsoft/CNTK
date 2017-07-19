#' @export
default_sample_dir <- function() {
	cntk$sample_installer$default_sample_dir()
}

#' @export
default_sample_url <- function() {
	cntk$sample_installer$default_sample_url()
}

#' @export
install_samples <- function(url = NULL, directory = NULL, quiet = FALSE) {
	cntk$sample_installer$install_samples(
		url = url,
		directory = directory,
		quiet = quiet
	)
}

#' @export
module_is_unreleased <- function() {
	cntk$sample_installer$module_is_unreleased()
}
