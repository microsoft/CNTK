#' Default Sample Dir
#'
#' @export
default_sample_dir <- function() {
	cntk$sample_installer$default_sample_dir()
}

#' Default Sample URL
#'
#' @export
default_sample_url <- function() {
	cntk$sample_installer$default_sample_url()
}

#' Install Samples
#'
#' Fetch the CNTK samples from a URL, extract to local directory, and install
#' Python package requirements.
#'
#' @param url
#' @param directory
#' @param quiet
#'
#' @export
install_samples <- function(url = NULL, directory = NULL, quiet = FALSE) {
	cntk$sample_installer$install_samples(
		url = url,
		directory = directory,
		quiet = quiet
	)
}

#' Module Is Unreleased
#'
#' @export
module_is_unreleased <- function() {
	cntk$sample_installer$module_is_unreleased()
}
