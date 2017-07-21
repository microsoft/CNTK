train <- reticulate::import("cntk.train")
np <- reticulate::import("numpy")

Trainer <- train$Trainer

.to_numpy <- function(x) {
	lapply(x, np$float32, dtype = np$float32)
}

.combine <- function(names, data) {
	combined <- vector("list")
	for (i in 1:length(names)) {
		combined[[names[i]]] <- np$array(data[i], dtype = np$float32)
	}
	return(combined)
}

#' @export
train_minibatch <- function(trainer, data, outputs = NULL, device = NULL) {
	#     if (class(data[[1]]) == 'list') {
	#         data <- .to_numpy(data)
	#     }
	trainer$train_minibatch(data, outputs, device)
}

#' @export
test_minibatch <- function(trainer, arguments, device = NULL) {
	trainer$test_minibatch(
		arguments,
		device = device
	)
}
