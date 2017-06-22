library(cntk)
library(magrittr)

generate_data <- function(sample_size, feature_dim, num_classes) {
	# create synthetic data
	Y <- replicate(1, sample(0:num_classes, sample_size, rep = TRUE))
	
	# make sure the data is separable
	M <- matrix(rnorm(sample_size * feature_dim, 3), ncol = feature_dim)
	X <- M * as.vector(Y + 1)

	# one-hot encoding
	class_ind <- matrix(nrow = sample_size, ncol = num_classes)
	for (class in 1:num_classes) {
		class_ind[,class] <- as.numeric(Y == class)
	}
	Y <- class_ind

	list(X, Y)
}

inputs <- 2
outputs <- 2
layers <- 2
hidden_dimension <- 50

# input variables denoting features and label data
features <- input_variable(inputs)
label <- input_variable(outputs)

# instantiate the feedforward classification model
my_model <- sequential(
	layer_dense(hidden_dimension, activation = sigmoid),
	layer_dense(outputs)
)
z <- my_model(features)

ce <- cross_entropy_with_softmax(z, label)
pe <- metric_classification_error(z, label)

# instantiate trainer object
lr_per_minibatch <- learning_rate_schedule(0.125, UnitType$minibatch)
progress_printer <- ProgressPrinter()
optimizer <- sgd(z$parameters, lr = lr_per_minibatch)
trainer <- Trainer(z, c(ce, pe), c(optimizer), c(progress_printer))

# get minibatches of training data and train model
minibatch_size <- 25
num_minibatches <- 1024

aggregate_loss = 0L
for (i in 1:num_minibatches) {
	training_data <- generate_data(minibatch_size, inputs, outputs)

	# map input variable to minibatch data
	trainer %>% train_minibatch(c("features", "label"), training_data)
	#     trainer$train_minibatch(dict(features = train[1], label = train[2]))
	sample_count <- trainer$previous_minibatch_sample_count
	loss <- trainer$previous_minibatch_loss_average * sample_count
	aggregate_loss <- aggregate_loss + loss
}

last_avg_error <- aggregate_loss / trainer$total_number_of_samples_seen

test <- generate_data(minibatch_size, inputs, outputs)
error <- trainer$test_minibatch(dict(features = test[1], label = test[1]))
sprintf('Error rate on an unseen minibatch: %f',  error)
