# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
import cntk as C

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "DataSets", "MNIST")
model_path = os.path.join(abs_path, "Output", "Models")

# Define the reader for both training and evaluation action.
def create_reader(path, is_training, input_dim, label_dim):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features  = C.io.StreamDef(field='features', shape=input_dim),
        labels    = C.io.StreamDef(field='labels',   shape=label_dim)
    )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


# Trains and tests a simple auto encoder for MNIST images using deconvolution
def deconv_mnist(max_epochs=3):
    image_height = 28
    image_width  = 28
    num_channels = 1
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    # Input variable and normalization
    input_var = C.ops.input_variable((num_channels, image_height, image_width), np.float32)
    scaled_input = C.ops.element_times(C.ops.constant(0.00390625), input_var, name="input_node")

    # Define the auto encoder model
    cMap = 1
    conv1   = C.layers.Convolution2D  ((5,5), cMap, pad=True, activation=C.ops.relu)(scaled_input)
    pool1   = C.layers.MaxPooling   ((4,4), (4,4), name="pooling_node")(conv1)
    unpool1 = C.layers.MaxUnpooling ((4,4), (4,4))(pool1, conv1)
    z       = C.layers.ConvolutionTranspose2D((5,5), num_channels, pad=True, bias=False, init=C.glorot_uniform(0.001), name="output_node")(unpool1)

    # define rmse loss function (should be 'err = C.ops.minus(deconv1, scaled_input)')
    f2        = C.ops.element_times(C.ops.constant(0.00390625), input_var)
    err       = C.ops.reshape(C.ops.minus(z, f2), (784))
    sq_err    = C.ops.element_times(err, err)
    mse       = C.ops.reduce_mean(sq_err)
    rmse_loss = C.ops.sqrt(mse)
    rmse_eval = C.ops.sqrt(mse)

    reader_train = create_reader(os.path.join(data_path, 'Train-28x28_cntk_text.txt'), True, input_dim, num_output_classes)

    # training config
    epoch_size = 60000
    minibatch_size = 64

    # Set learning parameters
    lr_schedule = C.learning_parameter_schedule_per_sample([0.00015], epoch_size=epoch_size)
    mm_schedule = C.learners.momentum_schedule_per_sample([0.9983347214509387], epoch_size=epoch_size)

    # Instantiate the trainer object to drive the model training
    learner = C.learners.momentum_sgd(z.parameters, lr_schedule, mm_schedule, unit_gain=True)
    progress_printer = C.logging.ProgressPrinter(tag='Training')
    trainer = C.Trainer(z, (rmse_loss, rmse_eval), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var : reader_train.streams.features
    }

    C.logging.log_number_of_parameters(z) ; print()

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += data[input_var].num_samples                     # count samples processed so far

        trainer.summarize_training_progress()
        z.save(os.path.join(model_path, "07_Deconvolution_PY_{}.model".format(epoch)))

    # rename final model
    last_model_name = os.path.join(model_path, "07_Deconvolution_PY_{}.model".format(max_epochs - 1))
    final_model_name = os.path.join(model_path, "07_Deconvolution_PY.model")
    try:
        os.remove(final_model_name)
    except OSError:
        pass
    os.rename(last_model_name, final_model_name)
    
    # Load test data
    reader_test = create_reader(os.path.join(data_path, 'Test-28x28_cntk_text.txt'), False, input_dim, num_output_classes)

    input_map = {
        input_var : reader_test.streams.features
    }

    # Test data for trained model
    epoch_size = 10000
    minibatch_size = 1024

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[input_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    deconv_mnist()

