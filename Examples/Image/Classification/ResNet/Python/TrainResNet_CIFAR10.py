# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import cntk as C
import numpy as np

from cntk import cross_entropy_with_softmax, classification_error, reduce_mean
from cntk import Trainer, cntk_py
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.learners import momentum_sgd, learning_parameter_schedule_per_sample, momentum_schedule
from cntk.debugging import *
from cntk.logging import *
from resnet_models import *
import cntk.io.transforms as xforms

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3 # RGB
num_classes  = 10

# Define the reader for both training and evaluation action.
def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=(0.8, 1.0), jitter_type='uniratio') # train uses jitter
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=num_classes))),     # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)


# Train and evaluate the network.
def train_and_evaluate(reader_train, reader_test, network_name, epoch_size, max_epochs, profiler_dir=None,
                       model_dir=None, log_dir=None, tensorboard_logdir=None, gen_heartbeat=False, fp16=False):

    set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width), name='features')
    label_var = C.input_variable((num_classes))

    dtype = np.float16 if fp16 else np.float32
    if fp16:
        graph_input = C.cast(input_var, dtype=np.float16)
        graph_label = C.cast(label_var, dtype=np.float16)
    else:
        graph_input = input_var
        graph_label = label_var

    with C.default_options(dtype=dtype):
        # create model, and configure learning parameters
        if network_name == 'resnet20':
            z = create_cifar10_model(graph_input, 3, num_classes)
            lr_per_mb = [1.0]*80 + [0.1]*40 + [0.01]
        elif network_name == 'resnet110':
            z = create_cifar10_model(graph_input, 18, num_classes)
            lr_per_mb = [0.1]*1 + [1.0]*80 + [0.1]*40 + [0.01]
        else:
            raise RuntimeError("Unknown model name!")

        # loss and metric
        ce = cross_entropy_with_softmax(z, graph_label)
        pe = classification_error(z, graph_label)

    if fp16:
        ce = C.cast(ce, dtype=np.float32)
        pe = C.cast(pe, dtype=np.float32)

    # shared training parameters
    minibatch_size = 128
    l2_reg_weight = 0.0001

    # Set learning parameters
    lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=epoch_size)
    mm_schedule = momentum_schedule(0.9, minibatch_size)

    # progress writers
    progress_writers = [ProgressPrinter(tag='Training', log_to_file=log_dir, num_epochs=max_epochs, gen_heartbeat=gen_heartbeat)]
    tensorboard_writer = None
    if tensorboard_logdir is not None:
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z)
        progress_writers.append(tensorboard_writer)

    # trainer object
    learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                           l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(z, (ce, pe), learner, progress_writers)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()

    # perform model training
    if profiler_dir:
        start_profiler(profiler_dir, True)

    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far

        trainer.summarize_training_progress()

        # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
        if tensorboard_writer:
            for parameter in z.parameters:
                tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)

        if model_dir:
            z.save(os.path.join(model_dir, network_name + "_{}.dnn".format(epoch)))
        enable_profiler() # begin to collect profiler data after first epoch

    if profiler_dir:
        stop_profiler()

    # Evaluation parameters
    test_epoch_size = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    sample_count = 0

    while sample_count < test_epoch_size:
        current_minibatch = min(minibatch_size, test_epoch_size - sample_count)
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples

    print("")
    trainer.summarize_test_progress()
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='network type, resnet20 or resnet110', required=False, default='resnet20')
    parser.add_argument('-e', '--epochs', help='total epochs', type=int, required=False, default='160')
    parser.add_argument('-es', '--epoch_size', help='Size of epoch in samples', type=int, required=False, default='50000')
    parser.add_argument('-p', '--profiler_dir', help='directory for saving profiler output', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where TensorBoard logs should be created', required=False, default=None)
    parser.add_argument('-datadir', '--datadir', help='Data directory where the CIFAR dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-genheartbeat', '--genheartbeat', help="Turn on heart-beat for philly", action='store_true', default=False)
    parser.add_argument('-fp16', '--fp16', help="use float16", action='store_true', default=False)

    args = vars(parser.parse_args())
    epochs = args['epochs']
    epoch_size = args['epoch_size']
    network_name = args['network']

    model_dir = args['outputdir']
    if not model_dir:
        model_dir = os.path.join(abs_path, "Models")

    data_path = args['datadir']

    reader_train = create_image_mb_source(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True, total_number_of_samples=epochs * epoch_size)
    reader_test = create_image_mb_source(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False, total_number_of_samples=C.io.FULL_DATA_SWEEP)

    train_and_evaluate(reader_train, reader_test, network_name, epoch_size, epochs, args['profiler_dir'], model_dir,
                       args['logdir'], args['tensorboard_logdir'], gen_heartbeat=args['genheartbeat'], fp16=args['fp16'])
