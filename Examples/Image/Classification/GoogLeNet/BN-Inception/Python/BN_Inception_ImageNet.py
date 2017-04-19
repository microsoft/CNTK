# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import argparse
import numpy as np
import cntk
import _cntk_py

import cntk.io.transforms as xforms
from cntk.logging import *
from cntk.ops import *
from cntk.distributed import data_parallel_distributed_learner, Communicator
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.debugging import *

from BN_Inception import bn_inception_model

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "..", "DataSets", "ImageNet")
config_path = abs_path
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 1000
model_name   = "BN-Inception.model"

# Create a minibatch source.
def create_image_mb_source(map_file, mean_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist." %
                          (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.875, jitter_type='uniratio') # train uses jitter
        ]
    else: 
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize = is_training, 
        multithreaded_deserializer = True)

# Create the network.
def create_bn_inception():

    # Input variables denoting the features and label data
    feature_var = input((num_channels, image_height, image_width))
    label_var = input((num_classes))

    bn_time_const = 4096
    z = bn_inception_model(feature_var, num_classes, bn_time_const)

    # loss and metric
    ce  = cross_entropy_with_softmax(z, label_var)
    pe  = classification_error(z, label_var)
    pe5 = classification_error(z, label_var, topN=5)

    log_number_of_parameters(z)
    print()

    return {
        'feature': feature_var,
        'label'  : label_var,
        'ce'     : ce,
        'pe'     : pe,
        'pe5'    : pe5, 
        'output' : z
    }

# Create trainer
def create_trainer(network, epoch_size, num_epochs, minibatch_size):

    # CNTK weights new gradient by (1-momentum) for unit gain, 
    # thus we divide Caffe's learning rate by (1-momentum)
    initial_learning_rate = 0.45 # equal to 0.045 in caffe
    initial_learning_rate *= minibatch_size / 32
    learn_rate_adjust_interval = 2
    learn_rate_decrease_factor = 0.94

    # Set learning parameters
    lr_per_mb = []
    learning_rate = initial_learning_rate
    for i in range(0, num_epochs, learn_rate_adjust_interval):
        lr_per_mb.extend([learning_rate] * learn_rate_adjust_interval)
        learning_rate *= learn_rate_decrease_factor

    lr_schedule       = cntk.learning_rate_schedule(lr_per_mb, unit=cntk.learner.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = cntk.learner.momentum_schedule(0.9)
    l2_reg_weight     = 0.0001 # CNTK L2 regularization is per sample, thus same as Caffe
    
    # Create learner
    learner = cntk.learner.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, 
                                            l2_regularization_weight=l2_reg_weight)

    # Create trainer
    return cntk.Trainer(network['output'], (network['ce'], network['pe']), learner)

# Train and test
def train_and_test(network, trainer, train_source, test_source, progress_printer, max_epochs, minibatch_size, epoch_size, restore, profiler_dir):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # perform model training
    if profiler_dir:
        start_profiler(profiler_dir, True)

    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = train_source.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        progress_printer.epoch_summary(with_metric=True)
        network['output'].save(os.path.join(model_path, "BN-Inception_{}.model".format(epoch)))
        enable_profiler() # begin to collect profiler data after first epoch

    if profiler_dir:
        stop_profiler()

    # Finished
    # Evaluation parameters
    test_epoch_size = 50000
    test_minibatch_size = 32

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0
    
    while sample_count < test_epoch_size:
        current_minibatch = min(test_minibatch_size, test_epoch_size - sample_count)
        # Fetch next test min batch.
        data = test_source.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += data[network['label']].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

# Train and evaluate the network.
def bn_inception_train_and_eval(train_data, test_data, mean_data, minibatch_size=32, epoch_size=1281167, max_epochs=300, 
                         restore=True, log_to_file=None, num_mbs_per_log=100, gen_heartbeat=False, profiler_dir=None):
    _cntk_py.set_computation_network_trace_level(1)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_bn_inception()
    trainer = create_trainer(network, epoch_size, max_epochs, minibatch_size)
    train_source = create_image_mb_source(train_data, mean_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, progress_printer, max_epochs, minibatch_size, epoch_size, restore, profiler_dir)
 
 
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the cifar-10 dataset is located', required=False, default=data_path)
    parser.add_argument('-configdir', '--configdir', help='Config directory where this python script is located', required=False, default=config_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-profilerdir', '--profilerdir', help='Directory for saving profiler output', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='300')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='32')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='1281167')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']
    if args['logdir'] is not None:
        log_dir = args['logdir']    
    if args['profilerdir'] is not None:
        profiler_dir = args['profilerdir']
    if args['device'] is not None:
        cntk.device.set_default_device(cntk.device.gpu(args['device']))

    mean_data = os.path.join(data_path, 'ImageNet1K_mean.xml')
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'val_map.txt')

    # Find the mean file
    if not os.path.exists(mean_data):
        mean_data = os.path.join(config_path, 'ImageNet1K_mean.xml')
    if not os.path.exists(mean_data):
        mean_data = os.path.join(abs_path, 'ImageNet1K_mean.xml')
    if not os.path.exists(mean_data):
        raise RuntimeError("Can not find the mean file. Please put the 'ImageNet1K_mean.xml' file in Data Directory or Config Directory.")

    bn_inception_train_and_eval(train_data, test_data, mean_data,
                             minibatch_size=args['minibatch_size'], 
                             epoch_size=args['epoch_size'],
                             max_epochs=args['num_epochs'],
                             restore=not args['restart'],
                             log_to_file=args['logdir'],
                             num_mbs_per_log=100,
                             gen_heartbeat=True,
                             profiler_dir=args['profilerdir'])
