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

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

# Define the reader for both training and evaluation action.
def create_reader(map_file, mean_file, train, total_number_of_samples, distributed_after=cntk.io.INFINITE_SAMPLES):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            cntk.io.ImageDeserializer.crop(crop_type='Random', ratio=0.8, jitter_type='uniRatio') # train uses jitter
        ]

    transforms += [
        cntk.io.ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        cntk.io.ImageDeserializer.mean(mean_file)
    ]

    # deserializer
    return cntk.io.MinibatchSource(
        cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
            features = cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = cntk.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        epoch_size=total_number_of_samples,
        multithreaded_deserializer = False,  # turn off omp as CIFAR-10 is not heavy for deserializer
        distributed_after = distributed_after)

# Train and evaluate the network.
def convnet_cifar10_dataaug(create_train_reader, test_reader, create_dist_learner, max_epochs=80, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=False):
    _cntk_py.set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = cntk.ops.input_variable((num_channels, image_height, image_width))
    label_var = cntk.ops.input_variable((num_classes))

    # apply model to input
    scaled_input = cntk.ops.element_times(cntk.ops.constant(0.00390625), input_var)
    
    with cntk.layers.default_options(activation=cntk.ops.relu, pad=True):
        z = cntk.models.Sequential([
            cntk.models.LayerStack(2, lambda : [
                cntk.layers.Convolution((3,3), 64),
                cntk.layers.Convolution((3,3), 64),
                cntk.layers.MaxPooling((3,3), (2,2))
            ]), 
            cntk.models.LayerStack(2, lambda i: [
                cntk.layers.Dense([256,128][i]), 
                cntk.layers.Dropout(0.5)
            ]), 
            cntk.layers.Dense(num_classes, activation=None)
        ])(scaled_input)

    # loss and metric
    ce = cntk.ops.cross_entropy_with_softmax(z, label_var)
    pe = cntk.ops.classification_error(z, label_var)

    # training config
    epoch_size = 50000  # for now we manually specify epoch size
    minibatch_size = 64

    # Set learning parameters
    lr_per_sample     = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046875]*10 + [0.000015625]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_sample, unit=cntk.learner.UnitType.sample, epoch_size=epoch_size)
    mm_time_constant  = [0]*20 + [600]*20 + [1200]
    mm_schedule       = cntk.learner.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)
    l2_reg_weight     = 0.002

    # trainer object
    learner = create_dist_learner(
        cntk.learner.momentum_sgd(z.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight))

    trainer = cntk.Trainer(z, ce, pe, learner)

    total_number_of_samples = max_epochs * epoch_size
    train_reader = create_train_reader(total_number_of_samples)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: train_reader.streams.features,
        label_var: train_reader.streams.labels
    }

    cntk.utils.log_number_of_parameters(z) ; print()
    progress_printer = cntk.utils.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        distributed_learner=learner,
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    # perform model training
    updated=True
    epoch=0
    
    while updated:
        data = train_reader.next_minibatch(minibatch_size, input_map=input_map)   # fetch minibatch.
        updated = trainer.train_minibatch(data)                                   # update model with it
        progress_printer.update_with_trainer(trainer, with_metric=True)           # log progress
        current_epoch = int(trainer.total_number_of_samples_seen/epoch_size)
        
        if epoch != current_epoch:
            progress_printer.epoch_summary(with_metric=True)
            epoch = current_epoch
            trainer.save_checkpoint(os.path.join(model_path, "ConvNet_CIFAR10_DataAug_{}.dnn".format(epoch)))

    ### Evaluation action
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    minibatch_index = 0

    while True:
        data = test_reader.next_minibatch(minibatch_size, input_map=input_map)
        if not data: break
        local_mb_samples=data[label_var].num_samples
        metric_numer += trainer.test_minibatch(data) * local_mb_samples
        metric_denom += local_mb_samples
        minibatch_index += 1


    fin_msg = "Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom)
    progress_printer.end_progress_print(fin_msg)

    print("")
    print(fin_msg)
    print("")

    return metric_numer/metric_denom

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', help='only interested in changes to that file');
    parser.add_argument('-logdir', help='only interested in changes by that user');
    parser.add_argument('-outputdir',  help='go straight to provided changelist');

    args = vars(parser.parse_args())

    if args['datadir'] != None:
        data_path = args['datadir']

    if args['logdir'] != None:
        log_dir = args['logdir']

    if args['outputdir'] != None:
        model_path = args['outputdir'] + "/models"

    distributed_after_samples = 0
    num_quantization_bits = 32

    create_dist_learner = \
        lambda learner: cntk.distributed.data_parallel_distributed_learner(learner,
                                                                           num_quantization_bits=num_quantization_bits,
                                                                           distributed_after=distributed_after_samples)

    mean=os.path.join(data_path, 'CIFAR-10_mean.xml')
    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'test_map.txt')

    create_train_reader = lambda data_size: create_reader(train_data, mean, True, data_size, distributed_after_samples)
    test_reader = create_reader(test_data, mean, False, cntk.io.FULL_DATA_SWEEP)

    convnet_cifar10_dataaug(create_train_reader, test_reader, create_dist_learner, log_to_file=log_dir, num_mbs_per_log=10, gen_heartbeat=False)

    cntk.distributed.Communicator.finalize()
