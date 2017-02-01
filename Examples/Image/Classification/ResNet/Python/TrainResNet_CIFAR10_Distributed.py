# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import math
import cntk
import numpy as np

from cntk.utils import *
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk import Trainer, cntk_py 
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from _cntk_py import set_computation_network_trace_level
from cntk.device import set_default_device, gpu
from cntk.distributed import data_parallel_distributed_learner, block_momentum_distributed_learner, Communicator

from resnet_models import *

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

# For this example we are using the same data source as for conv net - CIFAR
sys.path.append(os.path.join(abs_path, "..", "..", "ConvNet", "Python"))
from ConvNet_CIFAR10_DataAug_Distributed import create_image_mb_source

# model dimensions - these match the ones from convnet_cifar10_dataaug
# so we can use the same data source
image_height = 32
image_width  = 32
num_channels = 3  # RGB
num_classes  = 10

# Create network
def create_resnet_network(network_name):
    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    # create model, and configure learning parameters 
    if network_name == 'resnet20': 
        z = create_cifar10_model(input_var, 3, num_classes)
    elif network_name == 'resnet110': 
        z = create_cifar10_model(input_var, 18, num_classes)
    else: 
        return RuntimeError("Unknown model name!")

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    return {
        'name' : network_name,
        'feature': input_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }


# Create trainer
def create_trainer(network, minibatch_size, epoch_size, num_quantization_bits, block_size, warm_up):
    if network['name'] == 'resnet20': 
        lr_per_mb = [1.0]*80+[0.1]*40+[0.01]
    elif network['name'] == 'resnet110': 
        lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
    else: 
        return RuntimeError("Unknown model name!")

    momentum_time_constant = -minibatch_size/np.log(0.9)
    l2_reg_weight = 0.0001

    # Set learning parameters
    lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    
    # learner object
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    local_learner = momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule,
                                 l2_regularization_weight = l2_reg_weight)

    if block_size != None:
        learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)
    
    return Trainer(network['output'], network['ce'], network['pe'], learner)

# Train and test
def train_and_test(network, trainer, train_source, test_source, progress_printer, minibatch_size, epoch_size):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    training_session = cntk.training_session(train_source, trainer,
        cntk.minibatch_size_schedule(minibatch_size), progress_printer, input_map, "ConvNet_CIFAR10_DataAug_", epoch_size)
    training_session.train()

    # TODO: Stay tuned for an upcoming simpler EvalSession API for test/validation.
    epoch_size     = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while True:
        data = test_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data: break;

        local_mb_samples=data[network['label']].num_samples
        metric_numer += trainer.test_minibatch(data) * local_mb_samples
        metric_denom += local_mb_samples
        minibatch_index += 1

    fin_msg = "Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom)
    progress_printer.end_progress_print(fin_msg)

    print("")
    print(fin_msg)
    print("")

    return metric_numer/metric_denom

# Train and evaluate the network.
def resnet_cifar10(train_data, test_data, mean_data, network_name, epoch_size, num_quantization_bits=32, block_size=3200, warm_up=0, max_epochs=5, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=False, scale_up=False):

    set_computation_network_trace_level(0)
    
    # NOTE: scaling up minibatch_size increases sample throughput. In 8-GPU machine,
    # ResNet110 samples-per-second is ~7x of single GPU, comparing to ~3x without scaling
    # up. However, bigger minimatch size on the same number of samples means less updates, 
    # thus leads to higher training error. This is a trade-off of speed and accuracy
    minibatch_size = 128 * (Communicator.num_workers() if scale_up else 1)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_resnet_network(network_name)
    trainer = create_trainer(network, minibatch_size, epoch_size, num_quantization_bits, block_size, warm_up)
    train_source = create_image_mb_source(train_data, mean_data, train=True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, train=False, total_number_of_samples=cntk.io.FULL_DATA_SWEEP)
    return train_and_test(network, trainer, train_source, test_source, progress_printer, minibatch_size, epoch_size)


if __name__=='__main__':

    data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='network type, resnet20 or resnet110', required=False, default='resnet20')
    parser.add_argument('-s', '--scale_up', help='scale up minibatch size with #workers for better parallelism', type=bool, required=False, default='False')
    parser.add_argument('-d', '--datadir', help='Data directory where the CIFAR dataset is located', required=False, default=data_path)
    parser.add_argument('-o', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-l', '--log', help='Log file', required=False, default=None)
    parser.add_argument('-e', '--epochs', help='Total number of epochs to train', type=int, required=False, default='160')
    parser.add_argument('-es', '--epoch_size', help='Size of epoch in samples', type=int, required=False, default=None)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)


    args = vars(parser.parse_args())

    epoch_size = 50000
    if args['outputdir'] != None:
        model_path = args['o'] + "/models"
    if args['device'] != None:
        set_default_device(gpu(args['device']))
    if args['datadir'] is not None:
        data_path = args['datadir']
    if args['epoch_size'] is not None:
        epoch_size = args['epoch_size']

    mean_data=os.path.join(data_path, 'CIFAR-10_mean.xml')
    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'test_map.txt')

    num_quantization_bits = args['quantized_bits']
    epochs = args['epochs']
    warm_up = args['distributed_after']
    network_name = args['network']
    scale_up = bool(args['scale_up'])

    # Create distributed trainer factory
    print("Start training: quantize_bit = {}, epochs = {}, distributed_after = {}".format(num_quantization_bits, epochs, warm_up))

    try:
        resnet_cifar10(train_data, test_data, mean_data,
                       network_name, 
                       epoch_size,
                       num_quantization_bits,
                       block_size=args['block_samples'],
                       warm_up=args['distributed_after'],
                       max_epochs=epochs,
                       scale_up=scale_up,
                       log_to_file=args['log'])
    finally:
        # Must call MPI finalize when process exit
        Communicator.finalize()
