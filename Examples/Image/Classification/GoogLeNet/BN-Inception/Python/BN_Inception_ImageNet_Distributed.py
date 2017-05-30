# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from __future__ import division

import os
import math
import argparse
import numpy as np
import cntk
import _cntk_py

import cntk.io.transforms as xforms
from cntk.debugging import start_profiler, stop_profiler
from cntk.learners import learning_rate_schedule, momentum_schedule, momentum_sgd, UnitType
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.train.distributed import data_parallel_distributed_learner, Communicator
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.train import training_session, CheckpointConfig, TestConfig, Trainer

from BN_Inception_ImageNet import create_image_mb_source, create_bn_inception

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "..", "DataSets", "ImageNet")
config_path = abs_path
model_path = os.path.join(abs_path, "Models")
log_dir = None

model_name   = "BN-Inception.model"

# Create trainer
def create_trainer(network, epoch_size, num_epochs, minibatch_size, num_quantization_bits, progress_printer):
    
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

    lr_schedule       = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = momentum_schedule(0.9)
    l2_reg_weight     = 0.0001 # CNTK L2 regularization is per sample, thus same as Caffe
    
    # Create learner
    local_learner = momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule,
                                                l2_regularization_weight=l2_reg_weight)
    parameter_learner = data_parallel_distributed_learner(
        local_learner, 
        num_quantization_bits=num_quantization_bits,
        distributed_after=0)

    # Create trainer
    return Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_printer)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling=False):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    if profiling:
        start_profiler(sync_gpu=True)

    training_session(
        trainer=trainer, mb_source=train_source,
        model_inputs_to_streams = input_map,
        mb_size = minibatch_size,
        progress_frequency = epoch_size,
        checkpoint_config=CheckpointConfig(frequency=epoch_size, filename=os.path.join(model_path, model_name), restore=restore),
        test_config=TestConfig(test_source, minibatch_size=minibatch_size)
    ).train()
        
    if profiling:
        stop_profiler()

# Train and evaluate the network.
def bn_inception_train_and_eval(train_data, test_data, mean_data, num_quantization_bits=32, epoch_size=1281167, max_epochs=300, minibatch_size=None,
                         restore=True, log_to_file=None, num_mbs_per_log=100, gen_heartbeat=False, scale_up=False, profiling=False):
    _cntk_py.set_computation_network_trace_level(0)

    # NOTE: scaling up minibatch_size increases sample throughput. In 8-GPU machine,
    # ResNet110 samples-per-second is ~7x of single GPU, comparing to ~3x without scaling
    # up. However, bigger minibatch size on the same number of samples means less updates,
    # thus leads to higher training error. This is a trade-off of speed and accuracy
    if minibatch_size is None:
        mb_size = 32 * (Communicator.num_workers() if scale_up else 1)
    else:
        mb_size = minibatch_size

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_bn_inception()
    trainer = create_trainer(network, epoch_size, max_epochs, mb_size, num_quantization_bits, progress_printer)
    train_source = create_image_mb_source(train_data, mean_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, mb_size, epoch_size, restore, profiling)
 
 
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located', required=False, default=data_path)
    parser.add_argument('-configdir', '--configdir', help='Config directory where this python script is located', required=False, default=config_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='300')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='64')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='1281167')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-s', '--scale_up', help='scale up minibatch size with #workers for better parallelism', type=bool, required=False, default='True')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        cntk.device.try_set_default_device(cntk.device.gpu(args['device']))

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    os.chdir(data_path)

    mean_data = os.path.join(data_path, 'ImageNet1K_mean.xml')
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'val_map.txt')

    bn_inception_train_and_eval(train_data, test_data, mean_data,
                                epoch_size=args['epoch_size'],
                                num_quantization_bits=args['quantized_bits'],
                                max_epochs=args['num_epochs'],
                                minibatch_size=args["minibatch_size"],
                                restore=not args['restart'],
                                log_to_file=args['logdir'],
                                num_mbs_per_log=100,
                                gen_heartbeat=True,
                                scale_up=bool(args['scale_up']))

    # Must call MPI finalize when process exit without exceptions
    Communicator.finalize()
