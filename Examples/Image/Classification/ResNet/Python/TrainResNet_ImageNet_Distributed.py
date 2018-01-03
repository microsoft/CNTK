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

from cntk import input, cross_entropy_with_softmax, classification_error, Trainer, cntk_py
from cntk import data_parallel_distributed_learner, block_momentum_distributed_learner, Communicator
from cntk.learners import nesterov, learning_rate_schedule, momentum_schedule, UnitType
from cntk.device import try_set_default_device, gpu
from cntk.train.training_session import *
from cntk.debugging import *
from cntk.logging import *
from resnet_models import *
import cntk.io.transforms as xforms

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "ImageNet")
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3 # RGB
num_classes  = 1000
model_name   = "ResNet_ImageNet.model"

def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist." % (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomarea', area_ratio=(0.08, 1.0), aspect_ratio=(0.75, 1.3333), jitter_type='uniratio'), # train uses jitter
            xforms.color(brightness_radius=0.4, contrast_radius=0.4, saturation_radius=0.4)
        ]
    else:
        transforms += [
            C.io.transforms.crop(crop_type='center', side_ratio=0.875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='cubic'),
        xforms.mean(mean_file)
    ]

    # deserializer
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(map_file, C.io.StreamDefs(
            features=C.io.StreamDef(field='image', transforms=transforms), # 1st col in mapfile referred to as 'image'
            labels=C.io.StreamDef(field='label', shape=num_classes))),     # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)

# Create network
def create_resnet_network(network_name):
    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    stride1x1 = (1, 1)
    stride3x3 = (2, 2)

    # create model, and configure learning parameters
    if network_name == 'resnet18':
        z = create_imagenet_model_basic(input_var, [2, 1, 1, 2], num_classes)
    elif network_name == 'resnet34':
        z = create_imagenet_model_basic(input_var, [3, 3, 5, 2], num_classes)
    elif network_name == 'resnet50':
        z = create_imagenet_model_bottleneck(input_var, [2, 3, 5, 2], num_classes, stride1x1, stride3x3)
    elif network_name == 'resnet101':
        z = create_imagenet_model_bottleneck(input_var, [2, 3, 22, 2], num_classes, stride1x1, stride3x3)
    elif network_name == 'resnet152':
        z = create_imagenet_model_bottleneck(input_var, [2, 7, 35, 2], num_classes, stride1x1, stride3x3)
    else:
        return RuntimeError("Unknown model name!")

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    errs = classification_error(z, label_var, topN=1)
    top5Errs = classification_error(z, label_var, topN=5)

    return {
        'name' : network_name,
        'feature': input_var,
        'label': label_var,
        'ce' : ce,
        'errs' : errs,
        'top5Errs' : top5Errs,
        'output': z
    }


# Create trainer
def create_trainer(network, minibatch_size, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer):
    lr_per_mb = [1.0]*30 + [0.1]*30 + [0.01]*20 + [0.001]
    l2_reg_weight = 0.0001

    # adjust LR with minibatch size
    if minibatch_size != 256:
        for i in range(0, len(lr_per_mb)):
            lr_per_mb[i] *= minibatch_size / 256

    # Set learning parameters
    lr_schedule = learning_rate_schedule(lr_per_mb, epoch_size=epoch_size, unit=UnitType.minibatch)
    mm_schedule = momentum_schedule(0.9)

    local_learner = nesterov(network['output'].parameters, lr_schedule, mm_schedule,
                             l2_regularization_weight=l2_reg_weight)

    # learner object
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    if block_size != None:
        learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        learner = data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    return Trainer(network['output'], (network['ce'], network['errs']), learner, progress_printer)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore=True, profiling=False):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    if profiling:
        start_profiler(sync_gpu=True)

    training_session(
        trainer=trainer,
        mb_source=train_source,
        mb_size=minibatch_size,
        model_inputs_to_streams=input_map,
        progress_frequency=epoch_size,
        checkpoint_config=CheckpointConfig(frequency=epoch_size, filename=os.path.join(model_path, model_name), restore=restore),
        test_config=TestConfig(test_source, minibatch_size)
    ).train()

    if profiling:
        stop_profiler()

# Train and evaluate the network.
def resnet_imagenet(train_data, test_data, mean_data, network_name, epoch_size, num_quantization_bits=32, block_size=None, warm_up=0, 
                    max_epochs=90, restore=True, log_to_file=None, num_mbs_per_log=100, gen_heartbeat=False, scale_up=False, profiling=False):

    set_computation_network_trace_level(0)

    # NOTE: scaling up minibatch_size increases sample throughput. In 8-GPU machine,
    # ResNet110 samples-per-second is ~7x of single GPU, comparing to ~3x without scaling
    # up. However, bigger minimatch size on the same number of samples means less updates,
    # thus leads to higher training error. This is a trade-off of speed and accuracy
    minibatch_size = 32 * (Communicator.num_workers() if scale_up else 1)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_resnet_network(network_name)
    trainer = create_trainer(network, minibatch_size, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer)
    train_source = create_image_mb_source(train_data, mean_data, train=True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, mean_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling)


if __name__=='__main__':
    data_path = os.path.join(abs_path, "..", "..", "..", "DataSets", "ImageNet")

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='network type, resnet18, 34, 50, 101, 152', required=False, default='resnet50')
    parser.add_argument('-s', '--scale_up', help='scale up minibatch size with #workers for better parallelism', type=bool, required=False, default='True')
    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-e', '--epochs', help='Total number of epochs to train', type=int, required=False, default='90')
    parser.add_argument('-es', '--epoch_size', help='Size of epoch in samples', type=int, required=False, default='1281167')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true', default=False)
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)

    args = vars(parser.parse_args())

    if args['outputdir'] != None:
        model_path = args['outputdir'] + "/models"
    if args['device'] != None:
        try_set_default_device(gpu(args['device']))

    if args['epoch_size'] is not None:
        epoch_size = args['epoch_size']

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    mean_data = os.path.join(abs_path, 'ImageNet1K_mean.xml')
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'val_map.txt')

    num_quantization_bits = args['quantized_bits']
    epochs = args['epochs']
    warm_up = args['distributed_after']
    network_name = args['network']
    scale_up = bool(args['scale_up'])

    # Create distributed trainer factory
    print("Start training: quantize_bit = {}, epochs = {}, distributed_after = {}".format(num_quantization_bits, epochs, warm_up))

    resnet_imagenet(train_data, test_data, mean_data,
                    network_name,
                    epoch_size,
                    num_quantization_bits,
                    block_size=args['block_samples'],
                    warm_up=args['distributed_after'],
                    max_epochs=epochs,
                    restore=not args['restart'],
                    scale_up=scale_up,
                    log_to_file=args['logdir'],
                    profiling=args['profile'])

    # Must call MPI finalize when process exit without exceptions
    Communicator.finalize()
