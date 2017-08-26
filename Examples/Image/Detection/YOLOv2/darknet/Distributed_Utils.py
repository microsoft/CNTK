# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function

from PARAMETERS import *
import cntk
from cntk.train.training_session import *
from cntk.debugging import *
# from Utils import *
import cntk.io.transforms as xforms
import _cntk_py
import os
import numpy as np




def train_and_test(network, trainer, train_source, test_source, checkpoint_config, minibatch_size, epoch_size, restore, profiling=False):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # Train all minibatches
    if profiling:
        start_profiler(sync_gpu=True)

    training_session(
        trainer=trainer, mb_source = train_source,
        model_inputs_to_streams = input_map,
        mb_size = minibatch_size,
        progress_frequency = epoch_size,
        checkpoint_config = checkpoint_config,
        test_config = TestConfig(source = test_source, mb_size=minibatch_size)
    ).train()

    if profiling:
        stop_profiler()


# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers):
    # Set learning parameters
    lr_factor = 0.01
    lr_per_sample     = [1*lr_factor]*10 + [0.1*lr_factor]*10 + [0.01*lr_factor]*10 + [0.001*lr_factor]*10 + [0.0001*lr_factor]*20 + [0.00001*lr_factor]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_sample, unit=cntk.learners.UnitType.sample, epoch_size=epoch_size)
    mm_time_constant  = [-par_minibatch_size / np.log(0.9)]
    mm_schedule       = cntk.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)
    l2_reg_weight     = 0.0005

    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    # fails here since input is unknown I guess
    local_learner = cntk.learners.momentum_sgd(network['output'].parameters,
                                          lr_schedule, mm_schedule,
                                          l2_regularization_weight=l2_reg_weight, unit_gain=False)

    if block_size != None:
        parameter_learner = cntk.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = cntk.train.distributed.data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    # Create trainer
    return cntk.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_writers)


def add_model_input_and_output(model, input_shape, output_shape):
    # Input variables denoting the features and label data
    feature_var = cntk.input(input_shape)
    label_var = cntk.input(output_shape)

    z = model(feature_var)

    # loss and metric
    sse = cntk.squared_error(z, label_var)
    bce = cntk.binary_cross_entropy(z, label_var)
    ce = cntk.cross_entropy_with_softmax(z, label_var)
    pe = cntk.classification_error(z, label_var)

    cntk.logging.log_number_of_parameters(z);
    print()

    return {
        'feature': feature_var,
        'label': label_var,
        'sse' : sse,
        'bce' : bce,
        'ce': ce,
        'pe': pe,
        'output': z
    }


# Create a minibatch source.
def create_image_mb_source(map_file, train, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %
                           (map_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
        ]

    transforms += [
        xforms.scale(width=par_image_width, height=par_image_height, channels=par_num_channels, interpolations='linear')
    ]

    # deserializer
    return cntk.io.MinibatchSource(
        cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
            features = cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = cntk.io.StreamDef(field='label', shape=par_num_classes))),   # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer = True)


def run_distributed(model, train_data, test_data, result_path, minibatch_size=par_minibatch_size, epoch_size=50000, num_quantization_bits=32,
                            block_size=3200, warm_up=0, max_epochs=2, restore=False, log_to_file=None,
                            num_mbs_per_log=None, gen_heartbeat=False, profiling=False, tensorboard_logdir=None):

    _cntk_py.set_computation_network_trace_level(0)

    network = add_model_input_and_output(model, (par_num_channels , par_image_height, par_image_width), (par_num_classes))

    print("Created Model!")
    epoch_size = 1200000
    progress_writers = [cntk.logging.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=cntk.train.distributed.Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)]

    if tensorboard_logdir is not None:
        progress_writers.append(cntk.logging.TensorBoardProgressWriter(
            freq=num_mbs_per_log,
            log_dir=tensorboard_logdir,
            rank=cntk.train.distributed.Communicator.rank(),
            model=model['output']))


    import Utils
    trainer = create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers)
    # train_source = Utils.create_reader(train_data,is_training=True, is_distributed=True)
    # test_source = Utils.create_reader(test_data, is_training=False, is_distributed= True)
    train_source = create_image_mb_source(train_data, train=True,
                                          total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, train=False,
                                         total_number_of_samples=cntk.io.FULL_DATA_SWEEP)

    checkpoint_conf = CheckpointConfig(frequency=epoch_size,
                     filename=os.path.join(result_path, "Checkpoints" , str(model.name) + ".model"),
                     restore=restore)

    train_and_test(network, trainer, train_source, test_source, checkpoint_conf, minibatch_size, epoch_size, restore, profiling)


