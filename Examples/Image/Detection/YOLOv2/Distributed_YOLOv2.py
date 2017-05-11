# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import argparse
import _cntk_py
import cntk

from cntk.logging import *
from cntk.io import FULL_DATA_SWEEP
from cntk import *
from cntk import leaky_relu, reshape, softmax, param_relu, relu, user_function
from cntk.logging import ProgressPrinter

import YOLOv2 as yolo2
from TrainUDFyolov2 import *
from PARAMETERS import *


# Create a minibatch source.
def create_image_mb_source(image_file, gtb_file, is_training, total_number_of_samples):

    return yolo2.create_mb_source(par_image_height, par_image_width, par_num_channels, (5 * par_max_gtbs), image_file,
                                        gtb_file, multithreaded_deserializer=True, randomize=is_training, max_samples=total_number_of_samples)


# Create trainer
def create_trainer(to_train, epoch_size, minibatch_size, num_quantization_bits, printer, block_size, warm_up):
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    lr_schedule = cntk.learning_rate_schedule(par_lr_schedule, cntk.learners.UnitType.sample,
                                              epoch_size)
    mm_schedule = cntk.learners.momentum_as_time_constant_schedule([-minibatch_size / np.log(par_momentum)], epoch_size)

    # Instantiate the trainer object to drive the model training
    local_learner = cntk.learners.momentum_sgd(to_train['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                         l2_regularization_weight=0.0005)

    # Create trainer
    if block_size != None:
        parameter_learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = data_parallel_distributed_learner(local_learner,
                                                              num_quantization_bits=num_quantization_bits,
                                                              distributed_after=warm_up)

    return cntk.Trainer(None, (to_train['mse'], to_train['mse']), parameter_learner, printer)


# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore):

    input_map = {
        network['feature']: train_source["features"],
        network['gtb_in']: train_source["label"]
    }

    # Train all minibatches
    training_session(
        trainer=trainer, mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=None, # CheckpointConfig(filename=os.path.join(model_path, model_name), restore=restore),
        test_config=TestConfig(source=test_source, mb_size=minibatch_size)
    ).train()


# Train and evaluate the network.
def yolov2_train_and_eval(network, image_file, gtb_file, num_quantization_bits=32, block_size=3200, warm_up=0,
                           minibatch_size=64, epoch_size=5000, max_epochs=1,
                           restore=True, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=True):
    _cntk_py.set_computation_network_trace_level(0)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log if num_mbs_per_log is not None else int(epoch_size/10),
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)


    trainer = create_trainer(network, epoch_size, minibatch_size, num_quantization_bits, progress_printer, block_size, warm_up)
    train_source = create_image_mb_source(image_file, gtb_file, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(image_file, gtb_file, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore)

    return network['output']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    data_path = os.path.join(par_abs_path, "..", "..", "DataSets","Pascal", "mappings")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located',
                        required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False,
                        default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False,
                        default=par_max_epochs)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default=par_minibatch_size)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default=par_epoch_size)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation',
                        type=int, required=False, default='32')
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)
    parser.add_argument('-b', '--block_samples', type=int,
                        help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)",
                        required=False, default=None)
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed',
                        type=int, required=False, default='0')

    args = vars(parser.parse_args())

    output_dir = None
    if args['outputdir'] is not None:
        output_dir = args['outputdir']
        model_path = args['outputdir'] + "/models"
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        # Setting one worker on GPU and one worker on CPU. Otherwise memory consumption is too high for a single GPU.
        if Communicator.rank() == 0:
            cntk.device.try_set_default_device(cntk.device.gpu(args['device']))
        else:
            cntk.device.try_set_default_device(cntk.device.cpu())

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    train_data = os.path.join(data_path, 'trainval2007.txt')
    test_data = os.path.join(data_path, 'trainval2007_rois_center_rel.txt')

    output = None

    ####################################################################################################################
    model = yolo2.create_yolov2_net()

    image_input = input((par_num_channels, par_image_height, par_image_width))
    output = model(image_input)  # append model to image input

    # input for ground truth boxes
    num_gtb = par_max_gtbs
    gtb_input = input((num_gtb * 5))  # 5 for class, x,y,w,h


    from ErrorFunction import get_error
    mse = get_error(output, gtb_input, cntk_only=False)

    network = {
        'feature': image_input,
        'gtb_in': gtb_input,
        'mse': mse,
        'output': output,
        #'trainfunction': ud_tf
    }

    ####################################################################################################################

    try:
        logdir = args['logdir']
        nr_of_epoch = args['num_epochs']

        output = yolov2_train_and_eval(network, train_data, test_data,
                                       max_epochs=nr_of_epoch,
                                       restore=not args['restart'],
                                       log_to_file=logdir,
                                       num_mbs_per_log=50,
                                       num_quantization_bits=args['quantized_bits'],
                                       block_size=args['block_samples'],
                                       warm_up=args['distributed_after'],
                                       minibatch_size=args['minibatch_size'],
                                       epoch_size=args['epoch_size'],
                                       gen_heartbeat=True)

    finally:
        cntk.train.distributed.Communicator.finalize()
        print("Training finished!")

    if output is not None and output_dir is not None:
        save_path = os.path.join(output_dir, "YOLOv2.model")
        output.save_model(save_path)
        print("Saved model to " + save_path)