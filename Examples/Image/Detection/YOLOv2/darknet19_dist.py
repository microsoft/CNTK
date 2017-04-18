# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from cntk import leaky_relu, reshape, softmax
from cntk.layers import Convolution2D,BatchNormalization, MaxPooling, GlobalAveragePooling, Sequential, Activation, \
    default_options
from cntk.train.training_session import *
from cntk.logging import *
from cntk.debugging import *
from Utils import *
import os


# Creates the feature extractor shared by the classifier (Darknet19) and the Detector (YOLOv2)
def create_feature_extractor(filter_multiplier=32):
    with default_options(activation=leaky_relu):
        net = Sequential([
            Convolution2D(filter_shape=(3,3), num_filters=filter_multiplier, pad=True, name="feature_layer"),
            BatchNormalization(),
            MaxPooling(filter_shape=(2,2), strides=(2,2)),
            # Output: in_x/2 x in_y/2 x nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**1), pad=True, name="stage_1"),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output: in_x/4 x in_y/4 x 2*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**2), pad=True, name="stage_2"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**1), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**2), pad=True),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output in_x/8 x in_y/8 x 4*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**3), pad=True, name="stage_3"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**2), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**3), pad=True),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output in_x/16 x in_y/16 x 8*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**4), pad=True, name="stage_4"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**3), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**4), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**3), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**4), pad=True, name="YOLOv2PasstroughSource"),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output in_x/32 x in_y/32 x 16*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**5), pad=True, name="stage_5"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**4), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**5), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**4), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**5), pad=True),
            BatchNormalization(name="featureExtractor_output")
            # Output in_x/32 x in_y/32 x 32*nfilters
        ],'featureExtractor_darknet19')

    return net


# Puts a classifier end to any feature extractor
def put_classifier_on_feature_extractor(featureExtractor,nrOfClasses):
    return Sequential([
        reshape(x=Sequential([
            # [lambda x: x - 114],
            featureExtractor,
            Convolution2D(filter_shape=(1, 1), num_filters=nrOfClasses, pad=True, activation=identity,
                          name="classifier_input"),
            GlobalAveragePooling()
        ]), shape=(nrOfClasses)),
        Activation(activation=softmax, name="classifier_output")
    ], name="darknet19-classifier")


# Creates a Darknet19 classifier
def create_classification_model_darknet19(nrOfClasses, filter_mult=32):
    featureExtractor = create_feature_extractor(filter_mult)
    return put_classifier_on_feature_extractor(featureExtractor, nrOfClasses)


# Saves a model to the Output folder. If the models are already existing an ascending number is assigned to the model.
def save_model(model, name="darknet19"):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, "Output", name + ".model")
    if os.path.exists(model_path):
        i = 1
        while (os.path.exists(model_path)):
            i += 1
            model_path = os.path.join(abs_path, "Output", name + "_" + str(i) + ".model")

    model.save(model_path)
    print("Stored model " + name + " to " + model_path)
    return model_path


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################

def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling=False):

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
        progress_frequency=epoch_size,
        checkpoint_config = CheckpointConfig(frequency = epoch_size,
                                             filename = os.path.join(model_path, "ConvNet_CIFAR10_DataAug"),
                                             restore = restore),
        test_config = TestConfig(source = test_source, mb_size=minibatch_size)
    ).train()

    if profiling:
        stop_profiler()

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers):
    # Set learning parameters
    lr_per_sample     = [0.0015625]*20 + [0.00046875]*20 + [0.00015625]*20 + [0.000046875]*10 + [0.000015625]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_sample, unit=cntk.learners.UnitType.sample, epoch_size=epoch_size)
    mm_time_constant  = [0]*20 + [600]*20 + [1200]
    mm_schedule       = cntk.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)
    l2_reg_weight     = 0.002

    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    # fails here since input is unknown I guess
    local_learner = cntk.learners.momentum_sgd(network.parameters,
                                              lr_schedule, mm_schedule,
                                              l2_regularization_weight=l2_reg_weight)

    if block_size != None:
        parameter_learner = cntk.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = cntk.train.distributed.data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    criterion = create_criterion_function(network)

    # Create trainer
    return cntk.Trainer(network, criterion, parameter_learner, progress_writers)

def run_distributed(train_data, test_data, minibatch_size=64, epoch_size=50000, num_quantization_bits=32,
                            block_size=3200, warm_up=0, max_epochs=2, restore=False, log_to_file=None,
                            num_mbs_per_log=None, gen_heartbeat=False, profiling=False, tensorboard_logdir=None):

    #_cntk_py.set_computation_network_trace_level(0)

    # create
    model = create_classification_model_darknet19(num_classes)  # num_classes from Utils
    #  and normalizes the input features by subtracting 114 and dividing by 256
    model2 = Sequential([[lambda x: (x - par_input_bias)], [lambda x: (x / 256)], model])
    print("Created Model!")

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
            model=model2['output']))



    trainer = create_trainer(model2, epoch_size, num_quantization_bits, block_size, warm_up, progress_writers)
    train_source = create_reader(train_data,is_training=True, is_distributed=True)
    test_source = create_reader(test_data, is_training=False, is_distributed= True)
    train_and_test(model2, trainer, train_source, test_source, minibatch_size, epoch_size, restore, profiling)


    # train
    # reader = create_reader(os.path.join(data_path, par_trainset_label_file), is_training=True)
    # print("Created Readers!")

    # train_model(reader, model2, max_epochs=par_max_epochs, exponentShift=-1)
    # save
    # save_model(model, "darknet19_" + par_dataset_name)

    # from cntk.logging.graph import plot
    # plot(model, filename=os.path.join(par_abs_path, "darknet19_" + par_dataset_name + "_DataAug.pdf"))

    # test
    # reader = create_reader(os.path.join(data_path, par_testset_label_file), is_training=False)
    # evaluate_model(reader, model2)

    print("Done!")


if __name__ == '__main__':
    # from cntk.cntk_py import force_deterministic_algorithms
    # force_deterministic_algorithms()

    import argparse, cntk

    parser = argparse.ArgumentParser()
    data_path = par_data_path # os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the CIFAR dataset is located',
                        required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False,
                        default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False,
                        default='160')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='64')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='50000')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation',
                        type=int, required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed',
                        type=int, required=False, default='0')
    parser.add_argument('-b', '--block_samples', type=int,
                        help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)",
                        required=False, default=None)
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)
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

    # mean_data = os.path.join(data_path, 'CIFAR-10_mean.xml')
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'test_map.txt')

    try:
        run_distributed(train_data, test_data,
                                minibatch_size=args['minibatch_size'],
                                epoch_size=args['epoch_size'],
                                num_quantization_bits=args['quantized_bits'],
                                block_size=args['block_samples'],
                                warm_up=args['distributed_after'],
                                max_epochs=args['num_epochs'],
                                restore=not args['restart'],
                                log_to_file=args['logdir'],
                                num_mbs_per_log=100,
                                gen_heartbeat=False,
                                profiling=args['profile'],
                                tensorboard_logdir=args['tensorboard_logdir'])
    finally:
        cntk.train.distributed.Communicator.finalize()