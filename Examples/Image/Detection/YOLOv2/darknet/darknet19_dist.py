# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function

from cntk.layers import Sequential
from darknet.Distributed_Utils import  *
from darknet.darknet19 import create_classification_model_darknet19
from darknet.darknet19 import put_classifier_on_feature_extractor

import os



########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################


if __name__ == '__main__':
    # from cntk.cntk_py import force_deterministic_algorithms
    # force_deterministic_algorithms()

    import argparse, cntk

    parser = argparse.ArgumentParser()
    # data_path = os.path.join(abs_path, "..", "..", "..", "DataSets", "CIFAR-10")

    parser.add_argument('-datadir', '--datadir', help='Data directory where the CIFAR dataset is located',
                        required=False, default=par_data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False,
                        default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False,
                        default=par_max_epochs)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default=par_minibatch_size)
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

    data_path = args['datadir']
    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    if args['outputdir'] is not None:
        # model_path = args['outputdir'] + "/models"
        result_path = args['outputdir']
    else:
        result_path = par_abs_path
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        cntk.device.try_set_default_device(cntk.device.gpu(args['device']))

    train_data = os.path.join(data_path, par_trainset_label_file)
    test_data = os.path.join(data_path, par_testset_label_file)

    # create
    model = create_classification_model_darknet19(par_num_classes)  # num_classes from Utils
    #  and normalizes the input features by subtracting 114 and dividing by 256
    model2 = Sequential([[lambda x: (x - par_input_bias)], [lambda x: (x / 256)], model])

    if True:
        #pretrain on cifar-10
        pre_data_dir = os.path.join(data_path, "..", "cifar-10")
        pre_trainset_label_file = "train_map.txt"
        pre_testset_label_file = "test_map.txt"
        pre_train_data = os.path.join(pre_data_dir, pre_trainset_label_file)
        pre_test_data = os.path.join(pre_data_dir, pre_testset_label_file)

        without_classifier=cntk.logging.graph.find_by_name(model2,"featureExtractor_output")
        cifar_model = put_classifier_on_feature_extractor(without_classifier, 10)

        try:
            run_distributed(cifar_model, pre_train_data, pre_test_data, result_path,
                            minibatch_size=args['minibatch_size'],
                            epoch_size=args['epoch_size'],
                            num_quantization_bits=args['quantized_bits'],
                            block_size=args['block_samples'],
                            warm_up=args['distributed_after'],
                            max_epochs=args['num_epochs'],
                            restore=not args['restart'],
                            log_to_file=args['logdir'],
                            num_mbs_per_log=100,
                            gen_heartbeat=True,
                            profiling=args['profile'],
                            tensorboard_logdir=args['tensorboard_logdir'])

        finally:
            cntk.train.distributed.Communicator.finalize()



    try:
        run_distributed(model2, train_data, test_data, result_path,
                                minibatch_size=args['minibatch_size'],
                                epoch_size=args['epoch_size'],
                                num_quantization_bits=args['quantized_bits'],
                                block_size=args['block_samples'],
                                warm_up=args['distributed_after'],
                                max_epochs=args['num_epochs'],
                                restore=not args['restart'],
                                log_to_file=args['logdir'],
                                num_mbs_per_log=100,
                                gen_heartbeat=True,
                                profiling=args['profile'],
                                tensorboard_logdir=args['tensorboard_logdir'])

    finally:
        cntk.train.distributed.Communicator.finalize()

    # save final model!
    model.save(os.path.join(result_path, "Outputs", "darknet19_dist_" + par_dataset_name + ".model"))