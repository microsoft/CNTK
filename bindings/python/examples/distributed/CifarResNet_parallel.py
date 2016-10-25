# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer, distributed, device, persist
from cntk.learner import momentum_sgd, learning_rate_schedule
from cntk.ops import input_variable, constant, parameter, cross_entropy_with_softmax, combine, classification_error, times, element_times, pooling, AVG_POOLING, relu
from cntk.io import ReaderConfig, ImageDeserializer
from cntk.initializer import he_normal, glorot_uniform

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import conv_bn_relu_layer, conv_bn_layer, linear_layer, print_training_progress

TRAIN_MAP_FILENAME = 'train_map.txt'
MEAN_FILENAME = 'CIFAR-10_mean.xml'
TEST_MAP_FILENAME = 'test_map.txt'

# Instantiates the CNTK built-in minibatch source for reading images to be used for training the residual net
# The minibatch source is configured using a hierarchical dictionary of key:value pairs

def create_mb_source(features_stream_name, labels_stream_name, image_height,
                     image_width, num_channels, num_classes, cifar_data_path,
                     distributed_communicator):

    path = os.path.normpath(os.path.join(abs_path, cifar_data_path))
    map_file = os.path.join(path, TRAIN_MAP_FILENAME)
    mean_file = os.path.join(path, MEAN_FILENAME)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' does not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
                           (map_file, mean_file, cifar_py3, cifar_py3))

    image = ImageDeserializer(map_file)
    image.map_features(features_stream_name,
            [ImageDeserializer.crop(crop_type='Random', ratio=0.8,
                jitter_type='uniRatio'),
             ImageDeserializer.scale(width=image_width, height=image_height,
                 channels=num_channels, interpolations='linear'),
             ImageDeserializer.mean(mean_file)])
    image.map_labels(labels_stream_name, num_classes)

    rc = ReaderConfig(image, epoch_size=sys.maxsize)
    return rc.minibatch_source(distributed_communicator)

def create_test_mb_source(features_stream_name, labels_stream_name, image_height,
                     image_width, num_channels, num_classes, cifar_data_path):

    path = os.path.normpath(os.path.join(abs_path, cifar_data_path))

    map_file = os.path.join(path, TEST_MAP_FILENAME)
    mean_file = os.path.join(path, MEAN_FILENAME)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' does not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
                           (map_file, mean_file, cifar_py3, cifar_py3))

    image = ImageDeserializer(map_file)
    image.map_features(features_stream_name,
            [ImageDeserializer.scale(width=image_width, height=image_height,
                 channels=num_channels, interpolations='linear'),
             ImageDeserializer.mean(mean_file)])
    image.map_labels(labels_stream_name, num_classes)

    rc = ReaderConfig(image, epoch_size=sys.maxsize)
    return rc.minibatch_source()

def resnet_basic(input, out_feature_map_count, bn_time_const):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, [3, 3], [1, 1], bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, [3, 3], [1, 1], bn_time_const)
    p = c2 + input
    return relu(p)

def resnet_basic_inc(input, out_feature_map_count, strides, bn_time_const):
    c1 = conv_bn_relu_layer(input, out_feature_map_count, [3, 3], strides, bn_time_const)
    c2 = conv_bn_layer(c1, out_feature_map_count, [3, 3], [1, 1], bn_time_const)
    s  = conv_bn_layer(input, out_feature_map_count, [1, 1], strides, bn_time_const)
    p = c2 + s
    return relu(p)

def resnet_basic_stack2(input, out_feature_map_count, bn_time_const):
    r1 = resnet_basic(input, out_feature_map_count, bn_time_const)
    r2 = resnet_basic(r1, out_feature_map_count, bn_time_const)
    return r2

def resnet_basic_stack3(input, out_feature_map_count, bn_time_const):
    r12 = resnet_basic_stack2(input, out_feature_map_count, bn_time_const)
    r3 = resnet_basic(r12, out_feature_map_count, bn_time_const)
    return r3
    
# Defines the residual network model for classifying images
def resnet_classifer(input, num_classes):
    bn_time_const = 4096

    c_map1 = 16
    
    feat_scale = 0.00390625

    input_norm = element_times(feat_scale, input)

    conv = conv_bn_relu_layer(input, c_map1, [3, 3], [1, 1], bn_time_const)
    r1_1 = resnet_basic_stack3(conv, c_map1, bn_time_const)

    c_map2 = 32

    r2_1 = resnet_basic_inc(r1_1, c_map2, [2, 2], bn_time_const)
    r2_2 = resnet_basic_stack2(r2_1, c_map2, bn_time_const)

    c_map3 = 64
    r3_1 = resnet_basic_inc(r2_2, c_map3, [2, 2], bn_time_const)
    r3_2 = resnet_basic_stack2(r3_1, c_map3, bn_time_const)

    # Global average pooling
    poolw = 8
    poolh = 8
    poolh_stride = 1
    poolv_stride = 1

    pool = pooling(r3_2, AVG_POOLING, (1, poolh, poolw), (1, poolv_stride, poolh_stride))
    return linear_layer(pool, num_classes)

# Trains a residual network model on the Cifar image dataset
def cifar_resnet(base_path, run_test, num_epochs, communicator=None, save_model_filename=None, load_model_filename=None, debug_output=False):
    image_height = 32
    image_width = 32
    num_channels = 3
    num_classes = 10

    feats_stream_name = 'features'
    labels_stream_name = 'labels'

    minibatch_source = create_mb_source(feats_stream_name, labels_stream_name, 
                        image_height, image_width, num_channels, num_classes, base_path,
                        distributed_communicator = communicator)

    features_si = minibatch_source[feats_stream_name]
    labels_si = minibatch_source[labels_stream_name]

    # Instantiate the resnet classification model, or load from file
    
    if load_model_filename:
        print("Loading model:", load_model_filename)
        classifier_output = persist.load_model('float', load_model_filename)
        image_input = classifier_output.arguments[0]
    else:
        image_input = input_variable(
            (num_channels, image_height, image_width), features_si.m_element_type)
        classifier_output = resnet_classifer(image_input, num_classes)

    # Input variables denoting the features and label data
    label_var = input_variable((num_classes), features_si.m_element_type)

    ce = cross_entropy_with_softmax(classifier_output, label_var)
    pe = classification_error(classifier_output, label_var)

    # Instantiate the trainer object to drive the model training

    mb_size = 128
    num_mb_per_epoch = 100
    
    num_mbs = num_mb_per_epoch * num_epochs

    lr_per_sample = [1/mb_size]*80+[0.1/mb_size]*40+[0.01/mb_size]
    lr_schedule = learning_rate_schedule(lr_per_sample, units = mb_size * num_mb_per_epoch)
    momentum_time_constant = -mb_size/np.log(0.9)

    # create data parallel distributed trainer if needed
    dist_trainer = distributed.data_parallel_distributed_trainer(communicator, False) if communicator else None

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(classifier_output, ce, pe,
                      [momentum_sgd(classifier_output.parameters, lr_schedule, momentum_time_constant, l2_regularization_weight=0.0001)],
                      distributed_trainer = dist_trainer)
    
    # Get minibatches of images to train with and perform model training
    training_progress_output_freq = 100 if communicator else 20

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/4
        
    for i in range(0, num_mbs):
        mb = minibatch_source.next_minibatch(mb_size)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        arguments = {
                image_input: mb[features_si], 
                label_var: mb[labels_si]
                }
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)
        
    if save_model_filename:
        print("Saving model:", save_model_filename)
        persist.save_model(classifier_output, save_model_filename)

    if run_test:
        test_minibatch_source = create_test_mb_source(feats_stream_name, labels_stream_name,
                        image_height, image_width, num_channels, num_classes, base_path)
        features_si = test_minibatch_source[feats_stream_name]
        labels_si = test_minibatch_source[labels_stream_name]

        mb_size = 128
        num_mbs = 100

        total_error = 0.0
        for i in range(0, num_mbs):
            mb = test_minibatch_source.next_minibatch(mb_size)

            # Specify the mapping of input variables in the model to actual
            # minibatch data to be trained with
            arguments = {
                    image_input: mb[features_si], 
                    label_var: mb[labels_si]
                    }
            error = trainer.test_minibatch(arguments)
            total_error += error

        return total_error / num_mbs
    else:
        return 0

if __name__ == '__main__':
    base_path = os.path.abspath(os.path.normpath(os.path.join(
        *"../../../../Examples/Image/Datasets/CIFAR-10/".split("/"))))

    os.chdir(base_path)

    # Create distributed communicator for 1-bit SGD
    communicator = distributed.communicator(distributed.quantized_mpi_communicator(1))
    workers = communicator.workers()
    current_worker = communicator.current_worker()
    print("List all distributed workers")
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            print("* {} {}".format(wk.global_rank, wk.host_id))
        else:
            print("  {} {}".format(wk.global_rank, wk.host_id))

    print("Training on device type:{} id:{}".format('gpu' if device.default().type() else 'cpu', device.default().id()))

    start_model = "start_model.bin"
    num_start_epochs = 1
    num_parallel_epochs = 10

    # training the start model only in one worker
    if communicator.current_worker().global_rank == 0:
        cifar_resnet(base_path, save_model_filename=start_model, communicator=None, run_test=False, num_epochs=num_start_epochs)
    
    communicator.barrier()
    
    # train in parallel
    error = cifar_resnet(base_path, load_model_filename=start_model, communicator=communicator, run_test=True, num_epochs=num_parallel_epochs)
    
    print("Error: %f" % error)

    distributed.communicator.finalize()