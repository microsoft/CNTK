# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer, DeviceDescriptor
from cntk.learner import sgd
from cntk.ops import input_variable, constant, parameter, cross_entropy_with_softmax, combine, classification_error, times, pooling, AVG_POOLING
from cntk.io import ReaderConfig, ImageDeserializer
from cntk.initializer import glorot_uniform

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import conv_bn_relu_layer, conv_bn_layer, resnet_node2, resnet_node2_inc, print_training_progress

TRAIN_MAP_FILENAME = 'train_map.txt'
MEAN_FILENAME = 'CIFAR-10_mean.xml'
TEST_MAP_FILENAME = 'test_map.txt'

# Instantiates the CNTK built-in minibatch source for reading images to be used for training the residual net
# The minibatch source is configured using a hierarchical dictionary of key:value pairs

def create_mb_source(features_stream_name, labels_stream_name, image_height,
                     image_width, num_channels, num_classes, cifar_data_path):

    path = os.path.normpath(os.path.join(abs_path, cifar_data_path))
    map_file = os.path.join(path, TRAIN_MAP_FILENAME)
    mean_file = os.path.join(path, MEAN_FILENAME)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' do not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
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
    return rc.minibatch_source()

def create_test_mb_source(features_stream_name, labels_stream_name, image_height,
                     image_width, num_channels, num_classes, cifar_data_path):

    path = os.path.normpath(os.path.join(abs_path, cifar_data_path))

    map_file = os.path.join(path, TEST_MAP_FILENAME)
    mean_file = os.path.join(path, MEAN_FILENAME)

    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        cifar_py3 = "" if sys.version_info.major < 3 else "_py3"
        raise RuntimeError("File '%s' or '%s' do not exist. Please run CifarDownload%s.py and CifarConverter%s.py from CIFAR-10 to fetch them" %
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
    return rc.minibatch_source()

def get_projection_map(out_dim, in_dim):
    if in_dim > out_dim:
        raise ValueError(
            "Can only project from lower to higher dimensionality")

    projection_map_values = np.zeros(in_dim * out_dim, dtype=np.float32)
    for i in range(0, in_dim):
        projection_map_values[(i * in_dim) + i] = 1.0
        shape = (out_dim, in_dim, 1, 1)
        return constant(value=projection_map_values.reshape(shape))

# Defines the residual network model for classifying images
def resnet_classifer(input, num_classes):
    conv_w_scale = 7.07
    conv_b_value = 0

    fc1_w_scale = 0.4
    fc1_b_value = 0

    sc_value = 1
    bn_time_const = 4096

    kernel_width = 3
    kernel_height = 3

    conv1_w_scale = 0.26
    c_map1 = 16

    conv1 = conv_bn_relu_layer(input, c_map1, kernel_width, kernel_height,
                               1, 1, conv1_w_scale, conv_b_value, sc_value, bn_time_const)
    rn1_1 = resnet_node2(conv1, c_map1, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)
    rn1_2 = resnet_node2(rn1_1, c_map1, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)
    rn1_3 = resnet_node2(rn1_2, c_map1, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)

    c_map2 = 32
    rn2_1_wProj = get_projection_map(c_map2, c_map1)
    rn2_1 = resnet_node2_inc(rn1_3, c_map2, kernel_width, kernel_height,
                             conv1_w_scale, conv_b_value, sc_value, bn_time_const, rn2_1_wProj)
    rn2_2 = resnet_node2(rn2_1, c_map2, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)
    rn2_3 = resnet_node2(rn2_2, c_map2, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)

    c_map3 = 64
    rn3_1_wProj = get_projection_map(c_map3, c_map2)
    rn3_1 = resnet_node2_inc(rn2_3, c_map3, kernel_width, kernel_height,
                             conv1_w_scale, conv_b_value, sc_value, bn_time_const, rn3_1_wProj)
    rn3_2 = resnet_node2(rn3_1, c_map3, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)
    rn3_3 = resnet_node2(rn3_2, c_map3, kernel_width, kernel_height,
                         conv1_w_scale, conv_b_value, sc_value, bn_time_const)

    # Global average pooling
    poolw = 8
    poolh = 8
    poolh_stride = 1
    poolv_stride = 1

    pool = pooling(rn3_3, AVG_POOLING, (1, poolh, poolw), (1, poolv_stride, poolh_stride))
    out_times_params = parameter(shape=(c_map3, 1, 1, num_classes), init=glorot_uniform())
    out_bias_params = parameter(shape=(num_classes), init=0)
    t = times(pool, out_times_params)
    return t + out_bias_params

# Trains a residual network model on the Cifar image dataset
def cifar_resnet(base_path, debug_output=False):
    image_height = 32
    image_width = 32
    num_channels = 3
    num_classes = 10
    feats_stream_name = 'features'
    labels_stream_name = 'labels'

    minibatch_source = create_mb_source(feats_stream_name, labels_stream_name, 
                        image_height, image_width, num_channels, num_classes, base_path)
    features_si = minibatch_source[feats_stream_name]
    labels_si = minibatch_source[labels_stream_name]

    # Input variables denoting the features and label data
    image_input = input_variable(
        (num_channels, image_height, image_width), features_si.m_element_type)
    label_var = input_variable((num_classes), features_si.m_element_type)

    # Instantiate the resnet classification model
    classifier_output = resnet_classifer(image_input, num_classes)

    ce = cross_entropy_with_softmax(classifier_output, label_var)
    pe = classification_error(classifier_output, label_var)

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(classifier_output, ce, pe,
                      [sgd(classifier_output.parameters(), lr=0.0078125)])

    # Get minibatches of images to train with and perform model training
    mb_size = 32
    training_progress_output_freq = 60
    num_mbs = 1000

    if debug_output:
        training_progress_output_freq = training_progress_output_freq/3

    for i in range(0, num_mbs):
        mb = minibatch_source.get_next_minibatch(mb_size)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        arguments = {
                image_input: mb[features_si], 
                label_var: mb[labels_si]
                }
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)

    test_minibatch_source = create_test_mb_source(feats_stream_name, labels_stream_name,
                    image_height, image_width, num_channels, num_classes, base_path)
    features_si = test_minibatch_source[feats_stream_name]
    labels_si = test_minibatch_source[labels_stream_name]

    mb_size = 64
    num_mbs = 300

    total_error = 0.0
    for i in range(0, num_mbs):
        mb = test_minibatch_source.get_next_minibatch(mb_size)

        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        arguments = {image_input: mb[
            features_si].m_data, label_var: mb[labels_si].m_data}
        error = trainer.test_minibatch(arguments)
        total_error += error

    return total_error / num_mbs

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # target_device = DeviceDescriptor.cpu_device()
    # DeviceDescriptor.set_default_device(target_device)

    base_path = os.path.normpath(os.path.join(
        *"../../../../Examples/Image/Miscellaneous/CIFAR-10/cifar-10-batches-py".split("/")))

    os.chdir(os.path.join(base_path, '..'))

    error = cifar_resnet(base_path)
    print("Error: %f" % error)
