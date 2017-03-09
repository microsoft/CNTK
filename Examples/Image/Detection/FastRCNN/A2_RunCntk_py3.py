# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from cntk import Trainer, UnitType, load_model
from cntk.blocks import Placeholder, Constant
from cntk.graph import find_by_name, plot
from cntk.initializer import glorot_uniform
from cntk.io import ReaderConfig, ImageDeserializer, CTFDeserializer
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.ops import input_variable, parameter, cross_entropy_with_softmax, classification_error, times, combine
from cntk.ops import roipooling
from cntk.ops.functions import CloneMethod
from cntk.utils import log_number_of_parameters, ProgressPrinter
from PARAMETERS import *
import numpy as np
import os, sys

###############################################################
###############################################################
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))

# file and stream names
train_map_filename = 'train.txt'
test_map_filename = 'test.txt'
rois_filename_postfix = '.rois.txt'
roilabels_filename_postfix = '.roilabels.txt'
features_stream_name = 'features'
roi_stream_name = 'rois'
label_stream_name = 'roiLabels'

# from PARAMETERS.py
base_path = cntkFilesDir
num_channels = 3
image_height = cntk_padHeight
image_width = cntk_padWidth
num_classes = nrClasses
num_rois = cntk_nrRois
epoch_size = cntk_num_train_images
num_test_images = cntk_num_test_images
mb_size = cntk_mb_size
max_epochs = cntk_max_epochs
momentum_time_constant = cntk_momentum_time_constant

# model specific variables (only AlexNet for now)
base_model = "AlexNet"
if base_model == "AlexNet":
    model_file = "../../../../../PretrainedModels/AlexNet.model"
    feature_node_name = "features"
    last_conv_node_name = "conv5.y"
    pool_node_name = "pool3"
    last_hidden_node_name = "h2_d"
    roi_dim = 6
else:
    raise ValueError('unknown base model: %s' % base_model)
###############################################################
###############################################################


# Instantiates a composite minibatch source for reading images, roi coordinates and roi labels for training Fast R-CNN
def create_mb_source(img_height, img_width, img_channels, n_classes, n_rois, data_path, data_set):
    rois_dim = 4 * n_rois
    label_dim = n_classes * n_rois

    path = os.path.normpath(os.path.join(abs_path, data_path))
    if data_set == 'test':
        map_file = os.path.join(path, test_map_filename)
    else:
        map_file = os.path.join(path, train_map_filename)
    roi_file = os.path.join(path, data_set + rois_filename_postfix)
    label_file = os.path.join(path, data_set + roilabels_filename_postfix)

    if not os.path.exists(map_file) or not os.path.exists(roi_file) or not os.path.exists(label_file):
        raise RuntimeError("File '%s', '%s' or '%s' does not exist. "
                           "Please run install_fastrcnn.py from Examples/Image/Detection/FastRCNN to fetch them" %
                           (map_file, roi_file, label_file))

    # read images
    image_source = ImageDeserializer(map_file)
    image_source.ignore_labels()
    image_source.map_features(features_stream_name,
                              [ImageDeserializer.scale(width=img_width, height=img_height, channels=img_channels,
                                                       scale_mode="pad", pad_value=114, interpolations='linear')])

    # read rois and labels
    roi_source = CTFDeserializer(roi_file)
    roi_source.map_input(roi_stream_name, dim=rois_dim, format="dense")
    label_source = CTFDeserializer(label_file)
    label_source.map_input(label_stream_name, dim=label_dim, format="dense")

    # define a composite reader
    rc = ReaderConfig([image_source, roi_source, label_source], epoch_size=sys.maxsize, randomize=data_set == "train")
    return rc.minibatch_source()


# Defines the Fast R-CNN network model for detecting objects in images
def frcn_predictor(features, rois, n_classes):
    # Load the pretrained classification net and find nodes
    loaded_model = load_model(model_file)
    feature_node = find_by_name(loaded_model, feature_node_name)
    conv_node    = find_by_name(loaded_model, last_conv_node_name)
    pool_node    = find_by_name(loaded_model, pool_node_name)
    last_node    = find_by_name(loaded_model, last_hidden_node_name)

    # Clone the conv layers and the fully connected layers of the network
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: Placeholder()})
    fc_layers = combine([last_node.owner]).clone(CloneMethod.clone, {pool_node: Placeholder()})

    # Create the Fast R-CNN model
    feat_norm = features - Constant(114)
    conv_out  = conv_layers(feat_norm)
    roi_out   = roipooling(conv_out, rois, (roi_dim, roi_dim))
    fc_out    = fc_layers(roi_out)

    # z = Dense(rois[0], num_classes, map_rank=1)(fc_out)  # --> map_rank=1 is not yet supported
    W = parameter(shape=(4096, n_classes), init=glorot_uniform())
    b = parameter(shape=n_classes, init=0)
    z = times(fc_out, W) + b

    return z


# Trains a Fast R-CNN model
def train_fast_rcnn(debug_output=False):
    if debug_output:
        print("Storing graphs and intermediate models to %s." % os.path.join(abs_path, "Output"))

    # Create the minibatch source
    minibatch_source = create_mb_source(image_height, image_width, num_channels,
                                        num_classes, num_rois, base_path, "train")

    # Input variables denoting features, rois and label data
    image_input = input_variable((num_channels, image_height, image_width))
    roi_input   = input_variable((num_rois, 4))
    label_input = input_variable((num_rois, num_classes))

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[features_stream_name],
        roi_input: minibatch_source[roi_stream_name],
        label_input: minibatch_source[label_stream_name]
    }

    # Instantiate the Fast R-CNN prediction model and loss function
    frcn_output = frcn_predictor(image_input, roi_input, num_classes)
    ce = cross_entropy_with_softmax(frcn_output, label_input, axis=1)
    pe = classification_error(frcn_output, label_input, axis=1)
    if debug_output:
        plot(frcn_output, os.path.join(abs_path, "Output", "graph_frcn.png"))

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object
    learner = momentum_sgd(frcn_output.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(frcn_output, ce, pe, learner)

    # Get minibatches of images and perform model training
    print("Training Fast R-CNN model for %s epochs." % max_epochs)
    log_number_of_parameters(frcn_output)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress

        progress_printer.epoch_summary(with_metric=True)
        if debug_output:
            frcn_output.save_model(os.path.join(abs_path, "Output", "frcn_py_%s.model" % (epoch+1)))

    return frcn_output


# Tests a Fast R-CNN model
def test_fast_rcnn(model):
    test_minibatch_source = create_mb_source(image_height, image_width, num_channels,
                                             num_classes, num_rois, base_path, "test")
    input_map = {
        model.arguments[0]: test_minibatch_source[features_stream_name],
        model.arguments[1]: test_minibatch_source[roi_stream_name],
    }

    # evaluate test images and write netwrok output to file
    print("Evaluating Fast R-CNN model for %s images." % num_test_images)
    results_file_path = base_path + "test.z"
    with open(results_file_path, 'wb') as results_file:
        for i in range(0, num_test_images):
            data = test_minibatch_source.next_minibatch(1, input_map=input_map)
            output = model.eval(data)
            out_values = output[0, 0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")
            if (i+1) % 100 == 0:
                print("Evaluated %s images.." % (i+1))

    return


# The main method trains and evaluates a Fast R-CNN model.
# If a trained model is already available it is loaded an no training will be performed.
if __name__ == '__main__':
    os.chdir(base_path)
    model_path = os.path.join(abs_path, "Output", "frcn_py.model")

    # Train only is no model exists yet
    if os.path.exists(model_path):
        print("Loading existing model from %s" % model_path)
        trained_model = load_model(model_path)
    else:
        trained_model = train_fast_rcnn()
        trained_model.save_model(model_path)
        print("Stored trained model at %s" % model_path)

    # Evaluate the test set
    test_fast_rcnn(trained_model)
