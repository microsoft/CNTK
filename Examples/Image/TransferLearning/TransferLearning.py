# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import cv2
import warnings
import numpy as np
from cntk.blocks import Placeholder, Constant
from cntk.graph import find_by_name, plot
from cntk.initializer import glorot_uniform
from cntk.ops.functions import CloneMethod
from cntk import load_model, graph, Trainer, UnitType
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.graph import print_all_nodes
from cntk.layers import Dense
from cntk.ops import input_variable, parameter, cross_entropy_with_softmax, classification_error, times, combine
from cntk.utils import log_number_of_parameters, ProgressPrinter
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, momentum_schedule


# general settings
make_mode = False
base_folder = os.path.dirname(os.path.abspath(__file__))
tl_model_file = os.path.join(base_folder, "Output", "TransferLearning.model")
output_file = os.path.join(base_folder, "Output", "predOutput.txt")
features_stream_name = 'features'
label_stream_name = 'labels'
new_output_node_name = "prediction"

# define base model location and characteristics
base_model_file = os.path.join(base_folder, "..", "PretrainedModels", "AlexNet.model")
image_height = 224
image_width = 224
num_channels = 3
feature_node_name = "features"
last_hidden_node_name = "h2_d"


# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_height, image_width, num_channels, num_classes, randomize=False):
    transforms = [ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    image_source = ImageDeserializer(map_file)
    image_source.map_features(features_stream_name, transforms)
    image_source.map_labels(label_stream_name, num_classes)
    return MinibatchSource(image_source, randomize=randomize)


def get_cntk_image_input(image_path):
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    hwc_format = np.ascontiguousarray(np.array(resized, dtype=np.float32).transpose(2, 0, 1))
    return hwc_format


# Creates the network model for transfer learning
def create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, input_features):
    # Load the pretrained classification net and find nodes
    base_model = load_model(base_model_file)
    feature_node = find_by_name(base_model, feature_node_name)
    last_node    = find_by_name(base_model, last_hidden_node_name)

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(CloneMethod.freeze, {feature_node: Placeholder(name='features')})

    # Add new dense layer for class prediction
    feat_norm = input_features - Constant(114)
    cloned_out = cloned_layers(feat_norm)
    #z = Dense(num_classes, init=glorot_uniform(), activation=None, name=new_output_node_name) (cloned_out)
    z = Dense(num_classes, activation=None, name=new_output_node_name) (cloned_out)

    return z


def train_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, train_map_file):
    epoch_size = sum(1 for line in open(train_map_file))

    # Create the minibatch source and input variables
    minibatch_source = create_mb_source(train_map_file, image_height, image_width, num_channels, num_classes, randomize=True)
    image_input = input_variable((num_channels, image_height, image_width))
    label_input = input_variable(num_classes)

    # Define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[features_stream_name],
        label_input: minibatch_source[label_stream_name]
    }

    # Instantiate the transfer learning model and loss function
    tl_model = create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, image_input)
    ce = cross_entropy_with_softmax(tl_model, label_input)
    pe = classification_error(tl_model, label_input)

    # Set learning parameters
    max_epochs = 3
    mb_size = 50
    momentum_time_constant = 20
    momentum_per_mb = 0.9
    l2_reg_weight = 0.0005
    lr_per_sample = 0.0001
    lr_per_mb = 0.01

    # lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    lr_schedule = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch)
    # mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    mm_schedule = momentum_schedule(momentum_per_mb)

    # Instantiate the trainer object
    learner = momentum_sgd(tl_model.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(tl_model, ce, pe, learner)

    # Get minibatches of images and perform model training
    print("Training transfer learning model for %s epochs (epoch_size = %s)." % (max_epochs, epoch_size))
    log_number_of_parameters(tl_model)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
            if sample_count % (100 * mb_size) == 0:
                print ("Processed %s samples" % sample_count)

        progress_printer.epoch_summary(with_metric=True)

    return tl_model


def eval_and_write(loaded_model, output_file, test_map_file):
    num_images = sum(1 for line in open(test_map_file))
    minibatch_source = create_mb_source(test_map_file, image_height, image_width, num_channels, num_classes, randomize=True)

    # load model and pick desired node as output
    node_in_graph = loaded_model.find_by_name(new_output_node_name)
    output_nodes  = combine([node_in_graph.owner])

    # evaluate model and get desired node output
    print("Evaluating model output node '%s' for %s images." % (new_output_node_name, num_images))
    features_si = minibatch_source['features']
    with open(output_file, 'wb') as results_file:
        for i in range(0, num_images):
            mb = minibatch_source.next_minibatch(1)
            output = output_nodes.eval(mb[features_si])

            # compute softmax probabilities from raw predictions
            exp_out = np.exp(output[0, 0].astype(np.float64))
            sum_exp = np.sum(exp_out)
            probs = np.divide(exp_out, sum_exp)

            np.savetxt(results_file, probs[np.newaxis], fmt="%.3f")


def eval_from_image(loaded_model, output_file, test_map_file):
    num_images = sum(1 for line in open(test_map_file))

    # load model and pick desired node as output
    # TODO: try loaded model directly
    node_in_graph = loaded_model.find_by_name(new_output_node_name)
    output_nodes  = combine([node_in_graph.owner])

    # evaluate model and get desired node output
    print("Evaluating model output node '%s' for %s images." % (new_output_node_name, num_images))

    pred_count = 0
    correct_count = 0
    np.seterr(over='raise')
    with open(output_file, 'wb') as results_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                # format image and compute model output
                tokens = line.rstrip().split('\t')
                img_file = tokens[0]
                img_input = get_cntk_image_input(img_file)
                arguments = { output_nodes.arguments[0]: [img_input] }
                output = output_nodes.eval(arguments)

                # compute softmax probabilities from raw predictions
                try:
                    exp_out = np.exp(output[0, 0].astype(np.float64))
                except FloatingPointError:
                    exp_out = output[0, 0].astype(np.float64)

                sum_exp = np.sum(exp_out)
                probs = np.divide(exp_out, sum_exp)

                pred_count += 1
                true_label = int(tokens[1])
                predicted_label = np.argmax(probs)
                if predicted_label == true_label:
                    correct_count += 1

                # np.savetxt(results_file, probs[np.newaxis], fmt="%.3f")
                if pred_count % 500 == 0:
                    print("Processed %s samples (%s correct)" % (pred_count, (correct_count / pred_count)))

    print ("%s of %s prediction were correct (%s)." % (correct_count, pred_count, (correct_count / pred_count)))


if __name__ == '__main__':
    # define data location and characteristics
    data_folder = os.path.join(base_folder, "..", "DataSets", "CIFAR-10")
    os.chdir(os.path.join(data_folder))
    train_map_file = os.path.join(data_folder, "train_map.txt")
    test_map_file = os.path.join(data_folder, "test_map.txt")
    num_classes = 10

    # check for model and data existence
    if not (os.path.exists(base_model_file) and os.path.exists(train_map_file) and os.path.exists(test_map_file)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # Train only if no model exists yet or if make_mode is set to False
    # print_all_nodes(load_model(base_model_file))
    if os.path.exists(tl_model_file) and make_mode:
        print("Loading existing model from %s" % tl_model_file)
        trained_model = load_model(tl_model_file)
    else:
        trained_model = train_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, train_map_file)
        trained_model.save_model(tl_model_file)
        print("Stored trained model at %s" % tl_model_file)

    # Evaluate the test set
    # print_all_nodes(trained_model)
    eval_from_image(trained_model, output_file, test_map_file)


    print("Done. Wrote output to %s" % output_file)
