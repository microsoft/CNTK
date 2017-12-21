# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import cntk as C
from cntk import *
from cntk.initializer import glorot_uniform
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDefs, StreamDef
from cntk.io.transforms import scale
from cntk.layers import placeholder, Constant
from cntk.learners import momentum_sgd, learning_parameter_schedule_per_sample, momentum_schedule_per_sample
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot
import PARAMETERS
import numpy as np
import os, sys
from cntk import distributed

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

p = PARAMETERS.get_parameters_for_dataset()
base_path = p.cntkFilesDir
num_channels = 3
image_height = p.cntk_padHeight
image_width = p.cntk_padWidth
num_classes = p.nrClasses
num_rois = p.cntk_nrRois
epoch_size = p.cntk_num_train_images
num_test_images = p.cntk_num_test_images
mb_size = p.cntk_mb_size
max_epochs = p.cntk_max_epochs
distributed_flg = p.distributed_flg
num_quantization_bits = p.num_quantization_bits
warm_up = p.warm_up
momentum_per_sample = p.cntk_momentum_per_sample

# model specific variables (only AlexNet for now)
base_model = "AlexNet"
if base_model == "AlexNet":
    model_file = "../../../../../../../../PretrainedModels/AlexNet.model"
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
                           "Please run install_data_and_model.py from Examples/Image/Detection/FastRCNN to fetch them" %
                           (map_file, roi_file, label_file))

    # read images
    transforms = [scale(width=img_width, height=img_height, channels=img_channels,
                        scale_mode="pad", pad_value=114, interpolations='linear')]

    image_source = ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms)))

    # read rois and labels
    roi_source = CTFDeserializer(roi_file, StreamDefs(
        rois = StreamDef(field=roi_stream_name, shape=rois_dim, is_sparse=False)))
    label_source = CTFDeserializer(label_file, StreamDefs(
        roiLabels = StreamDef(field=label_stream_name, shape=label_dim, is_sparse=False)))

    # define a composite reader
    return MinibatchSource([image_source, roi_source, label_source], max_samples=sys.maxsize, randomize=data_set == "train")

# Defines the Fast R-CNN network model for detecting objects in images
def frcn_predictor(features, rois, n_classes, model_path):
    # Load the pretrained classification net and find nodes
    loaded_model = load_model(model_path)
    feature_node = find_by_name(loaded_model, feature_node_name)
    conv_node    = find_by_name(loaded_model, last_conv_node_name)
    pool_node    = find_by_name(loaded_model, pool_node_name)
    last_node    = find_by_name(loaded_model, last_hidden_node_name)

    # Clone the conv layers and the fully connected layers of the network
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: placeholder()})
    fc_layers = combine([last_node.owner]).clone(CloneMethod.clone, {pool_node: placeholder()})

    # Create the Fast R-CNN model
    feat_norm = features - Constant(114)
    conv_out  = conv_layers(feat_norm)
    roi_out   = roipooling(conv_out, rois, C.MAX_POOLING, (roi_dim, roi_dim), 0.0625)
    fc_out    = fc_layers(roi_out)

    # z = Dense(rois[0], num_classes, map_rank=1)(fc_out)  # --> map_rank=1 is not yet supported
    W = parameter(shape=(4096, n_classes), init=glorot_uniform())
    b = parameter(shape=n_classes, init=0)
    z = times(fc_out, W) + b

    return z

# Trains a Fast R-CNN model
def train_fast_rcnn(debug_output=False, model_path=model_file):
    if debug_output:
        print("Storing graphs and intermediate models to %s." % os.path.join(abs_path, "Output"))

    # Create the minibatch source
    minibatch_source = create_mb_source(image_height, image_width, num_channels,
                                        num_classes, num_rois, base_path, "train")

    # Input variables denoting features, rois and label data
    image_input = C.input_variable((num_channels, image_height, image_width))
    roi_input   = C.input_variable((num_rois, 4))
    label_input = C.input_variable((num_rois, num_classes))

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source.streams.features,
        roi_input: minibatch_source.streams.rois,
        label_input: minibatch_source.streams.roiLabels
    }

    # Instantiate the Fast R-CNN prediction model and loss function
    frcn_output = frcn_predictor(image_input, roi_input, num_classes, model_path)
    ce = cross_entropy_with_softmax(frcn_output, label_input, axis=1)
    pe = classification_error(frcn_output, label_input, axis=1)
    if debug_output:
        plot(frcn_output, os.path.join(abs_path, "Output", "graph_frcn.png"))

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    lr_schedule = learning_parameter_schedule_per_sample(lr_per_sample)
    mm_schedule = momentum_schedule_per_sample(momentum_per_sample)

    # Instantiate the trainer object as default
    learner = momentum_sgd(frcn_output.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    # Preparation for distributed learning, which is compatible for normal learner
    learner = distributed.data_parallel_distributed_learner(
        learner = learner,
        num_quantization_bits = num_quantization_bits,   # non-quantized gradient accumulation
        distributed_after = warm_up)                     # no warm start as default            
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs, rank=distributed.Communicator.rank())
    trainer = Trainer(frcn_output, (ce, pe), learner, progress_printer)

    # Get minibatches of images and perform model training
    print("Training Fast R-CNN model for %s epochs." % max_epochs)
    log_number_of_parameters(frcn_output)
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size * C.Communicator.num_workers(), epoch_size-sample_count), 
                input_map=input_map, 
                num_data_partitions=C.Communicator.num_workers(), 
                partition_index=C.Communicator.rank())     
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far

        trainer.summarize_training_progress()
        if debug_output:
            frcn_output.save(os.path.join(abs_path, "Output", "frcn_py_%s.model" % (epoch+1)))

    if distributed_flg:
        distributed.Communicator.finalize()

    return frcn_output

# Evaluate a Fast R-CNN model
def evaluate_fast_rcnn(model):
    test_minibatch_source = create_mb_source(image_height, image_width, num_channels,
                                             num_classes, num_rois, base_path, "test")
    input_map = {
        model.arguments[0]: test_minibatch_source[features_stream_name],
        model.arguments[1]: test_minibatch_source[roi_stream_name],
    }

    # evaluate test images and write netwrok output to file
    print("Evaluating Fast R-CNN model for %s images." % num_test_images)
    results_file_path = os.path.join(base_path, "test.z")
    with open(results_file_path, 'wb') as results_file:
        for i in range(0, num_test_images):
            data = test_minibatch_source.next_minibatch(1, input_map=input_map)
            output = model.eval(data)
            out_values = output[0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")
            if (i+1) % 100 == 0:
                print("Evaluated %s images.." % (i+1))

    return True


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
        trained_model.save(model_path)
        print("Stored trained model at %s" % model_path)

    # Evaluate the test set
    evaluate_fast_rcnn(trained_model)
