# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, sys

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", ".."))
sys.path.append(os.path.join(abs_path, "lib"))
sys.path.append(os.path.join(abs_path, "lib", "rpn"))
sys.path.append(os.path.join(abs_path, "lib", "nms"))
sys.path.append(os.path.join(abs_path, "lib", "nms", "gpu"))

from cntk import Trainer, UnitType, load_model, user_function, Axis
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDefs, StreamDef
from cntk.io.transforms import *
from cntk.initializer import glorot_uniform
from cntk.layers import Placeholder, Constant, Convolution
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import input_variable, parameter, times, combine, relu, softmax, roipooling, reduce_sum, slice, splice, reshape
from cntk.ops.functions import CloneMethod
from lib.rpn.cntk_anchor_target_layer import AnchorTargetLayer
from lib.rpn.cntk_proposal_layer import ProposalLayer
from lib.rpn.cntk_proposal_target_layer import ProposalTargetLayer
from lib.rpn.cntk_smoothL1_loss import SmoothL1Loss
from lib.rpn.cntk_ignore_label import IgnoreLabel

###############################################################
###############################################################
make_mode = False

# file and stream names
map_filename_postfix = '.imgMap.txt'
rois_filename_postfix = '.GTRois.txt'
features_stream_name = 'features'
roi_stream_name = 'roiAndLabel'

# from PARAMETERS.py
base_path = "C:/src/CNTK/Examples/Image/Detection/FastRCNN/proc/Grocery_100/rois/"
num_channels = 3
image_height = 1000
image_width = 1000
num_classes = 17
num_rois = 100
epoch_size = 25
num_test_images = 5
mb_size = 1
max_epochs = 3
momentum_time_constant = 10

# model specific variables (only AlexNet for now)
base_model = "AlexNet"
if base_model == "AlexNet":
    base_model_file = "C:/src/CNTK/Examples/Image/PretrainedModels/AlexNet.model"
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
def create_mb_source(img_height, img_width, img_channels, n_rois, data_path, data_set):
    rois_dim = 5 * n_rois

    path = os.path.normpath(os.path.join(abs_path, data_path))
    map_file = os.path.join(path, data_set + map_filename_postfix)
    roi_file = os.path.join(path, data_set + rois_filename_postfix)

    if not os.path.exists(map_file) or not os.path.exists(roi_file):
        raise RuntimeError("File '%s' or '%s' does not exist. "
                           "Please run install_fastrcnn.py from Examples/Image/Detection/FastRCNN to fetch them" %
                           (map_file, roi_file))

    # read images
    transforms = [scale(width=img_width, height=img_height, channels=img_channels,
                        scale_mode="pad", pad_value=114, interpolations='linear')]

    image_source = ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms)))

    # read rois and labels
    roi_source = CTFDeserializer(roi_file, StreamDefs(
        rois = StreamDef(field=roi_stream_name, shape=rois_dim, is_sparse=False)))

    # define a composite reader
    return MinibatchSource([image_source, roi_source], epoch_size=sys.maxsize, randomize=data_set == "train")


# Defines the Faster R-CNN network model for detecting objects in images
def faster_rcnn_predictor(features, gt_boxes, n_classes):
    im_info = [image_width, image_height, 1]

    # Load the pre-trained classification net and find nodes
    loaded_model = load_model(base_model_file)
    feature_node = find_by_name(loaded_model, feature_node_name)
    conv_node    = find_by_name(loaded_model, last_conv_node_name)
    pool_node    = find_by_name(loaded_model, pool_node_name)
    last_node    = find_by_name(loaded_model, last_hidden_node_name)

    # Clone the conv layers and the fully connected layers of the network
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: Placeholder()})
    fc_layers = combine([last_node.owner]).clone(CloneMethod.clone, {pool_node: Placeholder()})

    # Create the Faster R-CNN model
    feat_norm = features - Constant(114)
    conv_out  = conv_layers(feat_norm)

    # RPN network
    rpn_conv_3x3  = Convolution((3,3), 256, activation=relu, pad=True, strides=1)(conv_out)
    rpn_cls_score = Convolution((1,1), 18, activation=None) (rpn_conv_3x3) # 2(bg/fg)  * 9(anchors)
    rpn_bbox_pred = Convolution((1,1), 36, activation=None) (rpn_conv_3x3) # 4(coords) * 9(anchors)

    # RPN targets
    # Comment: rpn_cls_score is only passed   vvv   to get width and height of the conv feature map ...
    atl = user_function(AnchorTargetLayer(rpn_cls_score, gt_boxes, im_info=im_info))
    rpn_labels = atl.outputs[0]
    rpn_bbox_targets = atl.outputs[1]

    # getting rpn class scores and rpn targets into the correct shape for ce
    # i.e., (2, 33k), where each group of two corresponds to a (bg, fg) pair for score or target
    # Reshape scores
    num_anchors = int(rpn_cls_score.shape[0] / 2)
    num_predictions = int(np.prod(rpn_cls_score.shape) / 2)
    bg_scores = slice(rpn_cls_score, 0, 0, num_anchors)
    fg_scores = slice(rpn_cls_score, 0, num_anchors, num_anchors * 2)
    bg_scores_rshp = reshape(bg_scores, (1,num_predictions))
    fg_scores_rshp = reshape(fg_scores, (1,num_predictions))
    rpn_cls_score_rshp = splice(bg_scores_rshp, fg_scores_rshp, axis=0)
    rpn_cls_prob = softmax(rpn_cls_score_rshp, axis=0)
    # Reshape targets
    rpn_labels_rshp = reshape(rpn_labels, (1,num_predictions))

    # Ignore predictions for the 'ignore label', i.e. set target and prediction to 0 --> needs to be softmaxed before
    ignore = user_function(IgnoreLabel(rpn_cls_prob, rpn_labels_rshp, ignore_label=-1))
    rpn_cls_prob_ignore = ignore.outputs[0]
    fg_targets = ignore.outputs[1]
    bg_targets = 1 - fg_targets
    rpn_labels_ignore = splice(bg_targets, fg_targets, axis=0)

    # RPN losses
    rpn_loss_cls = cross_entropy_with_softmax(rpn_cls_prob_ignore, rpn_labels_ignore, axis=0)
    rpn_loss_bbox = user_function(SmoothL1Loss(rpn_bbox_pred, rpn_bbox_targets))

    # ROI proposal
    # - ProposalLayer:
    #    Outputs object detection proposals by applying estimated bounding-box
    #    transformations to a set of regular boxes (called "anchors").
    # - ProposalTargetLayer:
    #    Assign object detection proposals to ground-truth targets. Produces proposal
    #    classification labels and bounding-box regression targets.
    #  + adds gt_boxes to candidates and samples fg and bg rois for training

    # reshape predictions per (H, W) position to (2,9) ( == (bg, fg) per anchor)
    shp = (2,num_anchors,) + rpn_cls_score.shape[-2:]
    rpn_cls_prob_reshape = reshape(rpn_cls_prob, shp)

    rpn_rois = user_function(ProposalLayer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info=im_info))
    ptl = user_function(ProposalTargetLayer(rpn_rois, gt_boxes))
    rois = ptl.outputs[0]
    labels = ptl.outputs[1]
    bbox_targets = ptl.outputs[2]

    # RCNN
    # Comment: training uses 'rois' from ptl (sampled), eval uses 'rpn_rois' from proposal_layer
    roi_out = roipooling(conv_out, rois, (roi_dim, roi_dim))
    fc_out  = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, n_classes), init=glorot_uniform())
    b_pred = parameter(shape=n_classes, init=0)
    cls_score = times(fc_out, W_pred) + b_pred

    # regression head
    W_regr = parameter(shape=(4096, n_classes*4), init=glorot_uniform())
    b_regr = parameter(shape=n_classes*4, init=0)
    bbox_pred = times(fc_out, W_regr) + b_regr

    # loss function
    loss_cls = cross_entropy_with_softmax(cls_score, labels, axis=1)
    loss_box = user_function(SmoothL1Loss(bbox_pred, bbox_targets))

    loss_cls_scalar = reduce_sum(loss_cls)
    loss_box_scalar = reduce_sum(loss_box)
    rpn_loss_cls_scalar = reduce_sum(rpn_loss_cls)
    rpn_loss_bbox_scalar = reduce_sum(rpn_loss_bbox)

    loss = rpn_loss_cls_scalar + rpn_loss_bbox_scalar + loss_cls_scalar + loss_box_scalar
    pred_error = classification_error(cls_score, labels, axis=1)

    return cls_score, loss, pred_error


# Trains a Faster R-CNN model
def train_faster_rcnn(debug_output=False):
    if debug_output:
        print("Storing graphs and intermediate models to %s." % os.path.join(abs_path))

    # Create the minibatch source
    minibatch_source = create_mb_source(image_height, image_width, num_channels, num_rois, base_path, "train")

    # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()])
    roi_input   = input_variable((num_rois, 5), dynamic_axes=[Axis.default_batch_axis()])

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source.streams.features,
        roi_input: minibatch_source.streams.rois
    }

    # Instantiate the Faster R-CNN prediction model and loss function
    cls_score, loss, pred_error = faster_rcnn_predictor(image_input, roi_input, num_classes)
    if debug_output:
        plot(loss, os.path.join(abs_path, "graph_frcn.png"))

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object
    learner = momentum_sgd(cls_score.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    ##trainer = Trainer(frcn_output, (ce, pe), learner)
    trainer = Trainer(cls_score, (loss, pred_error), learner)

    # Get minibatches of images and perform model training
    print("Training Faster R-CNN model for %s epochs." % max_epochs)
    log_number_of_parameters(cls_score)
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
            cls_score.save(os.path.join(abs_path, "frcn_py_%s.model" % (epoch+1)))

    return cls_score


# Tests a Faster R-CNN model
def eval_faster_rcnn(model):
    test_minibatch_source = create_mb_source(image_height, image_width, num_channels, num_rois, base_path, "test")
    # !!! TODO: modify Faster RCNN model by excluding target layers and losses
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
    model_path = os.path.join(abs_path, "frcn_py.model")

    #import pdb; pdb.set_trace()

    # Train only if no model exists yet
    if os.path.exists(model_path) and make_mode:
        print("Loading existing model from %s" % model_path)
        trained_model = load_model(model_path)
    else:
        trained_model = train_faster_rcnn(debug_output=True)
        trained_model.save(model_path)
        print("Stored trained model at %s" % model_path)

    import pdb; pdb.set_trace()

    # Evaluate the test set
    eval_faster_rcnn(trained_model)
