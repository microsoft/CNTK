# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, sys
from matplotlib.pyplot import imshow, imsave
from PIL import ImageFont
from math import exp

available_font = "arial.ttf"
try:
    dummy = ImageFont.truetype(available_font, 16)
except:
    available_font = "FreeMono.ttf"

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", ".."))
sys.path.append(os.path.join(abs_path, "lib"))
sys.path.append(os.path.join(abs_path, "lib", "rpn"))
sys.path.append(os.path.join(abs_path, "lib", "nms"))
sys.path.append(os.path.join(abs_path, "lib", "nms", "gpu"))

from cntk import Trainer, UnitType, load_model, user_function, Axis, input, parameter, times, combine, relu, \
    softmax, roipooling, reduce_sum, slice, splice, reshape, plus, CloneMethod, minus, element_times, alias
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDefs, StreamDef, TraceLevel
from cntk.io.transforms import scale
from cntk.initializer import glorot_uniform
from cntk.layers import placeholder, Convolution, Constant
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from lib.rpn.anchor_target_layer import AnchorTargetLayer
from lib.rpn.proposal_layer import ProposalLayer
from lib.rpn.proposal_target_layer import ProposalTargetLayer
from lib.rpn.cntk_smoothL1_loss import SmoothL1Loss
from lib.rpn.cntk_ignore_label import IgnoreLabel
from cntk_helpers import visualizeResultsFaster
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.bbox_transform import bbox_transform_inv

###############################################################
###############################################################
make_mode = False
graph_type = "png" # "png" or "pdf"

# file and stream names
map_filename_postfix = '.imgMap.txt'
rois_filename_postfix = '.GTRois.txt'
features_stream_name = 'features'
roi_stream_name = 'roiAndLabel'

# from PARAMETERS.py
grocery = cfg["CNTK"].USE_GROCERY
if grocery:
    classes = ('__background__',  # always index 0
               'avocado', 'orange', 'butter', 'champagne', 'eggBox', 'gerkin', 'joghurt', 'ketchup',
               'orangeJuice', 'onion', 'pepper', 'tomato', 'water', 'milk', 'tabasco', 'mustard')
    base_path = "C:/src/CNTK/Examples/Image/Detection/FastRCNN/proc/Grocery_100/rois/"
    num_channels = 3
    image_height = 1000
    image_width = 1000
    num_classes = len(classes)
    num_rois = cfg["CNTK"].INPUT_ROIS_PER_IMAGE
    epoch_size = 20
    num_test_images = 5
    mb_size = 1
    max_epochs = cfg["CNTK"].MAX_EPOCHS
    momentum_time_constant = 10
else:
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    base_path = "C:/src/CNTK/Examples/Image/DataSets/Pascal/mappings/"
    map_filename_postfix = "val2007.txt"
    rois_filename_postfix = "val2007_rois_topleft_wh_rel.txt"
    num_channels = 3
    image_height = 1000
    image_width = 1000
    num_classes = len(classes)
    num_rois = cfg["CNTK"].INPUT_ROIS_PER_IMAGE
    epoch_size = 5010
    num_test_images = 4952
    mb_size = 1
    max_epochs = cfg["CNTK"].MAX_EPOCHS
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
def create_mb_source(img_height, img_width, img_channels, n_rois, data_path):
    rois_dim = 5 * n_rois

    path = os.path.normpath(os.path.join(abs_path, data_path))
    map_file = os.path.join(path, "train" + map_filename_postfix)
    roi_file = os.path.join(path, "train" + rois_filename_postfix)

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
        roiAndLabel = StreamDef(field=roi_stream_name, shape=rois_dim, is_sparse=False)))

    # define a composite reader
    return MinibatchSource([image_source, roi_source], epoch_size=sys.maxsize, randomize=True, trace_level=TraceLevel.Error)


def create_test_mb_source(img_height, img_width, img_channels, n_rois, data_path):
    path = os.path.normpath(os.path.join(abs_path, data_path))
    map_file = os.path.join(path, "test" + map_filename_postfix)

    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist. "
                           "Please run install_fastrcnn.py from Examples/Image/Detection/FastRCNN to fetch them" %
                           (map_file))

    # read images
    transforms = [scale(width=img_width, height=img_height, channels=img_channels,
                        scale_mode="pad", pad_value=114, interpolations='linear')]

    image_source = ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms)))

    # define a composite reader
    return MinibatchSource([image_source], epoch_size=sys.maxsize, randomize=False, trace_level=TraceLevel.Error)


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
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: placeholder()})
    # fc_layers = combine([last_node.owner]).clone(CloneMethod.clone, {pool_node: placeholder()})
    # TODO: reset to CloneMethod.clone. Setting to freeze for now to try learning rates
    fc_layers = combine([last_node.owner]).clone(CloneMethod.freeze, {pool_node: placeholder()})

    # Create the Faster R-CNN model
    feat_norm = features - Constant(114)
    conv_out  = conv_layers(feat_norm)

    # RPN network
    rpn_conv_3x3  = Convolution((3,3), 256, activation=relu, pad=True, strides=1)(conv_out)
    rpn_cls_score = Convolution((1,1), 18, activation=None, name="rpn_cls_score") (rpn_conv_3x3) # 2(bg/fg)  * 9(anchors)
    rpn_bbox_pred = Convolution((1,1), 36, activation=None, name="rpn_bbox_pred") (rpn_conv_3x3) # 4(coords) * 9(anchors)

    # RPN targets
    # Comment: rpn_cls_score is only passed   vvv   to get width and height of the conv feature map ...
    atl = user_function(AnchorTargetLayer(rpn_cls_score, gt_boxes, im_info=im_info))
    rpn_labels = atl.outputs[0]
    rpn_bbox_targets = atl.outputs[1]
    rpn_bbox_inside_weights = atl.outputs[2]

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
    rpn_cls_prob = softmax(rpn_cls_score_rshp, axis=0, name="objness_softmax")
    # Reshape targets
    rpn_labels_rshp = reshape(rpn_labels, (1,num_predictions))

    # Ignore label predictions for the 'ignore label', i.e. set target and prediction to 0 --> needs to be softmaxed before
    ignore = user_function(IgnoreLabel(rpn_cls_prob, rpn_labels_rshp, ignore_label=-1))
    rpn_cls_prob_ignore = ignore.outputs[0]
    fg_targets = ignore.outputs[1]
    bg_targets = 1 - fg_targets
    rpn_labels_ignore = splice(bg_targets, fg_targets, axis=0)

    # RPN losses
    rpn_loss_cls = cross_entropy_with_softmax(rpn_cls_prob_ignore, rpn_labels_ignore, axis=0)
    rpn_loss_bbox = user_function(SmoothL1Loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights))

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

    rpn_rois_raw = user_function(ProposalLayer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info=im_info))
    rpn_rois = alias(rpn_rois_raw, name='rpn_rois')
    ptl = user_function(ProposalTargetLayer(rpn_rois, gt_boxes))
    rois_raw = ptl.outputs[0]
    rois = alias(rois_raw, name='rpn_target_rois')
    labels = ptl.outputs[1]
    bbox_targets = ptl.outputs[2]
    bbox_inside_weights = ptl.outputs[3]

    # RCNN
    # Comment: training uses 'rois' from ptl (sampled), eval uses 'rpn_rois' from proposal_layer

    # for the roipooling layer we convert and scale roi coords back to x, y, w, h relative from x1, y1, x2, y2 absolute
    roi_xy1 = slice(rois, 1, 0, 2)
    roi_xy2 = slice(rois, 1, 2, 4)
    roi_wh = minus(roi_xy2, roi_xy1)
    roi_xywh = splice(roi_xy1, roi_wh, axis=1)
    scaled_rois = element_times(roi_xywh, (1.0 / image_width))

    roi_out = roipooling(conv_out, scaled_rois, (roi_dim, roi_dim))
    fc_out  = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, n_classes), init=glorot_uniform())
    b_pred = parameter(shape=n_classes, init=0)
    cls_score = plus(times(fc_out, W_pred), b_pred, name='cls_score')

    # regression head
    W_regr = parameter(shape=(4096, n_classes*4), init=glorot_uniform())
    b_regr = parameter(shape=n_classes*4, init=0)
    bbox_pred = plus(times(fc_out, W_regr), b_regr, name='bbox_regr')

    # loss function
    loss_cls = cross_entropy_with_softmax(cls_score, labels, axis=1)
    loss_box = user_function(SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights))

    loss_cls_scalar = reduce_sum(loss_cls)
    loss_box_scalar = reduce_sum(loss_box)
    rpn_loss_cls_scalar = reduce_sum(rpn_loss_cls)
    rpn_loss_bbox_scalar = reduce_sum(rpn_loss_bbox)

    loss = rpn_loss_cls_scalar + rpn_loss_bbox_scalar + loss_cls_scalar + loss_box_scalar
    pred_error = classification_error(cls_score, labels, axis=1)

    return cls_score, loss, pred_error


def create_eval_model(model, image_input):
    # modify Faster RCNN model by excluding target layers and losses
    feature_node = find_by_name(model, "img_input")
    conv_node = find_by_name(model, "conv5.y")
    rpn_roi_node = find_by_name(model, "rpn_rois")
    rpn_target_roi_node = find_by_name(model, "rpn_target_rois")
    cls_score_node = find_by_name(model, "cls_score")
    bbox_pred_node = find_by_name(model, "bbox_regr")

    conv_rpn_layers = combine([conv_node.owner, rpn_roi_node.owner])\
        .clone(CloneMethod.freeze, {feature_node: placeholder()})
    roi_fc_layers = combine([cls_score_node.owner, bbox_pred_node.owner])\
        .clone(CloneMethod.clone, {conv_node: placeholder(), rpn_target_roi_node: placeholder()})

    conv_rpn_net = conv_rpn_layers(image_input)
    conv_out = conv_rpn_net.outputs[0]
    rpn_rois = conv_rpn_net.outputs[1]

    pred_net = roi_fc_layers(conv_out, rpn_rois)
    cls_score = pred_net.outputs[0]
    bbox_regr = pred_net.outputs[1]

    cls_pred = softmax(cls_score, axis=1, name='cls_pred')
    return combine([cls_pred, rpn_rois, bbox_regr])


# Trains a Faster R-CNN model
def train_faster_rcnn(debug_output=False):
    if debug_output:
        print("Storing graphs and intermediate models to %s." % os.path.join(abs_path))

    # Create the minibatch source
    minibatch_source = create_mb_source(image_height, image_width, num_channels, num_rois, base_path)

    # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
    image_input = input((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name='img_input')
    roi_input   = input((num_rois, 5), dynamic_axes=[Axis.default_batch_axis()])

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[features_stream_name],
        roi_input: minibatch_source[roi_stream_name]
    }

    # Instantiate the Faster R-CNN prediction model and loss function
    cls_score, loss, pred_error = faster_rcnn_predictor(image_input, roi_input, num_classes)
    if debug_output:
        plot(loss, os.path.join(abs_path, "graph_frcn." + graph_type))

    # Set learning parameters
    # Caffe Faster R-CNN parameters are:
    #   base_lr: 0.001
    #   lr_policy: "step"
    #   gamma: 0.1
    #   stepsize: 50000
    #   momentum: 0.9
    #   weight_decay: 0.0005
    # ==> CNTK: lr_per_sample = [0.001] * 10 + [0.0001] * 10 + [0.00001]
    l2_reg_weight = 0.0005
    lr_per_sample = [0.001] * 10 + [0.0001] * 10 + [0.00001]
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object
    learner = momentum_sgd(loss.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(None, (loss, pred_error), learner)

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
            if sample_count % 100 == 0:
                print("Processed {} samples".format(sample_count))

        progress_printer.epoch_summary(with_metric=True)
        if debug_output:
            cls_score.save(os.path.join(abs_path, "frcn_py_%s.model" % (epoch+1)))

    return loss


def regress_rois(roi_proposals, roi_regression_factors, labels):
    for i in range(len(labels)):
        label = labels[i]
        if label > 0:
            #deltas = roi_regression_factors[i:i+1,label*4:(label+1)*4]
            deltas = roi_regression_factors[i:i+1,(label-1)*4:label*4]
            roi_coords = roi_proposals[i:i+1,:]

            regressed_rois = bbox_transform_inv(roi_coords, deltas)
            #import pdb; pdb.set_trace()

            roi_proposals[i,:] = regressed_rois
    return roi_proposals

# Tests a Faster R-CNN model
def eval_faster_rcnn(model, debug_output=False):
    # get image paths
    path = os.path.normpath(os.path.join(abs_path, base_path))
    map_file = os.path.join(path, "test" + map_filename_postfix)
    with open(map_file) as f:
        content = f.readlines()
    img_file_names = [x.split('\t')[1] for x in content]

    image_input = input((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name='img_input')
    frcn_eval = create_eval_model(model, image_input)

    if debug_output:
        plot(frcn_eval, os.path.join(abs_path, "graph_frcn_eval." + graph_type))

    test_minibatch_source = create_test_mb_source(image_height, image_width, num_channels, num_rois, base_path)
    input_map = {
        image_input: test_minibatch_source[features_stream_name],
    }

    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    visualize_with_regr = True
    visualize_without_regr = True
    save_results_as_text = True

    results_base_path = "c:/temp/grocery/" if grocery else "c:/temp/pascal/"
    results_file_path = results_base_path + "test.z"
    rois_file_path = results_base_path + "test.rois.txt"
    with open(results_file_path, 'wb') as results_file, \
        open(rois_file_path, 'wb') as rois_file:
        for i in range(0, num_test_images):
            data = test_minibatch_source.next_minibatch(1, input_map=input_map)
            output = frcn_eval.eval(data)

            out_dict = dict([(k.name, k) for k in output])
            out_cls_pred = output[out_dict['cls_pred']][0]
            out_rpn_rois = output[out_dict['rpn_rois']][0]
            out_bbox_regr = output[out_dict['bbox_regr']][0]

            imgPath = img_file_names[i]
            labels = out_cls_pred.argmax(axis=1)
            scores = out_cls_pred.max(axis=1).tolist()

            if save_results_as_text:
                out_values = out_cls_pred.flatten()
                np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")
                roi_values = out_rpn_rois.flatten()
                np.savetxt(rois_file, roi_values[np.newaxis], fmt="%.1f")

            if visualize_without_regr:
                imgDebug = visualizeResultsFaster(imgPath, labels, scores, out_rpn_rois, 1000, 1000,
                                            classes, nmsKeepIndices=None, boDrawNegativeRois=True)
                imsave("{}{}{}".format(results_base_path, i, os.path.basename(imgPath)), imgDebug)

            if visualize_with_regr:
                # apply regression to bbox coordinates
                regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels)

                imgDebug = visualizeResultsFaster(imgPath, labels, scores, regressed_rois, 1000, 1000,
                                            classes, nmsKeepIndices=None, boDrawNegativeRois=True)
                imsave("{}{}_regr_{}".format(results_base_path, i, os.path.basename(imgPath)), imgDebug)

    return

# The main method trains and evaluates a Fast R-CNN model.
# If a trained model is already available it is loaded an no training will be performed.
if __name__ == '__main__':
    os.chdir(base_path)
    model_path = os.path.join(abs_path, "faster_rcnn_py.model")

    # Train only if no model exists yet
    if os.path.exists(model_path) and make_mode:
        print("Loading existing model from %s" % model_path)
        trained_model = load_model(model_path)
    else:
        trained_model = train_faster_rcnn(debug_output=True)
        trained_model.save(model_path)
        print("Stored trained model at %s" % model_path)

    # Evaluate the test set
    eval_faster_rcnn(trained_model, debug_output=True)
