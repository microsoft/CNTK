# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, sys
import argparse
import yaml     # pip install pyyaml
import easydict # pip install easydict
import cntk
import easydict
from cntk import Trainer, UnitType, load_model, Axis, input_variable, parameter, times, combine, \
    softmax, roipooling, plus, element_times, CloneMethod, alias, Communicator, reduce_sum
from cntk.core import Value
from cntk.io import MinibatchData
from cntk.initializer import normal
from cntk.layers import placeholder, Constant, Sequential
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from _cntk_py import force_deterministic_algorithms

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.rpn.rpn_helpers import create_rpn, create_proposal_target_layer
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss
from utils.map.map_helpers import evaluate_detections
from utils.annotations.annotations_helper import parse_class_map_file
from config import cfg
from od_mb_source import ObjectDetectionMinibatchSource
from cntk_helpers import regress_rois

###############################################################
###############################################################
mb_size = 1
image_width = cfg["CNTK"].IMAGE_WIDTH
image_height = cfg["CNTK"].IMAGE_HEIGHT
num_channels = 3

# dims_input -- (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
dims_input_const = MinibatchData(Value(batch=np.asarray(
    [image_width, image_height, image_width, image_height, image_width, image_height], dtype=np.float32)), 1, 1, False)

# Color used for padding and normalization (Caffe model uses [102.98010, 115.94650, 122.77170])
img_pad_value = [103, 116, 123] if cfg["CNTK"].BASE_MODEL == "VGG16" else [114, 114, 114]
normalization_const = Constant([[[103]], [[116]], [[123]]]) if cfg["CNTK"].BASE_MODEL == "VGG16" else Constant([[[114]], [[114]], [[114]]])

globalvars = {}
globalvars['output_path'] = os.path.join(abs_path, "Output")

# dataset specific parameters
map_file_path = os.path.join(abs_path, cfg["CNTK"].MAP_FILE_PATH)
globalvars['class_map_file'] = cfg["CNTK"].CLASS_MAP_FILE
globalvars['train_map_file'] = cfg["CNTK"].TRAIN_MAP_FILE
globalvars['test_map_file'] = cfg["CNTK"].TEST_MAP_FILE
globalvars['train_roi_file'] = cfg["CNTK"].TRAIN_ROI_FILE
globalvars['test_roi_file'] = cfg["CNTK"].TEST_ROI_FILE
epoch_size = cfg["CNTK"].NUM_TRAIN_IMAGES
num_test_images = cfg["CNTK"].NUM_TEST_IMAGES

# model specific parameters
model_folder = os.path.join(abs_path, "..", "..", "PretrainedModels")
base_model_file = os.path.join(model_folder, cfg["CNTK"].BASE_MODEL_FILE)
feature_node_name = cfg["CNTK"].FEATURE_NODE_NAME
last_conv_node_name = cfg["CNTK"].LAST_CONV_NODE_NAME
start_train_conv_node_name = cfg["CNTK"].START_TRAIN_CONV_NODE_NAME
pool_node_name = cfg["CNTK"].POOL_NODE_NAME
last_hidden_node_name = cfg["CNTK"].LAST_HIDDEN_NODE_NAME
roi_dim = cfg["CNTK"].ROI_DIM
###############################################################
###############################################################

def set_global_vars(use_arg_parser = True):
    data_path = map_file_path

    # set and overwrite learning parameters
    globalvars['rpn_lr_factor'] = cfg["CNTK"].RPN_LR_FACTOR
    globalvars['frcn_lr_factor'] = cfg["CNTK"].FRCN_LR_FACTOR
    globalvars['e2e_lr_factor'] = cfg["CNTK"].E2E_LR_FACTOR
    globalvars['momentum_per_mb'] = cfg["CNTK"].MOMENTUM_PER_MB
    globalvars['e2e_epochs'] = 1 if cfg["CNTK"].FAST_MODE else cfg["CNTK"].E2E_MAX_EPOCHS
    globalvars['rpn_epochs'] = 1 if cfg["CNTK"].FAST_MODE else cfg["CNTK"].RPN_EPOCHS
    globalvars['frcn_epochs'] = 1 if cfg["CNTK"].FAST_MODE else cfg["CNTK"].FRCN_EPOCHS
    globalvars['rnd_seed'] = cfg.RNG_SEED
    globalvars['train_conv'] = cfg["CNTK"].TRAIN_CONV_LAYERS
    globalvars['train_e2e'] = cfg["CNTK"].TRAIN_E2E


    if use_arg_parser:
        parser = argparse.ArgumentParser()
        parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located',
                            required=False, default=data_path)
        parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models',
                            required=False, default=None)
        parser.add_argument('-logdir', '--logdir', help='Log file',
                            required=False, default=None)
        parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int,
                            required=False, default=cfg["CNTK"].E2E_MAX_EPOCHS)
        parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int,
                            required=False, default=mb_size)
        parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int,
                            required=False, default=epoch_size)
        parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int,
                            required=False, default='32')
        parser.add_argument('-r', '--restart',
                            help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                            action='store_true')
        parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                            required=False, default=None)
        parser.add_argument('-rpnLrFactor', '--rpnLrFactor', type=float, help="Scale factor for rpn lr schedule", required=False)
        parser.add_argument('-frcnLrFactor', '--frcnLrFactor', type=float, help="Scale factor for frcn lr schedule", required=False)
        parser.add_argument('-e2eLrFactor', '--e2eLrFactor', type=float, help="Scale factor for e2e lr schedule", required=False)
        parser.add_argument('-momentumPerMb', '--momentumPerMb', type=float, help="momentum per minibatch", required=False)
        parser.add_argument('-e2eEpochs', '--e2eEpochs', type=int, help="number of epochs for e2e training", required=False)
        parser.add_argument('-rpnEpochs', '--rpnEpochs', type=int, help="number of epochs for rpn training", required=False)
        parser.add_argument('-frcnEpochs', '--frcnEpochs', type=int, help="number of epochs for frcn training", required=False)
        parser.add_argument('-rndSeed', '--rndSeed', type=int, help="the random seed", required=False)
        parser.add_argument('-trainConv', '--trainConv', type=int, help="whether to train conv layers", required=False)
        parser.add_argument('-trainE2E', '--trainE2E', type=int, help="whether to train e2e (otherwise 4 stage)", required=False)

        args = vars(parser.parse_args())

        if args['rpnLrFactor'] is not None:
            globalvars['rpn_lr_factor'] = args['rpnLrFactor']
        if args['frcnLrFactor'] is not None:
            globalvars['frcn_lr_factor'] = args['frcnLrFactor']
        if args['e2eLrFactor'] is not None:
            globalvars['e2e_lr_factor'] = args['e2eLrFactor']
        if args['momentumPerMb'] is not None:
            globalvars['momentum_per_mb'] = args['momentumPerMb']
        if args['e2eEpochs'] is not None:
            globalvars['e2e_epochs'] = args['e2eEpochs']
        if args['rpnEpochs'] is not None:
            globalvars['rpn_epochs'] = args['rpnEpochs']
        if args['frcnEpochs'] is not None:
            globalvars['frcn_epochs'] = args['frcnEpochs']
        if args['rndSeed'] is not None:
            globalvars['rnd_seed'] = args['rndSeed']
        if args['trainConv'] is not None:
            globalvars['train_conv'] = True if args['trainConv']==1 else False
        if args['trainE2E'] is not None:
            globalvars['train_e2e'] = True if args['trainE2E']==1 else False

        if args['outputdir'] is not None:
            globalvars['output_path'] = args['outputdir']
        if args['logdir'] is not None:
            log_dir = args['logdir']
        if args['device'] is not None:
            # Setting one worker on GPU and one worker on CPU. Otherwise memory consumption is too high for a single GPU.
            if Communicator.rank() == 0:
                cntk.device.try_set_default_device(cntk.device.gpu(args['device']))
            else:
                cntk.device.try_set_default_device(cntk.device.cpu())

        if args['datadir'] is not None:
            data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    globalvars['class_map_file'] = os.path.join(data_path, globalvars['class_map_file'])
    globalvars['train_map_file'] = os.path.join(data_path, globalvars['train_map_file'])
    globalvars['test_map_file'] = os.path.join(data_path, globalvars['test_map_file'])
    globalvars['train_roi_file'] = os.path.join(data_path, globalvars['train_roi_file'])
    globalvars['test_roi_file'] = os.path.join(data_path, globalvars['test_roi_file'])

    if cfg["CNTK"].FORCE_DETERMINISTIC:
        force_deterministic_algorithms()
    np.random.seed(seed=globalvars['rnd_seed'])
    globalvars['classes'] = parse_class_map_file(globalvars['class_map_file'])
    globalvars['num_classes'] = len(globalvars['classes'])

    if cfg["CNTK"].DEBUG_OUTPUT:
        # report args
        print("Using the following parameters:")
        print("Flip image       : {}".format(cfg["TRAIN"].USE_FLIPPED))
        print("Train conv layers: {}".format(globalvars['train_conv']))
        print("Random seed      : {}".format(globalvars['rnd_seed']))
        print("Momentum per MB  : {}".format(globalvars['momentum_per_mb']))
        if globalvars['train_e2e']:
            print("E2E epochs       : {}".format(globalvars['e2e_epochs']))
        else:
            print("RPN lr factor    : {}".format(globalvars['rpn_lr_factor']))
            print("RPN epochs       : {}".format(globalvars['rpn_epochs']))
            print("FRCN lr factor   : {}".format(globalvars['frcn_lr_factor']))
            print("FRCN epochs      : {}".format(globalvars['frcn_epochs']))

###############################################################
###############################################################

def clone_model(base_model, from_node_names, to_node_names, clone_method):
    from_nodes = [find_by_name(base_model, node_name) for node_name in from_node_names]
    if None in from_nodes:
        print("Error: could not find all specified 'from_nodes' in clone. Looking for {}, found {}"
              .format(from_node_names, from_nodes))
    to_nodes = [find_by_name(base_model, node_name) for node_name in to_node_names]
    if None in to_nodes:
        print("Error: could not find all specified 'to_nodes' in clone. Looking for {}, found {}"
              .format(to_node_names, to_nodes))
    input_placeholders = dict(zip(from_nodes, [placeholder() for x in from_nodes]))
    cloned_net = combine(to_nodes).clone(clone_method, input_placeholders)
    return cloned_net

def clone_conv_layers(base_model):
    if not globalvars['train_conv']:
        conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
    elif feature_node_name == start_train_conv_node_name:
        conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], CloneMethod.clone)
    else:
        fixed_conv_layers = clone_model(base_model, [feature_node_name], [start_train_conv_node_name],
                                        CloneMethod.freeze)
        train_conv_layers = clone_model(base_model, [start_train_conv_node_name], [last_conv_node_name],
                                        CloneMethod.clone)
        conv_layers = Sequential([fixed_conv_layers, train_conv_layers])
    return conv_layers

# Please keep in sync with Readme.md
def create_fast_rcnn_predictor(conv_out, rois, fc_layers):
    # RCNN
    roi_out = roipooling(conv_out, rois, cntk.MAX_POOLING, (roi_dim, roi_dim), spatial_scale=1/16.0)
    fc_out = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, globalvars['num_classes']), init=normal(scale=0.01), name="cls_score.W")
    b_pred = parameter(shape=globalvars['num_classes'], init=0, name="cls_score.b")
    cls_score = plus(times(fc_out, W_pred), b_pred, name='cls_score')

    # regression head
    W_regr = parameter(shape=(4096, globalvars['num_classes']*4), init=normal(scale=0.001), name="bbox_regr.W")
    b_regr = parameter(shape=globalvars['num_classes']*4, init=0, name="bbox_regr.b")
    bbox_pred = plus(times(fc_out, W_regr), b_regr, name='bbox_regr')

    return cls_score, bbox_pred

# Please keep in sync with Readme.md
# Defines the Faster R-CNN network model for detecting objects in images
def create_faster_rcnn_predictor(base_model_file_name, features, scaled_gt_boxes, dims_input):
    # Load the pre-trained classification net and clone layers
    base_model = load_model(base_model_file_name)
    conv_layers = clone_conv_layers(base_model)
    fc_layers = clone_model(base_model, [pool_node_name], [last_hidden_node_name], clone_method=CloneMethod.clone)

    # Normalization and conv layers
    feat_norm = features - normalization_const
    conv_out = conv_layers(feat_norm)

    # RPN and prediction targets
    rpn_rois, rpn_losses = \
        create_rpn(conv_out, scaled_gt_boxes, dims_input, proposal_layer_param_string=cfg["CNTK"].PROPOSAL_LAYER_PARAMS)
    rois, label_targets, bbox_targets, bbox_inside_weights = \
        create_proposal_target_layer(rpn_rois, scaled_gt_boxes, num_classes=globalvars['num_classes'])

    # Fast RCNN and losses
    cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers)
    detection_losses = create_detection_losses(cls_score, label_targets, rois, bbox_pred, bbox_targets, bbox_inside_weights)
    loss = rpn_losses + detection_losses
    pred_error = classification_error(cls_score, label_targets, axis=1)

    return loss, pred_error

def create_detection_losses(cls_score, label_targets, rois, bbox_pred, bbox_targets, bbox_inside_weights):
    # classification loss
    cls_loss = cross_entropy_with_softmax(cls_score, label_targets, axis=1)

    p_cls_loss = placeholder()
    p_rois = placeholder()
    # The terms that are accounted for in the cls loss are those that correspond to an actual roi proposal --> do not count no-op (all-zero) rois
    roi_indicator = reduce_sum(p_rois, axis=1)
    cls_num_terms = reduce_sum(cntk.greater_equal(roi_indicator, 0.0))
    cls_normalization_factor = 1.0 / cls_num_terms
    normalized_cls_loss = reduce_sum(p_cls_loss) * cls_normalization_factor

    reduced_cls_loss = cntk.as_block(normalized_cls_loss,
                                     [(p_cls_loss, cls_loss), (p_rois, rois)],
                                     'Normalize', 'norm_cls_loss')

    # regression loss
    p_bbox_pred = placeholder()
    p_bbox_targets = placeholder()
    p_bbox_inside_weights = placeholder()
    bbox_loss = SmoothL1Loss(cfg["CNTK"].SIGMA_DET_L1, p_bbox_pred, p_bbox_targets, p_bbox_inside_weights, 1.0)
    # The bbox loss is normalized by the batch size
    bbox_normalization_factor = 1.0 / cfg["TRAIN"].BATCH_SIZE
    normalized_bbox_loss = reduce_sum(bbox_loss) * bbox_normalization_factor

    reduced_bbox_loss = cntk.as_block(normalized_bbox_loss,
                                     [(p_bbox_pred, bbox_pred), (p_bbox_targets, bbox_targets), (p_bbox_inside_weights, bbox_inside_weights)],
                                     'SmoothL1Loss', 'norm_bbox_loss')

    detection_losses = plus(reduced_cls_loss, reduced_bbox_loss, name="detection_losses")

    return detection_losses

def create_eval_model(model, image_input, dims_input, rpn_model=None):
    print("creating eval model")
    conv_layers = clone_model(model, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
    conv_out = conv_layers(image_input)

    model_with_rpn = model if rpn_model is None else rpn_model
    rpn = clone_model(model_with_rpn, [last_conv_node_name, "dims_input"], ["rpn_rois"], CloneMethod.freeze)
    rpn_rois = rpn(conv_out, dims_input)

    roi_fc_layers = clone_model(model, [last_conv_node_name, "rpn_target_rois"], ["cls_score", "bbox_regr"], CloneMethod.freeze)
    pred_net = roi_fc_layers(conv_out, rpn_rois)
    cls_score = pred_net.outputs[0]
    bbox_regr = pred_net.outputs[1]

    if cfg["TRAIN"].BBOX_NORMALIZE_TARGETS and cfg["TRAIN"].BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        num_boxes = int(bbox_regr.shape[1] / 4)
        bbox_normalize_means = np.array(cfg["TRAIN"].BBOX_NORMALIZE_MEANS * num_boxes)
        bbox_normalize_stds = np.array(cfg["TRAIN"].BBOX_NORMALIZE_STDS * num_boxes)
        bbox_regr = plus(element_times(bbox_regr, bbox_normalize_stds), bbox_normalize_means, name='bbox_regr')

    cls_pred = softmax(cls_score, axis=1, name='cls_pred')
    eval_model = combine([cls_pred, rpn_rois, bbox_regr])

    return eval_model

def train_model(image_input, roi_input, dims_input, loss, pred_error,
                lr_per_sample, mm_schedule, l2_reg_weight, epochs_to_train,
                rpn_rois_input=None, buffered_rpn_proposals=None):
    if isinstance(loss, cntk.Variable):
        loss = combine([loss])

    params = loss.parameters
    biases = [p for p in params if '.b' in p.name or 'b' == p.name]
    others = [p for p in params if not p in biases]
    bias_lr_mult = cfg["CNTK"].BIAS_LR_MULT

    if cfg["CNTK"].DEBUG_OUTPUT:
        print("biases")
        for p in biases: print(p)
        print("others")
        for p in others: print(p)
        print("bias_lr_mult: {}".format(bias_lr_mult))

    # Instantiate the learners and the trainer object
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    learner = momentum_sgd(others, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight,
                           unit_gain=False, use_mean_gradient=cfg["CNTK"].USE_MEAN_GRADIENT)

    bias_lr_per_sample = [v * bias_lr_mult for v in lr_per_sample]
    bias_lr_schedule = learning_rate_schedule(bias_lr_per_sample, unit=UnitType.sample)
    bias_learner = momentum_sgd(biases, bias_lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight,
                           unit_gain=False, use_mean_gradient=cfg["CNTK"].USE_MEAN_GRADIENT)
    trainer = Trainer(None, (loss, pred_error), [learner, bias_learner])

    # Get minibatches of images and perform model training
    print("Training model for %s epochs." % epochs_to_train)
    log_number_of_parameters(loss)

    # Create the minibatch source
    od_minibatch_source = ObjectDetectionMinibatchSource(
        globalvars['train_map_file'], globalvars['train_roi_file'],
        max_annotations_per_image=cfg["CNTK"].INPUT_ROIS_PER_IMAGE,
        pad_width=image_width, pad_height=image_height, pad_value=img_pad_value,
        randomize=True, use_flipping=cfg["TRAIN"].USE_FLIPPED,
        max_images=cfg["CNTK"].NUM_TRAIN_IMAGES,
        buffered_rpn_proposals=buffered_rpn_proposals)

    # define mapping from reader streams to network inputs
    input_map = {
        od_minibatch_source.image_si: image_input,
        od_minibatch_source.roi_si: roi_input,
        od_minibatch_source.dims_si: dims_input
    }

    use_buffered_proposals = buffered_rpn_proposals is not None
    progress_printer = ProgressPrinter(tag='Training', num_epochs=epochs_to_train, gen_heartbeat=True)
    for epoch in range(epochs_to_train):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data, proposals = od_minibatch_source.next_minibatch_with_proposals(min(mb_size, epoch_size-sample_count), input_map=input_map)
            if use_buffered_proposals:
                data[rpn_rois_input] = MinibatchData(Value(batch=np.asarray(proposals, dtype=np.float32)), 1, 1, False)
                # remove dims input if no rpn is required to avoid warnings
                del data[[k for k in data if '[6]' in str(k)][0]]

            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
            if sample_count % 100 == 0:
                print("Processed {} samples".format(sample_count))

        progress_printer.epoch_summary(with_metric=True)

def compute_rpn_proposals(rpn_model, image_input, roi_input, dims_input):
    num_images = cfg["CNTK"].NUM_TRAIN_IMAGES
    # Create the minibatch source
    od_minibatch_source = ObjectDetectionMinibatchSource(
        globalvars['train_map_file'], globalvars['train_roi_file'],
        max_annotations_per_image=cfg["CNTK"].INPUT_ROIS_PER_IMAGE,
        pad_width=image_width, pad_height=image_height, pad_value=img_pad_value,
        max_images=num_images,
        randomize=False, use_flipping=False)

    # define mapping from reader streams to network inputs
    input_map = {
        od_minibatch_source.image_si: image_input,
        od_minibatch_source.roi_si: roi_input,
        od_minibatch_source.dims_si: dims_input
    }

    # setting pre- and post-nms top N to training values since buffered proposals are used for further training
    test_pre = cfg["TEST"].RPN_PRE_NMS_TOP_N
    test_post = cfg["TEST"].RPN_POST_NMS_TOP_N
    cfg["TEST"].RPN_PRE_NMS_TOP_N = cfg["TRAIN"].RPN_PRE_NMS_TOP_N
    cfg["TEST"].RPN_POST_NMS_TOP_N = cfg["TRAIN"].RPN_POST_NMS_TOP_N

    buffered_proposals = [None for _ in range(num_images)]
    sample_count = 0
    while sample_count < num_images:
        data = od_minibatch_source.next_minibatch(1, input_map=input_map)
        output = rpn_model.eval(data)
        out_dict = dict([(k.name, k) for k in output])
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        buffered_proposals[sample_count] = np.round(out_rpn_rois).astype(np.int16)
        sample_count += 1
        if sample_count % 500 == 0:
            print("Buffered proposals for {} samples".format(sample_count))

    # resetting config values to original test values
    cfg["TEST"].RPN_PRE_NMS_TOP_N = test_pre
    cfg["TEST"].RPN_POST_NMS_TOP_N = test_post

    return buffered_proposals

# Trains a Faster R-CNN model end-to-end
def train_faster_rcnn_e2e(base_model_file_name, debug_output=False):
    # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    roi_input = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    dims_node = alias(dims_input, name='dims_input')

    # Instantiate the Faster R-CNN prediction model and loss function
    loss, pred_error = create_faster_rcnn_predictor(base_model_file_name, image_input, roi_input, dims_node)

    if debug_output:
        print("Storing graphs and models to %s." % globalvars['output_path'])
        plot(loss, os.path.join(globalvars['output_path'], "graph_frcn_train_e2e." + cfg["CNTK"].GRAPH_TYPE))

    # Set learning parameters
    e2e_lr_factor = globalvars['e2e_lr_factor']
    e2e_lr_per_sample_scaled = [x * e2e_lr_factor for x in cfg["CNTK"].E2E_LR_PER_SAMPLE]
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)

    print("Using base model:   {}".format(cfg["CNTK"].BASE_MODEL))
    print("lr_per_sample:      {}".format(e2e_lr_per_sample_scaled))

    train_model(image_input, roi_input, dims_input, loss, pred_error,
                e2e_lr_per_sample_scaled, mm_schedule, cfg["CNTK"].L2_REG_WEIGHT, globalvars['e2e_epochs'])

    return create_eval_model(loss, image_input, dims_input)

# Trains a Faster R-CNN model using 4-stage alternating training
def train_faster_rcnn_alternating(base_model_file_name, debug_output=False):
    '''
        4-Step Alternating Training scheme from the Faster R-CNN paper:
        
        # Create initial network, only rpn, without detection network
            # --> train only the rpn (and conv3_1 and up for VGG16)
        # buffer region proposals from rpn
        # Create full network, initialize conv layers with imagenet, use buffered proposals
            # --> train only detection network (and conv3_1 and up for VGG16)
        # Keep conv weights from detection network and fix them
            # --> train only rpn
        # buffer region proposals from rpn
        # Keep conv and rpn weights from step 3 and fix them
            # --> train only detection network
    '''

    # Learning parameters
    rpn_lr_factor = globalvars['rpn_lr_factor']
    rpn_lr_per_sample_scaled = [x * rpn_lr_factor for x in cfg["CNTK"].RPN_LR_PER_SAMPLE]
    frcn_lr_factor = globalvars['frcn_lr_factor']
    frcn_lr_per_sample_scaled = [x * frcn_lr_factor for x in cfg["CNTK"].FRCN_LR_PER_SAMPLE]

    l2_reg_weight = cfg["CNTK"].L2_REG_WEIGHT
    mm_schedule = momentum_schedule(globalvars['momentum_per_mb'])
    rpn_epochs = globalvars['rpn_epochs']
    frcn_epochs = globalvars['frcn_epochs']

    print("Using base model:   {}".format(cfg["CNTK"].BASE_MODEL))
    print("rpn_lr_per_sample:  {}".format(rpn_lr_per_sample_scaled))
    print("frcn_lr_per_sample: {}".format(frcn_lr_per_sample_scaled))
    if debug_output:
        print("Storing graphs and models to %s." % globalvars['output_path'])

    # Input variables denoting features, labeled ground truth rois (as 5-tuples per roi) and image dimensions
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()],
                                 name=feature_node_name)
    feat_norm = image_input - normalization_const
    roi_input = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    scaled_gt_boxes = alias(roi_input, name='roi_input')
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    dims_node = alias(dims_input, name='dims_input')
    rpn_rois_input = input_variable((cfg["TRAIN"].RPN_POST_NMS_TOP_N, 4), dynamic_axes=[Axis.default_batch_axis()])
    rpn_rois_buf = alias(rpn_rois_input, name='rpn_rois')

    # base image classification model (e.g. VGG16 or AlexNet)
    base_model = load_model(base_model_file_name)

    print("stage 1a - rpn")
    if True:
        # Create initial network, only rpn, without detection network
            #       initial weights     train?
            # conv: base_model          only conv3_1 and up
            # rpn:  init new            yes
            # frcn: -                   -

        # conv layers
        conv_layers = clone_conv_layers(base_model)
        conv_out = conv_layers(feat_norm)

        # RPN and losses
        rpn_rois, rpn_losses = create_rpn(conv_out, scaled_gt_boxes, dims_node, proposal_layer_param_string=cfg["CNTK"].PROPOSAL_LAYER_PARAMS)
        stage1_rpn_network = combine([rpn_rois, rpn_losses])

        # train
        if debug_output: plot(stage1_rpn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage1a_rpn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, rpn_losses, rpn_losses,
                    rpn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, epochs_to_train=rpn_epochs)

    print("stage 1a - buffering rpn proposals")
    buffered_proposals_s1 = compute_rpn_proposals(stage1_rpn_network, image_input, roi_input, dims_input)

    print("stage 1b - frcn")
    if True:
        # Create full network, initialize conv layers with imagenet, fix rpn weights
            #       initial weights     train?
            # conv: base_model          only conv3_1 and up
            # rpn:  stage1a rpn model   no --> use buffered proposals
            # frcn: base_model + new    yes

        # conv_layers
        conv_layers = clone_conv_layers(base_model)
        conv_out = conv_layers(feat_norm)

        # use buffered proposals in target layer
        rois, label_targets, bbox_targets, bbox_inside_weights = \
            create_proposal_target_layer(rpn_rois_buf, scaled_gt_boxes, num_classes=globalvars['num_classes'])

        # Fast RCNN and losses
        fc_layers = clone_model(base_model, [pool_node_name], [last_hidden_node_name], CloneMethod.clone)
        cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers)
        detection_losses = create_detection_losses(cls_score, label_targets, rois, bbox_pred, bbox_targets, bbox_inside_weights)
        pred_error = classification_error(cls_score, label_targets, axis=1, name="pred_error")
        stage1_frcn_network = combine([rois, cls_score, bbox_pred, detection_losses, pred_error])

        # train
        if debug_output: plot(stage1_frcn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage1b_frcn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, detection_losses, pred_error,
                    frcn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, epochs_to_train=frcn_epochs,
                    rpn_rois_input=rpn_rois_input, buffered_rpn_proposals=buffered_proposals_s1)
        buffered_proposals_s1 = None

    print("stage 2a - rpn")
    if True:
        # Keep conv weights from detection network and fix them
            #       initial weights     train?
            # conv: stage1b frcn model  no
            # rpn:  stage1a rpn model   yes
            # frcn: -                   -

        # conv_layers
        conv_layers = clone_model(stage1_frcn_network, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
        conv_out = conv_layers(image_input)

        # RPN and losses
        rpn = clone_model(stage1_rpn_network, [last_conv_node_name, "roi_input", "dims_input"], ["rpn_rois", "rpn_losses"], CloneMethod.clone)
        rpn_net = rpn(conv_out, dims_node, scaled_gt_boxes)
        rpn_rois = rpn_net.outputs[0]
        rpn_losses = rpn_net.outputs[1]
        stage2_rpn_network = combine([rpn_rois, rpn_losses])

        # train
        if debug_output: plot(stage2_rpn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage2a_rpn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, rpn_losses, rpn_losses,
                    rpn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, epochs_to_train=rpn_epochs)

    print("stage 2a - buffering rpn proposals")
    buffered_proposals_s2 = compute_rpn_proposals(stage2_rpn_network, image_input, roi_input, dims_input)

    print("stage 2b - frcn")
    if True:
        # Keep conv and rpn weights from step 3 and fix them
            #       initial weights     train?
            # conv: stage2a rpn model   no
            # rpn:  stage2a rpn model   no --> use buffered proposals
            # frcn: stage1b frcn model  yes                   -

        # conv_layers
        conv_layers = clone_model(stage2_rpn_network, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
        conv_out = conv_layers(image_input)

        # Fast RCNN and losses
        frcn = clone_model(stage1_frcn_network, [last_conv_node_name, "rpn_rois", "roi_input"],
                           ["cls_score", "bbox_regr", "rpn_target_rois", "detection_losses", "pred_error"], CloneMethod.clone)
        stage2_frcn_network = frcn(conv_out, rpn_rois_buf, scaled_gt_boxes)
        detection_losses = stage2_frcn_network.outputs[3]
        pred_error = stage2_frcn_network.outputs[4]

        # train
        if debug_output: plot(stage2_frcn_network, os.path.join(globalvars['output_path'], "graph_frcn_train_stage2b_frcn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, detection_losses, pred_error,
                    frcn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, epochs_to_train=frcn_epochs,
                    rpn_rois_input=rpn_rois_input, buffered_rpn_proposals=buffered_proposals_s2)
        buffered_proposals_s2 = None

    return create_eval_model(stage2_frcn_network, image_input, dims_input, rpn_model=stage2_rpn_network)

def eval_faster_rcnn_mAP(eval_model):
    img_map_file = globalvars['test_map_file']
    roi_map_file = globalvars['test_roi_file']
    classes = globalvars['classes']
    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    roi_input = input_variable((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    frcn_eval = eval_model(image_input, dims_input)

    # Create the minibatch source
    minibatch_source = ObjectDetectionMinibatchSource(
        img_map_file, roi_map_file,
        max_annotations_per_image=cfg["CNTK"].INPUT_ROIS_PER_IMAGE,
        pad_width=image_width, pad_height=image_height, pad_value=img_pad_value,
        randomize=False, use_flipping=False,
        max_images=cfg["CNTK"].NUM_TEST_IMAGES)

    # define mapping from reader streams to network inputs
    input_map = {
        minibatch_source.image_si: image_input,
        minibatch_source.roi_si: roi_input,
        minibatch_source.dims_si: dims_input
    }

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(globalvars['num_classes'])]

    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    all_gt_infos = {key: [] for key in classes}
    for img_i in range(0, num_test_images):
        mb_data = minibatch_source.next_minibatch(1, input_map=input_map)

        gt_row = mb_data[roi_input].asarray()
        gt_row = gt_row.reshape((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5))
        all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]

        for cls_index, cls_name in enumerate(classes):
            if cls_index == 0: continue
            cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
            all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                           'difficult': [False] * len(cls_gt_boxes),
                                           'det': [False] * len(cls_gt_boxes)})

        output = frcn_eval.eval({image_input: mb_data[image_input], dims_input: mb_data[dims_input]})
        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, mb_data[dims_input].asarray())

        labels.shape = labels.shape + (1,)
        scores.shape = scores.shape + (1,)
        coords_score_label = np.hstack((regressed_rois, scores, labels))

        #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        for cls_j in range(1, globalvars['num_classes']):
            coords_score_label_for_cls = coords_score_label[np.where(coords_score_label[:,-1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

        if (img_i+1) % 100 == 0:
            print("Processed {} samples".format(img_i+1))

    # calculate mAP
    aps = evaluate_detections(all_boxes, all_gt_infos, classes,
                              nms_threshold=cfg["CNTK"].RESULTS_NMS_THRESHOLD,
                              conf_threshold = cfg["CNTK"].RESULTS_NMS_CONF_THRESHOLD)
    ap_list = []
    for class_name in aps:
        ap_list += [aps[class_name]]
        print('AP for {:>15} = {:.4f}'.format(class_name, aps[class_name]))
    meanAP = np.nanmean(ap_list)
    print('Mean AP = {:.4f}'.format(meanAP))
    return meanAP

# The main method trains and evaluates a Fast R-CNN model.
# If a trained model is already available it is loaded an no training will be performed (if MAKE_MODE=True).
if __name__ == '__main__':
    running_locally = os.path.exists(map_file_path)
    if running_locally:
        os.chdir(map_file_path)
        if not os.path.exists(os.path.join(abs_path, "Output")):
            os.makedirs(os.path.join(abs_path, "Output"))
        if not os.path.exists(os.path.join(abs_path, "Output", cfg["CNTK"].DATASET)):
            os.makedirs(os.path.join(abs_path, "Output", cfg["CNTK"].DATASET))
    else:
        # disable debug and plot outputs when running on GPU cluster
        cfg["CNTK"].DEBUG_OUTPUT = False
        cfg["CNTK"].VISUALIZE_RESULTS = False

    set_global_vars()
    model_path = os.path.join(globalvars['output_path'], "faster_rcnn_eval_{}_{}.model"
                              .format(cfg["CNTK"].BASE_MODEL, "e2e" if globalvars['train_e2e'] else "4stage"))

    # Train only if no model exists yet
    if os.path.exists(model_path) and cfg["CNTK"].MAKE_MODE:
        print("Loading existing model from %s" % model_path)
        eval_model = load_model(model_path)
    else:
        if globalvars['train_e2e']:
            eval_model = train_faster_rcnn_e2e(base_model_file, debug_output=cfg["CNTK"].DEBUG_OUTPUT)
        else:
            eval_model = train_faster_rcnn_alternating(base_model_file, debug_output=cfg["CNTK"].DEBUG_OUTPUT)

        eval_model.save(model_path)
        if cfg["CNTK"].DEBUG_OUTPUT:
            plot(eval_model, os.path.join(globalvars['output_path'], "graph_frcn_eval_{}_{}.{}"
                                          .format(cfg["CNTK"].BASE_MODEL, "e2e" if globalvars['train_e2e'] else "4stage", cfg["CNTK"].GRAPH_TYPE)))

        print("Stored eval model at %s" % model_path)

    # Compute mean average precision on test set
    eval_faster_rcnn_mAP(eval_model)

    # Plot results on test set
    if cfg["CNTK"].VISUALIZE_RESULTS:
        from plot_helpers import eval_and_plot_faster_rcnn
        num_eval = min(num_test_images, 100)
        img_shape = (num_channels, image_height, image_width)
        results_folder = os.path.join(globalvars['output_path'], cfg["CNTK"].DATASET)
        eval_and_plot_faster_rcnn(eval_model, num_eval, globalvars['test_map_file'], img_shape,
                                  results_folder, feature_node_name, globalvars['classes'],
                                  drawUnregressedRois=cfg["CNTK"].DRAW_UNREGRESSED_ROIS,
                                  drawNegativeRois=cfg["CNTK"].DRAW_NEGATIVE_ROIS,
                                  nmsThreshold=cfg["CNTK"].RESULTS_NMS_THRESHOLD,
                                  nmsConfThreshold=cfg["CNTK"].RESULTS_NMS_CONF_THRESHOLD,
                                  bgrPlotThreshold=cfg["CNTK"].RESULTS_BGR_PLOT_THRESHOLD)

