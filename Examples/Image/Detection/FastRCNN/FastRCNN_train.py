# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, sys
import argparse
import easydict # pip install easydict
import cntk
from cntk import Trainer, load_model, Axis, input_variable, parameter, times, combine, \
    softmax, roipooling, plus, element_times, CloneMethod, alias, Communicator, reduce_sum
from cntk.core import Value
from cntk.initializer import normal
from cntk.layers import placeholder, Constant, Sequential
from cntk.learners import momentum_sgd, learning_parameter_schedule_per_sample, momentum_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import find_by_name, plot
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from _cntk_py import force_deterministic_algorithms

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss
from utils.annotations.annotations_helper import parse_class_map_file
from utils.od_mb_source import ObjectDetectionMinibatchSource
from utils.proposal_helpers import ProposalProvider

def prepare(cfg, use_arg_parser=True):
    cfg.MB_SIZE = 1
    cfg.NUM_CHANNELS = 3
    cfg.OUTPUT_PATH = os.path.join(abs_path, "Output")
    cfg["DATA"].MAP_FILE_PATH = os.path.join(abs_path, cfg["DATA"].MAP_FILE_PATH)
    running_locally = os.path.exists(cfg["DATA"].MAP_FILE_PATH)
    if running_locally:
        os.chdir(cfg["DATA"].MAP_FILE_PATH)
        if not os.path.exists(os.path.join(abs_path, "Output")):
            os.makedirs(os.path.join(abs_path, "Output"))
        if not os.path.exists(os.path.join(abs_path, "Output", cfg["DATA"].DATASET)):
            os.makedirs(os.path.join(abs_path, "Output", cfg["DATA"].DATASET))
    else:
        # disable debug and plot outputs when running on GPU cluster
        cfg["CNTK"].DEBUG_OUTPUT = False
        cfg.VISUALIZE_RESULTS = False

    if use_arg_parser:
        parse_arguments(cfg)

    data_path = cfg["DATA"].MAP_FILE_PATH
    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    cfg["DATA"].CLASS_MAP_FILE = os.path.join(data_path, cfg["DATA"].CLASS_MAP_FILE)
    cfg["DATA"].TRAIN_MAP_FILE = os.path.join(data_path, cfg["DATA"].TRAIN_MAP_FILE)
    cfg["DATA"].TEST_MAP_FILE = os.path.join(data_path, cfg["DATA"].TEST_MAP_FILE)
    cfg["DATA"].TRAIN_ROI_FILE = os.path.join(data_path, cfg["DATA"].TRAIN_ROI_FILE)
    cfg["DATA"].TEST_ROI_FILE = os.path.join(data_path, cfg["DATA"].TEST_ROI_FILE)
    if cfg.USE_PRECOMPUTED_PROPOSALS:
        try:
            cfg["DATA"].TRAIN_PRECOMPUTED_PROPOSALS_FILE = os.path.join(data_path, cfg["DATA"].TRAIN_PRECOMPUTED_PROPOSALS_FILE)
        except:
            print("To use precomputed proposals please specify the following parameters in your configuration:\n"
                  "__C.DATA.TRAIN_PRECOMPUTED_PROPOSALS_FILE\n"
                  "__C.DATA.TEST_PRECOMPUTED_PROPOSALS_FILE")
            exit(-1)

    cfg['MODEL_PATH'] = os.path.join(cfg.OUTPUT_PATH, "fast_rcnn_eval_{}.model".format(cfg["MODEL"].BASE_MODEL))
    cfg['BASE_MODEL_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "PretrainedModels",
                                          cfg["MODEL"].BASE_MODEL_FILE)

    cfg["DATA"].CLASSES = parse_class_map_file(cfg["DATA"].CLASS_MAP_FILE)
    cfg["DATA"].NUM_CLASSES = len(cfg["DATA"].CLASSES)

    if cfg["CNTK"].FAST_MODE:
        cfg["CNTK"].MAX_EPOCHS = 1

    if cfg["CNTK"].FORCE_DETERMINISTIC:
        force_deterministic_algorithms()
    np.random.seed(seed=cfg.RND_SEED)

def parse_arguments(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located',
                        required=False, default=cfg["DATA"].MAP_FILE_PATH)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models',
                        required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file',
                        required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int,
                        required=False, default=cfg["CNTK"].MAX_EPOCHS)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int,
                        required=False, default=cfg.MB_SIZE)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int,
                        required=False, default=cfg["DATA"].NUM_TRAIN_IMAGES)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation',
                        type=int,
                        required=False, default='32')
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)
    parser.add_argument('-lrFactor', '--lrFactor', type=float, help="Scale factor for the lr schedule",
                        required=False)
    parser.add_argument('-momentumPerMb', '--momentumPerMb', type=float, help="momentum per minibatch", required=False)
    parser.add_argument('-rndSeed', '--rndSeed', type=int, help="the random seed", required=False)
    parser.add_argument('-trainConv', '--trainConv', type=int, help="whether to train conv layers", required=False)

    args = vars(parser.parse_args())

    if args['lrFactor'] is not None:
        cfg["CNTK"].LR_FACTOR = args['lrrFactor']
    if args['num_epochs'] is not None:
        cfg["CNTK"].MAX_EPOCHS = args['num_epochs']
    if args['momentumPerMb'] is not None:
        cfg.MOMENTUM_PER_MB = args['momentumPerMb']
    if args['rndSeed'] is not None:
        cfg.RND_SEED = args['rndSeed']
    if args['trainConv'] is not None:
        cfg["CNTK"].TRAIN_CONV_LAYERS = True if args['trainConv'] == 1 else False

    if args['datadir'] is not None:
        cfg["DATA"].MAP_FILE_PATH = args['datadir']
    if args['outputdir'] is not None:
        cfg.OUTPUT_PATH = args['outputdir']
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        # Setting one worker on GPU and one worker on CPU. Otherwise memory consumption is too high for a single GPU.
        if Communicator.rank() == 0:
            cntk.device.try_set_default_device(cntk.device.gpu(args['device']))
        else:
            cntk.device.try_set_default_device(cntk.device.cpu())

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

def clone_conv_layers(base_model, cfg):
    feature_node_name = cfg["MODEL"].FEATURE_NODE_NAME
    start_train_conv_node_name = cfg["MODEL"].START_TRAIN_CONV_NODE_NAME
    last_conv_node_name = cfg["MODEL"].LAST_CONV_NODE_NAME
    if not cfg.TRAIN_CONV_LAYERS:
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
def create_fast_rcnn_predictor(conv_out, rois, fc_layers, cfg):
    # RCNN
    roi_out = roipooling(conv_out, rois, cntk.MAX_POOLING, (cfg["MODEL"].ROI_DIM, cfg["MODEL"].ROI_DIM), spatial_scale=1/16.0)
    fc_out = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, cfg["DATA"].NUM_CLASSES), init=normal(scale=0.01), name="cls_score.W")
    b_pred = parameter(shape=cfg["DATA"].NUM_CLASSES, init=0, name="cls_score.b")
    cls_score = plus(times(fc_out, W_pred), b_pred, name='cls_score')

    # regression head
    W_regr = parameter(shape=(4096, cfg["DATA"].NUM_CLASSES*4), init=normal(scale=0.001), name="bbox_regr.W")
    b_regr = parameter(shape=cfg["DATA"].NUM_CLASSES*4, init=0, name="bbox_regr.b")
    bbox_pred = plus(times(fc_out, W_regr), b_regr, name='bbox_regr')

    return cls_score, bbox_pred

# Defines the Fast R-CNN network model for detecting objects in images
def create_fast_rcnn_model(features, roi_proposals, label_targets, bbox_targets, bbox_inside_weights, cfg):
    # Load the pre-trained classification net and clone layers
    base_model = load_model(cfg['BASE_MODEL_PATH'])
    conv_layers = clone_conv_layers(base_model, cfg)
    fc_layers = clone_model(base_model, [cfg["MODEL"].POOL_NODE_NAME], [cfg["MODEL"].LAST_HIDDEN_NODE_NAME], clone_method=CloneMethod.clone)

    # Normalization and conv layers
    feat_norm = features - Constant([[[v]] for v in cfg["MODEL"].IMG_PAD_COLOR])
    conv_out = conv_layers(feat_norm)

    # Fast RCNN and losses
    cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, roi_proposals, fc_layers, cfg)
    detection_losses = create_detection_losses(cls_score, label_targets, bbox_pred, roi_proposals, bbox_targets, bbox_inside_weights, cfg)
    pred_error = classification_error(cls_score, label_targets, axis=1)

    return detection_losses, pred_error

def create_detection_losses(cls_score, label_targets, bbox_pred, rois, bbox_targets, bbox_inside_weights, cfg):
    # The losses are normalized by the batch size
    # classification loss
    p_cls_score = placeholder()
    p_label_targets = placeholder()
    cls_loss = cross_entropy_with_softmax(p_cls_score, p_label_targets, axis=1)
    cls_normalization_factor = 1.0 / cfg.NUM_ROI_PROPOSALS
    normalized_cls_loss = reduce_sum(cls_loss) * cls_normalization_factor

    reduced_cls_loss = cntk.as_block(normalized_cls_loss,
                                     [(p_cls_score, cls_score), (p_label_targets, label_targets)],
                                     'CrossEntropyWithSoftmax', 'norm_cls_loss')

    # regression loss
    p_bbox_pred = placeholder()
    p_bbox_targets = placeholder()
    p_bbox_inside_weights = placeholder()
    bbox_loss = SmoothL1Loss(cfg.SIGMA_DET_L1, p_bbox_pred, p_bbox_targets, p_bbox_inside_weights, 1.0)
    bbox_normalization_factor = 1.0 / cfg.NUM_ROI_PROPOSALS
    normalized_bbox_loss = reduce_sum(bbox_loss) * bbox_normalization_factor

    reduced_bbox_loss = cntk.as_block(normalized_bbox_loss,
                                     [(p_bbox_pred, bbox_pred), (p_bbox_targets, bbox_targets), (p_bbox_inside_weights, bbox_inside_weights)],
                                     'SmoothL1Loss', 'norm_bbox_loss')

    detection_losses = plus(reduced_cls_loss, reduced_bbox_loss, name="detection_losses")

    return detection_losses

def create_fast_rcnn_eval_model(model, image_input, roi_proposals, cfg):
    print("creating eval model")
    predictor = clone_model(model, [cfg["MODEL"].FEATURE_NODE_NAME, "roi_proposals"], ["cls_score", "bbox_regr"], CloneMethod.freeze)
    pred_net = predictor(image_input, roi_proposals)
    cls_score = pred_net.outputs[0]
    bbox_regr = pred_net.outputs[1]

    if cfg.BBOX_NORMALIZE_TARGETS:
        num_boxes = int(bbox_regr.shape[1] / 4)
        bbox_normalize_means = np.array(cfg.BBOX_NORMALIZE_MEANS * num_boxes)
        bbox_normalize_stds = np.array(cfg.BBOX_NORMALIZE_STDS * num_boxes)
        bbox_regr = plus(element_times(bbox_regr, bbox_normalize_stds), bbox_normalize_means, name='bbox_regr')

    cls_pred = softmax(cls_score, axis=1, name='cls_pred')
    eval_model = combine([cls_pred, bbox_regr])

    if cfg["CNTK"].DEBUG_OUTPUT:
        plot(eval_model, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_eval." + cfg["CNTK"].GRAPH_TYPE))

    return eval_model

# Trains a Fast R-CNN model
def train_fast_rcnn(cfg):
    # Train only if no model exists yet
    model_path = cfg['MODEL_PATH']
    if os.path.exists(model_path) and cfg["CNTK"].MAKE_MODE:
        print("Loading existing model from %s" % model_path)
        return load_model(model_path)
    else:
        # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
        image_input = input_variable(shape=(cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                                     dynamic_axes=[Axis.default_batch_axis()],
                                     name=cfg["MODEL"].FEATURE_NODE_NAME)
        roi_proposals = input_variable((cfg.NUM_ROI_PROPOSALS, 4), dynamic_axes=[Axis.default_batch_axis()], name = "roi_proposals")
        label_targets = input_variable((cfg.NUM_ROI_PROPOSALS, cfg["DATA"].NUM_CLASSES), dynamic_axes=[Axis.default_batch_axis()])
        bbox_targets = input_variable((cfg.NUM_ROI_PROPOSALS, 4*cfg["DATA"].NUM_CLASSES), dynamic_axes=[Axis.default_batch_axis()])
        bbox_inside_weights = input_variable((cfg.NUM_ROI_PROPOSALS, 4*cfg["DATA"].NUM_CLASSES), dynamic_axes=[Axis.default_batch_axis()])

        # Instantiate the Fast R-CNN prediction model and loss function
        loss, pred_error = create_fast_rcnn_model(image_input, roi_proposals, label_targets, bbox_targets, bbox_inside_weights, cfg)
        if isinstance(loss, cntk.Variable):
            loss = combine([loss])

        if cfg["CNTK"].DEBUG_OUTPUT:
            print("Storing graphs and models to %s." % cfg.OUTPUT_PATH)
            plot(loss, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_train." + cfg["CNTK"].GRAPH_TYPE))

        # Set learning parameters
        lr_factor = cfg["CNTK"].LR_FACTOR
        lr_per_sample_scaled = [x * lr_factor for x in cfg["CNTK"].LR_PER_SAMPLE]
        mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)
        l2_reg_weight = cfg["CNTK"].L2_REG_WEIGHT
        epochs_to_train = cfg["CNTK"].MAX_EPOCHS

        print("Using base model:   {}".format(cfg["MODEL"].BASE_MODEL))
        print("lr_per_sample:      {}".format(lr_per_sample_scaled))

        # --- train ---
        # Instantiate the learners and the trainer object
        params = loss.parameters
        biases = [p for p in params if '.b' in p.name or 'b' == p.name]
        others = [p for p in params if not p in biases]
        bias_lr_mult = cfg["CNTK"].BIAS_LR_MULT
        lr_schedule = learning_parameter_schedule_per_sample(lr_per_sample_scaled)
        learner = momentum_sgd(others, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight, unit_gain=False, use_mean_gradient=True)

        bias_lr_per_sample = [v * bias_lr_mult for v in cfg["CNTK"].LR_PER_SAMPLE]
        bias_lr_schedule = learning_parameter_schedule_per_sample(bias_lr_per_sample)
        bias_learner = momentum_sgd(biases, bias_lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight, unit_gain=False, use_mean_gradient=True)
        trainer = Trainer(None, (loss, pred_error), [learner, bias_learner])

        # Get minibatches of images and perform model training
        print("Training model for %s epochs." % epochs_to_train)
        log_number_of_parameters(loss)

        # Create the minibatch source
        if cfg.USE_PRECOMPUTED_PROPOSALS:
            proposal_provider = ProposalProvider.fromfile(cfg["DATA"].TRAIN_PRECOMPUTED_PROPOSALS_FILE, cfg.NUM_ROI_PROPOSALS)
        else:
            proposal_provider = ProposalProvider.fromconfig(cfg)

        od_minibatch_source = ObjectDetectionMinibatchSource(
            cfg["DATA"].TRAIN_MAP_FILE, cfg["DATA"].TRAIN_ROI_FILE,
            max_annotations_per_image=cfg.INPUT_ROIS_PER_IMAGE,
            pad_width=cfg.IMAGE_WIDTH,
            pad_height=cfg.IMAGE_HEIGHT,
            pad_value=cfg["MODEL"].IMG_PAD_COLOR,
            randomize=True,
            use_flipping=cfg["TRAIN"].USE_FLIPPED,
            max_images=cfg["DATA"].NUM_TRAIN_IMAGES,
            num_classes=cfg["DATA"].NUM_CLASSES,
            proposal_provider=proposal_provider,
            provide_targets=True,
            proposal_iou_threshold = cfg.BBOX_THRESH,
            normalize_means = None if not cfg.BBOX_NORMALIZE_TARGETS else cfg.BBOX_NORMALIZE_MEANS,
            normalize_stds = None if not cfg.BBOX_NORMALIZE_TARGETS else cfg.BBOX_NORMALIZE_STDS)

        # define mapping from reader streams to network inputs
        input_map = {
            od_minibatch_source.image_si: image_input,
            od_minibatch_source.proposals_si: roi_proposals,
            od_minibatch_source.label_targets_si: label_targets,
            od_minibatch_source.bbox_targets_si: bbox_targets,
            od_minibatch_source.bbiw_si: bbox_inside_weights
        }

        progress_printer = ProgressPrinter(tag='Training', num_epochs=epochs_to_train, gen_heartbeat=True)
        for epoch in range(epochs_to_train):  # loop over epochs
            sample_count = 0
            while sample_count < cfg["DATA"].NUM_TRAIN_IMAGES:  # loop over minibatches in the epoch
                data = od_minibatch_source.next_minibatch(min(cfg.MB_SIZE, cfg["DATA"].NUM_TRAIN_IMAGES - sample_count), input_map=input_map)

                trainer.train_minibatch(data)  # update model with it
                sample_count += trainer.previous_minibatch_sample_count  # count samples processed so far
                progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
                if sample_count % 100 == 0:
                    print("Processed {} samples".format(sample_count))

            progress_printer.epoch_summary(with_metric=True)

        eval_model = create_fast_rcnn_eval_model(loss, image_input, roi_proposals, cfg)
        eval_model.save(cfg['MODEL_PATH'])
        return eval_model
