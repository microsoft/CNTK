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
from cntk.io import MinibatchData
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
from utils.rpn.rpn_helpers import create_rpn, create_proposal_target_layer, create_proposal_layer
from utils.annotations.annotations_helper import parse_class_map_file
from utils.od_mb_source import ObjectDetectionMinibatchSource
from utils.proposal_helpers import ProposalProvider
from FastRCNN.FastRCNN_train import clone_model, clone_conv_layers, create_fast_rcnn_predictor, \
    create_detection_losses

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

    cfg['MODEL_PATH'] = os.path.join(cfg.OUTPUT_PATH, "faster_rcnn_eval_{}_{}.model"
                                     .format(cfg["MODEL"].BASE_MODEL, "e2e" if cfg["CNTK"].TRAIN_E2E else "4stage"))
    cfg['BASE_MODEL_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "PretrainedModels",
                                          cfg["MODEL"].BASE_MODEL_FILE)

    cfg["DATA"].CLASSES = parse_class_map_file(cfg["DATA"].CLASS_MAP_FILE)
    cfg["DATA"].NUM_CLASSES = len(cfg["DATA"].CLASSES)

    if cfg["CNTK"].FAST_MODE:
        cfg["CNTK"].E2E_MAX_EPOCHS = 1
        cfg["CNTK"].RPN_EPOCHS = 1
        cfg["CNTK"].FRCN_EPOCHS = 1

    if cfg["CNTK"].FORCE_DETERMINISTIC:
        force_deterministic_algorithms()
    np.random.seed(seed=cfg.RND_SEED)

    if False and cfg["CNTK"].DEBUG_OUTPUT:
        # report args
        print("Using the following parameters:")
        print("Flip image       : {}".format(cfg["TRAIN"].USE_FLIPPED))
        print("Train conv layers: {}".format(cfg.TRAIN_CONV_LAYERS))
        print("Random seed      : {}".format(cfg.RND_SEED))
        print("Momentum per MB  : {}".format(cfg["CNTK"].MOMENTUM_PER_MB))
        if cfg["CNTK"].TRAIN_E2E:
            print("E2E epochs       : {}".format(cfg["CNTK"].E2E_MAX_EPOCHS))
        else:
            print("RPN lr factor    : {}".format(cfg["CNTK"].RPN_LR_FACTOR))
            print("RPN epochs       : {}".format(cfg["CNTK"].RPN_EPOCHS))
            print("FRCN lr factor   : {}".format(cfg["CNTK"].FRCN_LR_FACTOR))
            print("FRCN epochs      : {}".format(cfg["CNTK"].FRCN_EPOCHS))

def parse_arguments(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located',
                        required=False, default=cfg["DATA"].MAP_FILE_PATH)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models',
                        required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file',
                        required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int,
                        required=False, default=cfg["CNTK"].E2E_MAX_EPOCHS)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int,
                        required=False, default=cfg["CNTK"].MB_SIZE)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int,
                        required=False, default=cfg["CNTK"].NUM_TRAIN_IMAGES)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation',
                        type=int,
                        required=False, default='32')
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device",
                        required=False, default=None)
    parser.add_argument('-rpnLrFactor', '--rpnLrFactor', type=float, help="Scale factor for rpn lr schedule",
                        required=False)
    parser.add_argument('-frcnLrFactor', '--frcnLrFactor', type=float, help="Scale factor for frcn lr schedule",
                        required=False)
    parser.add_argument('-e2eLrFactor', '--e2eLrFactor', type=float, help="Scale factor for e2e lr schedule",
                        required=False)
    parser.add_argument('-momentumPerMb', '--momentumPerMb', type=float, help="momentum per minibatch", required=False)
    parser.add_argument('-e2eEpochs', '--e2eEpochs', type=int, help="number of epochs for e2e training", required=False)
    parser.add_argument('-rpnEpochs', '--rpnEpochs', type=int, help="number of epochs for rpn training", required=False)
    parser.add_argument('-frcnEpochs', '--frcnEpochs', type=int, help="number of epochs for frcn training",
                        required=False)
    parser.add_argument('-rndSeed', '--rndSeed', type=int, help="the random seed", required=False)
    parser.add_argument('-trainConv', '--trainConv', type=int, help="whether to train conv layers", required=False)
    parser.add_argument('-trainE2E', '--trainE2E', type=int, help="whether to train e2e (otherwise 4 stage)",
                        required=False)

    args = vars(parser.parse_args())

    if args['rpnLrFactor'] is not None:
        cfg["MODEL"].RPN_LR_FACTOR = args['rpnLrFactor']
    if args['frcnLrFactor'] is not None:
        cfg["MODEL"].FRCN_LR_FACTOR = args['frcnLrFactor']
    if args['e2eLrFactor'] is not None:
        cfg["MODEL"].E2E_LR_FACTOR = args['e2eLrFactor']
    if args['e2eEpochs'] is not None:
        cfg["CNTK"].E2E_MAX_EPOCHS = args['e2eEpochs']
    if args['rpnEpochs'] is not None:
        cfg["CNTK"].RPN_EPOCHS = args['rpnEpochs']
    if args['frcnEpochs'] is not None:
        cfg["CNTK"].FRCN_EPOCHS = args['frcnEpochs']
    if args['momentumPerMb'] is not None:
        cfg["CNTK"].MOMENTUM_PER_MB = args['momentumPerMb']
    if args['rndSeed'] is not None:
        cfg.RND_SEED = args['rndSeed']
    if args['trainConv'] is not None:
        cfg.TRAIN_CONV_LAYERS = True if args['trainConv'] == 1 else False
    if args['trainE2E'] is not None:
        cfg.TRAIN_E2E = True if args['trainE2E'] == 1 else False

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

# Defines the Faster R-CNN network model for detecting objects in images
def create_faster_rcnn_model(features, scaled_gt_boxes, dims_input, cfg):
    # Load the pre-trained classification net and clone layers
    base_model = load_model(cfg['BASE_MODEL_PATH'])
    conv_layers = clone_conv_layers(base_model, cfg)
    fc_layers = clone_model(base_model, [cfg["MODEL"].POOL_NODE_NAME], [cfg["MODEL"].LAST_HIDDEN_NODE_NAME], clone_method=CloneMethod.clone)

    # Normalization and conv layers
    feat_norm = features - Constant([[[v]] for v in cfg["MODEL"].IMG_PAD_COLOR])
    conv_out = conv_layers(feat_norm)

    # RPN and prediction targets
    rpn_rois, rpn_losses = create_rpn(conv_out, scaled_gt_boxes, dims_input, cfg)
    rois, label_targets, bbox_targets, bbox_inside_weights = \
        create_proposal_target_layer(rpn_rois, scaled_gt_boxes, cfg)

    # Fast RCNN and losses
    cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers, cfg)
    detection_losses = create_detection_losses(cls_score, label_targets, bbox_pred, rois, bbox_targets, bbox_inside_weights, cfg)
    loss = rpn_losses + detection_losses
    pred_error = classification_error(cls_score, label_targets, axis=1)

    return loss, pred_error

def create_faster_rcnn_eval_model(model, image_input, dims_input, cfg, rpn_model=None):
    print("creating eval model")
    last_conv_node_name = cfg["MODEL"].LAST_CONV_NODE_NAME
    conv_layers = clone_model(model, [cfg["MODEL"].FEATURE_NODE_NAME], [last_conv_node_name], CloneMethod.freeze)
    conv_out = conv_layers(image_input)

    model_with_rpn = model if rpn_model is None else rpn_model
    rpn = clone_model(model_with_rpn, [last_conv_node_name], ["rpn_cls_prob_reshape", "rpn_bbox_pred"], CloneMethod.freeze)
    rpn_out = rpn(conv_out)
    # we need to add the proposal layer anew to account for changing configs when buffering proposals in 4-stage training
    rpn_rois = create_proposal_layer(rpn_out.outputs[0], rpn_out.outputs[1], dims_input, cfg)

    roi_fc_layers = clone_model(model, [last_conv_node_name, "rpn_target_rois"], ["cls_score", "bbox_regr"], CloneMethod.freeze)
    pred_net = roi_fc_layers(conv_out, rpn_rois)
    cls_score = pred_net.outputs[0]
    bbox_regr = pred_net.outputs[1]

    if cfg.BBOX_NORMALIZE_TARGETS:
        num_boxes = int(bbox_regr.shape[1] / 4)
        bbox_normalize_means = np.array(cfg.BBOX_NORMALIZE_MEANS * num_boxes)
        bbox_normalize_stds = np.array(cfg.BBOX_NORMALIZE_STDS * num_boxes)
        bbox_regr = plus(element_times(bbox_regr, bbox_normalize_stds), bbox_normalize_means, name='bbox_regr')

    cls_pred = softmax(cls_score, axis=1, name='cls_pred')
    eval_model = combine([cls_pred, rpn_rois, bbox_regr])

    return eval_model

def store_eval_model_with_native_udf(eval_model, cfg):
    import copy
    sys.path.append(os.path.join(abs_path, "..", "..", "Extensibility", "ProposalLayer"))
    cntk.ops.register_native_user_function('ProposalLayerOp',
                                           'Cntk.ProposalLayerLib-' + cntk.__version__.rstrip('+'),
                                           'CreateProposalLayer')

    def filter(x):
        return type(x) == cntk.Function and x.op_name == 'UserFunction' and x.name == 'ProposalLayer'

    def converter(x):
        layer_config = copy.deepcopy(x.attributes)
        return cntk.ops.native_user_function('ProposalLayerOp', list(x.inputs), layer_config, 'native_proposal_layer')


    model_w_native_udf = cntk.misc.convert(eval_model, filter, converter)
    model_path = cfg['MODEL_PATH']
    new_model_path = model_path[:-6] + '_native.model'
    model_w_native_udf.save(new_model_path)
    print("Stored eval model with native UDF to {}".format(new_model_path))

def compute_rpn_proposals(rpn_model, image_input, roi_input, dims_input, cfg):
    num_images = cfg["DATA"].NUM_TRAIN_IMAGES
    # Create the minibatch source
    od_minibatch_source = ObjectDetectionMinibatchSource(
        cfg["DATA"].TRAIN_MAP_FILE, cfg["DATA"].TRAIN_ROI_FILE,
        num_classes=cfg["DATA"].NUM_CLASSES,
        max_annotations_per_image=cfg.INPUT_ROIS_PER_IMAGE,
        pad_width=cfg.IMAGE_WIDTH,
        pad_height=cfg.IMAGE_HEIGHT,
        pad_value=cfg["MODEL"].IMG_PAD_COLOR,
        max_images=num_images,
        randomize=False, use_flipping=False,
        proposal_provider=None)

    # define mapping from reader streams to network inputs
    input_map = {
        od_minibatch_source.image_si: image_input,
        od_minibatch_source.roi_si: roi_input,
        od_minibatch_source.dims_si: dims_input
    }

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

    return buffered_proposals

# If a trained model is already available it is loaded an no training will be performed (if MAKE_MODE=True).
def train_faster_rcnn(cfg):
    # Train only if no model exists yet
    model_path = cfg['MODEL_PATH']
    if os.path.exists(model_path) and cfg["CNTK"].MAKE_MODE:
        print("Loading existing model from %s" % model_path)
        eval_model = load_model(model_path)
    else:
        if cfg["CNTK"].TRAIN_E2E:
            eval_model = train_faster_rcnn_e2e(cfg)
        else:
            eval_model = train_faster_rcnn_alternating(cfg)

        eval_model.save(model_path)
        if cfg["CNTK"].DEBUG_OUTPUT:
            plot(eval_model, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_eval_{}_{}.{}"
                                          .format(cfg["MODEL"].BASE_MODEL, "e2e" if cfg["CNTK"].TRAIN_E2E else "4stage", cfg["CNTK"].GRAPH_TYPE)))

        print("Stored eval model at %s" % model_path)
    return eval_model

# Trains a Faster R-CNN model end-to-end
def train_faster_rcnn_e2e(cfg):
    # Input variables denoting features and labeled ground truth rois (as 5-tuples per roi)
    image_input = input_variable(shape=(cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                                 dynamic_axes=[Axis.default_batch_axis()],
                                 name=cfg["MODEL"].FEATURE_NODE_NAME)
    roi_input = input_variable((cfg.INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    dims_node = alias(dims_input, name='dims_input')

    # Instantiate the Faster R-CNN prediction model and loss function
    loss, pred_error = create_faster_rcnn_model(image_input, roi_input, dims_node, cfg)

    if cfg["CNTK"].DEBUG_OUTPUT:
        print("Storing graphs and models to %s." % cfg.OUTPUT_PATH)
        plot(loss, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_train_e2e." + cfg["CNTK"].GRAPH_TYPE))

    # Set learning parameters
    e2e_lr_factor = cfg["MODEL"].E2E_LR_FACTOR
    e2e_lr_per_sample_scaled = [x * e2e_lr_factor for x in cfg["CNTK"].E2E_LR_PER_SAMPLE]
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)

    print("Using base model:   {}".format(cfg["MODEL"].BASE_MODEL))
    print("lr_per_sample:      {}".format(e2e_lr_per_sample_scaled))

    train_model(image_input, roi_input, dims_input, loss, pred_error,
                e2e_lr_per_sample_scaled, mm_schedule, cfg["CNTK"].L2_REG_WEIGHT, cfg["CNTK"].E2E_MAX_EPOCHS, cfg)

    return create_faster_rcnn_eval_model(loss, image_input, dims_input, cfg)

# Trains a Faster R-CNN model using 4-stage alternating training
def train_faster_rcnn_alternating(cfg):
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

    # setting pre- and post-nms top N to training values since buffered proposals are used for further training
    test_pre = cfg["TEST"].RPN_PRE_NMS_TOP_N
    test_post = cfg["TEST"].RPN_POST_NMS_TOP_N
    cfg["TEST"].RPN_PRE_NMS_TOP_N = cfg["TRAIN"].RPN_PRE_NMS_TOP_N
    cfg["TEST"].RPN_POST_NMS_TOP_N = cfg["TRAIN"].RPN_POST_NMS_TOP_N

    # Learning parameters
    rpn_lr_factor = cfg["MODEL"].RPN_LR_FACTOR
    rpn_lr_per_sample_scaled = [x * rpn_lr_factor for x in cfg["CNTK"].RPN_LR_PER_SAMPLE]
    frcn_lr_factor = cfg["MODEL"].FRCN_LR_FACTOR
    frcn_lr_per_sample_scaled = [x * frcn_lr_factor for x in cfg["CNTK"].FRCN_LR_PER_SAMPLE]

    l2_reg_weight = cfg["CNTK"].L2_REG_WEIGHT
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)
    rpn_epochs = cfg["CNTK"].RPN_EPOCHS
    frcn_epochs = cfg["CNTK"].FRCN_EPOCHS

    feature_node_name = cfg["MODEL"].FEATURE_NODE_NAME
    last_conv_node_name = cfg["MODEL"].LAST_CONV_NODE_NAME
    print("Using base model:   {}".format(cfg["MODEL"].BASE_MODEL))
    print("rpn_lr_per_sample:  {}".format(rpn_lr_per_sample_scaled))
    print("frcn_lr_per_sample: {}".format(frcn_lr_per_sample_scaled))

    debug_output=cfg["CNTK"].DEBUG_OUTPUT
    if debug_output:
        print("Storing graphs and models to %s." % cfg.OUTPUT_PATH)

    # Input variables denoting features, labeled ground truth rois (as 5-tuples per roi) and image dimensions
    image_input = input_variable(shape=(cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                                 dynamic_axes=[Axis.default_batch_axis()],
                                 name=feature_node_name)
    feat_norm = image_input - Constant([[[v]] for v in cfg["MODEL"].IMG_PAD_COLOR])
    roi_input = input_variable((cfg.INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    scaled_gt_boxes = alias(roi_input, name='roi_input')
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    dims_node = alias(dims_input, name='dims_input')
    rpn_rois_input = input_variable((cfg["TRAIN"].RPN_POST_NMS_TOP_N, 4), dynamic_axes=[Axis.default_batch_axis()])
    rpn_rois_buf = alias(rpn_rois_input, name='rpn_rois')

    # base image classification model (e.g. VGG16 or AlexNet)
    base_model = load_model(cfg['BASE_MODEL_PATH'])

    print("stage 1a - rpn")
    if True:
        # Create initial network, only rpn, without detection network
            #       initial weights     train?
            # conv: base_model          only conv3_1 and up
            # rpn:  init new            yes
            # frcn: -                   -

        # conv layers
        conv_layers = clone_conv_layers(base_model, cfg)
        conv_out = conv_layers(feat_norm)

        # RPN and losses
        rpn_rois, rpn_losses = create_rpn(conv_out, scaled_gt_boxes, dims_node, cfg)
        stage1_rpn_network = combine([rpn_rois, rpn_losses])

        # train
        if debug_output: plot(stage1_rpn_network, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_train_stage1a_rpn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, rpn_losses, rpn_losses,
                    rpn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, rpn_epochs, cfg)

    print("stage 1a - buffering rpn proposals")
    buffered_proposals_s1 = compute_rpn_proposals(stage1_rpn_network, image_input, roi_input, dims_input, cfg)

    print("stage 1b - frcn")
    if True:
        # Create full network, initialize conv layers with imagenet, fix rpn weights
            #       initial weights     train?
            # conv: base_model          only conv3_1 and up
            # rpn:  stage1a rpn model   no --> use buffered proposals
            # frcn: base_model + new    yes

        # conv_layers
        conv_layers = clone_conv_layers(base_model, cfg)
        conv_out = conv_layers(feat_norm)

        # use buffered proposals in target layer
        rois, label_targets, bbox_targets, bbox_inside_weights = \
            create_proposal_target_layer(rpn_rois_buf, scaled_gt_boxes, cfg)

        # Fast RCNN and losses
        fc_layers = clone_model(base_model, [cfg["MODEL"].POOL_NODE_NAME], [cfg["MODEL"].LAST_HIDDEN_NODE_NAME], CloneMethod.clone)
        cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers, cfg)
        detection_losses = create_detection_losses(cls_score, label_targets, bbox_pred, rois, bbox_targets, bbox_inside_weights, cfg)
        pred_error = classification_error(cls_score, label_targets, axis=1, name="pred_error")
        stage1_frcn_network = combine([rois, cls_score, bbox_pred, detection_losses, pred_error])

        # train
        if debug_output: plot(stage1_frcn_network, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_train_stage1b_frcn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, detection_losses, pred_error,
                    frcn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, frcn_epochs, cfg,
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
        if debug_output: plot(stage2_rpn_network, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_train_stage2a_rpn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, rpn_losses, rpn_losses,
                    rpn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, rpn_epochs, cfg)

    print("stage 2a - buffering rpn proposals")
    buffered_proposals_s2 = compute_rpn_proposals(stage2_rpn_network, image_input, roi_input, dims_input, cfg)

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
        if debug_output: plot(stage2_frcn_network, os.path.join(cfg.OUTPUT_PATH, "graph_frcn_train_stage2b_frcn." + cfg["CNTK"].GRAPH_TYPE))
        train_model(image_input, roi_input, dims_input, detection_losses, pred_error,
                    frcn_lr_per_sample_scaled, mm_schedule, l2_reg_weight, frcn_epochs, cfg,
                    rpn_rois_input=rpn_rois_input, buffered_rpn_proposals=buffered_proposals_s2)
        buffered_proposals_s2 = None

    # resetting config values to original test values
    cfg["TEST"].RPN_PRE_NMS_TOP_N = test_pre
    cfg["TEST"].RPN_POST_NMS_TOP_N = test_post

    return create_faster_rcnn_eval_model(stage2_frcn_network, image_input, dims_input, cfg, rpn_model=stage2_rpn_network)

def train_model(image_input, roi_input, dims_input, loss, pred_error,
                lr_per_sample, mm_schedule, l2_reg_weight, epochs_to_train, cfg,
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
    lr_schedule = learning_parameter_schedule_per_sample(lr_per_sample)
    learner = momentum_sgd(others, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight,
                           unit_gain=False, use_mean_gradient=True)

    bias_lr_per_sample = [v * bias_lr_mult for v in lr_per_sample]
    bias_lr_schedule = learning_parameter_schedule_per_sample(bias_lr_per_sample)
    bias_learner = momentum_sgd(biases, bias_lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight,
                           unit_gain=False, use_mean_gradient=True)
    trainer = Trainer(None, (loss, pred_error), [learner, bias_learner])

    # Get minibatches of images and perform model training
    print("Training model for %s epochs." % epochs_to_train)
    log_number_of_parameters(loss)

    # Create the minibatch source
    if buffered_rpn_proposals is not None:
        proposal_provider = ProposalProvider.fromlist(buffered_rpn_proposals, requires_scaling=False)
    else:
        proposal_provider = None

    od_minibatch_source = ObjectDetectionMinibatchSource(
        cfg["DATA"].TRAIN_MAP_FILE, cfg["DATA"].TRAIN_ROI_FILE,
        num_classes=cfg["DATA"].NUM_CLASSES,
        max_annotations_per_image=cfg.INPUT_ROIS_PER_IMAGE,
        pad_width=cfg.IMAGE_WIDTH,
        pad_height=cfg.IMAGE_HEIGHT,
        pad_value=cfg["MODEL"].IMG_PAD_COLOR,
        randomize=True,
        use_flipping=cfg["TRAIN"].USE_FLIPPED,
        max_images=cfg["DATA"].NUM_TRAIN_IMAGES,
        proposal_provider=proposal_provider)

    # define mapping from reader streams to network inputs
    input_map = {
        od_minibatch_source.image_si: image_input,
        od_minibatch_source.roi_si: roi_input,
    }
    if buffered_rpn_proposals is not None:
        input_map[od_minibatch_source.proposals_si] = rpn_rois_input
    else:
        input_map[od_minibatch_source.dims_si] = dims_input

    progress_printer = ProgressPrinter(tag='Training', num_epochs=epochs_to_train, gen_heartbeat=True)
    for epoch in range(epochs_to_train):       # loop over epochs
        sample_count = 0
        while sample_count < cfg["DATA"].NUM_TRAIN_IMAGES:  # loop over minibatches in the epoch
            data = od_minibatch_source.next_minibatch(min(cfg.MB_SIZE, cfg["DATA"].NUM_TRAIN_IMAGES-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
            if sample_count % 100 == 0:
                print("Processed {} samples".format(sample_count))

        progress_printer.epoch_summary(with_metric=True)

