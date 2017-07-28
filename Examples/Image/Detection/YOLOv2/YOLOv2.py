# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os

import cntk.io.transforms as xforms
from cntk import *
from cntk import reshape, relu, user_function
from cntk.io import ImageDeserializer, CTFDeserializer, MinibatchSource, TraceLevel
from cntk.io import StreamDefs, StreamDef
from cntk.layers import Convolution2D, BatchNormalization, Sequential
from cntk.logging.graph import find_by_name, plot
from cntk_debug_single import DebugLayerSingle

from extensions.depth_increasing_pooling import depth_increasing_pooling

import pdb
Debug = False

# def new_create_output_activation_layer(network, par):
#
#
#     anchor_box_scales = par.par_anchorbox_scales
#     n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
#     n_gridcells_vertical = int(par.par_image_height / par.par_downsample)
#     n_anchorboxes = len(anchor_box_scales)
#     output_width = n_gridcells_horizontal * n_gridcells_vertical * n_anchorboxes
#     output_height = par.par_num_classes + 5
#
#     if False:
#         from TrainUDFyolov2 import LambdaFunc
#         lf = LambdaFunc(network, lambda arg: arg.shape != (par.par_minibatch_size,output_height*n_anchorboxes,n_gridcells_vertical,n_gridcells_horizontal))
#         network = user_function(lf)
#
#     #net arrives as c_h_w (vector*ab,y,x)
#     tp = transpose(network, (2,1,0))
#     #tp is w_h_c (x,y,ab*vector)
#     reshaped = reshape(tp,(n_gridcells_horizontal, n_gridcells_vertical, n_anchorboxes, output_height))
#     # a vector is now reshaped[x][y][ab]
#
#     xy_cords = alias(reshaped[:,:,:,0:2],name="XY")
#     wh_mults = alias(reshaped[:,:,:,2:4],name="WH")
#     objectnesses = alias(reshaped[:,:,:,4:5], name="OBJ")
#     cls_outs = alias(reshaped[:,:,:,5:],name="CLS")
#
#     # classes: 245*20; softmax should be applied to each row of 20 preds
#     cls_preds = ops.softmax(cls_outs, axis=3)
#
#     # objectness: 245*1; sigmoid to each
#     obj_preds = ops.sigmoid(objectnesses)
#
#     # xy_cords: 245*2; the offset for each coordinate
#     xy_rel_in_grid = ops.sigmoid(xy_cords)
#
#     xys = np.zeros((n_gridcells_horizontal, n_gridcells_vertical, n_anchorboxes,2))
#     for i in range(n_gridcells_horizontal):
#         xys[i,:,:,0]=i
#     for i in range(n_gridcells_vertical):
#         xys[:,i,:,1]=i
#     grid_numbers = ops.constant(np.ascontiguousarray(xys, np.float32), name="Grid_Pos")
#     xy_in_gridcells = xy_rel_in_grid + grid_numbers
#     xy_preds = xy_in_gridcells / ops.constant(np.ascontiguousarray([1/n_gridcells_horizontal, 1/n_gridcells_vertical], np.float32), name="grid_dims")
#
#     # wh_mults: 245*2 for the anchorboxes
#     wh_rels = ops.exp(wh_mults)
#     ab_scales = ops.constant(np.asarray(anchor_box_scales, dtype=np.float32),name="ab_scales")
#
#     wh_preds = wh_rels * ab_scales
#
#     # Splice the parts back together!
#     full_output = ops.splice(xy_preds, wh_preds, obj_preds, cls_preds, axis=3, name="yolo_results")
#
#     # full output format: x,y,ab,vector
#     # transform to output format h*w*ab,vector
#     tp_out = transpose(full_output, (1,0,2,3)) # thereafter: y,x,ab,vector
#     rs_out = reshape(tp_out, (output_width, output_height))
#     return rs_out
#
# def apply_xy_output_func(xy_in, par, w_h_ab_unflattened=False):
#     """
#     Applies sigmoid to input, adds the number of the gc, divides by gc_size to keep values in [0..1]
#     :param xy_in:
#     :param par:
#     :param w_h_ab_unflattened:
#     :return:
#     """
#     n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
#     n_gridcells_vertical = int(par.par_image_height / par.par_downsample)
#     output_width = n_gridcells_horizontal * n_gridcells_vertical * par.par_num_anchorboxes
#
#
#     if w_h_ab_unflattened:
#         ""
#     else:
#         # xy_cords: 245*2; the offset for each coordinate
#         xy_rel_in_grid = ops.sigmoid(xy_in)
#         ## create constants for offset
#         ### offsets
#         xs = []
#         ys = []
#         cur_x = -1
#         cur_y = -1
#         x_div = par.par_num_anchorboxes
#         y_div = par.par_num_anchorboxes * n_gridcells_horizontal
#         for i in range(output_width):
#             if (i % x_div == 0): cur_x += 1
#             if (i % y_div == 0): cur_y += 1
#             if (cur_x == n_gridcells_horizontal): cur_x = 0
#             xs.append([cur_x])
#             ys.append([cur_y])
#         xs = np.asarray(xs, np.float32)
#         ys = np.asarray(ys, np.float32)
#         xys = np.concatenate((xs, ys), axis=1)
#         grid_numbers = ops.constant(np.ascontiguousarray(xys, np.float32), name="Grid_Pos")
#         ### scales
#         scales = [[1.0 / n_gridcells_horizontal] * output_width, [1.0 / n_gridcells_vertical] * output_width]
#         scales = np.asarray(scales, np.float32).transpose(1, 0)
#         # scale_imagedim_per_gridcell = ops.constant(scales, name="Scale_gridcells_to_relative")
#         scale_imagedim_per_gridcell = ops.constant(np.ascontiguousarray(scales, np.float32),
#                                                    name="Scale_gridcells_to_relative")
#         ## constants done!
#
#         xy_in_gridcells = ops.plus(xy_rel_in_grid, grid_numbers)
#         xy_preds = xy_in_gridcells * scale_imagedim_per_gridcell
#     return xy_preds
#
# def apply_wh_output_func(wh_in, par):
#     wh_rels = ops.exp(wh_in)
#     ## create constants for anchorbox scales
#     ab_scales = []
#     for i in range(par.par_num_anchorboxes):
#         ab_scales.append([par.par_anchorbox_scales[i][0], par.par_anchorbox_scales[i][1]])
#
#     n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
#     n_gridcells_vertical = int(par.par_image_height / par.par_downsample)
#     ab_scales = ab_scales * n_gridcells_horizontal * n_gridcells_vertical
#     ab_scales = np.asarray(ab_scales, np.float32)
#     anchorbox_scale_mults = ops.constant(np.ascontiguousarray(ab_scales, np.float32), name="Anchorbox-scales")
#     ## constants done
#     wh_preds = wh_rels * anchorbox_scale_mults
#     return wh_preds
#
# def apply_obj_output_func(obj_in):
#     obj_preds = ops.sigmoid(obj_in)
#     return obj_preds
#
# def apply_cls_output_func(cls_in, axis):
#     cls_preds = ops.softmax(cls_in, axis=axis)
#     return cls_preds
#
# def create_output_activation_layer_subfuncs(network, par):
#     anchor_box_scales=par.par_anchorbox_scales
#     n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
#     n_gridcells_vertical =  int(par.par_image_height / par.par_downsample)
#     n_anchorboxes=len(anchor_box_scales)
#     output_width = n_gridcells_horizontal * n_gridcells_vertical * n_anchorboxes
#     output_height =  par.par_num_classes + 5
#
#     if False:
#         from TrainUDFyolov2 import LambdaFunc
#         lf = LambdaFunc(network, lambda arg: arg.shape != (par.par_minibatch_size,output_height*n_anchorboxes,n_gridcells_vertical,n_gridcells_horizontal))
#         network = user_function(lf)
#
#     # reorder array! now 125*7*7
#     # tp1 = ops.transpose(network, 0,2) # 7*7*125
#     # tp2 = ops.transpose(tp1, 0, 1) # 7*7*125
#     # network is coded: c_h_w
#     tp = ops.transpose(network, (1,2,0))
#     # tp is coded h_w_c(1,2,0:w_h_c) #changed
#     reshaped = ops.reshape(tp, (output_width, output_height))
#     #shape is now 245 * 25
#
#     # slicing the array into subarrays
#     xy_cords = ops.slice(reshaped, axis=1, begin_index=0, end_index=2, name="XY-Out")
#     wh_mults = ops.slice(reshaped, axis=1, begin_index=2, end_index=4, name="WH-Out")
#     objectnesses =  ops.slice(reshaped, axis=1, begin_index=4, end_index=5, name="Obj_Out")
#     cls_outs = ops.slice(reshaped, axis=1, begin_index=5, end_index=output_height, name="Cls_Out")
#
#     if False: # Test-case to see if bounding-box placement is ok!
#         xy_cords = xy_cords * [0]
#         wh_mults = wh_mults * [0]
#
#     #applying output functions to the parts
#
#     # classes: 245*20; softmax should be applied to each row of 20 preds
#     cls_preds = apply_cls_output_func(cls_outs, axis=1)
#
#     # objectness: 245*1; sigmoid to each
#     obj_preds = apply_obj_output_func(objectnesses)
#
#     # xy_cords: 245*2; the offset for each coordinate
#     xy_preds = apply_xy_output_func(xy_cords, False)
#     wh_preds = apply_wh_output_func(wh_mults, par)
#
#     # Splice the parts back together!
#     full_output = ops.splice(xy_preds, wh_preds, obj_preds, cls_preds, axis=1, name="yolo_results")
#     return full_output

def create_output_activation_layer(network, par):
    #pdb.set_trace()
    anchor_box_scales=par.par_anchorbox_scales
    n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
    n_gridcells_vertical =  int(par.par_image_height / par.par_downsample)
    n_anchorboxes=len(anchor_box_scales)
    output_width = n_gridcells_horizontal * n_gridcells_vertical * n_anchorboxes
    output_height =  par.par_num_classes + 5

    # reorder array! now 125*7*7
    # tp1 = ops.transpose(network, 0,2) # 7*7*125
    # tp2 = ops.transpose(tp1, 0, 1) # 7*7*125
    # network is coded: c_h_w

    tp = ops.transpose(network, (1,2,0))
    # tp is coded h_w_c(1,2,0:w_h_c) #changed
    reshaped = ops.reshape(tp, (output_width, output_height))
    #shape is now 245 * 25


    # slicing the array into subarrays
    xy_cords = ops.slice(reshaped, axis=1, begin_index=0, end_index=2, name="XY-Out")
    wh_mults = ops.slice(reshaped, axis=1, begin_index=2, end_index=4, name="WH-Out")
    objectnesses =  ops.slice(reshaped, axis=1, begin_index=4, end_index=5, name="Obj_Out")
    cls_outs = ops.slice(reshaped, axis=1, begin_index=5, end_index=output_height, name="Cls_Out")

    #xy_cords = user_function(DebugLayerSingle(xy_cords, debug_name='XY_Out_d'))
    #wh_mults = user_function(DebugLayerSingle(wh_mults, debug_name='WH_Out_d'))
    #wh_mults = alias(wh_mults, name="WH_Out_d_alias")
    #objectnesses = user_function(DebugLayerSingle(objectnesses, debug_name='Obj_Out_d'))
    #cls_outs = user_function(DebugLayerSingle(cls_outs, debug_name='Cls_Out_d', split_line=True))

    if False: # Test-case to see if bounding-box placement is ok!
        xy_cords = xy_cords * [0]
        wh_mults = wh_mults * [0]

    #applying output functions to the parts

    # classes: 245*20; softmax should be applied to each row of 20 preds
    cls_preds = ops.softmax(cls_outs, axis=1)

    # objectness: 245*1; sigmoid to each
    obj_preds = ops.sigmoid(objectnesses)

    # xy_cords: 245*2; the offset for each coordinate
    xy_rel_in_grid = ops.sigmoid(xy_cords)
    ## create constants for offset
    ### offsets
    xs = []
    ys = []
    cur_x = -1
    cur_y = -1
    x_div = n_anchorboxes
    y_div = n_anchorboxes * n_gridcells_horizontal
    for i in range (output_width):
        if(i % x_div == 0): cur_x += 1
        if(i % y_div == 0): cur_y += 1
        if(cur_x == n_gridcells_horizontal): cur_x = 0
        xs.append([cur_x])
        ys.append([cur_y])
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    xys = np.concatenate((xs,ys), axis=1)
    grid_numbers = ops.constant(np.ascontiguousarray(xys, np.float32), name="Grid_Pos")
    ### scales
    scales = [[1.0 / n_gridcells_horizontal] * output_width, [1.0 / n_gridcells_vertical] * output_width]
    scales = np.asarray(scales, np.float32).transpose(1, 0)
    #scale_imagedim_per_gridcell = ops.constant(scales, name="Scale_gridcells_to_relative")
    scale_imagedim_per_gridcell = ops.constant(np.ascontiguousarray(scales, np.float32), name="Scale_gridcells_to_relative")
    ## constants done!

    xy_in_gridcells = ops.plus(xy_rel_in_grid, grid_numbers)
    xy_preds = xy_in_gridcells * scale_imagedim_per_gridcell

    # wh_mults: 245*2 for the anchorboxes
    wh_rels = ops.exp(wh_mults)
    ## create constants for anchorbox scales
    ab_scales = []
    for i in range(n_anchorboxes):
        ab_scales.append([anchor_box_scales[i][0]
                          # * 1.0/n_gridcells_horizontal
                             , anchor_box_scales[i][1]
                          # *1.0/n_gridcells_vertical
                         ])
    ab_scales = ab_scales*n_gridcells_horizontal*n_gridcells_vertical
    ab_scales = np.asarray(ab_scales, np.float32)
    #anchorbox_scale_mults = ops.constant(ab_scales, name="Anchorbox-scales")
    anchorbox_scale_mults = ops.constant(np.ascontiguousarray(ab_scales, np.float32), name="Anchorbox-scales")
    ## constants done
    # wh_preds = cntk.ops.times(wh_rels, anchorbox_scale_mults) # this is mat-mul not element-wise!
    wh_preds = wh_rels * anchorbox_scale_mults

    #wh_preds = user_function(DebugLayerSingle(wh_preds, debug_name='wh_preds_d'))
    #xy_preds = user_function(DebugLayerSingle(xy_preds, debug_name='xy_preds_d'))
    #obj_preds = user_function(DebugLayerSingle(obj_preds, debug_name='obj_preds_d'))
    #cls_preds = user_function(DebugLayerSingle(cls_preds, debug_name='cls_preds_d', split_line=True))

    # Splice the parts back together!
    full_output = ops.splice(xy_preds, wh_preds, obj_preds, cls_preds, axis=1, name="yolo_results")
    return full_output


def create_detector(par, regular_input, bypass_input=None):
    #first_conv_name = "z.x.x.x.x.x.x.x.x.x.x._.x.c"
    #first_conv_node = regular_input.find_by_name(first_conv_name)
    #dummy = user_function(DebugLayerSingle(first_conv_node, debug_name='conv1'))
    #zero = reduce_mean(dummy * 0)

    #first_bnrelu_name = "z.x.x.x.x.x.x.x.x.x.x"
    #first_bnrelu_node = regular_input.find_by_name(first_bnrelu_name)
    #dummy2 = user_function(DebugLayerSingle(first_bnrelu_node, debug_name='bnrelu1'))
    #zero = reduce_mean(dummy2 * 0)

    #regular_input = user_function(DebugLayerSingle(regular_input, debug_name='regular_input_d'))
    from cntk.layers import LayerNormalization


    output_depth = (5+par.par_num_classes)*par.par_num_anchorboxes
    first_net = Sequential([
        Convolution2D((3, 3), num_filters=par.par_dense_size, init=he_normal(), activation=lambda x: 0.1*x + 0.9*relu(x), pad=True),
        BatchNormalization(),
        #LayerNormalization(),
        Convolution2D((3, 3), num_filters=par.par_dense_size, init=he_normal(), activation=lambda x: 0.1*x + 0.9*relu(x), pad=True)
    ])(regular_input)

    if bypass_input is not None:
        bypass_output = splice(first_net, bypass_input, axis=0)
    else:
        bypass_output = first_net


    net2=Sequential([
        BatchNormalization(),
        #LayerNormalization(),
        Convolution2D((3, 3), num_filters=par.par_dense_size, init=he_normal(), activation=lambda x: 0.1*x + 0.9*relu(x), pad=True),
        #BatchNormalization(), #C-impl says no!

        Convolution2D((1, 1), num_filters=output_depth),
    ], "YOLOv2_Detector")(bypass_output)

    if Debug: #for debug purposes, prints all the inputs!
        from TrainUDFyolov2 import LambdaFunc
        lf = LambdaFunc(net2,execute=lambda arg: print(arg.shape))
        net2 = user_function(lf)

    net3 = create_output_activation_layer(net2, par)# + zero
    return net3


def load_pretrained_darknet_feature_extractor():
    loaded_model = load_model(".\Output\darknet19_CIFAR10_7.model")

    feature_layer = find_by_name(loaded_model, "feature_layer")
    fe_output_layer = find_by_name(loaded_model, "featureExtractor_output")

    return combine([fe_output_layer.owner]).clone(CloneMethod.clone, {feature_layer: placeholder()})

def load_pretrained_alexnet_feature_extractor():
    # Load the pretrained classification net and find nodes
    loaded_model = load_model("../../PretrainedModels/AlexNet.model")
    feature_node = find_by_name(loaded_model, "features")
    conv_node = find_by_name(loaded_model,  "conv5.y")
    pool_node = find_by_name(loaded_model, "pool3")
    last_node = find_by_name(loaded_model, "h2_d")

    # Clone the conv layers and the fully connected layers of the network
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: placeholder()})
    fc_layers = combine([last_node.owner]).clone(CloneMethod.clone, {pool_node: placeholder()})
    plot(conv_layers, "alexnet_conv_layers.png")
    plot(fc_layers, "alexnet_fc_layerrs.png")

    usable_layers = combine([pool_node.owner]).clone(CloneMethod.freeze, {feature_node: placeholder()})


def load_pretrained_resnet18_feature_extractor(par):
    model_fn = os.path.normpath(os.path.join(par.par_abs_path, "..", "..",
        "PretrainedModels", "ResNet18_ImageNet_CNTK.model"))
    if not os.path.exists(model_fn):
        raise ValueError('Model %s does not exist'%model_fn)
    loaded_model = load_model(model_fn)

    feature_layer = find_by_name(loaded_model, "features")
    fe_output_layer = find_by_name(loaded_model, "z.x.x.r")
    #ph = placeholder(shape=(par.par_num_channels, par.par_image_width, par.par_image_height), name="input_ph")
    #net = combine([fe_output_layer.owner]).clone(CloneMethod.clone, {feature_layer: ph})
    ph = placeholder(shape=(100, 100, 100), name="input_ph")
    net = combine([fe_output_layer.owner]).clone(CloneMethod.freeze, {feature_layer: ph})

    #plot(net, "ResNet18_s.pdf")

    return Sequential([
        [lambda x: x - par.par_input_bias]
        ,net])

def load_pretrained_resnet101_feature_extractor(par):
    loaded_model = load_model(os.path.join(par.par_abs_path, "..", "..", "PretrainedModels", "ResNet101_ImageNet.model"))
    feature_layer = find_by_name(loaded_model, "data")
    #first_conv = find_by_name(loaded_model, "conv1")
    #first_conv = first_conv(placeholder(shape=(par.par_num_channels, par.par_image_width, par.par_image_height)))
    fe_output_layer = find_by_name(loaded_model, "res5c_relu")
    #return combine([fe_output_layer.owner]).clone(CloneMethod.freeze, {first_conv: placeholder()})
    # return combine([fe_output_layer.owner]).clone(CloneMethod.clone, {feature_layer: placeholder()})
    ph = placeholder(shape=(par.par_num_channels, par.par_image_width, par.par_image_height),name="input_ph")
    net = combine([fe_output_layer.owner]).clone(CloneMethod.clone, {feature_layer: ph})

    return net
    # make resnet a block if desired
    #return as_block(net,[(net.arguments[0], net.arguments[0])],"ResNet_FE","ResNet_FE")


def create_untrained_darknet19_fe():
    import darknet.darknet19 as dn19
    return dn19.create_feature_extractor(32)


def create_feature_extractor(par, use_model = "pre_ResNet18_ImageNet"):
    if(use_model == "pre_Darknet_Cifar"):
        fe = load_pretrained_darknet_feature_extractor()
        rl = find_by_name(fe, "YOLOv2PasstroughSource")
    elif(use_model == "pre_ResNet101_ImageNet"):
        fe = load_pretrained_resnet101_feature_extractor(par)
        rl = None
    elif(use_model == "pre_ResNet18_ImageNet"):
        fe = load_pretrained_resnet18_feature_extractor(par)
        rl = find_by_name(fe, "z.x.x.x.x.r")

    if not par.par_use_reorg_bypass:
        rl=None

    return fe, rl


def create_yolov2_net(par):
    output, reorg_input = create_feature_extractor(par)
    reorg_output = depth_increasing_pooling(reorg_input, (2,2)) if reorg_input is not None else None
    return create_detector(par, output, reorg_output)


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################

def predict_image(model, path, par, conf_threshold = 0.5, show=True):
    import matplotlib.pyplot as mp
    import cv2

    cv_img = cv2.imread(path)
    resized = cv2.resize(cv_img, (par.par_image_width, par.par_image_height), interpolation=cv2.INTER_NEAREST)
    bgr_image = np.asarray(resized, dtype=np.float32)
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    arguments = {model.arguments[0]: [hwc_format]}

    predictions = model.eval(arguments)[0]
    # print(predictions[0:5])

    box_list = []

    for j in range(predictions.shape[0]):
        box = predictions[j]
        if (box[4] > conf_threshold):
            box_list.append(box)
    # print("box" + ":" + str(len(box_list)))

    # draw rectangles!
    image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    box_list_len = len(box_list)
    for j in range(box_list_len):
        box = box_list[j]
        xmin = int(par.par_image_width * box[0] - par.par_image_width * box[2] / 2)
        xmax = int(xmin + par.par_image_width * box[2])
        ymin = int(par.par_image_height * box[1] - par.par_image_height * box[3] / 2)
        ymax = int(ymin + par.par_image_height * box[3])
        if(xmax >= par.par_image_width or ymax >= par.par_image_height or xmin < 0 or ymin < 0):
            print("Box out of bounds: (" + str(xmin) +","+ str(ymin) +") ("+ str(xmax) +","+ str(ymax) +")")
            # print(box[5:])
        xmax = par.par_image_width-1 if xmax >= par.par_image_width else xmax
        ymax = par.par_image_height-1 if ymax >= par.par_image_height else ymax
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin

        color = (255, 255 - int(j*255/box_list_len), int(j*255/box_list_len))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        # print(box[5:])
        detected_class = np.argmax(box[5:]) + 1


        cv2.putText(image, str(detected_class), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color,1)

    if show:
        mp.imshow(image)
        mp.plot()
        mp.show()
    return image

def create_mb_source(img_height, img_width, img_channels, output_size, image_file, roi_file, is_training = False, multithreaded_deserializer=False, max_samples=io.INFINITELY_REPEAT, max_epochs = io.INFINITELY_REPEAT):
    transforms = []

    if False and is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
        ]

    if True:
        transforms += [
            xforms.scale(width=img_width, height=img_height, channels=img_channels, interpolations='linear',
                         scale_mode='fill'),
        ]
    else:
        # TODO resize bounding boxes
        transforms += [
            xforms.scale(width=img_width, height=img_height, channels=img_channels, interpolations='linear',
                         scale_mode='pad', pad_value=114),
        ]

    image_source = ImageDeserializer(image_file, StreamDefs(features=StreamDef(field='image', transforms=transforms)))


    # read rois and labels
    roi_source = CTFDeserializer(roi_file, StreamDefs(label=StreamDef(field='roiAndLabel', shape=output_size)))

    rc = MinibatchSource([image_source, roi_source], randomize=False, trace_level=TraceLevel.Error,
                         multithreaded_deserializer=multithreaded_deserializer, max_samples=max_samples)#, max_epochs=max_epochs)
    return rc



# if __name__ == '__main__':
#     import PARAMETERS as par
#     model = create_yolov2_net(par)
#     print("created model!")
#     #model = load_pretrained_resnet101_feature_extractor()
#     # plot
#
#
#     image_input= input_variable((par.par_num_channels, par.par_image_width, par.par_image_height))
#     output = model(image_input) # append model to image input
#     plot(output, filename=os.path.join(par.par_abs_path, "YOLOv2_in.pdf"))
#     exit()
#     """
#     # input for ground truth boxes
#     num_gtb = par.par_max_gtbs
#     gtb_input = input((num_gtb*5)) # 5 for  x,y,w,h,class
#
#     from ErrorFunction import get_error
#     mse = get_error(output, gtb_input, cntk_only=False)
#
#     plot(mse, filename=os.path.join(par.par_abs_path,"Trainnet.pdf"))
#
#
#     # trainig params
#     max_epochs = par.par_max_epochs
#     epoch_size = par.par_epoch_size
#     minibatch_size = par.par_minibatch_size
#
#     lr_schedule = learning_rate_schedule(par.par_lr_schedule, learners.UnitType.sample, epoch_size)
#     mm_schedule = learners.momentum_schedule([-minibatch_size / np.log(par.par_momentum)], epoch_size)
#
#     # Instantiate the trainer object to drive the model training
#     learner = learners.momentum_sgd(mse.parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=0.0005)
#     trainer = Trainer(None, (mse, mse), [learner])
#
#     image_file = os.path.join(par.par_abs_path, "..", "..", "DataSets", "Pascal", "mappings" , "trainval2007.txt")
#     roi_file = os.path.join(par.par_abs_path, "..", "..", "DataSets", "Pascal", "mappings" , "trainval2007_rois_center_rel.txt")
#
#     minibatch_source = create_mb_source(par.par_image_height, par.par_image_width, par.par_num_channels, (5 * num_gtb), image_file, roi_file)
#
#     # define mapping from reader streams to network inputs
#     input_map = {
#         image_input: minibatch_source["features"],
#         gtb_input: minibatch_source["label"]
#     }
#
#     progress_printer = ProgressPrinter(freq= int(epoch_size / 10), tag='Training', rank=Communicator.rank(), gen_heartbeat=True,
#                                        num_epochs=max_epochs)
#
#
#     for epoch in range(max_epochs):  # loop over epochs
#         print("---Start new epoch---")
#         sample_count = 2
#         while sample_count < epoch_size :#- minibatch_size:  # loop over minibatches in the epoch
#
#             # get next minibatch data
#             data = minibatch_source.next_minibatch(min(minibatch_size, epoch_size - sample_count),
#                                                    input_map=input_map)  # fetch minibatch.
#             gtbs = (data[gtb_input]).asarray()
#
#             trainer.train_minibatch({image_input: data[image_input].asarray(), gtb_input:gtbs}, device=gpu(0))  # update model with it
#             sample_count += data[image_input].num_samples  # count samples processed so far
#             progress_printer.update_with_trainer(trainer=trainer, with_metric=True)  # log progress
#             print(progress_printer.avg_metric_since_start())
#
#         progress_printer.epoch_summary(with_metric=True)
#
#     save_path = os.path.join(par.par_abs_path,  "outputdir")
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     output.save_model(os.path.join(save_path, "YOLOv2.model"))
#     """
#     # alternatively to training a model you can reload a pretrained!
#     model = load_model(r".\outputdir\YOLOv2.model")
#     #image_input = input((par.par_num_channels, par.par_image_height, par_image_width))
#
#     plot(model, filename=os.path.join(par.par_abs_path, "YOLOv2.pdf"))
#
#     import ipdb;ipdb.set_trace()
#     output = model#(image_input)  # append model to image input
#
#     #predict a few images
#     names = ["000018.jpg","000118.jpg","001118.jpg","002118.jpg"]
#     threshold_objectness = 0.505
#     for i in range(len(names)):
#         image = os.path.join(par.par_abs_path, "..", "..", "DataSets", "Pascal", "VOCdevkit" , "VOC2007", "JPEGImages" ,names[i])
#         predict_image(output, image, conf_threshold=threshold_objectness)
#
#     print("Done!")
#
#     import ipdb;ipdb.set_trace()
#     predict_image(output,r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages\000001.jpg",0.501)
#
# def test_create_output_activation_layer():
#     """
#     Test for create_output_activation_layer()
#     :return: Nothing
#     """
#     assert False, "Not implemented yet"
#
# def test_new_create_output_activation_layer():
#     """
#     Test for new_create_output_activation_layer()
#     :return: Nothing
#     """
#     assert False, "Not implemented yet"
#
# def test_create_mb_source():
#     """
#     Test for create_mb_source()
#     :return: Nothing
#     """
#     assert False, "Not implemented yet"
#
#
# def test_apply_xy_output_func():
#     """
#     Test for apply_xy_output_func()
#     :return: Nothing
#     """
#     import PARAMETERS as par
#     n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
#     n_gridcells_vertical = int(par.par_image_height / par.par_downsample)
#     n_ab = par.par_num_anchorboxes
#     output_width = n_gridcells_horizontal * n_gridcells_vertical * n_ab
#     output_height = par.par_num_classes + 5
#
#
#
#     # Testing zero input --> center of the box
#     data_flat = np.zeros((output_width, 2))
#
#     predicted = apply_xy_output_func(data_flat, par, False).eval()
#
#     expected = np.zeros((output_width, 2))
#     for x in range(n_gridcells_horizontal):
#         for y in range(n_gridcells_vertical):
#             vector_start=(y*n_gridcells_horizontal+x)*n_ab
#             expected[vector_start:vector_start+n_ab]+=[x+.5, y+.5]
#     expected[:,0]/=n_gridcells_horizontal
#     expected[:, 1] /= n_gridcells_vertical
#
#     assert np.allclose(predicted, expected)
#
#     #Testing high value --> lower, right end
#     data_flat = np.ones((output_width,2))*1e9
#     predicted = apply_xy_output_func(data_flat, par, False).eval()
#
#     expected[:,0]+=.5/n_gridcells_horizontal
#     expected[:,1]+=.5/n_gridcells_vertical
#     assert np.allclose(predicted, expected)
#
#     # Testing low value --> upper, left end
#     data_flat = np.ones((output_width, 2)) * -1e9
#     predicted = apply_xy_output_func(data_flat, par, False).eval()
#
#     expected[:, 0] -= 1 / n_gridcells_horizontal
#     expected[:, 1] -= 1 / n_gridcells_vertical
#     assert np.allclose(predicted, expected)
#
#
# def test_apply_wh_output_func():
#     """
#     Test for apply_wh_output_func()
#     :return: Nothing
#     """
#     assert False, "Not implemented yet"
#
# def test_apply_obj_output_func():
#     """
#     Test for apply_obj_output_func()
#     :return: Nothing
#     """
#     assert False, "Not implemented yet"
#
# def test_apply_cls_output_func():
#     """
#     Test for apply_cls_output_func()
#     :return: Nothing
#     """
#     assert False, "Not implemented yet"
