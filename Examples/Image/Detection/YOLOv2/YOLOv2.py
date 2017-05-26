# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
# import cntk
from cntk import *
from cntk.io import StreamDefs, StreamDef
from cntk import leaky_relu, reshape, softmax, param_relu, relu, user_function
from cntk.layers import Convolution2D, BatchNormalization, MaxPooling, GlobalAveragePooling, Sequential, Activation, \
    default_options
from cntk.logging import ProgressPrinter
from cntk.logging.graph import find_by_name, plot, get_node_outputs
from cntk.io import ImageDeserializer, CTFDeserializer, MinibatchSource, TraceLevel
import cntk.io.transforms as xforms
import PARAMETERS


def create_output_activation_layer(network, par):
    anchor_box_scales=par.par_anchorbox_scales
    n_gridcells_horizontal = int(par.par_image_width / par.par_downsample)
    n_gridcells_vertical =  int(par.par_image_height / par.par_downsample)
    n_anchorboxes=len(anchor_box_scales)
    output_width = n_gridcells_horizontal * n_gridcells_vertical * n_anchorboxes
    output_height =  par.par_num_classes + 5

    # reorder array! now 125*7*7
    # tp1 = ops.transpose(network, 0,2) # 7*7*125
    # tp2 = ops.transpose(tp1, 0, 1) # 7*7*125
    tp = ops.transpose(network, (1,2,0))
    reshaped = ops.reshape(tp, (output_width, output_height))
    #shape is now 245 * 25


    # slicing the array into subarrays
    xy_cords = ops.slice(reshaped, axis=1, begin_index=0, end_index=2, name="XY-Out")
    wh_mults = ops.slice(reshaped, axis=1, begin_index=2, end_index=4, name="WH-Out")
    objectnesses =  ops.slice(reshaped, axis=1, begin_index=4, end_index=5, name="Obj_Out")
    cls_outs = ops.slice(reshaped, axis=1, begin_index=5, end_index=output_height, name="Cls_Out")

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

    # Splice the parts back together!
    full_output = ops.splice(xy_preds, wh_preds, obj_preds, cls_preds, axis=1, name="yolo_results")
    return full_output


def create_detector(par, batchnormalization = False):
    output_depth = (5+par.par_num_classes)*par.par_num_anchorboxes
    net = Sequential([
        Convolution2D((3, 3), num_filters=par.par_dense_size, init=he_normal(), activation=lambda x: 0.1*x + 0.9*relu(x), pad=True),
        BatchNormalization(),
        Convolution2D((3, 3), num_filters=par.par_dense_size, init=he_normal(), activation=lambda x: 0.1*x + 0.9*relu(x), pad=True),
        BatchNormalization(),
        Convolution2D((3, 3), num_filters=par.par_dense_size, init=he_normal(), activation=lambda x: 0.1*x + 0.9*relu(x), pad=True),
        #BatchNormalization(), #C-impl says no!

        Convolution2D((1, 1), num_filters=output_depth),
    ], "YOLOv2_Detector")

    net2 = create_output_activation_layer(net, par)
    return net2


def load_pretrained_darknet_feature_extractor():
    loaded_model = load_model(".\Output\darknet19_CIFAR10_7.model")

    feature_layer = find_by_name(loaded_model, "feature_layer")
    fe_output_layer = find_by_name(loaded_model, "featureExtractor_output")

    return combine([fe_output_layer.owner]).clone(CloneMethod.clone, {feature_layer: placeholder()})

def load_pretrained_resnet18_feature_extractor(par):
    loaded_model = load_model(os.path.join(par.par_abs_path, "..", "..", "PretrainedModels", "ResNet18_ImageNet_CNTK.model"))


    feature_layer = find_by_name(loaded_model, "features")
    fe_output_layer = find_by_name(loaded_model, "z.x.x.r")
    ph = placeholder(shape=(par.par_num_channels, par.par_image_width, par.par_image_height), name="input_ph")
    net = combine([fe_output_layer.owner]).clone(CloneMethod.clone, {feature_layer: ph})

    #plot(net, "ResNet18_s.pdf")

    return net

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
    elif(use_model == "pre_ResNet101_ImageNet"):
        fe = load_pretrained_resnet101_feature_extractor(par)
    elif(use_model == "pre_ResNet18_ImageNet"):
        fe = load_pretrained_resnet18_feature_extractor(par)
    return fe


def create_yolov2_net(par):
    return create_detector(par)(create_feature_extractor(par))


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

def create_mb_source(img_height, img_width, img_channels, output_size, image_file, roi_file, randomize = False, multithreaded_deserializer=False, max_samples=io.INFINITELY_REPEAT, max_epochs = io.INFINITELY_REPEAT):
    transforms = []
    transforms += [
        xforms.scale(width=img_width, height=img_height, channels=img_channels, interpolations='linear',
                     scale_mode='fill'),
    ]

    image_source = ImageDeserializer(image_file, StreamDefs(features=StreamDef(field='image', transforms=transforms)))


    # read rois and labels
    roi_source = CTFDeserializer(roi_file, StreamDefs(label=StreamDef(field='rois', shape=output_size)))

    rc = MinibatchSource([image_source, roi_source], randomize=randomize, trace_level=TraceLevel.Error, multithreaded_deserializer=multithreaded_deserializer, max_samples=max_samples)#, max_epochs=max_epochs)
    return rc



if __name__ == '__main__':
    import PARAMETERS as par
    model = create_yolov2_net(par)
    print("created model!")
    #model = load_pretrained_resnet101_feature_extractor()
    # plot
    print("beep")
    plot(model, filename=os.path.join(par.par_abs_path, "YOLOv2.pdf"))
    print("buup")

    image_input= input((par.par_num_channels, par.par_image_height, par.par_image_width))
    output = model(image_input) # append model to image input
    plot(output, filename=os.path.join(par.par_abs_path, "YOLOv2_in.pdf"))
    """
    # input for ground truth boxes
    num_gtb = par.par_max_gtbs
    gtb_input = input((num_gtb*5)) # 5 for  x,y,w,h,class

    from ErrorFunction import get_error
    mse = get_error(output, gtb_input, cntk_only=False)

    plot(mse, filename=os.path.join(par.par_abs_path,"Trainnet.pdf"))


    # trainig params
    max_epochs = par.par_max_epochs
    epoch_size = par.par_epoch_size
    minibatch_size = par.par_minibatch_size

    lr_schedule = learning_rate_schedule(par.par_lr_schedule, learners.UnitType.sample, epoch_size)
    mm_schedule = learners.momentum_schedule([-minibatch_size / np.log(par.par_momentum)], epoch_size)

    # Instantiate the trainer object to drive the model training
    learner = learners.momentum_sgd(mse.parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=0.0005)
    trainer = Trainer(None, (mse, mse), [learner])

    image_file = os.path.join(par.par_abs_path, "..", "..", "DataSets", "Pascal", "mappings" , "trainval2007.txt")
    roi_file = os.path.join(par.par_abs_path, "..", "..", "DataSets", "Pascal", "mappings" , "trainval2007_rois_center_rel.txt")

    minibatch_source = create_mb_source(par.par_image_height, par.par_image_width, par.par_num_channels, (5 * num_gtb), image_file, roi_file)

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source["features"],
        gtb_input: minibatch_source["label"]
    }

    progress_printer = ProgressPrinter(freq= int(epoch_size / 10), tag='Training', rank=Communicator.rank(), gen_heartbeat=True,
                                       num_epochs=max_epochs)


    for epoch in range(max_epochs):  # loop over epochs
        print("---Start new epoch---")
        sample_count = 2
        while sample_count < epoch_size :#- minibatch_size:  # loop over minibatches in the epoch

            # get next minibatch data
            data = minibatch_source.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                                   input_map=input_map)  # fetch minibatch.
            gtbs = (data[gtb_input]).asarray()

            trainer.train_minibatch({image_input: data[image_input].asarray(), gtb_input:gtbs}, device=gpu(0))  # update model with it
            sample_count += data[image_input].num_samples  # count samples processed so far
            progress_printer.update_with_trainer(trainer=trainer, with_metric=True)  # log progress
            print(progress_printer.avg_metric_since_start())

        progress_printer.epoch_summary(with_metric=True)

    save_path = os.path.join(par.par_abs_path,  "outputdir")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output.save_model(os.path.join(save_path, "YOLOv2.model"))
    """
    # alternatively to training a model you can reload a pretrained!
    model = load_model(r".\outputdir\YOLOv2.model")
    #image_input = input((par.par_num_channels, par.par_image_height, par_image_width))

    plot(model, filename=os.path.join(par_abs_path, "YOLOv2.pdf"))

    import ipdb;ipdb.set_trace()
    output = model#(image_input)  # append model to image input

    #predict a few images
    names = ["000018.jpg","000118.jpg","001118.jpg","002118.jpg"]
    threshold_objectness = 0.505
    for i in range(len(names)):
        image = os.path.join(par.par_abs_path, "..", "..", "DataSets", "Pascal", "VOCdevkit" , "VOC2007", "JPEGImages" ,names[i])
        predict_image(output, image, conf_threshold=threshold_objectness)

    print("Done!")

    import ipdb;ipdb.set_trace()
    predict_image(output,r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages\000001.jpg",0.501)
