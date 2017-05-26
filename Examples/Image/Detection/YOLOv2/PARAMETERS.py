# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os.path

# Set to the DataSet you want to train
par_dataset_name = "Pascal_VOC_2007"
#par_dataset_name = "Logos158"


par_trainset_label_file = "train_map.txt"
par_testset_label_file = "test_map.txt"
par_abs_path = os.path.dirname(os.path.abspath(__file__))
par_max_epochs = 100
#par_dataset_name = "Grocery"

_par_lr_dataset_adoption = 1
par_downsample = 32


par_dense_size = 1024

if(par_dataset_name == "CIFAR10"):
    par_image_height = 416 # 32*3  # Darknet19 scales input image down over all by a factor of 32. \\
    par_image_width = 416 # 32*3   # It needs at least a 3x3 shape for the last conv layer. So 32*3 is required at least.
    par_num_channels = 3  # RGB
    par_input_bias = 114
    par_num_classes = 10
    par_data_path = os.path.join(par_abs_path, "..", "..", "DataSets", "CIFAR-10")

    par_minibatch_size = 24


elif(par_dataset_name == "ImageNet50k"):
    par_image_height = 416  # Darknet19 scales input image down over all by a factor of 32. \\
    par_image_width = 416   # It needs at least a 3x3 shape for the last conv layer. So 32*3 is required at least.
    par_num_channels = 3 # RGB
    par_input_bias = 114
    par_num_classes = 1000
    par_data_path = "C:\\Data\\private\\Image\\ResNet\\Data\\v0" #local path for test

    par_minibatch_size = 24

elif(par_dataset_name == "Pascal_VOC_2007"):
    #par_image_width = 288       # width the image is scaled to
    #par_image_height = 224      # height the image is scaled to
    par_minibatch_size = 32# 16    # minibatch size
    par_image_width = 416       # width the image is scaled to
    par_image_height = 416      # height the image is scaled to
    par_input_bias = 114        # average input value
    par_num_channels = 3        # nr of color-channels of the input
    par_num_classes = 20        # nr of classes displayed
    par_epoch_size = 5011       # nr of input images
    par_max_gtbs = 50
    par_boxes_centered = True
    par_train_data_file = 'trainval2007.txt'
    par_train_roi_file = 'trainval2007_rois_center_rel.txt'


elif(par_dataset_name == "Logos158"):
    par_minibatch_size = 32    # minibatch size
    par_image_width = 416       # width the image is scaled to
    par_image_height = 416      # height the image is scaled to
    par_input_bias = 114        # average input value
    par_num_channels = 3        # nr of color-channels of the input
    par_num_classes = 158        # nr of classes displayed
    par_epoch_size = 4278       # nr of input images
    par_max_gtbs = 52
    par_boxes_centered = True
    par_train_data_file = 'logo_v5_img.txt'
    par_train_roi_file = 'logo_v5_gts.txt'

elif(par_dataset_name == "Grocery"):
    par_minibatch_size = 20  # minibatch size
    par_image_width = 288  # width the image is scaled to
    par_image_height = 512  # height the image is scaled to
    par_input_bias = 114  # average input value
    par_num_channels = 3  # nr of color-channels of the input
    par_num_classes = 16  # nr of classes displayed
    par_epoch_size = 20  # nr of input images
    par_max_gtbs = 50
    par_boxes_centered = False
    par_train_data_file = 'train.txt'
    par_train_roi_file = 'train.GTRois.txt'
    par_max_epochs=1250
    _par_lr_dataset_adoption = 2

# Priors from k-means
par_anchorbox_scales = [[ 0.09635106,  0.14264049],
 [ 0.35856731,  0.73630027],
 [ 0.82745373,  0.84205688],
 [ 0.2346392,   0.37655989],
 [ 0.69155048,  0.44537981]]

# Priors from other sources
par_anchorbox_scales_old =[[1.08/13, 1.19/13], # priors [width, height] for the box predictions
                         [3.42/13, 4.41/13],
                         [6.63/13, 11.38/13],
                         [9.42/13, 5.11/13],
                         [16.62/13, 10.52/13]]
par_num_anchorboxes = len(par_anchorbox_scales)

par_lambda_coord = 1
par_lambda_obj = 5
par_lambda_no_obj = 1
par_lambda_cls = 1
par_objectness_threshold=0.6
par_box_default_mbs = int(100 * 128/par_minibatch_size)
par_scale_default_boxes = 0.01

# apply custom learning rate here
par_base_lr = 0.001 *_par_lr_dataset_adoption
par_lr_schedule = [par_base_lr*.1] * 10+[par_base_lr] * 60+ [par_base_lr*0.1] * 30 + [par_base_lr*0.01]*60 + [par_base_lr*0.001]
par_lr_schedule = [par_base_lr*.1] * 10+[par_base_lr] * 600+ [par_base_lr*0.1] * 300 + [par_base_lr*0.01]*600 + [par_base_lr*0.001]
par_momentum = 0.9
par_weight_decay = 0.0005
