# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os.path

# Set to the DataSet you want to train
par_dataset_name = "Pascal_VOC_2007"


par_trainset_label_file = "train_map.txt"
par_testset_label_file = "test_map.txt"
par_abs_path = os.path.dirname(os.path.abspath(__file__))
par_max_epochs = 1
par_downsample = 32
par_max_gtbs = 50

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
    par_minibatch_size = 64
    par_image_width = 224
    par_image_height = 224
    par_input_bias = 114
    par_num_channels = 3
    par_num_classes = 20
    par_num_images = 500
    par_epoch_size = 5

par_anchorbox_scales =  [[1.08/13, 1.19/13],
                         [3.42/13, 4.41/13],
                         [6.63/13, 11.38/13],
                         [9.42/13, 5.11/13],
                         [16.62/13, 10.52/13]]
par_num_anchorboxes = len(par_anchorbox_scales)


