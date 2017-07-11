# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import cntk as C
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imresize

################################################
################################################
# base model settings
base_folder = os.path.dirname(os.path.abspath(__file__))
_base_model_file = os.path.join(base_folder, "..", "TransferLearning", "Output", "TransferLearning.model")
_last_hidden_node_name = "z.x"
_num_classes = 2
_image_height = 224
_image_width = 224

# Class activation maps configuration
_normalize = True  # If maps should be normalized
_max_classes = 5 # Maximum number of classes

# define data location and characteristics
_data_folder = os.path.join(base_folder, "..", "DataSets", "Animals")
_class_maps = ["Sheep", "Wolf"]
################################################
################################################


def configure_nodes(model):
    # Find feature node by its name (this is has to be a average pooling layer)
    feature_node = C.logging.find_by_name(model, _last_hidden_node_name)
    # Selecing the output node (assuming a single fc layer after GAP)
    output_node = model.outputs[0].owner
    for param in output_node.parameters:
        if param.name == 'W':
            weights = param.asarray()
    # We´ll need the output of the layer before the average pooling layer
    feature_node = feature_node.inputs[0].owner
    # Generating the a new model with two outputs: the original output from the dense layer
    # and the feature maps used to compute the class activation maps.
    combined_model = C.combine([feature_node, output_node])

    return combined_model, weights, feature_node.output, output_node.output


def compute_cams(feature, prediction, weights):
    # Selecting (at most) N most probable outputs
    prediction = prediction.squeeze()
    top_classes = prediction.argsort()[-1:-_max_classes:-1]
    # Computing maps
    activation_maps = []
    for class_id in top_classes:
        weighted_feat = np.zeros_like(feature)
        for row in range(feature.shape[2]):
            for col in range(feature.shape[3]):
                weighted_feat[0, :, row, col] = feature[0, :, row, col] \
                                                * weights[:, 0, 0, class_id]
        out_map = np.squeeze(np.sum(weighted_feat, axis=1))
        activation_maps.append(out_map)
    if _normalize:
        min_value = min(list(map(np.min, activation_maps)))
        max_value = max(list(map(np.max, activation_maps)))
        # Normalizing maps to [0, 1] interval
        activation_maps = [(activation_map - min_value) / (max_value - min_value)
                            for activation_map in activation_maps]

    return activation_maps, top_classes


def plot_cams(activation_maps, image, selected_classes):

    # Upsampling maps
    reshaped_maps = [imresize(activation_map, size=(_image_height, _image_width), mode='F')
                        for activation_map in activation_maps]
    # Showing each map
    for class_id, activation_map in zip(selected_classes, reshaped_maps):
        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(activation_map, vmin=0, vmax=1, alpha=0.5)
        plt.title(_class_maps[class_id])
    plt.show()


def load_image(image_path):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    img = Image.open(image_path)
    if image_path.endswith("png"):
        temp = Image.new("RGB", img.size, (255, 255, 255))
        temp.paste(img, img)
        img = temp
    resized = img.resize((_image_height, _image_height), Image.ANTIALIAS)
    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    return hwc_format, resized


# Evaluates a single image and plots its activation map
def eval_and_plot_map(base_model, image_path):

    model, weights, feature_node, pred_node = configure_nodes(base_model)
    # Load image
    image_hwc, image_rgb = load_image(image_path)
    # Run network
    arguments = {model.arguments[0]: [image_hwc]}
    output = model.eval(arguments)
    # Parsing output
    features = output[feature_node]
    predictions = output[pred_node]
    # Compute class activation maps
    maps, classes = compute_cams(features, predictions, weights=weights)
    # Showing maps
    plot_cams(maps, image_rgb, classes)


if __name__ == '__main__':
    C.try_set_default_device(C.gpu(0))
    # check for model and data existence
    if not (os.path.exists(_base_model_file)):
        print("Please run TransferLearning_ext.py example script.")
        exit(0)
    image_path = os.path.join(_data_folder, 'Test', 'Wolf', 'Canis_lupus_occidentalis.jpg')
    # Loading model and checking number of outputs
    model = C.load_model(_base_model_file)
    if model.output.shape[0] != _num_classes:
        print("Please run TransferLearning_ext.py example script.")
        exit(0)

    eval_and_plot_map(model, image_path)