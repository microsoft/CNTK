# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs

#####################################################
#####################################################
# helpers to print all node names
def dfs_walk(node, visited):
    if node in visited:
        return
    visited.add(node)
    print("visiting %s"%node.name)
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visited)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visited)

def print_all_node_names(model_file):
    loaded_model = load_model(model_file)
    dfs_walk(loaded_model, set())
#####################################################
#####################################################


def create_mb_source(image_height, image_width, num_channels, map_file):
    transforms = [ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),  # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=1000))))             # and second as 'label'. TODO: add option to ignore labels


def eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects):
    # load model and pick desired node as output
    loaded_model  = load_model(model_file)
    node_in_graph = loaded_model.find_by_name(node_name)
    output_nodes  = combine([node_in_graph.owner])

    # evaluate model and get desired node output
    features_si = minibatch_source['features']
    with open(output_file, 'wb') as results_file:
        for i in range(0, num_objects):
            mb = minibatch_source.next_minibatch(1)
            output = output_nodes.eval(mb[features_si])

            # write results to file
            out_values = output[0,0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")
    return

if __name__ == '__main__':
    # define location of model and data
    base_folder = os.path.join(os.getcwd(), "..")
    model_file  = os.path.join(base_folder, "PretrainedModels/AlexNetBS.model")
    map_file    = os.path.join(base_folder, "DataSets/grocery/test.txt")

    # create minibatch source
    image_height = 227
    image_width  = 227
    num_channels = 3
    minibatch_source = create_mb_source(image_height, image_width, num_channels, map_file)

    # use this to print all node names of the model (and knowledge of the model to pick the correct one)
    # print_all_node_names(model_file)

    # use t his to get 1000 class predictions (not yet softmaxed!)
    # node_name = "z"
    # out_file_name = "predOutput.txt"

    # use this to get 4096 features from the last fc layer
    node_name = "z.x._._"
    out_file_name = "fcOutput.txt"

    # evaluate model and write out the desired layer output
    os.chdir(os.path.join(base_folder, "DataSets/grocery/"))
    output_file = os.path.join(base_folder, out_file_name)
    eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects=5)

    print("Done. Wrote output to %s" % output_file)
