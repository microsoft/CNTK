# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
from cntk import load_model, graph
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk import graph


# helper to print all node names
def print_all_node_names(model_file, is_BrainScript=True):
    loaded_model = load_model(model_file)
    if is_BrainScript:
        loaded_model = combine([loaded_model.outputs[0]])
    node_list = graph.depth_first_search(loaded_model, lambda x: isinstance(x, Function))
    print("printing node information in the format")
    for node in node_list: 
        print("Node name:", node.name)
        for out in node.outputs: 
            print("Output name and shape:", out.name, out.shape)


def create_mb_source(image_height, image_width, num_channels, map_file):
    transforms = [ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),  # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=1000))),             # and second as 'label'. TODO: add option to ignore labels
        randomize=False)


def eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects):
    # load model and pick desired node as output
    loaded_model  = load_model(model_file)
    node_in_graph = loaded_model.find_by_name(node_name)
    output_nodes  = combine([node_in_graph.owner])

    # evaluate model and get desired node output
    print("Evaluating model for output node %s" % node_name)
    features_si = minibatch_source['features']
    with open(output_file, 'wb') as results_file:
        for i in range(0, num_objects):
            mb = minibatch_source.next_minibatch(1)
            output = output_nodes.eval(mb[features_si])

            # write results to file
            out_values = output[0,0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")

if __name__ == '__main__':
    # define location of model and data and check existence
    base_folder = os.path.dirname(os.path.abspath(__file__))
    model_file  = os.path.join(base_folder, "..", "PretrainedModels", "ResNet_18.model")
    map_file    = os.path.join(base_folder, "..", "DataSets", "grocery", "test.txt")
    os.chdir(os.path.join(base_folder, "..", "DataSets", "grocery"))
    if not (os.path.exists(model_file) and os.path.exists(map_file)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # create minibatch source
    image_height = 224
    image_width  = 224
    num_channels = 3
    minibatch_source = create_mb_source(image_height, image_width, num_channels, map_file)

    # use this to print all node names of the model (and knowledge of the model to pick the correct one)
    # print_all_node_names(model_file)

    # use this to get 1000 class predictions (not yet softmaxed!)
    # node_name = "z"
    # output_file = os.path.join(base_folder, "predOutput.txt")

    # use this to get 512 features from the last but one layer of ResNet_18
    node_name = "z.x"
    output_file = os.path.join(base_folder, "layerOutput.txt")

    # evaluate model and write out the desired layer output
    eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects=5)

    print("Done. Wrote output to %s" % output_file)
