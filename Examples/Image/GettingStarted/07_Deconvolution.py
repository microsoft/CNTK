# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP

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
    print("loading model...")
    loaded_model = load_model(model_file)
    print("walking tree...")
    dfs_walk(loaded_model, set())
#####################################################
#####################################################

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "DataSets", "MNIST")
model_path = os.path.join(abs_path, "Output", "Models")


# Define the reader for both training and evaluation action.
def create_reader(path, is_training, input_dim, label_dim):
    return cntk.io.MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='features', shape=input_dim),
        labels    = StreamDef(field='labels',   shape=label_dim)
    )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)


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
    #import pdb
    #pdb.set_trace()

    # define location of model and data and check existence
    model_file  = os.path.join(model_path, "07_Deconvolution.model")
    loaded_model = load_model(model_file)

    data_file    = os.path.join(data_path, "Dummy.txt")
    #os.chdir(os.path.join(base_folder, "..", "DataSets", "grocery"))
    if not (os.path.exists(model_file) and os.path.exists(data_file)):
        print("Cannot find required data or model.")
        exit(0)

    # use this to print all node names of the model (and knowledge of the model to pick the correct one)
    print("Printing node names...")
    print_all_node_names(model_file)

    # create minibatch source
    image_height = 28
    image_width  = 28
    num_channels = 1
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10
    minibatch_source = create_reader(data_file, False, input_dim, num_output_classes)

    # use this to get 28 * 28 output features from the decoder
    node_name = "z.x._._"
    output_file = os.path.join(abs_path, "Output", "decoderOutput.txt")

    # evaluate model and write out the desired layer output
    eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects=3)

    print("Done. Wrote output to %s" % output_file)
