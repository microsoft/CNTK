# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
import cntk as C
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs

def create_mb_source(image_height, image_width, num_channels, map_file):
    transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(map_file, C.io.StreamDefs(
            features=C.io.StreamDef(field='image', transforms=transforms),
            labels=C.io.StreamDef(field='label', shape=1000))),
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
            out_values = output[0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")

if __name__ == '__main__':
    # define location of model and data and check existence
    base_folder = os.path.dirname(os.path.abspath(__file__))
    model_file  = os.path.join(base_folder, "..", "..", "..", "PretrainedModels", "ResNet18_ImageNet_CNTK.model")
    map_file    = os.path.join(base_folder, "..", "DataSets", "Grocery", "test.txt")
    os.chdir(os.path.join(base_folder, "..", "DataSets", "Grocery"))
    if not (os.path.exists(model_file) and os.path.exists(map_file)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # create minibatch source
    image_height = 224
    image_width  = 224
    num_channels = 3
    minibatch_source = create_mb_source(image_height, image_width, num_channels, map_file)

    # use this to print all node names of the model (and knowledge of the model to pick the correct one)
    # node_outputs = get_node_outputs(load_model(model_file))
    # for out in node_outputs: print("{0} {1}".format(out.name, out.shape))

    # use this to get 1000 class predictions (not yet softmaxed!)
    # node_name = "z"
    # output_file = os.path.join(base_folder, "predOutput.txt")

    # use this to get 512 features from the last but one layer of ResNet_18
    node_name = "z.x"
    output_file = os.path.join(base_folder, "layerOutput.txt")

    # evaluate model and write out the desired layer output
    eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects=5)

    print("Done. Wrote output to %s" % output_file)
