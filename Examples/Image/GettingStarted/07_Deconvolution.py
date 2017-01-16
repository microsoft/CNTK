# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, FULL_DATA_SWEEP
from PIL import Image


# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "..", "DataSets", "MNIST")
model_path = os.path.join(abs_path, "Output", "Models")


# Helper to save array as grayscale image
def save_as_png(val_array, img_file_name):
    img_array = val_array.reshape((28, 28))
    img_array = np.clip(img_array, 0, img_array.max())
    img_array *= 255.0 / img_array.max()
    img_array = np.rint(img_array).astype('uint8')

    im = Image.fromarray(img_array)
    im.save(img_file_name)


if __name__ == '__main__':
    num_objects_to_eval = 5

    # define location of output, model and data and check existence
    output_file = os.path.join(abs_path, "Output", "imageAutoEncoder.txt")
    model_file = os.path.join(model_path, "07_Deconvolution.model")
    data_file = os.path.join(data_path, "Test-28x28_cntk_text.txt")
    if not (os.path.exists(model_file) and os.path.exists(data_file)):
        print("Cannot find required data or model. "
              "Please get the MNIST data set and run 'cntk configFile=07_Deconvolution.cnkt' to create the model.")
        exit(0)

    # create minibatch source
    minibatch_source = MinibatchSource(CTFDeserializer(data_file, StreamDefs(
        features  = StreamDef(field='features', shape=(28*28)),
        labels    = StreamDef(field='labels',   shape=10)
    )), randomize=False, epoch_size = FULL_DATA_SWEEP)

    # load model and pick desired nodes as output
    loaded_model = load_model(model_file)
    output_nodes = combine([loaded_model.find_by_name('f1').owner, loaded_model.find_by_name('z').owner])

    # evaluate model save output
    features_si = minibatch_source['features']
    with open(output_file, 'wb') as results_file:
        for i in range(0, num_objects_to_eval):
            mb = minibatch_source.next_minibatch(1)
            raw_dict = output_nodes.eval(mb[features_si])
            output_dict = {}
            for key in raw_dict.keys(): output_dict[key.name] = raw_dict[key]

            decoder_input = output_dict['f1']
            decoder_output = output_dict['z']
            in_values = (decoder_input[0,0].flatten())[np.newaxis]
            out_values = (decoder_output[0,0].flatten())[np.newaxis]

            # write results as text and png
            np.savetxt(results_file, out_values, fmt="%.6f")
            save_as_png(in_values,  "%s_%s_input.png"  % (output_file, i))
            save_as_png(out_values, "%s_%s_output.png" % (output_file, i))

    print("Done. Wrote output to %s" % output_file)
