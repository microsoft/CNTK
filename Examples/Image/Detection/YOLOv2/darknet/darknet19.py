# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from cntk import leaky_relu, reshape, softmax
from cntk.layers import Convolution2D,BatchNormalization, MaxPooling, GlobalAveragePooling, Sequential, Activation, \
    default_options

from darknet.Utils import *
import os


# Creates the feature extractor shared by the classifier (Darknet19) and the Detector (YOLOv2)
def create_feature_extractor(filter_multiplier=32):
    with default_options(activation=leaky_relu):
        net = Sequential([
            Convolution2D(filter_shape=(3,3), num_filters=filter_multiplier, pad=True, name="feature_layer"),
            BatchNormalization(),
            MaxPooling(filter_shape=(2,2), strides=(2,2)),
            # Output: in_x/2 x in_y/2 x nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**1), pad=True, name="stage_1"),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output: in_x/4 x in_y/4 x 2*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**2), pad=True, name="stage_2"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**1), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**2), pad=True),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output in_x/8 x in_y/8 x 4*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**3), pad=True, name="stage_3"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**2), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**3), pad=True),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output in_x/16 x in_y/16 x 8*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**4), pad=True, name="stage_4"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**3), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**4), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**3), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**4), pad=True, name="YOLOv2PasstroughSource"),
            BatchNormalization(),
            MaxPooling(filter_shape=(2, 2), strides=(2, 2)),
            # Output in_x/32 x in_y/32 x 16*nfilters


            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**5), pad=True, name="stage_5"),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**4), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**5), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(1, 1), num_filters=(filter_multiplier * 2**4), pad=True),
            BatchNormalization(),
            Convolution2D(filter_shape=(3, 3), num_filters=(filter_multiplier * 2**5), pad=True),
            BatchNormalization(name="featureExtractor_output")
            # Output in_x/32 x in_y/32 x 32*nfilters
        ],'featureExtractor_darknet19')

    return net


# Puts a classifier end to any feature extractor
def put_classifier_on_feature_extractor(featureExtractor,nrOfClasses):
    return Sequential([
        reshape(x=Sequential([
            # [lambda x: x - 114],
            featureExtractor,
            Convolution2D(filter_shape=(1, 1), num_filters=nrOfClasses, pad=True, activation=identity,
                          name="classifier_input"),
            GlobalAveragePooling()
        ]), shape=(nrOfClasses)),
        Activation(activation=softmax, name="classifier_output")
    ], name="darknet19-classifier")


# Creates a Darknet19 classifier
def create_classification_model_darknet19(nrOfClasses, filter_mult=32):
    featureExtractor = create_feature_extractor(filter_mult)
    return put_classifier_on_feature_extractor(featureExtractor, nrOfClasses)


# Saves a model to the Output folder. If the models are already existing an ascending number is assigned to the model.
def save_model(model, name="darknet19"):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, "Output", name + ".model")
    if os.path.exists(model_path):
        i = 1
        while (os.path.exists(model_path)):
            i += 1
            model_path = os.path.join(abs_path, "Output", name + "_" + str(i) + ".model")

    model.save(model_path)
    print("Stored model " + name + " to " + model_path)
    return model_path


########################################################################################################################
#   Main                                                                                                               #
########################################################################################################################

if __name__ == '__main__':
    from cntk.cntk_py import force_deterministic_algorithms
    force_deterministic_algorithms()

    data_path = par_data_path # from PARAMETERS

    # create
    model = create_classification_model_darknet19(num_classes) # num_classes from Utils
    #  and normalizes the input features by subtracting 114 and dividing by 256
    model2 = Sequential([[lambda x: (x - par_input_bias)], [lambda x: (x / 256)] , model])
    print("Created Model!")

    # train
    reader = create_reader(os.path.join(data_path, par_trainset_label_file),  is_training=True)
    print("Created Readers!")

    train_model(reader, model2, max_epochs=par_max_epochs, exponentShift=-1)
    # save
    save_model(model, "darknet19_" + par_dataset_name)

    from cntk.logging.graph import plot
    plot(model, filename=os.path.join(par_abs_path, "darknet19_" + par_dataset_name + "_DataAug.pdf"))

    # test
    reader = create_reader(os.path.join(data_path, par_testset_label_file), is_training=False)
    evaluate_model(reader, model2)

    print("Done!")

