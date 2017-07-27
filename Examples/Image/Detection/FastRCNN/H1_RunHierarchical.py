# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys, os
import cntk as C
from cntk import cntk_py, reduce_sum, ops, user_function, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule, momentum_sgd, Trainer
from cntk.logging import ProgressPrinter, log_number_of_parameters
from A2_RunWithPyModel import create_mb_source, p, image_height, image_width, num_channels, num_classes, num_rois, base_path, frcn_predictor, momentum_time_constant, max_epochs, epoch_size, mb_size, model_file
from htree_helper import get_tree_str

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.hierarchical_classification.hierarchical_classification_helper import HierarchyHelper, Target_Creator

cntk_py.force_deterministic_algorithms()

USE_HIERARCHICAL_CLASSIFICATION = True
HCH = HierarchyHelper(get_tree_str(p.datasetName, USE_HIERARCHICAL_CLASSIFICATION))


# Trains a Fast R-CNN model
def train_fast_rcnn_h(debug_output=False, model_path=model_file):
    if debug_output:
        print("Storing graphs and intermediate models to %s." % os.path.join(abs_path, "Output"))

    # Create the minibatch source
    minibatch_source = create_mb_source(image_height, image_width, num_channels,
                                        num_classes, num_rois, base_path, "train")

    # Input variables denoting features, rois and label data
    image_input = C.input_variable((num_channels, image_height, image_width))
    roi_input = C.input_variable((num_rois, 4))
    label_input = C.input_variable((num_rois, num_classes))

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source.streams.features,
        roi_input: minibatch_source.streams.rois,
        label_input: minibatch_source.streams.roiLabels
    }

    # Instantiate the Fast R-CNN prediction model and loss function
    def cross_entropy(output, target):
        return -reduce_sum(target * ops.log(output))

    num_neurons = HCH.tree_map.get_nr_of_required_neurons()
    target = user_function(Target_Creator(label_input, p.cntk_nrRois, HCH))
    frcn_output = frcn_predictor(image_input, roi_input, num_neurons, model_path)
    softmaxed = HCH.apply_softmax(frcn_output, axis=1, offset=0)

    ce = cross_entropy(softmaxed, target)
    error = softmaxed - target
    pe = reduce_sum(error * error)
    frcn_output = softmaxed

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_multiplier = .6
    lr_per_sample = [0.00001 * lr_multiplier] * 20 + [0.000001 * lr_multiplier] * 5 + [0.0000001 * lr_multiplier]
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object
    learner = momentum_sgd(frcn_output.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs, gen_heartbeat=True, freq=50)
    trainer = Trainer(frcn_output, (ce, pe), learner, progress_printer)

    # Get minibatches of images and perform model training
    print("Training Fast R-CNN model for %s epochs." % max_epochs)
    log_number_of_parameters(frcn_output)
    for epoch in range(max_epochs):  # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size - sample_count), input_map=input_map)
            trainer.train_minibatch(data)  # update model with it
            sample_count += trainer.previous_minibatch_sample_count  # count samples processed so far

        trainer.summarize_training_progress()
        if debug_output:
            frcn_output.save(os.path.join(abs_path, "Output", "hfrcn_py_%s.model" % (epoch + 1)))

    return frcn_output


def create_and_save_model(model_file):
    trained_model = train_fast_rcnn_h()
    trained_model.save(model_file)
    return trained_model


if __name__ == '__main__':
    os.chdir(base_path)
    model_path = os.path.join(abs_path, "Output", p.datasetName + "_hfrcn_py.model")

    # Train only if no model exists yet
    if os.path.exists(model_path):
        print("Model was already trained - model file: " + model_path)
    else:
        create_and_save_model(model_path)
        print("Stored trained model at %s" % model_path)
