# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
import sys
import time
import argparse
from TransferLearning import (create_mb_source, create_model,
                              eval_single_image,
                              mb_size, lr_per_mb, momentum_per_mb,
                              l2_reg_weight,
                              features_stream_name, label_stream_name)
from cntk import distributed
from PIL import Image
from cntk.device import try_set_default_device, gpu
from cntk import load_model, placeholder, Constant
from cntk import Trainer, UnitType
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk.layers import Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.ops import input, combine, softmax
from cntk.ops.functions import CloneMethod
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import log_number_of_parameters, ProgressPrinter


# Define base model location and characteristics.
base_folder = os.path.dirname(os.path.abspath(__file__))
tl_model_file = os.path.join(
    base_folder, "..", "PretrainedModels", "ResNet_18.model")
now = time.strftime("%F_%H-%M-%S")
new_model_file = os.path.join(base_folder,
                              "Output",
                              "TransferLearning_{0}.model".format(now))
feature_node_name = "features"
last_hidden_node_name = "z.x"
image_height = 224
image_width = 224
num_channels = 3

# Overwrite the default. Comment it out to use the default defined in
# TransferLearning.py.
lr_per_mb = [0.01]


def train_model(base_model_file, num_classes, train_map_file, num_epochs,
                batch_size=mb_size, max_images=-1, distributed_after=-1,
                freeze=False, VERBOSE=True):
    """Modified version of the TransferLearning function, made distributed by
    following these instructions: https://git.io/v9bYw
    """

    with open(train_map_file) as train_file_obj:
        epoch_size = sum(1 for line in train_file_obj)
    if max_images > 0:
        epoch_size = min(epoch_size, max_images)

    if distributed_after < 0:
        distributed_after = epoch_size

    # Create the minibatch source and input variables.
    minibatch_source = create_mb_source(
        train_map_file, image_width, image_height, num_channels, num_classes)

    image_input = input((num_channels, image_height, image_width))
    label_input = input(num_classes)

    # Define mapping from reader streams to network inputs.
    input_map = {
        image_input: minibatch_source[features_stream_name],
        label_input: minibatch_source[label_stream_name]
    }

    # Instantiate the transfer learning model and loss function.
    tl_model = create_model(base_model_file, feature_node_name,
                            last_hidden_node_name, num_classes,
                            image_input, freeze)
    ce = cross_entropy_with_softmax(tl_model, label_input)
    pe = classification_error(tl_model, label_input)

    # Instantiate the trainer object.
    lr_schedule = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch)
    mm_schedule = momentum_schedule(momentum_per_mb)
    learner = momentum_sgd(tl_model.parameters, lr_schedule,
                           mm_schedule, l2_regularization_weight=l2_reg_weight)

    progress_printer = ProgressPrinter(tag="Training", num_epochs=num_epochs)
    distributed_learner = distributed.data_parallel_distributed_learner(
        learner=learner,
        # `32` is the non-quantized gradient accumulation; `1` is the
        # 1-bit sgd microsoft technique: https://git.io/v9bYk .
        num_quantization_bits=32,
        # Warm start: no parallelization is used for the first
        # 'distributed_after' samples.
        distributed_after=distributed_after)
    trainer = Trainer(tl_model, (ce, pe),
                      distributed_learner, progress_printer)

    if VERBOSE and distributed.Communicator.rank() == 0:
        print("Training transfer learning model for {0} epochs "
              "(epoch_size = {1}).".format(num_epochs, epoch_size))

    # Get minibatches of images and perform model training.
    log_number_of_parameters(tl_model)
    # Loop over epochs.
    for epoch in range(num_epochs):
        sample_count = 0
        # Loop over minibatches in the epoch.
        while sample_count < epoch_size:
            data = minibatch_source.next_minibatch(
                min(batch_size, epoch_size - sample_count),
                input_map=input_map)
            # Update model.
            trainer.train_minibatch(data)
            # Count samples processed so far.
            sample_count += trainer.previous_minibatch_sample_count
            if (VERBOSE and sample_count % 1000 == 0 and
                    distributed.Communicator.rank() == 0):
                print("Trained the model on {0} samples".format(sample_count))

        trainer.summarize_training_progress()

    # Must be called to finalize MPI in case of successful distributed
    # training.
    distributed.Communicator.finalize()

    return tl_model


def positive_int(x):
    if int(x) < 0:
        raise argparse.ArgumentTypeError("{}: invalid positive int "
                                         "value.".format(value))
    return int(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_or_evaluate = parser.add_mutually_exclusive_group()
    train_or_evaluate.add_argument("-tr", "--train-map-file", help="Path to file "
                        "where paths to train images are written into "
                        "(https://git.io/vH4XD)", required=False,
                        default=argparse.SUPPRESS)
    train_or_evaluate.add_argument("-ev", "--evaluate-only",
                        help="Run only an evaluation on the model provided",
                        action="store_true", required=False, default=False)
    parser.add_argument("-ts", "--test-map-file", help="Path to file where "
                        "paths to test images are written into "
                        "(https://git.io/vH4XD)", required=True,
                        default=argparse.SUPPRESS)
    parser.add_argument("-l", "--labels-list-file", help="Path to file where "
                        "the images labels/categories are written into "
                        "(https://git.io/vH4XD)", required=True,
                        default=argparse.SUPPRESS)
    parser.add_argument("-m", "--model-file", help="Path to pre-trained "
                        "model file", required=False, default=tl_model_file)
    parser.add_argument("-e", "--epochs",
                        help="Total number of epochs to train the model on",
                        type=int, required=False, default=5)
    parser.add_argument("-bs", "--batch-size", help="Batch size dimension",
                        type=int, required=False, default=mb_size)
    parser.add_argument("-a", "--distributed-after",
                        help="Number of samples to train the model on before "
                        "running distributed", type=int, required=False,
                        default=argparse.SUPPRESS)
    parser.add_argument("-not", "--disable-timing",
                        help="Disable timing of the training",
                        action="store_true", required=False, default=False)
    parser.add_argument("-q", "--quiet",
                        help="Disable output (errors are still printed)",
                        action="store_true", required=False, default=False)

    args = vars(parser.parse_args())
    VERBOSE = not args["quiet"]


    if not (args.get("train_map_file", False) or 
            args.get("evaluate_only", False)):
        print("\n ERROR: Choose whether to transfer learning on a new model "
              "(`-tr' option) or to evaluate the performance of the model on "
              "some images (`-ev' option).",
              file=sys.stderr)
        exit(1)
    elif (distributed.Communicator.num_workers() < 2 and 
          distributed.Communicator.rank() == 0):
        print("\n ERROR: Run this script with at least two workers.",
              file=sys.stderr)
        exit(1)

    distributed_after = args.get("distributed_after", False)    
    if not distributed_after:
        # Read the amount the user inserted, or set it to one epoch-size.
        # -1 stands for "distribute after one epoch size".
        distributed_after = -1

    with open(args["labels_list_file"]) as cmf:
        class_mapping = cmf.readline()
        if not class_mapping and distributed.Communicator.rank() == 0:
            print("\n ERROR: empty class mapping file '{}'.".format(
                args["labels_list_file"]), file=sys.stderr)
            exit(1)

    class_mapping = np.asarray(class_mapping.strip("[] ").split(", "))

    if not args["evaluate_only"]:
        if not args["disable_timing"] and distributed.Communicator.rank() == 0:
            start = time.clock()

        # Distributed retraining.
        trained_model = train_model(base_model_file=args["model_file"],
                                    num_classes=len(class_mapping),
                                    train_map_file=args["train_map_file"],
                                    batch_size=args["batch_size"],
                                    distributed_after=distributed_after,
                                    num_epochs=args["epochs"],
                                    freeze=True,
                                    VERBOSE=VERBOSE)

        if distributed.Communicator.rank() == 0:
            if VERBOSE and not args["disable_timing"]:
                print("Training time: {} seconds.".format(time.clock() - start))

            trained_model.save(new_model_file)

            if VERBOSE:
                print("Stored trained model at '{}'.".format(new_model_file))

    # Evaluation phase. We don't need to use multiple GPUs to evaluate, so we
    # make only the first worker evaluate.
    if distributed.Communicator.rank() == 0:

        if args["evaluate_only"] and args["model_file"] == tl_model_file:
            print("\n ERROR: Please provide a trained model to be used to "
                  "evaluate, we cannot use the default model: '{}'.".format(
                        tl_model_file
                    ),
                  file=sys.stderr)
            exit(1)
        elif args["evaluate_only"]:
            trained_model = load_model(args["model_file"])

        predicted_labels_list = []
        true_labels_list = []

        with open(args["test_map_file"], "r") as input_file:
            for line_idx, line in enumerate(input_file):
                # Every line in the test input file is made by the path to an
                # image, a tab, and the category it belongs to.
                img_file, true_label = line.rstrip().split("\t")

                if VERBOSE and line_idx % 100 == 0:
                    print("Evaluating image no. {}".format(line_idx))

                probs = eval_single_image(
                    trained_model, img_file, image_width, image_height)

                predicted_labels_list.append(np.argmax(probs))
                true_labels_list.append(int(true_label))

        assert len(predicted_labels_list) == len(true_labels_list)

        accuracy = sum(1. for p, t in zip(predicted_labels_list,
                                          true_labels_list)
                       if p == t) / len(true_labels_list)

        print("Training accuracy: {}".format(accuracy))
