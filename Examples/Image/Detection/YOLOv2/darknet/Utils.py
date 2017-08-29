# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function

import cntk.io.transforms as xforms

from cntk.layers import identity
from cntk.layers.typing import Tensor, Signature
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk import Trainer
from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk import cross_entropy_with_softmax, classification_error
from cntk.ops import Function
from cntk.logging import ProgressPrinter, log_number_of_parameters

from PARAMETERS import *

########################
# variables and paths  #
########################

# paths (are relative to current python file)
abs_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")

# model dimensions
image_height = par_image_height # Darknet19 scales input image down over all by a factor of 32. \\
image_width = par_image_width  # It needs at least a 3x3 shape for the last conv layer. So 32*3 is required at least.
num_channels = par_num_channels  # RGB
num_classes = par_num_classes

#training parameters
__mb_size = par_minibatch_size # copy on import

########################
# define the reader    #
########################

def create_reader(map_file, is_training, is_distributed = False):
    if not os.path.exists(map_file):
        raise RuntimeError(
            "File '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
            (map_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.color(0.5,0.0,0.5),
            xforms.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio')  # train uses jitter
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),  # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=num_classes))),  # and second as 'label'
                           randomize=is_training, max_sweeps=INFINITELY_REPEAT if is_training else 1,
                           multithreaded_deserializer=is_distributed)


########################
# define the criteria  #
########################

# compose model function and criterion primitives into a criterion function
#  takes:   Function: features -> prediction
#  returns: Function: (features, labels) -> (loss, metric)
def create_criterion_function(model, normalize=identity):
    # @Function    # Python 3
    # def criterion(x: Tensor[(num_channels, image_height, image_width)], y: Tensor[num_classes]):
    @Function
    @Signature(x=Tensor[(num_channels, image_height, image_width)], y=Tensor[num_classes])
    def criterion(x, y):
        z = model(normalize(x))
        ce = cross_entropy_with_softmax(z, y)
        errs = classification_error(z, y)
        return (ce, errs)

    return criterion


########################
# train action         #
########################

def train_model(reader, model, epoch_size=50000, max_epochs=par_max_epochs, save_progress=False, mb_size=__mb_size, exponentShift = 0):
    # declare the model's input dimension
    # Training does not require this, but it is needed for deployment.
    model.update_signature((num_channels, image_height, image_width))

    # criterion function. This is what is being trained trained.
    # Model gets "sandwiched" between normalization (not part of model proper) and criterion.
    criterion = create_criterion_function(model)

    # learning parameters
    learner = momentum_sgd(model.parameters,
                           lr=learning_rate_schedule([0.001 * 10**exponentShift] * 15 + [0.0001 * 10**exponentShift] * 15 + [0.00001 * 10**exponentShift] * 15),
                               #[0.0015625 * 10**exponentShift] * 20 + [0.00046875 * 10**exponentShift] * 20
                               # + [0.00015625 * 10**exponentShift] * 20 + [0.000046875 * 10**exponentShift] * 10
                               # + [0.000015625 * 10**exponentShift], unit=UnitType.sample, epoch_size=epoch_size),
                           momentum=momentum_as_time_constant_schedule([0] * 20 + [600] * 20 + [1200],
                                                                       epoch_size=epoch_size),
                           l2_regularization_weight=0.002)


    # trainer object
    trainer = Trainer(None, criterion, learner)

    # perform model training
    log_number_of_parameters(model);
    print()
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    print("Minibatch size is " + str(mb_size))
    for epoch in range(max_epochs):  # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            mb = reader.next_minibatch(min(mb_size, epoch_size - sample_count))  # fetch minibatch.
            # trainer.train_minibatch(mb[reader.streams.features], mb[reader.streams.labels])
            trainer.train_minibatch({criterion.arguments[0]: mb[reader.streams.features],
                                     criterion.arguments[1]: mb[reader.streams.labels]})
            sample_count += mb[reader.streams.labels].num_samples  # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
        if save_progress:
            model.save(os.path.join(model_path, "train_model_DataAug_{}.dnn".format(epoch)))

    # return evaluation error.
    return loss, metric  # return values from last epoch


########################
# eval action          #
########################

# helper function to create a dummy Trainer that one can call test_minibatch() on
# TODO: replace by a proper such class once available
def Evaluator(model, criterion):
    loss, metric = Trainer._get_loss_metric(criterion)
    parameters = set(loss.parameters)
    if model:
        parameters |= set(model.parameters)
    if metric:
        parameters |= set(metric.parameters)
    dummy_learner = momentum_sgd(tuple(parameters),
                                 lr=learning_rate_schedule(1, UnitType.minibatch),
                                 momentum=momentum_as_time_constant_schedule(0))
    return Trainer(model, (loss, metric), dummy_learner)


def evaluate_model(reader, model):
    # criterion function. This is what is being evaluated
    criterion = create_criterion_function(model)

    # process minibatches and perform evaluation
    evaluator = Evaluator(None, criterion)

    progress_printer = ProgressPrinter(tag='Evaluation', num_epochs=1)

    while True:
        minibatch_size = __mb_size
        mb = reader.next_minibatch(minibatch_size)  # fetch minibatch
        if not mb:  # until we hit the end
            break

        metric = evaluator.test_minibatch({criterion.arguments[0]: mb[reader.streams.features],
                                           criterion.arguments[1]: mb[reader.streams.labels]})  # evaluate minibatch
        progress_printer.update(0, mb[reader.streams.labels].num_samples, metric)  # log progress
    loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)

    return loss, metric

