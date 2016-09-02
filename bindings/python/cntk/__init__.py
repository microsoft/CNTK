# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

__version__ = '2.0'

import os

from . import ops
from .cntk_py import *
from .utils import *

import numpy as np

DATATYPE = np.float32

class Trainer(Trainer):
    """
    Trainer to train the specified 'model' with the specified `training_loss` as the training criterion,
    the specified `evaluation_function` as the criterion for evaluating the trained model's quality, and using the specified set
    of `parameters` for updating the model's parameters using computed gradients.
    """
    def __init__(self, model, loss_function, eval_function, parameters):
        if isinstance(model, Variable):
            model = model.owner
        if isinstance(loss_function, Variable):
            loss_function = loss_function.owner
        if isinstance(eval_function, Variable):
            eval_function = eval_function.owner        
        super(Trainer, self).__init__(model, loss_function, eval_function, parameters)