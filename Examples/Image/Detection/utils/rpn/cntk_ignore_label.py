# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
import scipy as sp

DEBUG = False

class IgnoreLabel(UserFunction):
    '''
    Sets entries to zero in target and prediction for the label to ignore
    '''

    def __init__(self, arg1, arg2, name='IgnoreLabel', ignore_label=None):
        super(IgnoreLabel, self).__init__([arg1, arg2], name=name)

        self._ignore_label = ignore_label

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, name='rpn_obj_prob'),
                output_variable(self.inputs[1].shape, self.inputs[1].dtype, self.inputs[1].dynamic_axes, name='rpn_obj_targets', needs_gradient=False)]

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        # set entries to zero in target and prediction for the label to ignore
        predictions = arguments[0][0,:]
        targets = arguments[1][0,0,:]

        bg_pred = predictions[0,:]
        fg_pred = predictions[1,:]

        ignore_ind = np.where(targets == self._ignore_label)
        bg_pred[ignore_ind] = 1.0
        fg_pred[ignore_ind] = 0.0
        targets[ignore_ind] = 0

        clean_pred = np.vstack((bg_pred, fg_pred))
        clean_pred.shape = (1,) + clean_pred.shape
        targets.shape = (1,) + targets.shape

        outputs[self.outputs[0]] = clean_pred
        outputs[self.outputs[1]] = targets

        # since we set target = pred the gradients for ignored entries should already be zero.
        # hence, no state is required
        return None

    def backward(self, state, root_gradients, variables):
        # gradients for prediction: propagate only for those that were not ignored
        if self.inputs[0] in variables:
            # since we set target = pred in forward the gradients for ignored entries should already be zero
            variables[self.inputs[0]] = root_gradients[self.outputs[0]]

    def clone(self, cloned_inputs):
        return IgnoreLabel(cloned_inputs[0], cloned_inputs[1], ignore_label=self._ignore_label)

    def serialize(self):
        internal_state = {}
        internal_state['ignore_label'] = self._ignore_label
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        ignore_label = state['ignore_label']
        return IgnoreLabel(inputs[0], inputs[1], name=name, ignore_label=ignore_label)
