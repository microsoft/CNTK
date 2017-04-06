# --------------------------------------------------------
# Copyright (c) 2017 Microsoft
# --------------------------------------------------------

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
import scipy as sp
from fast_rcnn.config import cfg

DEBUG = cfg["CNTK"].DEBUG_LAYERS
debug_fwd = cfg["CNTK"].DEBUG_FWD
debug_bkw = cfg["CNTK"].DEBUG_BKW

class IgnoreLabel(UserFunction):
    """
    Sets entries to zero in target and prediction for the label to ignore
    """

    def __init__(self, arg1, arg2, name='IgnoreLabel', ignore_label=None):
        super(IgnoreLabel, self).__init__([arg1, arg2], name=name)

        self._ignore_label = ignore_label

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, name='rpn_obj_prob'),
                output_variable(self.inputs[1].shape, self.inputs[1].dtype, self.inputs[1].dynamic_axes, name='rpn_obj_targets', needs_gradient=False)]

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        if debug_fwd: print("--> Entering forward in {}".format(self.name))

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

        return ignore_ind

    def backward(self, state, root_gradients, variables):
        if debug_bkw: print("<-- Entering backward in {}".format(self.name))

        # gradients for prediction: propagate only for those that were not ignored
        if self.inputs[0] in variables:
            # since we set target = pred in forward the gradients for ignored entries should already be zero
            variables[self.inputs[0]] = root_gradients[self.outputs[0]]

