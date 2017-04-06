# --------------------------------------------------------
# Copyright (c) 2017 Microsoft
# --------------------------------------------------------

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
from fast_rcnn.config import cfg

DEBUG = cfg["CNTK"].DEBUG_LAYERS
debug_fwd = cfg["CNTK"].DEBUG_FWD
debug_bkw = cfg["CNTK"].DEBUG_BKW

class SmoothL1Loss(UserFunction):
    """
    Computes a smooth L1 loss
    """

    def __init__(self, arg1, arg2, arg3, name='SmoothL1Loss', sigma=None):
        super(SmoothL1Loss, self).__init__([arg1, arg2, arg3], name=name)
        self._sigma = sigma

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)] #, name="SmoothL1Loss")]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        if debug_fwd: print("--> Entering forward in {}".format(self.name))
        # Algorithm:
        #
        # (According to Fast R-CNN paper, formula (3))
        # The smooth L1 loss is defined per dimension as
        #
        # smooth_L1(x) = | 0.5 * x^2     , if |x| < 1 ## corresponds to \simga/2 * x^2 in huber loss
        #                | |x| - 0.5     , otherwise

        predictions = arguments[0]#[0,:]
        targets = arguments[1]#[0,0,:]
        bbox_inside_weights = arguments[2]#[0, 0, :]
        sigma = self._sigma ## sigma is one for Faster R-CNN and ignored here for now

        diff = predictions - targets
        diff = diff * bbox_inside_weights

        #import pdb; pdb.set_trace()

        x = np.abs(diff)
        lt1 = np.where(x < 1)
        loss = x - .5
        l2 = x * x * .5
        loss[lt1] = l2[lt1]

        return diff, loss

    def backward(self, state, root_gradients, variables):
        if debug_bkw: print("<-- Entering backward in {}".format(self.name))
        # Derivative of smooth L1 loss:
        #
        # - root_gradients      , if diff <= -1
        # diff * root_gradients , if -1 < diff < 1
        # root_gradients        , else

        # A gradient is only required for predictions, not for targets
        if self.inputs[0] in variables:
            diff = state
            item_gradients = root_gradients[0,:]

            assert(item_gradients.size == diff.size)
            diff = diff.reshape(item_gradients.shape)

            le_minus_one = np.where(diff <= -1)
            ge_plus_one = np.where(diff >= 1)

            gradients = item_gradients * diff
            gradients[le_minus_one] = -1 * item_gradients[le_minus_one]
            gradients[ge_plus_one] = item_gradients [ge_plus_one]
            gradients.shape = (1,) + gradients.shape
            variables[self.inputs[0]] = gradients
