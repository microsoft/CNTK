# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
import pdb
np.set_printoptions(linewidth=9999, precision=4, suppress=True)
class DebugLayerSingle(UserFunction):
    def __init__(self, arg1, name='DebugLayerSingle', debug_name="", split_line=False, print_grads=True):
        super(DebugLayerSingle, self).__init__([arg1], name=debug_name)
        self._debug_name = debug_name
        self._split_char = "\n" if split_line else " "
        self._img1 = True
        self._img2 = True
        self._print_grads = print_grads
        self._verbose_fwd = True
        self._verbose_bkw = False

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, name=self._debug_name)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        if self._verbose_fwd:
            #print("udf_cntk: {}".format(arguments[0, 527, :]))
            if self._debug_name=="image_input_d":
                #pdb.set_trace()
                print("udf features:\n{}".format(arguments[:, :, 22:23, :10]))
            elif self._debug_name == "regular_input_d":
                # pdb.set_trace()
                print("resnet out:\n{}".format(arguments[:, 3:5, 2:4, :10]))
            elif self._debug_name == "conv1" or self._debug_name == "bnrelu1":
                #pdb.set_trace()
                print("{}:\n{}".format(self._debug_name, arguments[:, 3:5, 2:4, :10]))
            else:
                #pass
                print(self.format_line(arguments))

        return None, arguments

    def backward(self, state, root_gradients):
        if self._verbose_bkw:
            #pdb.set_trace()
            arguments = root_gradients
            if self._debug_name=="image_input_d":
                pass
            elif self._debug_name=="regular_input_d":
                pass
            elif self._debug_name == "conv1" or self._debug_name == "bnrelu1":
                pass
            else:
                print(self.format_line(arguments, fwd=False))

        return root_gradients

    def clone(self, cloned_inputs):
        return DebugLayerSingle(cloned_inputs[0], debug_name=self._debug_name)

    def serialize(self):
        internal_state = {}
        internal_state["debug_name"] = self._debug_name
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        debug_name = state['debug_name']
        return DebugLayerSingle(inputs[0], name=name, debug_name=debug_name)

    def format_line(self, arguments, fwd=True):
        #pdb.set_trace()
        # responsible boxes [527, 527], [453, 453, 523, 523, 630, 630]
        boxes = []
        #if self._debug_name=='WH_Out_d':
        #    import pdb;pdb.set_trace()

        if self._img1:
            b1 = arguments[0,527,:]
            boxes.append(b1)
        if self._img2:
            index = arguments.shape[0] - 1
            b2 = arguments[index,453,:]
            b3 = arguments[index,523,:]
            b4 = arguments[index,630,:]
            boxes.append(b2)
            boxes.append(b3)
            boxes.append(b4)

        line = "{}_{}:".format("fwd" if fwd else "bkw", self._debug_name)
        for i, item in enumerate(boxes):
            line = "{}{}b{}: {}".format(line, self._split_char, i+1, item)
        return line