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
        self._print_grads = print_grads
        self._verbose_fwd = True
        self._verbose_bkw = True

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes, name=self._debug_name)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        if self._verbose_fwd:
            print(self.format_line(arguments))

        return None, arguments

    def backward(self, state, root_gradients):
        if self._verbose_bkw:
            arguments = root_gradients
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
        slices = []
        #s1 = arguments[0, 527, :]
        #slices.append(s1)

        line = "{}_{}:".format("fwd" if fwd else "bkw", self._debug_name)
        for i, item in enumerate(slices):
            line = "{}{}b{}: {}".format(line, self._split_char, i+1, item)
        return line