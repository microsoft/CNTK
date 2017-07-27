# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable
from cntk.ops.functions import UserFunction
import pdb

class DebugLayer(UserFunction):
    def __init__(self, arg1, arg2, arg3, name='DebugLayer', debug_name=""):
        super(DebugLayer, self).__init__([arg1, arg2, arg3], name=name)
        self._debug_name = debug_name

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes),
                output_variable(self.inputs[1].shape, self.inputs[1].dtype, self.inputs[1].dynamic_axes),
                output_variable(self.inputs[2].shape, self.inputs[2].dtype, self.inputs[2].dynamic_axes)]

    def forward(self, arguments, device=None, outputs_to_retain=None):
        pdb.set_trace()
        image = arguments[0]
        output = arguments[1]
        gt = arguments[2]

        print("-- {} -- shapes".format(self._debug_name))
        print("image shape {}".format(image.shape))
        print("output shape {}".format(output.shape))
        print("gt shape {}".format(gt.shape))

        outputs[self.outputs[0]] = arguments[0]
        outputs[self.outputs[1]] = arguments[1]
        outputs[self.outputs[2]] = arguments[2]

        return None

    def backward(self, state, root_gradients, variables):
        pdb.set_trace()
        variables[self.inputs[0]] = root_gradients[self.outputs[0]]
        variables[self.inputs[1]] = root_gradients[self.outputs[1]]
        variables[self.inputs[2]] = root_gradients[self.outputs[2]]

    def clone(self, cloned_inputs):
        return DebugLayer(cloned_inputs[0], cloned_inputs[1], cloned_inputs[2], debug_name=self._debug_name)

    def serialize(self):
        internal_state = {}
        internal_state["debug_name"] = self._debug_name
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        debug_name = state['debug_name']
        return DebugLayer(inputs[0], inputs[1], inputs[2], name=name, debug_name=debug_name)

