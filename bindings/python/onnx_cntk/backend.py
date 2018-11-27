# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import numpy as np
from onnx import helper, TensorProto
from onnx.backend.base import Backend
from onnx.backend.base import BackendRep

class CNTKBackend(Backend):
    @staticmethod
    def set_device(device):
        if device == 'CPU':
            C.try_set_default_device(C.device.cpu())
        elif device == 'GPU' or device == 'CUDA':
            try:
                C.try_set_default_device(C.device.gpu(0))
            except:
                C.use_default_device()
        else:
            C.use_default_device()

    @classmethod
    def run_node(cls, node, input, device='CPU'):
        input_tensors = []

        if len(node.input) != len(input):
            raise ValueError(
                "Unexpected Input Size: Op_Type = {0}, Expected = {1}, Received = {2}"
                .format(node.op_type, len(node.input), len(input))
                ) 

        for i in range(len(input)):
            input_tensors.append(helper.make_tensor_value_info(node.input[i], TensorProto.FLOAT, input[i].shape))

        onnx_graph = helper.make_graph([node], "test_{}".format(node.op_type), input_tensors, [])
        onnx_model = helper.make_model(onnx_graph)
        return CNTKBackend.run_model(onnx_model, input, device)

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        expected_out_types = [out.type.tensor_type.elem_type for out in model.graph.output]
        with open(r'tmp_model.pb', 'wb') as f:
            f.write(model.SerializeToString())
        
        CNTKBackend.set_device(device)
        c_model = C.Function.load(r'tmp_model.pb', format=C.ModelFormat.ONNX)
        return CNTKBackendRep(c_model, expected_out_types)

    @classmethod
    def supports_device(cls, device='CPU'):
        return device in ['CPU', 'GPU', 'CUDA']


class CNTKBackendRep(BackendRep):
    def __init__(self, model, expected_out_types):
        self.model = model
        self.expected_out_types = expected_out_types

    def run(self, inputs, **kwargs):
        input = {self.model.arguments[i]:inputs[i] for i in range(len(inputs))}
        res = self.model.eval(input)
        # TODO: make this work for multiple output case.
        # TODO: support more types.
        if self.expected_out_types[0] == TensorProto.BOOL:
            res = res.astype("bool")
        return [res]


prepare = CNTKBackend.prepare
run_model = CNTKBackend.run_model
run_node = CNTKBackend.run_node
supports_device = CNTKBackend.supports_device
