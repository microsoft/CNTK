#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
import cntk as C
from onnx.backend.base import Backend
from onnx import helper, TensorProto
from .backend_rep import CNTKBackendRep

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
    def run_model(cls, model, input, device='CPU'):
        with open(r'tmp_model.pb', 'wb') as f:
            f.write(model.SerializeToString())
                   
        CNTKBackend.set_device(device)
        c_model = C.Function.load(r'tmp_model.pb', format=C.ModelFormat.ONNX)
        c_inputs = {c_model.arguments[i]:input[i] for i in range(len(input))} 
        res = c_model.eval(c_inputs)
        return [res]

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        with open(r'tmp_model.pb', 'wb') as f:
            f.write(model.SerializeToString())
        
        CNTKBackend.set_device(device)
        c_model = C.Function.load(r'tmp_model.pb', format=C.ModelFormat.ONNX)
        return CNTKBackendRep(c_model)

    @classmethod
    def supports_device(cls, device='CPU'):
        return device in ['CPU', 'GPU', 'CUDA']

run_node = CNTKBackend.run_node
supports_device = CNTKBackend.supports_device
prepare = CNTKBackend.prepare