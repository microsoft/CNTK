# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C
import numpy as np
from onnx.backend.base import BackendRep

class CNTKBackendRep(BackendRep):
    def __init__(self, model):
        self.model = model

    def run(self, inputs, **kwargs):
        input = {self.model.arguments[i]:inputs[i] for i in range(len(inputs))} 
        res = self.model.eval(input)
        return [res]
