# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

from ..layers import *

def test_layers_name(device_id): 
    from cntk import placeholder_variable, combine
    I = placeholder_variable(name='input')
    p = Dense(10, name='dense10')(I)
    assert(I.name == 'input')
    assert(p.root_function.name == 'dense10')
    
    q = Convolution((3,3), 3, name='conv33')(I)
    assert(q.root_function.name == 'conv33')

