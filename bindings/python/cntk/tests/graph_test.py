# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..graph import *
from ..ops import *

def _graph_dict():
    d = {}
    
    d['i1'] = input_variable(shape=(2,3), name='i1')
    d['i2'] = input_variable(shape=(2,3), name='i2')

    d['p1'] = parameter(shape=(3,2), name='p1')

    
    d['op1'] = plus(d['i1'], d['i2'], name='op1')
    d['op2'] = times(d['op1'], d['p1'], name='op2')

    # no name
    d['p2'] = parameter(shape=(2,2))

    # duplicate names
    d['op3a'] = plus(d['op2'], d['p2'], name='op3')
    d['op3b'] = plus(d['op3a'], d['p2'], name='op3')
    
    d['past'] = past_value(d['op3b'], name='past')

    d['root'] = d['past']

    return d
    

def test_find_nodes():
    d = _graph_dict()

    for name in ['i1', 'i2', 'p1', 'op1', 'op2', 'past']:
        n = find_nodes_by_name(d['root'], name)
        assert len(n) == 1, name
        assert n[0].name == name, name
    
    n = find_nodes_by_name(d['root'], 'op3')
    assert len(n) == 2, 'op3'
    assert n[0].name == 'op3' and n[1].name == 'op3', 'op3'

    none = find_nodes_by_name(d['root'], 'none')
    assert none == []


