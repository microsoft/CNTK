# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ..graph import *
from ..ops import *
from ..axis import Axis


def _graph_dict():
    # This function creates a graph that has no real meaning other than
    # providing something to traverse.
    d = {}

    batch_axis = Axis.default_batch_axis()
    input_seq_axis = Axis('ia')
    input_dynamic_axes = [batch_axis, input_seq_axis]

    d['i1'] = input_variable(
        shape=(2, 3), dynamic_axes=input_dynamic_axes, name='i1')
    d['i2'] = input_variable(
        shape=(2, 3), dynamic_axes=input_dynamic_axes, name='i2')

    d['p1'] = parameter(shape=(3, 2), name='p1')

    d['op1'] = plus(d['i1'], d['i2'], name='op1')
    d['op2'] = times(d['op1'], d['p1'], name='op2')

    #d['slice'] = slice(d['i2'], Axis.default_dynamic_axis(), 0, 3)
    #label_sentence_start = sequence.first(raw_labels)

    # no name
    d['p2'] = parameter(shape=(2, 2))

    # duplicate names
    d['op3a'] = plus(d['op2'], d['p2'], name='op3')
    d['op3b'] = plus(d['op3a'], d['p2'], name='op3')

    d['first'] = sequence.first(d['op3b'], name='past')

    d['root'] = d['first']

    return d


def _simple_dict():
    d = {}

    d['i1'] = input_variable(shape=(2, 3), name='i1')
    d['i2'] = input_variable(shape=(2, 3), name='i2')
    d['p1'] = parameter(shape=(3, 2), name='p1')
    d['op1'] = plus(d['i1'], d['i2'], name='op1')
    d['op2'] = times(d['op1'], d['p1'], name='op2')
    d['root'] = d['op2']

    d['target'] = input_variable((), name='label')
    d['all'] = combine([d['root'], minus(d['target'], 1)], name='all')

    return d


def test_find_nodes():
    d = _graph_dict()

    for name in ['i1', 'i2', 'p1', 'op1', 'op2', 'past']:
        n = find_all_with_name(d['root'], name)
        assert len(n) == 1, name
        assert n[0].name == name, name

        n = d['root'].find_all_with_name(name)
        assert len(n) == 1, name
        assert n[0].name == name, name

        n = find_by_name(d['root'], name)
        assert n.name == name, name
        assert n != None

        n = d['root'].find_by_name(name)
        assert n.name == name, name

    n = find_all_with_name(d['root'], 'op3')
    assert len(n) == 2, 'op3'
    assert n[0].name == 'op3' and n[1].name == 'op3', 'op3'

    none = find_all_with_name(d['root'], 'none')
    assert none == []

    assert find_by_name(d['root'], 'none') is None


def test_plot():
    d = _simple_dict()

    m = plot(d['all'])
    p = "Plus"
    t = "Times"

    assert len(m) != 0
    assert p in m
    assert t in m
    assert m.find(p) < m.find(t)

def test_depth_first_search():
    d = _simple_dict()

    found = depth_first_search(d['op2'], lambda x:True)
    found_names = [v.name for v in found]
    assert found_names == ['op2', 'op1', 'i1', 'i2', 'p1']
