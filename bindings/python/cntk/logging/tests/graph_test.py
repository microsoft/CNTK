# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
from ..graph import *
from cntk.ops import *
from cntk.axis import Axis


def _graph_dict():
    # This function creates a graph that has no real meaning other than
    # providing something to traverse.
    d = {}

    d['i1'] = sequence.input(shape=(2, 3), sequence_axis=Axis('ia'), name='i1')
    d['c1'] = constant(shape=(2, 3), value=6, name='c1')
    d['p1'] = parameter(shape=(3, 2), init=7, name='p1')

    d['op1'] = plus(d['i1'], d['c1'], name='op1')
    d['op2'] = times(d['op1'], d['p1'], name='op2')

    #d['slice'] = slice(d['c1'], Axis.default_dynamic_axis(), 0, 3)
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

    d['i1'] = input(shape=(2, 3), name='i1')
    d['c1'] = constant(shape=(2, 3), value=6, name='c1')
    d['p1'] = parameter(shape=(3, 2), init=7, name='p1')
    d['op1'] = plus(d['i1'], d['c1'], name='op1')
    d['op2'] = times(d['op1'], d['p1'], name='op2')
    d['root'] = d['op2']

    d['target'] = input((), name='label')
    d['all'] = combine([d['root'], minus(
        d['target'], constant(1, name='c2'), name='minus')], name='all')

    return d


def test_find_nodes():
    d = _graph_dict()

    for name in ['i1', 'c1', 'p1', 'op1', 'op2', 'past']:
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


def test_find_nodes_returning_proper_types():
    d = _graph_dict()

    c1 = find_by_name(d['root'], 'c1')
    assert isinstance(c1, Constant)
    assert np.allclose(c1.value, np.zeros((2, 3)) + 6)

    p1 = find_by_name(d['root'], 'p1')
    assert isinstance(p1, Parameter)
    assert np.allclose(p1.value, np.zeros((3, 2)) + 7)


def test_plot():
    d = _simple_dict()

    m = plot(d['all'])
    p = "Plus"
    t = "Times"

    assert len(m) != 0
    assert p in m
    assert t in m
    assert m.find(p) < m.find(t)


@pytest.mark.parametrize("depth", [
    (-1), (0), (1), (5)])
def test_depth_first_search(depth):
    '''
    For graphs without blocks, depth should not make any difference.
    '''
    d = _simple_dict()

    found = depth_first_search(d['all'], lambda x: True, depth=depth)
    found_names = [v.name for v in found]
    assert found_names == ['all', 'op2', 'op1',
                           'i1', 'c1', 'p1', 'minus', 'label', 'c2']


@pytest.mark.parametrize("depth,prefix_count", [
    (0, {
            "Input('image'":1,
            "Dense:":1,
            "MaxPooling:":1,
            "Convolution:":1,
            "Parameter('W'":2,
            "Parameter('b'":2,
            }),
     (-1, {
            "Input('image'":1,
            "Dense:":1,
            "MaxPooling:":1,
            "Convolution:":1,
            "Parameter('W'":2,
            "Parameter('b'":2,
            # in addition to depth=0...
            "Plus:":1,
            "Times":1,
            }),
     (1, {
            "Input('image'":1,
            "Dense:":1,
            "MaxPooling:":1,
            "Convolution:":1,
            "Parameter('W'":2,
            "Parameter('b'":2,
            "Plus:":1,
            "Times":1,
            }),
     (2, {
            "Input('image'":1,
            "Dense:":1,
            "MaxPooling:":1,
            "Convolution:":1,
            "Parameter('W'":2,
            "Parameter('b'":2,
            "Plus:":1,
            "Times":1,
            # in addition to depth=1...
            "Pooling: Placeholder(": 1,
            }),
     ])
def test_depth_first_search_blocks(depth, prefix_count):
    from cntk.layers import Sequential, Convolution, MaxPooling, Dense
    from cntk.default_options import default_options

    with default_options(activation=relu):
        image_to_vec = Sequential ([
            Convolution((5,5), 32, pad=True),
            MaxPooling((3,3), strides=(2,2)),
            Dense(10, activation=None)
            ]
        )

    in1 = input(shape=(3, 256, 256), name='image')
    img = image_to_vec(in1)

    found = depth_first_search(img, lambda x: True, depth=depth)
    found_str = [str(v) for v in found]

    assert len(found) == sum(prefix_count.values())
    for prefix, count in prefix_count.items():
        assert sum(f.startswith(prefix) for f in found_str) == count
