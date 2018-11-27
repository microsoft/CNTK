# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest
import cntk as C
from cntk.axis import Axis


def _graph_dict():
    # This function creates a graph that has no real meaning other than
    # providing something to traverse.
    d = {}

    d['i1'] = C.sequence.input_variable(shape=(2, 3), sequence_axis=Axis('ia'), name='i1')
    d['c1'] = C.constant(shape=(2, 3), value=6, name='c1')
    d['p1'] = C.parameter(shape=(3, 2), init=7, name='p1')

    d['op1'] = C.plus(d['i1'], d['c1'], name='op1')
    d['op2'] = C.times(d['op1'], d['p1'], name='op2')

    #d['slice'] = slice(d['c1'], Axis.default_dynamic_axis(), 0, 3)
    #label_sentence_start = sequence.first(raw_labels)

    # no name
    d['p2'] = C.parameter(shape=(2, 2))

    # duplicate names
    d['op3a'] = C.plus(d['op2'], d['p2'], name='op3')
    d['op3b'] = C.plus(d['op3a'], d['p2'], name='op3')

    d['first'] = C.sequence.first(d['op3b'], name='past')

    d['root'] = d['first']

    return d


def _simple_dict():
    d = {}

    d['i1'] = C.input_variable(shape=(2, 3), name='i1')
    d['c1'] = C.constant(shape=(2, 3), value=6, name='c1')
    d['p1'] = C.parameter(shape=(3, 2), init=7, name='p1')
    d['op1'] = C.plus(d['i1'], d['c1'], name='op1')
    d['op2'] = C.times(d['op1'], d['p1'], name='op2')
    d['root'] = d['op2']

    d['target'] = C.input_variable((), name='label')
    d['all'] = C.combine([d['root'], C.minus(
        d['target'], C.constant(1, name='c2'), name='minus')], name='all')

    return d


def test_find_nodes():
    d = _graph_dict()
    graph_uids = []

    for name in ['i1', 'c1', 'p1', 'op1', 'op2', 'past']:
        n = C.logging.graph.find_all_with_name(d['root'], name)
        assert len(n) == 1, name
        assert n[0].name == name, name

        n = d['root'].find_all_with_name(name)
        assert len(n) == 1, name
        assert n[0].name == name, name

        n = C.logging.graph.find_by_name(d['root'], name)
        assert n.name == name, name
        assert n != None

        graph_uids.append(n.uid)

        n = d['root'].find_by_name(name)
        assert n.name == name, name

        for uid in graph_uids:
            n2 = C.logging.graph.find_by_uid(d['root'], uid)
            assert n2.uid == uid, uid
            assert n2 != None

    n = C.logging.graph.find_all_with_name(d['root'], 'op3')
    assert len(n) == 2, 'op3'
    assert n[0].name == 'op3' and n[1].name == 'op3', 'op3'

    none = C.logging.graph.find_all_with_name(d['root'], 'none')
    assert none == []

    assert C.logging.graph.find_by_name(d['root'], 'none') is None


def test_find_nodes_returning_proper_types():
    d = _graph_dict()

    c1 = C.logging.graph.find_by_name(d['root'], 'c1')
    assert isinstance(c1, C.Constant)
    assert np.allclose(c1.value, np.zeros((2, 3)) + 6)

    p1 = C.logging.graph.find_by_name(d['root'], 'p1')
    assert isinstance(p1, C.Parameter)
    assert np.allclose(p1.value, np.zeros((3, 2)) + 7)


def test_plot():
    d = _simple_dict()

    m = C.logging.graph.plot(d['all'])
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

    found = C.logging.graph.depth_first_search(d['all'], lambda x: True, depth=depth)
    found_names = [v.name for v in found]
    assert found_names == ['all', 'op2', 'op1',
                           'i1', 'c1', 'p1', 'minus', 'label', 'c2']


@pytest.mark.parametrize("depth,prefix_count", [
    (0, {
            "Input('image'":1,
            "blocked_dense:":1,
            "Dense(":1,
            "MaxPooling(":1,
            "Convolution(":1,
            "Parameter('W'":3,
            "Parameter('b'":3,
            }),
     (1, {
            "Input('image'":1,
            "blocked_dense:":1,
            "Dense(":2,
            "MaxPooling(":1,
            "Convolution(":1,
            "Parameter('W'":3,
            "Parameter('b'":3,
            }),
     (2, {
            "Input('image'":1,
            "blocked_dense:":1,
            "Dense(":2,
            "MaxPooling(":1,
            "Convolution(":1,
            "Parameter('W'":3,
            "Parameter('b'":3,
            # in addition to depth=1...
            "Plus(":2,
            "Times(":2,
            }),
     (-1, {
            "Input('image'":1,
            "blocked_dense:":1,
            "Dense(":2,
            "MaxPooling(":1,
            "Convolution(":2,
            "Parameter('W'":3,
            "Parameter('b'":3,
            "Times(":2,
            # in addition to depth=2...
            "Plus(":3,
            "ReLU(":1,
            "Pooling(Tensor":1,
            }),
     ])
def test_depth_first_search_blocks(depth, prefix_count):
    from cntk.layers import Sequential, Convolution, MaxPooling, Dense
    from cntk.default_options import default_options
    
    def Blocked_Dense(dim, activation=None):
        dense = Dense(dim, activation=activation)
        @C.layers.BlockFunction('blocked_dense', 'blocked_dense')
        def func(x):
            return dense(x)
        return func

    with default_options(activation=C.relu):
        image_to_vec = Sequential ([
            Convolution((5,5), 32, pad=True),
            MaxPooling((3,3), strides=(2,2)),
            Dense(10, activation=None),
            Blocked_Dense(10)
            ]
        )

    in1 = C.input_variable(shape=(3, 256, 256), name='image')
    img = image_to_vec(in1)

    found = C.logging.graph.depth_first_search(img, lambda x: True, depth=depth)
    found_str = [str(v) for v in found]
    
    assert len(found) == sum(prefix_count.values())
    for prefix, count in prefix_count.items():
        assert sum(f.startswith(prefix) for f in found_str) == count
