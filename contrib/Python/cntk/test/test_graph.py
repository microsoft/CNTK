from ..context import get_new_context, _CONTEXT
from ..graph import *
from ..graph import _tensor_to_text_format

import pytest

import scipy.sparse

# keeping things short
A = np.asarray
C = constant
I = input


# testing whether operator overloads result in proper type
@pytest.mark.parametrize('root_node, expected', [
    # __add__ / __radd__
    (C(0) + C(1), Plus),
    (C(0) + 1, Plus),
    (0 + C(1), Plus),
    (0 + 1, int),

    # __sub__ / __rsub__
    (C(0) - C(1), Minus),
    (C(0) - 1, Minus),
    (0 - C(1), Minus),
    (0 - 1, int),

    # __mul__ / __rmul__ --> element-wise (!) multiplication
    (C(0) * C(1), ElementTimes),
    (C(0) * 1, ElementTimes),
    (0 * C(1), ElementTimes),
    (0 * 1, int),

    # __abs__
    (abs(C(0)), Abs),

    # __getitem__
    (C(np.arange(0, 10))[2:5], RowSlice),
    (C(np.arange(0, 10))[:5], RowSlice),

])
def test_overload_types(root_node, expected):
    assert isinstance(root_node, expected)


def test_overload_exception():
    with pytest.raises(ValueError):
        C(range(0, 10))[:]

    with pytest.raises(ValueError):
        C(range(0, 10))[0:3:2]


def _to_list(desc):
    return [line.strip() for line in desc.split('\n')]


def test_graph_with_same_node_twice():
    v0 = C(1)
    root_node = Plus(v0, v0)
    description, has_inputs, readers = root_node.to_config()
    assert len(_to_list(description)) == 2


@pytest.mark.parametrize("alias, idx, data, expected", [
    ('', 0, [A([1, 0]), A([0, 0, 1, 0])], ValueError),  # no alias given
    ('A', 0, [object()], ValueError),
])
def test_tensor_conversion_exceptions(alias, idx, data, expected):
    with pytest.raises(expected):
        _tensor_to_text_format(idx, alias, data)


@pytest.mark.parametrize("alias, idx, data, expected", [
    ('W', 0, A([]), "0	|W "),
    ('W', 0, A([[1, 0, 0, 0], [0, 0, 1, 0]]), """\
0	|W 1 0 0 0 0 0 1 0\
"""),
])
def test_tensor_conversion_dense(alias, idx, data, expected):
    assert _tensor_to_text_format(idx, alias, data,
            has_sequence_dimension=False) == expected

if False:
    @pytest.mark.parametrize("alias, data, expected", [
        ('W', [A({})], ""),
        ('W', [{3: 1, 50: 1, 2: 0}, {1: -5}], """\
    0	|W 2:0 3:1 50:1
    1	|W 1:-5\
    """),
    ])
    def test_tensor_conversion_sparse(alias, data, expected):
        # We use the dictionary in data to create a SciPy sparse dictionary of
        # keys, which we then feed to the converter.
        dok_data = []
        for idx, data_elem in enumerate(data):
            d = scipy.sparse.dok_matrix((100, 1))
            for k, v in data_elem.items():
                d[k] = v
            dok_data.append(d)
        assert _tensor_to_text_format(idx, alias, dok_data) == expected


@pytest.mark.parametrize("data, expected", [
    ([], True),
    ([1], True),
    ([[1, 2]], True),
    ([[]], True),
    ([[A([1, 2])]], False),
    ([A([1, 2])], False),
    ([A([1, 2]), A([])], False),
])
def test_is_tensor(data, expected):
    assert is_tensor(data) == expected


@pytest.mark.parametrize("data, expected", [
    ([], False),
    ([1], False),
    ([[1, 2]], False),
    ([[]], False),
    ([[A([1, 2])]], False),
    ([A([1, 2])], True),
    ([A([1, 2]), A([])], True),
])
def test_is_tensor_list(data, expected):
    assert is_tensor_list(data) == expected

def test_loose_coupling():
    from cntk.ops.cntk1 import PastValue
    dh = PastValue(1, 'outnode')
    out = Times(dh, Constant(2), var_name='outnode')

    expected = ['v0 = PastValue(1, outnode, timeStep=1, defaultHiddenActivation=0.1)', 
            'v1 = Constant(2, rows=1, cols=1)',
            'outnode = Times(v0, v1, outputRank=1)']

    description, has_inputs, readers = out.to_config()
    assert _to_list(description) == expected


