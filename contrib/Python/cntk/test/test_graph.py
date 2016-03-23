from ..context import get_new_context, _CONTEXT
from ..graph import *
from ..graph import _seq_to_text_format

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


@pytest.mark.parametrize("root_node, expected", [
    (C(2, var_name='c0'), "c0 = Constant(2, rows=1, cols=1)"),
	# Input should behave as Constant in case of scalars
	(I([1,2], var_name='i1'), "i1 = Input(2:1, tag='feature')"), 
    (Plus(C(0), C(1)),
     "v0 = Constant(0, rows=1, cols=1)\nv1 = Constant(1, rows=1, cols=1)\nv2 = Plus(v0, v1)"),
])
def test_description(root_node, expected):
    description, has_inputs, readers = root_node.to_config() 
    assert description == expected

def test_graph_with_same_node_twice():
    v0 = C(1)
    root_node = Plus(v0, v0)
    expected = 'v0 = Constant(1, rows=1, cols=1)\nv1 = Plus(v0, v0)'
    description, has_inputs, readers = root_node.to_config() 
    assert description == expected
    assert readers == []

@pytest.mark.parametrize("alias, data, expected", [
	('', [A([1,0]), A([0,0,1,0])], ValueError), # no alias given
	('A', [object()], ValueError), 
	])
def test_sequence_conversion_exceptions(alias, data, expected):
	with pytest.raises(expected):
		_seq_to_text_format(data, alias=alias)

def test_constant_var_name():
	var_name = 'NODE'
	node = C([A([])], var_name=var_name)
	assert node.var_name == var_name

@pytest.mark.parametrize("alias, data, expected", [
	('W', [A([])], """\
0|W \
"""),
	('W', [A([1,0]), A([0,0,1,0])], """\
0|W 1 0
1|W 0 0 1 0\
"""),
	])
def test_sequence_conversion_dense(alias, data, expected):
	assert _seq_to_text_format(data, alias=alias) == expected

if False:
	@pytest.mark.parametrize("alias, data, expected", [
		('W', [A({})], """\
	0|W \
	"""),
		('W', [{3:1, 50:1, 2:0}, {1:-5}], """\
	0|W 2:0 3:1 50:1
	1|W 1:-5\
	"""),
		])
	def test_sequence_conversion_sparse(alias, data, expected):
		# We use the dictionary in data to create a SciPy sparse dictionary of
		# keys, which we then feed to the converter.
		dok_data = []
		for data_elem in data:
			d = scipy.sparse.dok_matrix((100,1))
			for k,v in data_elem.items():
				d[k] = v
			dok_data.append(d)
		assert _seq_to_text_format(dok_data, alias=alias) == expected

@pytest.mark.parametrize("data, expected", [
    ([], True),
    ([1], True),
    ([[1,2]], True),
    ([[]], True),
    ([[A([1,2])]], False),
    ([A([1,2])], False),
    ([A([1,2]), A([])], False),
	])
def test_is_tensor(data, expected):
	#import ipdb;ipdb.set_trace()
	assert is_tensor(data) == expected

@pytest.mark.parametrize("data, expected", [
    ([], False),
    ([1], False),
    ([[1,2]], False),
    ([[]], False),
    ([[A([1,2])]], False),
    ([A([1,2])], True),
    ([A([1,2]), A([])], True),
	])
def test_is_sequence(data, expected):
	assert is_sequence(data) == expected

