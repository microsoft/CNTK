from ..context import *

def test_parse_shapes():
	output = '''\
FormNestedNetwork: WARNING: Was called twice for v3 Plus operation

Validating network. 5 nodes to process in pass 1.

Validating --> dummy_node = InputValue() :  -> [2 {1} x *]
Validating --> v0 = LearnableParameter() :  -> [4 x 1 {1,4}]
Validating --> v1 = Reshape (v0) : [4 x 1 {1,4}] -> [2 x 2 {1,2}]
Validating --> v2 = LearnableParameter() :  -> [1 x 1 {1,1}]
Validating --> v3 = Plus (v1, v2) : [2 x 2 {1,2}], [1 x 1 {1,1}] -> [2 x 2 {1,2}]

Validating network. 2 nodes to process in pass 2.


Validating network, final pass.



5 out of 5 nodes do not share the minibatch layout with the input data.

Post-processing network complete.
'''

	expected = { 
			'dummy_node':(2,),
			'v0':(4,1),
			'v1':(2,2),
			'v2':(1,1),
			'v3':(2,2)
			}

	assert Context._parse_shapes_from_output(output) == expected

