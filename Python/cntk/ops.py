from graph import Node

# Because CNTK stores the sample in a transposed form, we need to
# switch parameters for some operators
BIN_OPS_WITH_REVERSED_PARAMETERS = {'Times'}

class Operator(Node):
    def __init__(self, name, params, **kwargs):
        super(Operator, self).__init__(name, params, **kwargs)

    def get_cntk_param_string(self, param_variable_names=None):
        if len(param_variable_names)==0:
            raise ValueError("expected one or more parameter variable names")

        if self.name in BIN_OPS_WITH_REVERSED_PARAMETERS: 
            assert len(param_variable_names)==2 # not sure what to do otherwise
            param_variable_names = reversed(param_variable_names)

        params = ", ".join(param_variable_names) if self.params is not None else ""

        return params

def plus_check(a,b):
    if not hasattr(a, 'get_shape') or not hasattr(b, 'get_shape'):
        return True

    a_shape = a.get_shape()
    b_shape = b.get_shape()

    if not a_shape or not b_shape:
        return True

    if a_shape[0]==None and len(b_shape)==1 and a_shape[1]==b_shape[0]:
        return True

    return a_shape==b_shape

def times_check(a,b):
    a_shape = a.get_shape()
    b_shape = b.get_shape()
    if not a_shape or not b_shape:
        return True

    return a_shape[1]==b_shape[0]

def times(left, right):
    return Operator("Times", (left, right),
            get_output_shape=lambda a,b: (a.get_shape()[0], b.get_shape()[1]),
            check=times_check
            )

def softmax(x):
    return Operator("Softmax", (x,), 
            get_output_shape=lambda x: x.get_shape()
            )

def mean(x, axis=None, keepdims=False):
    # TODO check axes
    return cn.Operator("Mean", (x,),
            #TODO axis
            get_output_shape=lambda a : a.get_shape()[:-1] # TODO
            )

def categorical_crossentropy(output, target):
    return cn.Operator("CrossEntropy", (output, target), 
            get_output_shape=lambda a,b: a.get_shape()[:-1]
            ) 

