
class Node(object):
    def __init__(self, name, params=None, value=None, get_output_shape=None,
            var_name=None, check=None):
        self.name = name
        self.params = params
        self.value = value
        self.get_output_shape = get_output_shape
        self.var_name = var_name

        if check:
            #print("name=%s params=%s"%(name, str(params)))
            assert check(*params)

    def __add__(self, other):
        return Operator("Plus", (self, other),
                get_output_shape=lambda a,b: a.get_shape(),
                check=plus_check
                )

    def __radd__(self, other):
        return Operator("Plus", (other, self),
                get_output_shape=lambda a,b: a.get_shape(),
                check=plus_check
                )

    def __mul__(self, other):
        return times(self, other)

    def __truediv__(self, other):
        return Operator("**Divide**", (self, other),
                get_output_shape=lambda a,b: np.asarray(a).shape
                )

    def __rtruediv__(self, other):
        return Operator("**Divide**", (other, self),
                get_output_shape=lambda a,b: np.asarray(a).shape
                )

    def get_cntk_param_string(self, param_variable_names=None):
        return ""

    def get_value(self):
        return self.value

    def get_shape(self):
        if self.value is not None:
            return self.value.shape
        else:
            if self.params:
                print("params: "+str(self.params))

                return self.get_output_shape(*self.params)
            else:
                return self.get_output_shape()

    def eval(self, **kw):
        raise NotImplementedError

    def __str__(self):
        return "%s / params=%s / value=%s"%(self.name, self.params, self.value)


class Input(Node):
    def __init__(self, shape, **kwargs):
        super(Input, self).__init__('Input', **kwargs)
        self.get_output_shape=lambda : shape

    def get_cntk_param_string(self, param_variable_names=None):

class LearnableParameter(Node):
    def __init__(self, **kwargs):
        super(LearnableParameter, self).__init__('LearnableParameter', **kwargs)
        self.get_output_shape=lambda : kwargs['value'].shape

    def get_cntk_param_string(self, param_variable_names=None):
        if len(param_variable_names)!=0:
            raise ValueError("expected no parameter variable names",
                    param_variable_names)

        shape = self.get_output_shape()

        # TODO this makes only sense as the first layer for a
        # classification problem.
        if len(shape)==1:
            params = "$NumOfClasses$" 
        elif len(shape)==2:
            # TODO have layer's output_dim and input_dim a word on this
            rows = shape[0] 
            cols = shape[0]
            params = "%s, %s"%(rows, cols)
        else:
            raise ValueError("expected either 1 or 2-dimensional shape", shape) 

        return params

