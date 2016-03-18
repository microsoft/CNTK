import numpy as np

class ComputationNode(object):
    '''
    Base class for all nodes and operators. Provides a NumPy-like interface
    with operators that are converted to CNTK operators.
    '''

    def __init__(self, name, params=None, var_name=None):
        if not isinstance(name, str):
            raise ValueError("Parameter 'name' has to be a string and not '%s'"%type(name))
        if var_name is not None and not isinstance(var_name, str):
            raise ValueError("Parameter 'var_name' has to be a string and not '%s'"%type(var_name))
                
        self.name = name
        self.params = params
        self.var_name = var_name
        self.consumers = []
        for p in self.params:
            if hasattr(p, 'consumers'):
                p.consumers.append(self)

        # Some nodes require a special reader. In that case, they can set the
        # reader
        # FIXME currently is only UCIFastReader supported
        self.reader = None

    def _is_input(self):
        return isinstance(self, Input)

    def __add__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        return Plus(self, other)

    def __radd__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        return Plus(other, self)

    def __sub__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        return Minus(self, other)

    def __rsub__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        return Minus(other, self)

    def __mul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        return ElementTimes(self, other)

    def __rmul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        return ElementTimes(other, self)

    def __matmul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        # NOTE supported in Python 3.5
        return Times(self, other)

    def __rmatmul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = Constant(other)
        # NOTE supported in Python 3.5
        return Times(other, self)

    def __abs__(self):
        return Abs(self)

    def __getitem__(self, so):
        if so.stop == None:
            raise ValueError('The stop index has to be provided')

        if isinstance(so, int):
            return RowSlice(self, so, 1)

        elif isinstance(so, slice):
            if so.step not in {1, None}:
                raise ValueError("RowSlice does not support strides")

            start = so.start or 0

            return RowSlice(self, start, so.stop - start)

    # TODO more __operators__

    def _get_cntk_param_string(self, param_variable_names=None):
        return ", ".join(param_variable_names)

    def __str__(self):
        return "%s / params=%s" % (self.name, self.params)

    def _param_to_brainscript(self, p_name, p_value):
        if isinstance(p_value, bool):
            p_value = str(p_value).lower()
        elif isinstance(p_value, str):
            p_value = "'%s'" % p_value
        elif type(p_value) in [list, tuple]:
            # FIXME here we assume that all dims are of TensorShape
            if p_name in ['dims', 'inputs']:
                p_value = ":".join(v for v in p_value)
            else:
                raise ValueError('Sequence initialization is only allowed for' +
                                 ' parameters dims and not "%s"' % p_name)
        else:
            p_value = str(p_value)

        if p_name in self.params_with_defaults:
            param = '%s=%s' % (p_name, p_value)
        else:
            param = p_value

        return param

    def _to_description_unroll(self, desc, unrolled_nodes, inputs,
            readers, node_counter=0):
        param_variable_names = []
        if self.params:
            for p_name in self.params:
                p_value = self.__dict__[p_name]
                if hasattr(p_value, '_to_description') and p_name or \
                        p_name == 'inputs':
                        # TODO this is under the assumption that RowStack's
                        # inputs parameter gets a tuple of inputs

                    if p_name == 'inputs' and isinstance(self, RowStack):
                        # Special treatment for special operator.
                        # Used like RowStack(v0:v1:v2)
                        inputs_param = p_value
                    else:                        
                        inputs_param = [p_value]

                    input_nodes_vars = []
                    for p_value in inputs_param:
                        if p_value in unrolled_nodes:
                            # we have seen this node already, so just retrieve its
                            # name
                            child_var = unrolled_nodes[p_value]
                        else:
                            child_var, node_counter, child_desc = p_value._to_description_unroll(
                                desc, unrolled_nodes, inputs, readers, node_counter)
                            unrolled_nodes[p_value] = child_var
                        input_nodes_vars.append(child_var)

                    param_variable_names.append(':'.join(input_nodes_vars))
                else:
                    param_variable_names.append(
                        self._param_to_brainscript(p_name, p_value))

        if self.reader:
            readers.append(self.reader)

        if self._is_input():
            inputs.add(self)

        if hasattr(self, 'tag') and 'tag' not in self.params:
            param_variable_names.append("tag='%s'" % self.tag)

        self.var_name = self.var_name or "v%i" % node_counter
        node_counter += 1

        params = self._get_cntk_param_string(param_variable_names)

        line = "%s = %s(%s)" % (self.var_name, self.name, params)
        desc.append(line)

        return self.var_name, node_counter, desc

    def _to_description(self):
        unrolled_nodes = {}
        inputs=set()
        readers=[]
        var_name, node_counter, desc = self._to_description_unroll(
            desc=[], 
            unrolled_nodes=unrolled_nodes, 
            inputs=inputs,
            readers=readers)

        return var_name, node_counter, desc, len(inputs)>0, readers

    def to_description(self):
        '''
        Generate CNTK configuration for this node including the configuration
        for all dependent child nodes.
        '''
        var_name, node_counter, desc, has_inputs, readers = self._to_description()

        return "\n".join(desc), has_inputs, readers

# importing after defining ComputationNode to work around circular imports
from cntk.ops import *
import cntk.ops # to have a separate namespace when we want to override below

# redefine some operators to work with NumPy and sequences as input

def _get_input_node(value):
    # FIXME We need to better manage the context. How can we get hold
    # of the overall context without having to always pass it
    # explicitly?
    if len(value.shape)>2:
        raise ValueError('NumPy arrays with more than 2 dimensions are not yet supported')

    from cntk.context import get_context 
    import tempfile
    # We have to use NamedTemporaryFile and close it, because when using the
    # obvious first choice, mkstemp(), would later fail in cntk.exe because the
    # file would still be locked.
    f = tempfile.NamedTemporaryFile(prefix='_input_', suffix='.txt', dir=get_context().directory, delete = False)
    f.close()

    #import ipdb;ipdb.set_trace()
    from cntk.reader import NumPyReader
    input_node = Input(2)
    input_node.reader = NumPyReader(value, f.name)
    input_node.reader.add_input(input_node, 0,2)

    return input_node

def Constant(value, **kw):
    '''
    Defining Constant as a factor override that creates either a Constant()
    operator or an Input() operator based on the type of the `value`.

    In case the `value` is a scalar, a normal CNTK Constant() operator is
    returned.

    In case the `value` is a list of NumPy arrays, a CNTK Input() operator is
    returned, interpreting every element as a sequence of tensors.
      
    In case the `value` is a NumPy array, a CNTK Input() operator is
    returned, interpreting it as a dense tensor.

    '''

    if np.isscalar(value):
        return cntk.ops.Constant(value, **kw)
    else:
        if isinstance(value, list):
            if len(value)!=1:
                raise ValueError('Right now, only instances of length 1 are allowed')

            return _get_input_node(value[0])

        elif isinstance(value, np.ndarray):
            return _get_input_node(value)

        else:
            raise ValueError('value type is not supported: %s'%type(value))

