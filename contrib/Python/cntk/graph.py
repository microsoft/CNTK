# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

#TODO: Formalize the naming convention and the transformation rules from C++ to Python

from abc import ABCMeta, abstractmethod
import numpy as np

# Workaround until we have switched to Anaconde with scipy support
#import scipy.sparse as sparse 
class sparse(object):
    @staticmethod
    def issparse(obj):
        return hasattr(obj, 'todense')

from .utils import MODEL_INDENTATION
from .utils import aggregate_readers

def _tuple_to_cntk_shape(shape):
    return ':'.join(str(v) for v in shape)


class ComputationNode(object):

    '''
    Base class for all nodes and operators. Provides a NumPy-like interface
    with operators that are converted to CNTK operators.
    '''

    def __init__(self, name, params=None, var_name=None, reader=None):
        if not isinstance(name, str):
            raise ValueError(
                "Parameter 'name' has to be a string and not '%s'" % type(name))
        if var_name is not None and not isinstance(var_name, str):
            raise ValueError(
                "Parameter 'var_name' has to be a string and not '%s'" % type(var_name))

        self.name = name
        self.params = params
        self.var_name = var_name
        self.consumers = []
        for p in self.params:
            if hasattr(p, 'consumers'):
                p.consumers.append(self)

        # Create sub-class construtor and more these 
        self.reader = None
        self.has_sequence_dimensions = False

    def _is_input(self):
        return isinstance(self, InputComputationNodeBase)

    # operator overload for (+) where self is the left operand
    def __add__(self, other):
        if not isinstance(other, ComputationNode):            
            other = constant(other)
        return Plus(self, other)

    # operator overload for (+) where self is the right operand
    def __radd__(self, other):
        if not isinstance(other, ComputationNode):            
            other = constant(other)
        return Plus(other, self)
    
    # operator overload for (-) where self is the left operand
    def __sub__(self, other):
        if not isinstance(other, ComputationNode):
            other = constant(other)
        return Minus(self, other)

    # operator overload for (-) where self is the right operand
    def __rsub__(self, other):
        if not isinstance(other, ComputationNode):            
            other = constant(other)
        return Minus(other, self)

    # operator overload for (*) where self is the left operand
    def __mul__(self, other):
        if not isinstance(other, ComputationNode):
            other = constant(other)
        return ElementTimes(self, other)

    # operator overload for (*) where self is the right operand
    def __rmul__(self, other):
        if not isinstance(other, ComputationNode):
            other = constant(other)
        return ElementTimes(other, self)

    # operator overload for (@) where self is the left operand
    def __matmul__(self, other):
        if not isinstance(other, ComputationNode):
            other = constant(other)
        # NOTE supported in Python 3.5
        return Times(self, other)

    # operator overload for (@) where self is the right operand
    def __rmatmul__(self, other):
        if not isinstance(other, ComputationNode):            
            other = constant(other)
        # NOTE supported in Python 3.5
        return Times(other, self)

    # operator overload for (\) where self is the left operand
    def __truediv__(self, other):
        if not isinstance(other, ComputationNode):
            other = constant(other)
        self.__div__ = self.__truediv__
        return ElementDivide(self, other)

    # operator overload for (\) where self is the right operand
    def __rtruediv__(self, other):
        if not isinstance(other, ComputationNode):
            other = constant(other)
        self.__rdiv__ = self.__rtruediv__
        return ElementDivide(other, self)

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
                p_value = _tuple_to_cntk_shape(p_value)
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


    def _is_forward_ref(self, p_name, p_value): 
        '''
        Although the unrolled graph is a DAG, when we specify recurrence we
        naturally have loops. We can resolve this by using forward references.
        This method is checking whether the particular name and value of this
        instance are actually one of those forward references.
        '''
        is_loop_node = self.name in ('Delay', 'PastValue', 'FutureValue')
        return is_loop_node and p_name == 'input' and isinstance(p_value, str)

    def _to_config_recursively(self, input_reader, desc, unrolled_nodes, inputs,
                               readers, dep_inputs, node_counter, reconciled_cache):

        param_variable_names = []
        # In case we have multiple unreconciled inputs, we will reconcile each
        # of them to the layout of the first input.
        first_unreconciled_input = None
        if self.params:
            for p_name in self.params:
                p_value = self.__dict__[p_name]
                if hasattr(p_value, '_to_config') and p_name or \
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
                    for pv in inputs_param:

                        if pv in unrolled_nodes:
                            # We have seen this node already, so just retrieve its
                            # name.
                            child_var, child_dep_inputs = unrolled_nodes[pv]
                        else:
                            child_var, node_counter, child_desc, child_dep_inputs = pv._to_config_recursively(
                                input_reader, desc, unrolled_nodes, inputs, readers,
                                dep_inputs, node_counter, reconciled_cache)

                            unrolled_nodes[pv] = child_var, dep_inputs

                        # Whenever two unreconciled inputs meet, we need
                        # reconcile them to have the same MB layout. This is a
                        # temporary necessity that should go away with future
                        # CNTK versions.
                        if dep_inputs != child_dep_inputs:

                            if first_unreconciled_input is None:
                                first_unreconciled_input = pv

                            else:
                                if (pv, first_unreconciled_input) in reconciled_cache:
                                    child_var, dep_inputs = reconciled_cache[(pv, first_unreconciled_input)]
                                else:
                                    unrec_pv = pv
                                    pv = ReconcileMBLayout(unrec_pv, first_unreconciled_input)
                                    child_var, node_counter, child_desc, dep_inputs = pv._to_config_recursively(
                                        input_reader, desc, unrolled_nodes, inputs, readers,
                                        dep_inputs, node_counter,
                                        reconciled_cache)
                                    reconciled_cache[(unrec_pv, first_unreconciled_input)] = pv.var_name, dep_inputs

                                unrolled_nodes[pv] = child_var, dep_inputs

                            dep_inputs = child_dep_inputs

                        input_nodes_vars.append(child_var)

                    param_variable_names.append(
                        _tuple_to_cntk_shape(input_nodes_vars))
                else:
                    if self._is_forward_ref(p_name, p_value):
                        # We have a forward reference to a node that will be
                        # later on defined. p_value is the var_name of the
                        # later defined node.
                        param_variable_names.append(p_value)
                    else:
                        param_variable_names.append(
                            self._param_to_brainscript(p_name, p_value))

        if hasattr(self, 'tag') and 'tag' not in self.params:
            param_variable_names.append("tag='%s'" % self.tag)
        
        has_var_name = False
        if (self.var_name):
            has_var_name = True
            
        self.var_name = self.var_name or "v%i" % node_counter
        node_counter += 1

        if self._is_input():
            inputs.add(self)
            
            if self.reader:                
                node_reader = self.reader
            elif self in input_reader:
                node_reader = input_reader[self]
            elif has_var_name and self.var_name in input_reader:
                node_reader = input_reader[self.var_name]
            else:
                raise RuntimeError("No reader was found for input node: {0}".format(self.var_name))

            readers.add(node_reader._to_aggregate_form(self))

        params = self._get_cntk_param_string(param_variable_names)

        line = ' ' * MODEL_INDENTATION + \
            "%s = %s(%s)" % (self.var_name, self.name, params)
        desc.append(line)

        if self._is_input():
            dep_inputs += (self.var_name,)

        return self.var_name, node_counter, desc, dep_inputs

    def _to_config(self, input_reader, description, unrolled_nodes, inputs, readers,
            dep_inputs, node_counter, reconciled_cache):
        '''
        Helper method to generate the CNTK configuration for this node.
        '''

        var_name, node_counter, desc, dep_inputs = self._to_config_recursively(
            input_reader,
            description,
            unrolled_nodes=unrolled_nodes,
            inputs=inputs,
            readers=readers, 
            dep_inputs=dep_inputs,
            node_counter=node_counter, 
            reconciled_cache=reconciled_cache)

        return var_name, node_counter, desc, len(inputs) > 0, readers, dep_inputs

    def to_config(self):
        '''
        Generate CNTK configuration for this node including the configuration
        for all dependent child nodes.
        '''
        var_name, node_counter, desc, has_inputs, readers, dep_inputs = \
            self._to_config(input_reader={},
					description=[], 
                    unrolled_nodes={},
                    inputs=set(),
                    readers=set(), 
                    dep_inputs=tuple(), 
                    node_counter=0,
                    reconciled_cache={})

        return "\n".join(desc), has_inputs, aggregate_readers(readers)


class InputComputationNodeBase(ComputationNode, metaclass=ABCMeta):
    '''
    Base class for all non-image input nodes nodes and operators. 
    '''
    pass


class ImageInputComputationNodeBase(ComputationNode, metaclass=ABCMeta):

    '''
    Base class for all image input nodes nodes and operators. 
    '''
    pass

# importing after defining ComputationNode to work around circular imports
from cntk.ops.cntk1 import *
# to have a separate namespace when we want to override below
from cntk.ops import cntk1 as cntk1_ops
from .reader import CNTKTextFormatReader

# redefine some operators to work with NumPy and sequences as input


def _dense_to_str(data):
    return ' '.join(data.ravel(order='F').astype(np.str))

def _sparse_to_str(data):
    # return ' '.join('%s:%s'%(k,data[k]) for k in sorted(data.items()))
    raise NotImplementedError


def _tensor_to_text_format(idx, alias, tensor, has_sequence_dimension=True):
    '''
    Converts a NumPy array representing tensor of one input into a format that
    is readable by `CNTKTextReader`.

    :param `alias`: alias to be used in the temporary file
    :param `tensor`: a NumPy array having sequence as its innermost dimension
    '''
    if not alias:
        raise ValueError('alias is missing')

    if isinstance(tensor, np.ndarray):
        to_str = _dense_to_str
    elif sparse.issparse(tensor):
        raise ValueError('sparse is not yet supported')
        #to_str = _sparse_to_str
    else:
        raise ValueError('sequence elements have to be of type numpy.ndarray' +
                ' (dense) or dictionary (sparse), you gave "%s"' % \
                str(type(tensor)))

    if has_sequence_dimension:
        num_seq_elements = tensor.shape[0]
        lines = []
        for seq_idx in range(0, num_seq_elements):
            lines.append('%i\t|%s %s'%(idx, alias, to_str(tensor[seq_idx])))

        return '\n'.join(lines)
    else:
        return '%i\t|%s %s'%(idx, alias, to_str(tensor))


def _get_constant_node(value, **kw):
    '''
    This function creates a node that represents `value` as a constant tensor
    in the graph.

    To be as generic as possible, we 
     - flatten the data 
     - initialize a ParameterTensor operator with it
     - ensure that the graph does not backprob to it.  
     - Finally we to reshape it.
    '''

    # FIXME We need to better manage the context. How can we get hold
    # of the overall context without having to always pass it
    # explicitly?

    if isinstance(value, list) or np.isscalar(value):
        value = np.asarray(value)

    if sparse.issparse(value):
        raise ValueError('only dense data is supported')

    param_shape = value.shape if value.shape else (1,)
    literal_shape = (param_shape[0], np.multiply.reduce(param_shape[1:]))
    
    literal_array = np.reshape(value, literal_shape)        

    from io import BytesIO
    s = BytesIO()    
    np.savetxt(s, literal_array, '%.4f')

    # TODO switch to ConstantTensor once it is in the core.bs file
    node = cntk1_ops.ParameterTensor(
        dims=param_shape,
        learningRateMultiplier=0.0,
        init='fromLiteral',
        initFromLiteral=s.getvalue().decode(),
        **kw)

    return node


def _get_input_node(list_of_tensors, has_sequence_dimension, **kw):
    '''
    :param list_of_tensors: list of tensors potentially having sequences of
    different lengths.
    '''

    # FIXME We need to better manage the context. How can we get hold
    # of the overall context without having to always pass it
    # explicitly?

    from cntk.context import get_context
    import tempfile

    # We have to use NamedTemporaryFile and close it, because the obvious first
    # choice, mkstemp(), would later fail in cntk.exe because the file would
    # still be locked.
    tf = tempfile.NamedTemporaryFile(prefix='_input_', suffix='.txt',
                                     dir=get_context().directory, delete=False)
    tf.close()

    if 'alias' in kw:
        alias = kw['alias']
        del kw['alias']  # don't confuse with constructor's parameters
    else:
        # TODO make sure we don't have clashes
        alias = '_I_%i' % np.random.randint(1000)

    shapes = set()
    with open(tf.name, 'w') as f:
        for idx,tensor in enumerate(list_of_tensors):
            if isinstance(tensor, list):
                tensor = np.asarray(tensor)

            if has_sequence_dimension:
                # collecting the shapes ignoring the sequence dimension
                shapes.add(tensor.shape[1:])
            else:
                shapes.add(tensor.shape)

            f.write(_tensor_to_text_format(idx, alias, tensor,
                has_sequence_dimension) + '\n')

    # ignoring the sequence dimension, all shapes should be equal
    if len(shapes)!=1:
        raise ValueError('except for the sequence dimensions all shapes ' +
                'should be the same - instead we have: %s'%(", ".join(str(s) for s in shapes)))

    # shapes now contains only one shape, which has the sequence dimension
    # removed.
    value_shape = shapes.pop()

    cntk_shape = value_shape if value_shape else (1,)
    
    node = cntk1_ops.Input(cntk_shape, **kw)
    node.reader = CNTKTextFormatReader(tf.name, alias)
        
    return node


def is_tensor_list(data):
    '''
    Checks whether the data is a CNTK sequence, which is expressed in Python as
    a list of varying sized NumPy objects.
    '''
    is_list = isinstance(data, list)
    return is_list and len(data) > 0 and isinstance(data[0], np.ndarray)


def is_tensor(data):
    '''
    Checks whether the data is a tensor, i.e. whether it is a NumPy array or a
    list of NumPy arrays.

    :param `data`: data to check
    '''
    if isinstance(data, np.ndarray):
        return True

    if not isinstance(data, list):
        return False

    while len(data) > 0:
        # All but the innermost dimension's values have to be lists
        try:
            data[0][0]
        except:
            # We reached the innermost dimension
            break

        if not isinstance(data[0], list):
            return False

        data = data[0]

    return True


def input_reader(value, has_sequence_dimension=True, **kw):
    '''
    creates an input node.

    Args:
        value: is a list of NumPy tensors.  Currently, only dense tensors are supported. Sparse will come soon by the power of scipy.
        has_sequence_dimension: If True, the outermost dimension is treated as the sequence dimension. If False, it will wrap each sample into its own 1-dimensional array.
        alias: optional the alias to be used when serializing the data into an intermediate file
        kw: will be passed on to the input operator [TODO: specify most commonly used options]

    Returns:
        :class:`cntk.graph.ComputationNode`
    '''
    if is_tensor_list(value) or is_tensor(value):
        return _get_input_node(value, has_sequence_dimension, **kw)
    else:
        raise ValueError('value type is not supported: %s' % type(value))


def constant(value, **kw):
    '''
    creates a constant tensor node.
    '''
    if np.isscalar(value) or is_tensor(value):
        return _get_constant_node(value, **kw)
    else:
        raise ValueError('value type is not supported: %s' % type(value))
