# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# TODO: Formalize the naming convention and the transformation rules from
# C++ to Python

from abc import ABCMeta, abstractmethod
import numpy as np

from .utils import MODEL_INDENTATION
from .utils import aggregate_readers
from .utils import with_metaclass, is_string


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

    def is_input(self):
        '''
        Returns: True if this node is an input node.
        '''
        return isinstance(self, InputComputationNodeBase)

    # operator overload for (+) where self is the left operand
    def __add__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        return ops.plus(self, other)

    # operator overload for (+) where self is the right operand
    def __radd__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        return ops.plus(other, self)

    # operator overload for (-) where self is the left operand
    def __sub__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        return ops.minus(self, other)

    # operator overload for (-) where self is the right operand
    def __rsub__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        return ops.minus(other, self)

    # operator overload for (*) where self is the left operand
    def __mul__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        return ops.element_times(self, other)

    # operator overload for (*) where self is the right operand
    def __rmul__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        return ops.element_times(other, self)

    # operator overload for (@) where self is the left operand
    def __matmul__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        # NOTE supported in Python 3.5
        return ops.times(self, other)

    # operator overload for (@) where self is the right operand
    def __rmatmul__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        # NOTE supported in Python 3.5
        return ops.times(other, self)

    # operator overload for (\) where self is the left operand
    def __truediv__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        self.__div__ = self.__truediv__
        return ops.element_divide(self, other)

    # operator overload for (\) where self is the right operand
    def __rtruediv__(self, other):
        if not isinstance(other, ComputationNode):
            other = ops.constant(other)
        self.__rdiv__ = self.__rtruediv__
        return ops.element_divide(other, self)

    # Python2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __abs__(self):
        return ops.abs(self)

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

    def _param_to_brainscript(self, p_name, p_value, is_node=False):
        if isinstance(p_value, bool):
            p_value = str(p_value).lower()
        elif is_string(p_value) and not is_node:
            p_value = "'%s'" % p_value
        elif type(p_value) in [list, tuple]:
            # FIXME here we assume that all dims are of TensorShape
            if p_name in ['dims', 'inputs', 'z']:
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
                                    child_var, dep_inputs = reconciled_cache[
                                        (pv, first_unreconciled_input)]
                                else:
                                    unrec_pv = pv
                                    from .ops.cntk1 import ReconcileMBLayout
                                    pv = ReconcileMBLayout(
                                        unrec_pv, first_unreconciled_input)
                                    child_var, node_counter, child_desc, dep_inputs = pv._to_config_recursively(
                                        input_reader, desc, unrolled_nodes, inputs, readers,
                                        dep_inputs, node_counter,
                                        reconciled_cache)
                                    reconciled_cache[
                                        (unrec_pv, first_unreconciled_input)] = pv.var_name, dep_inputs

                                unrolled_nodes[pv] = child_var, dep_inputs

                            dep_inputs = child_dep_inputs

                        input_nodes_vars.append(child_var)

                    param_variable_names.append(self._param_to_brainscript(p_name,
                        _tuple_to_cntk_shape(input_nodes_vars), True))
                else:
                    if self._is_forward_ref(p_name, p_value):
                        # We have a forward reference to a node that will be
                        # later on defined. p_value is the var_name of the
                        # later defined node.
                        param_variable_names.append(self._param_to_brainscript
                        (p_name, p_value, True))
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

        if self.is_input():
            inputs.add(self)

            num_readers_mapping = 0

            if self.reader:
                node_reader = self.reader
                num_readers_mapping += 1

            if input_reader:
                if self in input_reader:
                    node_reader = input_reader[self]
                    num_readers_mapping += 1
                if has_var_name and self.var_name in input_reader:
                    node_reader = input_reader[self.var_name]
                    num_readers_mapping += 1

            if num_readers_mapping == 0:
                raise RuntimeError(
                    "No reader was found for input node: {0}".format(self.var_name))

            if num_readers_mapping > 1:
                raise RuntimeError(
                    "More than one reader found for input node: {0}".format(self.var_name))

            readers.add(node_reader)

        params = self._get_cntk_param_string(param_variable_names)

        line = ' ' * MODEL_INDENTATION + \
            "%s = %s(%s)" % (self.var_name, self.name, params)
        desc.append(line)

        if self.is_input():
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

        return var_name, node_counter, desc, inputs, readers, dep_inputs

    def to_config(self):
        '''
        Generate CNTK configuration for this node including the configuration
        for all dependent child nodes.
        '''
        var_name, node_counter, desc, inputs, readers, dep_inputs = \
            self._to_config(input_reader={},
                            description=[],
                            unrolled_nodes={},
                            inputs=set(),
                            readers=set(),
                            dep_inputs=tuple(),
                            node_counter=0,
                            reconciled_cache={})

        return "\n".join(desc), inputs, aggregate_readers(readers)


class InputComputationNodeBase(with_metaclass(ABCMeta, ComputationNode)):

    '''
    Base class for all non-image input nodes nodes and operators. 
    '''
    pass


class ImageInputComputationNodeBase(with_metaclass(ABCMeta, ComputationNode)):

    '''
    Base class for all image input nodes nodes and operators. 
    '''
    pass

def eval(node):        
    """
    It evaluates a node that has taken a numpy array as input. Note that sequences
    are not supported yet by this method
    
    Examples:
        Plus with two matrices
        >>> print (cntk.eval(cntk.ops.plus([[-30.,40.], [1.,2.]], [[-30.,40.], [1.,2.]])))
        #   [array([[[-60., 80.], [2., 4.]]])]
        
        Times with broadcast of a scalar over a matrix
        >>> print (cntk.eval(cntk.ops.element_times([[-30.,40.], [1.,2.]], 5)))
        #   [array([[[-150., 200.], [5., 10.]]])]        

    Args:
        node (cntk.graph.ComputationNode): the node to evaluate        
    Returns:
        numpy array containing the result
    """    
    
    from cntk.context import get_context        
    # call a helper method to get a context
    ctx = get_context()    
    first = True    
    
    # the params are passed as arryas e.g. plus([1,2], [3,4]), we need to 
    # wrap them with input and parameter nodes
    if node.params:
        for p in node.params:
            if p in node.inputs:
                val = getattr(node, p)
                # one param needs to be an Input() node. This is being fixed in 
                #CNTK we will remove this workaround onces we can evaluate a 
                #network with no inputs
                if first:
                    if not isinstance(val, list):                
                        # inputs have the outmost dimension for sequences
                        val = [val]
                    setattr(node, p, input_reader([val], False, var_name=p, alias=p))            
                    first = False
                else:
                    setattr(node, p, constant(getattr(node, p), name=p))
    return ctx.eval(node)

class LazyInput(InputComputationNodeBase):

    '''
    Lazy reader that takes an NumPy array and serializes it to disk only when
    the complete graph is specified. This is necessary in case of multiple
    inputs, because they have to reside in the same file.

    Note:
        All readers of this type need to have the exact same number of samples,
        as they will be aligned by the first index.

    Note:
        This class will be deprecated once the reader bundlers have arrived in
        CNTK.

    Args:
        value (ndarray): the data to be serialized.
        input_alias (str): a short name for the input, it is how inputs are
        referenced in the data files. If not provided, it will be automatically
        assigned.
        has_dynamic_axis (bool): whether the tensor already has the data
        packaged as sequences. If not, it will be wrapped again in a sequence of
        length 1.

    '''

    def __init__(self, value, input_alias=None, has_dynamic_axis=True):
        self.value = value
        self.input_alias = input_alias
        self.has_dynamic_axis = has_dynamic_axis

# At the bottom to avoid circular import
from . import ops
