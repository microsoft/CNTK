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

    def __init__(self, op_name, params=None, name=None, reader=None):
        if not isinstance(op_name, str):
            raise ValueError(
                "Parameter 'op_name' has to be a string and not '%s'" %
                type(op_name))
        if name is not None and not isinstance(name, str):
            raise ValueError(
                "Parameter 'name' has to be a string and not '%s'" % type(name))

        self.op_name = op_name
        self.params = params
        self.name = name
        self.consumers = []
        for p in self.params:
            if hasattr(p, 'consumers'):
                p.consumers.append(self)

        # Create sub-class construtor and more these
        self.reader = None

    def _is_input(self):
        '''
        Returns: True if this node is an input node.
        '''
        return isinstance(self, _InputComputationNodeBase)

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

    def __getitem__(self, key):
        if isinstance(key, int):
            # Case 1: e.g. data[3] -> key=3
            return ops.slice(self, key, key+1, axis=0)

        elif isinstance(key, slice):
            # Case 2: e.g. data[2:4] -> key will be a slice object
            if key.step is not None:
                raise TypeError('step argument is not supported')
            if not isinstance(key.stop, int):
                raise TypeError('end index has to be of type int, not "%s"'%type(key.stop))

            if isinstance(key.start, int):
                if key.stop<=key.start:
                    raise ValueError('end index has to be greater than start index')
            return ops.slice(self, key.start or 0, key.stop or 0, axis=0)

        elif isinstance(key, (tuple, list)):
            # Case 3: e.g. data[2:4,1:,1:7] -> key will be an iterable of ints
            # (case 1) or slices (case 2)
            # objects.
            # FIXME: we need to check that len(key) equals the node's rank
            node = self
            for ax_counter, so in enumerate(key):
                if isinstance(so, int):
                    # Proceed as case 1
                    node = ops.slice(node, so, so+1, axis=ax_counter)

                elif isinstance(so, slice):
                    # Proceed as case 2
                    if so.step is not None:
                        raise TypeError('step argument is not supported')
                    if isinstance(so.start, int) and isinstance(so.stop, int):
                        if so.stop<=so.start:
                            raise ValueError('end index has to be greater than start index')
                    if so.start is None and so.stop is None:
                        continue
                    node = ops.slice(node, so.start or 0, so.stop or 0, axis=ax_counter)
                elif isinstance(so, list):
                    # Case 3b: e.g. data[[0],[2,3]] aka "advanced indexing" ->
                    # so = ([0], [2,3])
                    # In NumPy we would have another dimension, but since
                    # data[0].shape != data[[0]].shape == data[[[0]]].shape ==
                    # we decided to have all shapes like data[0] in this case
                    for idx in so:
                        if not isinstance(idx, int):
                            raise IndexError('indices have to be of type int and not "%s"'%type(idx))
                        node = ops.slice(node, idx, idx+1, axis=ax_counter)
                else:
                    raise IndexError('type "%s" is not supported as index'%type(so))

            return node
        else:
            raise TypeError('index must be int or slice, not {}'.format(type(key).__name__))


    # TODO more __operators__

    def _get_cntk_param_string(self, param_variable_names=None):
        return ", ".join(param_variable_names)

    def __str__(self):
        return "%s / params=%s" % (self.op_name, self.params)

    def _param_to_brainscript(self, p_name, p_value, is_node=False):
        if isinstance(p_value, bool):
            p_value = str(p_value).lower()
        elif is_string(p_value) and not is_node:
            p_value = "'%s'" % p_value
        elif type(p_value) in [list, tuple]:
            # FIXME here we assume that all dims are of TensorShape

            if p_name in ['_', 'shape', 'dims', 'inputs', 'z']:
                p_value = _tuple_to_cntk_shape(p_value)
            else:
                msg = 'tuple or list initialization is only allowed for' +\
                      ' parameters shape, dims, inputs, z and _, but' +\
                      ' not "%s". If this is an input or' % p_name + \
                      ' parameter, please use parameter() or input_numpy().'
                raise ValueError(msg)
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
        is_loop_node = self.op_name in ('Delay', 'PastValue', 'FutureValue', 
                'CNTK2.Delay', 'CNTK2.PastValue', 'CNTK2.FutureValue')
        return is_loop_node and p_name == 'input' and isinstance(p_value, str)

    def _to_config_recursively(self, input_map, desc, unrolled_nodes, inputs,
                               node_counter):

        param_variable_names = []
        if self.params:
            for p_name in self.params:
                p_value = self.__dict__[p_name]
                if hasattr(p_value, '_to_config') and p_name or \
                        p_name in ['inputs', '_']:
                        # TODO this is under the assumption that RowStack's
                        # or Splice's inputs parameter gets a tuple of inputs

                    if p_name in ['inputs', '_'] and self.op_name in ['RowStack', 'CNTK2.Splice']:
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
                            child_var = unrolled_nodes[pv]
                        else:
                            child_var, node_counter, child_desc = pv._to_config_recursively(
                                input_map, desc, unrolled_nodes, inputs, 
                                node_counter)

                            unrolled_nodes[pv] = child_var

                        input_nodes_vars.append(child_var)

                    param_variable_names.append(self._param_to_brainscript(p_name,
                        _tuple_to_cntk_shape(input_nodes_vars), True))
                else:
                    if self._is_forward_ref(p_name, p_value):
                        # We have a forward reference to a node that will be
                        # later on defined. p_value is the name of the
                        # later defined node.
                        param_variable_names.append(self._param_to_brainscript
                        (p_name, p_value, True))
                    else:
                        param_variable_names.append(
                            self._param_to_brainscript(p_name, p_value))

        if hasattr(self, 'tag') and 'tag' not in self.params:
            param_variable_names.append("tag='%s'" % self.tag)

        has_name = False
        if (self.name):
            has_name = True

        self.name = self.name or "v%i" % node_counter
        node_counter += 1

        params = self._get_cntk_param_string(param_variable_names)

        line = ' ' * MODEL_INDENTATION + \
            "%s = %s(%s)" % (self.name, self.op_name, params)
        desc.append(line)

        if self._is_input():
            if not self in input_map:
                input_map._add_unmapped(self)
            inputs.add(self)

        return self.name, node_counter, desc

    def _to_config(self, input_map, description, unrolled_nodes, inputs, node_counter):
        '''
        Helper method to generate the CNTK configuration for this node.
        '''

        name, node_counter, desc = self._to_config_recursively(
            input_map,
            description,
            unrolled_nodes=unrolled_nodes,
            inputs=inputs,
            node_counter=node_counter)

        return name, node_counter, desc, inputs

    def _to_config_description(self, input_map=None):
        '''
        Generate CNTK configuration for this node including the configuration
        for all dependent child nodes.

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
        '''
        name, node_counter, desc, inputs = \
            self._to_config(input_map=input_map,
                            description=[],
                            unrolled_nodes={},
                            inputs=set(),
                            node_counter=0)

        return "\n".join(desc), inputs


class _InputComputationNodeBase(with_metaclass(ABCMeta, ComputationNode)):

    '''
    Base class for all non-image input nodes nodes and operators. 
    '''
    pass


class _ImageInputComputationNodeBase(with_metaclass(ABCMeta, ComputationNode)):

    '''
    Base class for all image input nodes nodes and operators. 
    '''
    pass

# At the bottom to avoid circular import
from . import ops
