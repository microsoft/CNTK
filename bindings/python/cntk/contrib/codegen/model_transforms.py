# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Classes and functions for building a NetworkX graph of a CNTK model.
"""

from cntk import *
from cntk import cntk_py
from pdb import set_trace
import cntk.variables
import networkx as nx
import numpy as np

class Node:
    '''
    Node in the graph. Mostly a wrapper with additional information about the model node.
    '''
    def __init__(self, graph, id, shape, quantize=False):
        self.graph = graph
        self._id = id
        self._shape = shape
        self.quantize = quantize

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def cm_shape(self):
        '''
        Column major shape.
        '''
        if len(self.shape) == 2:
            return (str(self.shape[0]), str(self.shape[1]))
        elif len(self.shape) == 1:
            return (str(self.shape[0]),)
        elif len(self.shape) == 0:
            return (str(1),)
        else:
            raise ValueError('Unexpected shape encountered, only 1 and 2D are currently supported %s' % self.shape)

    @property
    def predecessors(self):
        '''
        Utility function to return all predecessors of a node from the graph honouring the order.
        The order is taken from the 'order' attribut of the corresponding (predecessor, node) edge.
        Args:
            graph(list): nx model graph
            node(str): uid of the model node for which to find the predecessors
     
        Returns:
            list of predecessor nodes in the order according to the 'order' attribute of the edges.
        '''
        predecessors = self.graph.predecessors(self)     
        ordered = [(p, self.graph.get_edge_data(p, self)['order']) for p in predecessors]
        ordered = sorted(ordered, key=lambda o: o[1])
        return [o[0] for o in ordered]

    @property
    def successors(self):
        return self.graph.successors(self)

    def __hash__(self):
        return hash(self.id)

class ModelNode(Node):
    '''
    Node in the graph. Mostly a wrapper with additional information about the model node.
    '''
    def __init__(self, graph, model_node, original_model_node=None):
        super(ModelNode, self).__init__(graph, model_node.uid, model_node.shape)
        self.model_node = model_node
        self.original_model_node = original_model_node

    @property
    def original(self):
        return self.original_model_node

    @original.setter
    def original(self, value):
        self.original_model_node = value

    @property
    def model(self):
        return self.model_node

    @property
    def name(self):
        return self.model_node.name

    @property
    def op_name(self):
        return self.model_node.op_name

    @property
    def is_primitive(self):
        return self.model_node.is_primitive

    @property
    def is_variable(self):
        return isinstance(self.model_node, cntk.variables.Variable)

    @property
    def type(self):
        if self.dtype == np.float32:
            return 'float'
        elif self.dtype == np.float64:
            return 'double'
        else:
            raise ValueError('Unsupported type %s' % self.dtype)

    @property
    def is_function(self):
        return isinstance(self.model_node, cntk_py.Function)

    @property
    def is_output(self):
        return self.is_variable and self.model_node.is_output

    @property
    def is_input(self):
        return self.model_node.is_input

    @property
    def is_placeholder(self):
        return self.model_node.is_placeholder

    @property
    def is_parameter(self):
        return self.model_node.is_parameter

    @property
    def is_constant(self):
        return self.model_node.is_constant

    @property
    def dynamic_axes(self):
        return self.model_node.dynamic_axes

    @property
    def shape(self):
        return self.model_node.shape

    @property
    def dtype(self):
        return self.model_node.dtype

    @property
    def op_name(self):
        return self.model_node.op_name

class ModelToGraphConverter:
    '''
    Converts a CNTK model to a NX graph. Eliminates block functions.
    Each node in the graph contains the uid of the corresponding entity of the model.
    The original model node is stored in the attached 'data' attribute.
    Because the order of edges is not defined for NX graphs, each edge has an additional 
    numeric attribute 'order' that corresponds to the order of parameters.
    TODO: Currently does not handle nested blocks
    '''
    def __init__(self):
        super(ModelToGraphConverter, self).__init__()

    def convert(self, model):
        '''
        Converts CNTK model to the NX graph.
        Args:
            model: CNTK model

        Returns:
            NX graph that corresponds to the model.
        '''
        from cntk import cntk_py
        outputs = []
        if isinstance(model, cntk_py.Function):
            if model.is_composite:
                model = model.root_function
            outputs.extend(model.outputs)
        elif isinstance(model, cntk_py.Variable):
            outputs = [model]
        else:
            raise ValueError('Model is expected to be an output variable or a function')

        g = nx.OrderedDiGraph()
        visited = {}
        for output in model.outputs:
            self._convert(g, output, None, 0, set(), {})
        return g

    def get_node_by_id(self, g, id):
        for n in g.nodes():
            if n.id == id:
                return n
        return None

    def _convert(self, g, node, child, order, visited, placeholder_mapping):
        from cntk import cntk_py
        is_function = isinstance(node, cntk_py.Function)

        # First thing - add an edge between the child and the node
        # skipping blocks if needed
        # BUGBUG: Two nested blocks is probably not supported?
        if child is not None:
            if not self.get_node_by_id(g, child.uid):
                g.add_node(ModelNode(g, child))
            cur = dict(node.block_outputs_mapping)[child] if is_function and node.is_block else node
            if not self.get_node_by_id(g, cur.uid):
                g.add_node(ModelNode(g, cur))

            # Unfortunately, nx does not preserve the order of edges, so we need
            # to remember the order in which edges are added because order of parameters
            # to a CNTK function matters.
            g.add_edge(self.get_node_by_id(g, cur.uid), self.get_node_by_id(g, child.uid), order=order)

        if node.uid in visited:
            return

        visited.add(node.uid)

        if is_function:
            if node.is_block:
                placeholder_mapping.update(node.block_arguments_mapping)
                outputs_mapping = dict(node.block_outputs_mapping)
                inner_output_variable = outputs_mapping[child]
                self._convert(g, inner_output_variable, child, order, visited, placeholder_mapping)
            elif node.is_primitive:
                for order, i in enumerate(node.inputs):
                    i = placeholder_mapping[i] if i.is_placeholder else i
                    self._convert(g, i, node, order, visited, placeholder_mapping)
            else:
                set_trace()
                raise ValueError("Unexpected function node type %s" % node)

        elif node.is_parameter or node.is_constant or node.is_input:
            pass
        elif node.is_output:
            self._convert(g, node.owner, node, order, visited, placeholder_mapping)
        elif node.is_placeholder:
            actual_node = placeholder_mapping[node]
            self._convert(g, actual_node, order, visited, placeholder_mapping)
        else:
            set_trace()
            raise ValueError("Unexpected node type %s" % node)


def remove_intermediate_output_nodes(graph):
    '''
    Utility function to remove intermediate output variables from the graph.
    Only actual outputs of the graph are preserved.
    Args:
        graph: nx model graph
    '''
    # Remove all output variables in the graph
    # except for the actual end outputs (that have no children).
    removed = True
    while removed:
        removed = False
        for node in graph.nodes():
            if not node.is_output:
                continue
     
            successors = graph.successors(node)
            if len(successors) == 0: # No successors - actual output
                continue
     
            if len(node.predecessors) != 1:
                raise ValueError("Unexpected output node with no ancestors")

            p = node.predecessors[0] 
            for s in successors:
                graph.add_edge(p, s, label=node.id, order=graph.get_edge_data(node, s)['order'])

            graph.remove_node(node)
            removed = True

class PastStateSelectorNode(ModelNode):
    def __init__(self, graph, model_node):
        super(PastStateSelectorNode, self).__init__(graph, model_node)

def split_past_values(graph):
    '''
    Splits each past value into (input + state + state_selector) and (output) nodes.
    For all these nodes the attribute 'original' points to the original past value node.
    Args:
        graph: nx model graph
    '''
    nodes = list(graph.nodes())
    for node in nodes:
        if not node.is_function or node.op_name != 'PastValue':
            continue
        external_output = ModelNode(graph, cntk.output_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name='Input' + node.id), node.model)
        external_input = ModelNode(graph, cntk.input_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name='Output' + node.id), node.model)

        graph.add_node(external_input)
        graph.add_node(external_output)

        if len(node.predecessors) != 2:
            raise ValueError('Past value is expected to have two operands')

        state = node.predecessors[1]
        if not state.is_constant:
            raise ValueError('Currently only constant initial state of past values is supported')
        state.original = node.model

        state_selector = PastStateSelectorNode(graph, state.original)
        graph.add_node(state_selector)

        graph.add_edge(external_input, state_selector, order = 0)
        graph.add_edge(state, state_selector, order = 1)

        for successor in node.successors:
            graph.add_edge(state_selector, successor, order = graph.get_edge_data(node, successor)['order'])

        graph.add_edge(node.predecessors[0], external_output, order = 0)
        graph.remove_node(node)
