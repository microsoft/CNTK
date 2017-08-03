# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Classes and functions for expression generation from a CNTK model.
"""
from pdb import set_trace
import networkx as nx
import itertools
import functools
from model_transforms import *

class NodeVisitor:
    '''
    Visitor for the model nodes.
    '''
    def __init__(self, graph):
        '''
        Constructor.
        Args:
            graph: nx graph of the model     
        '''
        super(NodeVisitor, self).__init__()
        self.graph = graph

    def visit(self, nodes):
        '''
        Visits the nodes in order.
        Args:
            nodes(list): uids of nodes for evaluation. Nodes are evaluated
              in the given list order.
        '''
        for node in nodes:
            if not isinstance(node, ModelNode):
                self.visit_node(node)
                continue

            if node.is_function:
                if not node.is_primitive:
                    raise ValueError('Unexpected non primitive function %s' % node)
                self.visit_primitive_function(node)
            elif node.is_parameter:
                self.visit_parameter(node)
            elif node.is_constant:
                self.visit_constant(node)
            elif node.is_input:
                self.visit_input(node)
            elif node.is_output:
                self.visit_output(node)
            else:
                raise ValueError('Unexpected node')

    def visit_parameter(self, node):
        raise NotImplemented()

    def visit_constant(self, node):
        raise NotImplemented()

    def visit_input(self, node):
        raise NotImplemented()

    def visit_output(self, node):
        raise NotImplemented()

    def visit_primitive_function(self, node):
        raise NotImplemented()

    def visit_node(self, node):
        raise NotImplemented()

class EmptyNodeVisitor(NodeVisitor):
    '''
    Empty node visitor for the model nodes.
    '''
    def __init__(self, graph):
        super(EmptyNodeVisitor, self).__init__(graph)

    def visit_parameter(self, node):
        pass

    def visit_constant(self, node):
        pass

    def visit_input(self, node):
        pass

    def visit_output(self, node):
        pass

    def visit_primitive_function(self, node):
        pass

    def visit_node(self, node):
        pass

