# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Utility functions for CNTK model represented as NetworkX graph.
"""

from cntk import *
from cntk import cntk_py
import networkx as nx
import matplotlib.pyplot as plt
import os

def nx_plot(g, filename):
    '''
    Utility function for visualizing CNTK model represented as a NetworkX graph.
    The output will contain the full information about nodes, including their uids.
    Args:
        g(list): model graph
        filename (str): the name of the ouptut file
    '''
    if filename:
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in ('.svg', '.pdf', '.png', '.dot'):
            raise ValueError('only file extensions ".svg", ".pdf", ".png", and ".dot" are supported')
    else:
        raise ValueError('Please specify the output filename')

    if filename:
        try:
            import pydot_ng as pydot
        except ImportError:
            raise ImportError("Unable to import pydot_ng, which is required to output SVG, PDF, PNG, and DOT format.")

        # initialize a dot object to store vertices and edges
        dot_object = pydot.Dot(graph_name="network_graph", rankdir='TB')
        dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                     style='filled',
                                     fillcolor='lightgray',
                                     height=.85, width=.85, fontsize=12)
        dot_object.set_edge_defaults(fontsize=10)

    primitive_op_map = {
        'Plus': '+',
        'Minus': '-',
        'ElementTimes': '*',
        'Times': '@',
    }
    dot_nodes = {}  # [uid] -> dot node

    def node_desc(node):
        return '<' + node.uid + '>'

    def shape_desc(node):
        dyn_axes = node.dynamic_axes
        dyn = '[#' + ',*' * (len(dyn_axes) - 1) + ']' if len(dyn_axes) > 0 else ''
        return dyn + str(node.shape)

    # add current Function node
    def create_node(node):
        if node.uid in dot_nodes: # dot node already exists
            raise ValueError('Node is already created')

        if node.is_primitive and not node.is_block and len(node.outputs) == 1 and node.output.name == node.name:     # skip the node name if redundant
            op_name = primitive_op_map.get(node.op_name, node.op_name)
            render_as_primitive = len(op_name) <= 4
            size = 0.4 if render_as_primitive else 0.6
            cur_node = pydot.Node(node.uid[:6], label='"' + op_name + node_desc(node) + '"',
                                  shape='ellipse'  if render_as_primitive else 'box',
                                  fixedsize='true' if render_as_primitive else 'false', height=size, width=size,
                                  fontsize=20  if render_as_primitive and len(op_name) == 1 else 12 ,
                                  penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
        else:
            f_name = '\n' + node.name + '()' if node.name else ''
            cur_node = pydot.Node(node.uid, label='"' + node.op_name + f_name + node_desc(node) + '"',
                                  fixedsize='true', height=1, width=1.3,
                                  penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
        dot_object.add_node(cur_node)
        dot_nodes[node.uid] = cur_node
        return cur_node

    # Add all nodes
    for n in g.nodes():
        node = g.node[n]['data']
        from cntk import cntk_py
        if isinstance(node, cntk_py.Function):
            # add current node
            cur_node = create_node(node)
            dot_object.add_node(cur_node)
            continue
        elif node.is_input:
            shape = 'invhouse'
            color = 'yellow'
        elif node.is_placeholder:
            shape = 'invhouse'
            color = 'grey'
        elif node.is_parameter:
            shape = 'diamond'
            color = 'green'
        elif node.is_constant:
            shape = 'rectangle'
            color = 'lightblue'
        else: # is_output
            shape = 'invhouse'
            color = 'grey'

        name = 'Parameter' if node.is_parameter else 'Constant' if node.is_constant else 'Input' if node.is_input else 'Placeholder' if node.is_placeholder else 'Output'
        if node.name:
            if name == 'Parameter':  # don't say 'Parameter' for named parameters, it's already indicated by being a box
                name = node.name
            else:
                name = name + '\n' + node.name
        name += '\n' + shape_desc(node) + '\n' + node_desc(node)
        if node.is_input or node.is_placeholder: # graph inputs are eggs (since dot has no oval)
            cur_node = pydot.Node(node.uid, shape='egg', label=name, fixedsize='true', height=1, width=1.3, penwidth=4) # wish it had an oval
        elif not node.name and node.is_constant and (node.shape == () or node.shape == (1,)): # unnamed scalar constants are just shown as values
            cur_node = pydot.Node(node.uid, shape='box', label=str(node.as_constant().value), color='white', fillcolor='white', height=0.3, width=0.4)
        else:                                      # parameters and constants are boxes
            cur_node = pydot.Node(node.uid, shape='box', label=name, height=0.6, width=1)

        dot_object.add_node(cur_node)
        dot_nodes[node.uid] = cur_node

    # Add edges
    for n in g.nodes():
        node = g.node[n]['data']
        successors = g.successors(node.uid)
        for successor in successors:
            label = node.name if node.name else node.uid # the Output variables have no name if the function has none
            label += '\n' + shape_desc(node) + '\n' + node_desc(node)

            dot_object.add_edge(pydot.Edge(dot_nodes[node.uid], dot_nodes[successor], label=label))

    if filename:
        if suffix == '.svg':
            dot_object.write_svg(filename, prog='dot')
        elif suffix == '.pdf':
            dot_object.write_pdf(filename, prog='dot')
        elif suffix == '.png':
            dot_object.write_png(filename, prog='dot')
        else:
            dot_object.write_raw(filename)
