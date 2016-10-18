# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def dfs_walk(node, visitor, accum, visited):
    '''
    Generic function to walk the graph.

    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns `True` if that node should be returned.
        accum (`list`): accumulator of nodes while traversing the graph
        visited (`set`): set of nodes that have already been visited.
         Initialize with empty set.
    '''
    if node in visited:
        return
    visited.add(node)
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visitor, accum, visited)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visitor, accum, visited)

    if visitor(node):
        accum.append(node)

def visit(node, visitor):
    '''
    Generic function that walks through the graph starting at `node` and
    applies function `visitor` on each of those.

    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns `True` if that node should be returned.

    Returns:
        List of nodes, for which `visitor` was `True`
    '''
    nodes = []
    dfs_walk(node, visitor, nodes, set())
    return nodes

def find_nodes_by_name(node, node_name):
    '''
    Finds nodes in the graph starting from `node` and doing a depth-first
    search.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        List of nodes having the specified name
    '''
    return visit(node, lambda x: x.name == node_name)

