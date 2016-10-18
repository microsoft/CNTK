# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def dfs_walk(node, visitor, accum):
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visitor, accum)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visitor, accum)

    if visitor(node):
        accum.append(node)

def visit(root_node, visitor):
    nodes = []
    dfs_walk(root_node, visitor, nodes)
    return nodes

def find_nodes_by_name(root_node, node_name):
    '''
    Return a list of nodes having the given name

    Args:
        root_node (node in the graph): root node from where the search should
         start
        node_name (`str`): name of the nodes

    Returns:
       a list of nodes having the given name

    '''
    return visit(root_node, lambda x: x.name == node_name)
