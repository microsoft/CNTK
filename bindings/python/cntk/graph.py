# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def dfs_walk(node, visitor):
    '''
    Generic function that walks through the graph starting at ``node`` and
    uses function ``visitor`` on each node to check whether it should be
    returned.
 
    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns ``True`` if that node should be returned.

    Returns:
        List of nodes, for which ``visitor`` was ``True``
    '''
    stack = [node]
    accum = []
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        try:
            # Function node
            stack.extend(node.root_function.inputs)
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

        if visitor(node):
            accum.append(node)

        visited.add(node)

    return accum

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
    return dfs_walk(node, lambda x: x.name == node_name)

