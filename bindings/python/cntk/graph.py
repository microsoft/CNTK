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

def build_graph(node,visitor,path):
    import pydot_ng as pydot
    
    # initialize a dot object to store vertices and edges
    dot_object = pydot.Dot(graph_name="network_graph",rankdir='TB')
    dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                             height=.85, width=.85, fontsize=12)
    dot_object.set_edge_defaults(fontsize=10)

    # walk the graph iteratively
    stack = [node]
    accum = []
    visited = set()

    while stack:
        node = stack.pop()
        
        if node in visited:
            continue

        try:
            # Function node
            node = node.root_function
            stack.extend(node.inputs)
            cur_node = pydot.Node(node.op_name+' '+node.uid, label=node.op_name,shape='circle',
                                    fixedsize='true', height=1, width=1)
            dot_object.add_node(cur_node)
            out_node = pydot.Node(node.outputs[0].uid)#,shape="rectangle")#,label=node.outputs[0].name)
            dot_object.add_node(out_node)
            dot_object.add_edge(pydot.Edge(cur_node,out_node,label=str(node.outputs[0].shape)))
            for child in node.inputs:
                child_node = pydot.Node(child.uid)#,shape="rectangle")#,label=child.name)
                dot_object.add_node(child_node)
                dot_object.add_edge(pydot.Edge(child_node, cur_node,label=str(child.shape)))

        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

        if visitor(node):
            accum.append(node)

    # save to png
    dot_object.write_png(path + '\\network_graph.png', prog='dot')

