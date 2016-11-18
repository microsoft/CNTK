# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import ipdb
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

def output_function_graph(node,dot_file_path=None,png_file_path=None):
    '''
    Walks through every node of the graph starting at ``node``,
    creates a network graph, and saves it as a string. If dot_file_name or 
    png_file_name specified corresponding files will be saved.
    
    
    Requirements for DOT output: pydot_ng
    Requirements for PNG output: pydot_ng and graphviz

    Args:
        node (graph node): the node to start the journey from
        dot_file_path (`str`, optional): DOT file path
        png_file_path (`str`, optional): PNG file path

    Returns:
        `str` containing all nodes and edges
    '''

    dot = (dot_file_path != None)
    png = (png_file_path != None)

    if (dot or png):

        try:
            import pydot_ng as pydot
        except ImportError:
            raise ImportError("PNG and DOT format requires pydot_ng package. Unable to import pydot_ng.")

        # initialize a dot object to store vertices and edges
        dot_object = pydot.Dot(graph_name="network_graph",rankdir='TB')
        dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                 height=.85, width=.85, fontsize=12)
        dot_object.set_edge_defaults(fontsize=10)
    
    # string to store model 
    model = ''

    # walk every node of the graph iteratively
    visitor = lambda x: True
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

            # add current node
            model += node.op_name + '('
            if (dot or png):
                cur_node = pydot.Node(node.op_name+' '+node.uid,label=node.op_name,shape='circle',
                                        fixedsize='true', height=1, width=1)
                dot_object.add_node(cur_node)

            # add node's inputs
            for i in range(len(node.inputs)):
                child = node.inputs[i]
                
                model += child.uid
                if (i != len(node.inputs) - 1):
                    model += ", "

                if (dot or png):
                    child_node = pydot.Node(child.uid)
                    dot_object.add_node(child_node)
                    dot_object.add_edge(pydot.Edge(child_node, cur_node,label=str(child.shape)))

            # ad node's output
            model += ") -> " + node.outputs[0].uid +'\n'

            if (dot or png):
                out_node = pydot.Node(node.outputs[0].uid)
                dot_object.add_node(out_node)
                dot_object.add_edge(pydot.Edge(cur_node,out_node,label=str(node.outputs[0].shape)))

        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

    if visitor(node):
        accum.append(node)

    if (png):
        dot_object.write_png(png_file_path, prog='dot')
    if (dot):
        dot_object.write_raw(dot_file_path)

    # return lines in reversed order
    return "\n".join(model.split("\n")[::-1])

ops_dict = {
    "Plus": "Add",
    "Times": "MatMul",
    "ElementTimes": "Mul",
    "ReLU": "Relu"
    }

def create_tensorflow_graph(node,graph):

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Tensorflow module is required to create a tensorflow graph object")

    # walk every node of the graph iteratively
    visitor = lambda x: True
    stack = [node]
    accum = []
    visited = set()
    outputs = {}
    tf_ops = []

    while stack:
        node = stack.pop()
        
        if node in visited:
            continue

        tf_inputs = []
        try:
            # Function node
            node = node.root_function
            stack.extend(node.inputs)

            # add node's inputs
            for i in range(len(node.inputs)):
                child = node.inputs[i]

                try:
                    tf_input = graph.get_operation_by_name(child.uid)

                    
                except  KeyError:
                    shape = convert_shape(child.shape, child.is_parameter, node.op_name=="Times")
                    tf_input = tf.placeholder(child.dtype, shape=shape, name=child.uid)
                  
                tf_inputs.append(tf_input)
                
            try: 
                graph.create_op(compute_shapes=True,op_type=ops_dict[node.op_name], inputs=tf_inputs, 
                    dtypes=[node.outputs[0].dtype], name=node.outputs[0].uid)
            except Exception:
                if (node.op_name=='Times'):
                    from tensorflow.core.framework import attr_value_pb2
                    attr = attr_value_pb2.AttrValue()
                    attr.b = False
                    attrs = {"transpose_a": attr, "transpose_b": attr}
                    try:
                        graph.create_op(op_type=ops_dict[node.op_name], inputs=tf_inputs, 
                            dtypes=[node.outputs[0].dtype], name=node.outputs[0].uid, attrs=attrs) 
                    except TypeError:

                        graph.create_op(op_type=ops_dict["ElementTimes"], inputs=tf_inputs, 
                            dtypes=[node.outputs[0].dtype], name=node.outputs[0].uid, attrs=attrs) 
           
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

    if visitor(node):
        accum.append(node)

    update_outputs(graph.get_operations())
    graph.finalize()

def convert_shape(shape, param, times):
    if (len(shape)==0):
        shape = (1,1)
    if (len(shape)==1):
        shape += (1,)
    if (param and times):
        shape = shape[::-1]
    return shape

def update_outputs(ops):
    for i in range(len(ops)):
        for j in range(len(ops)):
            if (i != j and ops[i].name in ops[j].name):
                ops[i]._add_input(ops[j].outputs[0])



    