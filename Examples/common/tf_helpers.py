# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
def create_tensorflow_graph(network,graph):
    '''
    Converts a function from CNTK to the Tensorflow graph format
    
    Args:
        network: CNTK function that defines the network structure
        graph: destinaiton Tensorflow graph
    ''' 
    try:
        import tensorflow as tf
        from tensorflow.core.framework import attr_value_pb2
        from tensorflow.core.framework import tensor_shape_pb2
    except ImportError:
        raise ImportError("Tensorflow module is required to create a tensorflow graph object")

    # Walk every node of the netwrok iteratively
    visitor = lambda x: True
    stack = [network]
    visited = set()

    while (stack):
        node = stack.pop()
        
        if node in visited:
            continue

        try:

            # Function node
            node = node.root_function
            stack.extend(node.inputs)
            try:
                # TF graph already has the current node 
                graph.get_operation_by_name(node.uid.split('_')[0])
                continue
            
            except KeyError:
                # New netwrok node that has to be converted to TF format
                
                # define TF operation attributes based on CNTK network node  
                try:
                    dimX = tensor_shape_pb2.TensorShapeProto.Dim(size=node.outputs[0].shape[0])
                except IndexError:
                    dimX = tensor_shape_pb2.TensorShapeProto.Dim(size=1)
                try:
                    dimY = tensor_shape_pb2.TensorShapeProto.Dim(size=node.outputs[0].shape[1])
                except IndexError:
                    dimY = tensor_shape_pb2.TensorShapeProto.Dim(size=1)     
                shape = tensor_shape_pb2.TensorShapeProto(dim=(dimX,dimY))
                shape_attr = attr_value_pb2.AttrValue(shape=shape)
                attrs = {"shape": shape_attr}
                
                # Use name scope based on the node's name (e.g. Plus1) to 
                # group the operation and its inputs
                with graph.name_scope(node.uid) as scope:

                    # Create a TF placeholder operation with type, name and shape of the current node
                    op = graph.create_op("Placeholder", inputs =[],
                             dtypes=[node.outputs[0].dtype], attrs=attrs, 
                             name=node.uid)

                    # Add inputs to the created TF operation
                    for i in range(len(node.inputs)):
                        child = node.inputs[i]
                        name = child.uid
                        try:
                            # The input tensor already exists in the graph
                            tf_input = graph.get_tensor_by_name(name +":0")
                        except KeyError:
                            # A new tensor that needs to be converted from CNTK to TF
                            shape = convert_shape(child.shape)
                            dtype = child.dtype
                            # Create a new placeholder tensor with the corresponding attributes
                            tf_input = tf.placeholder(shape=shape, dtype=dtype, name=name)
                        
                        # Update TF operator's inputs
                        op._add_input(tf_input)

                # Update TF operation's outputs
                output = node.outputs[0]
                for o in graph.get_operations():
                    if (output.uid in o.name):
                        o._add_input(op.outputs[0])
 
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    try:
                        # Owner of the node is already added to the TF graph
                        owner_name = node.owner.uid + '/' + node.owner.uid
                        op = graph.get_operation_by_name(owner_name)
                    except KeyError:
                        # Unknown network node
                        stack.append(node.owner)
                    
            except AttributeError:
                    pass

    # Add missing connections in the graph
    update_outputs(graph.get_operations())

    graph.finalize()

def convert_shape(shape):
    if (len(shape)==0):
        shape = (1,1)
    else:
        if (len(shape)==1):
            shape += (1,)
    return shape

def update_outputs(ops):
    '''
    Updates the inputs/outputs of the Tensorflow operations
    by adding missing connections

    Args:
        ops: a list of Tensorflow operations
    '''
    for i in range(len(ops)):
        for j in range(i+1,len(ops)):
            if (ops[i].name.split('/')[1] in ops[j].name.split('/')[1]):
                ops[i]._add_input(ops[j].outputs[0])

def summary_message(tag,value):
    '''
    Creates a Tensorflow summary protobuf object

    Args:
        tag(string): a tag that describes the object's content
        value: a scalar value it stores

    Returns: 
        Tensorflow summary protobuf object
    '''
    from tensorflow.core.framework import summary_pb2
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value
            (tag=tag,simple_value=value)])
