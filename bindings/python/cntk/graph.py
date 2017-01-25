# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def depth_first_search(root, visitor, max_depth=None, sort_by_distance=False):
    '''
    Generic function that walks through the graph starting at ``node`` and
    uses function ``visitor`` on each node to check whether it should be
    returned.
 
    Args:
        root (Function or Variable): the root to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns ``True`` if that node should be returned.
        max_depth: maximum number of BlockFunction levels to traverse into.
        sort_by_distance: result list is sorted by how far away they are from the root
    Returns:
        List of functions, for which ``visitor`` was ``True``
    '''
    #stack = [(root,0,0)] # node, distance, Block depth
    # was changed to:
    stack = [(root.root_function,0,0)] # node, distance, Block depth
    accum = []         # final result (all unique nodes in a list) (node, distance)
    visited = set()    # [node]

    while stack:
        node, distance, depth = stack.pop()
        if node in visited:
            continue
        if max_depth is None or depth < max_depth:
            try:
                # TODO: This is still broken, needs to be cleaned up after debugging/desperate trying-around.
                composite = node.block_root
                # BlockFunction node
                mapping = node.block_arguments_mapping
                # redirect the composite's inputs to the true inputs
                stack.extend([(actual_input, distance+1, depth) for _, actual_input in mapping]) # traverse into actual composite inputs
                visited |= {comp_input for comp_input, _ in mapping} # don't traverse into the mapped-away inputs
                #stack.extend((input, distance+1, depth+1) for input in composite.root_function.inputs) # and short-circuit the root composite instead
                stack.append((composite, distance+1, depth+1))
                visited.add(node)
                if visitor(node):
                    accum.append((node, distance))
                continue
                # BlockFunctions are short-circuited until max_depth is hit, and not added to accum[]
            except:
                pass
        try:
            # Function node
            stack.extend((input, distance+1, depth) for input in node.root_function.inputs)
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append((node.owner, distance+1, depth))
                    visited.add(node)
                    continue
            except AttributeError:
                pass

        if visitor(node):
            accum.append((node, distance))

        visited.add(node)

    if sort_by_distance:
        accum.sort(key=lambda tpl: tpl[1]) # [1] is distance

    return [node for node, distance in accum]

# BUGBUG: Doesn't this require @typemap, so ?
def find_all_with_name(node, node_name, max_depth=None):
    '''
    Finds functions in the graph starting from ``node`` and doing a depth-first
    search.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        List of primitive functions having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_all_with_name` in class
        :class:`~cntk.ops.functions.Function`.
    '''
    return depth_first_search(node, lambda x: x.name == node_name, max_depth)

def find_by_name(node, node_name, max_depth=None):
    '''
    Finds a function in the graph starting from ``node`` and doing a depth-first
    search. It assumes that the name occurs only once.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        Primitive function having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_by_name` in class
        :class:`~cntk.ops.functions.Function`.

    '''
    if not isinstance(node_name, str):
        raise ValueError('node name has to be a string. You gave '
                'a %s'%type(node_name))

    result = depth_first_search(node, lambda x: x.name == node_name, max_depth)

    if len(result)>1:
        raise ValueError('found multiple functions matching "%s". '
                'If that was expected call find_all_with_name'%node_name)

    if not result: # TODO: a better name would be try_find_by_name()
        return None

    return result[0]

def try_find_closest_by_name(node, node_name, max_depth=None):
    '''
    Finds the closest function or variable in the graph starting from ``node`` and doing a depth-first
    search. Closest means that if there are multiple, the one with the shortest path is returned.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        Primitive function or variable having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_by_name` in class
        :class:`~cntk.ops.functions.Function`.

    '''
    if not isinstance(node_name, str):
        raise ValueError('node name has to be a string. You gave '
                'a %s'%type(node_name))

    result = depth_first_search(node, lambda x: x.name == node_name, max_depth, sort_by_distance=True)

    if not result:
        return None

    return result[0]

# TODO: This seems to have lots of overlap with depth_first_search() above
def output_function_graph(root, dot_file_path=None, png_file_path=None, pdf_file_path=None, svg_file_path=None, scale=1.5):
    '''
    Walks through every node of the graph starting at ``root``,
    creates a network graph, and saves it as a string. If dot_file_name, 
    png_file_name, pdf_file_name, or svg_file_name specified corresponding files will be saved.
    
    Requirements:

     * for DOT output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`_
     * for PNG, PDF, and SVG output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`_ 
       and `graphviz <http://graphviz.org>`_

    Args:
        root (graph node): the root to start the journey from
        dot_file_path (`str`, optional): DOT file path
        png_file_path (`str`, optional): PNG file path
        pdf_file_path (`str`, optional): PDF file path
        svg_file_path (`str`, optional): PDF file path
        scale: a factor on the font sizes, if it comes out too small to read

    Returns:
        `str` containing all nodes and edges
    '''

    # TODO: known issues:
    #  - Combine() shows with loops
    #  - duplicate connections into a BlockFunction should be unique'd
    #  - singleton dots should be short-circuited
    #  - scalar Constants should be shown as such

    # TODO: a better interface would just derive the format from the filename extension of a format parameter
    write_to_file = (dot_file_path != None) or (png_file_path != None) or (pdf_file_path != None) or (svg_file_path != None)
    if (write_to_file):

        try:
            import pydot_ng as pydot
        except ImportError:
            raise ImportError("SVG, PDF, PNG, and DOT format requires pydot_ng package. Unable to import pydot_ng.")

        # initialize a dot object to store vertices and edges
        dot_object = pydot.Dot(graph_name="network_graph",rankdir='TB')
        dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                 height=.85, width=.85, fontsize=12 * scale)
        dot_object.set_edge_defaults(fontsize=10 * scale)
    
    # string to store model 
    model = ''

    # walk every node of the graph iteratively
    visitor = lambda x: True
    stack = [root]
    accum = []
    visited = set()

    primitive_op_map = {
        'Plus': '+',
        'Minus': '-',
        'ElementTimes': '*',
        'Times': '@',
    }
    def map_primitive_op(op_name):
        if op_name in primitive_op_map:
            op_name = primitive_op_map[op_name]
        return op_name

    while stack:
        node = stack.pop()
        
        if node in visited:
            continue

        try:
            # Function node
            is_root = node is root
            node = node.root_function

            stack.extend(node.inputs)

            # add current Function node
            model += node.op_name + '('
            if (write_to_file):
                if node.is_primitive and not node.is_block and len(node.outputs) == 1 and node.output.name == node.name:     # skip the node name if redundant
                    op_name = map_primitive_op(node.op_name)
                    render_as_primitive = len(op_name) <= 4
                    size = 0.4 if render_as_primitive else 0.6
                    cur_node = pydot.Node(node.op_name + ' ' + node.uid, label='"' + op_name + '"',
                                          shape='ellipse'  if render_as_primitive else 'box',
                                          fixedsize='true' if render_as_primitive else 'false', height=size, width=size,
                                          fontsize=20 * scale if render_as_primitive and len(op_name) == 1 else 12 * scale,
                                          penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
                else:
                    cur_node = pydot.Node(node.op_name + ' ' + node.uid, label='"' + node.op_name + '\n' + node.name + '()"',
                                          fixedsize='true', height=1, width=1.3,
                                          penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
                dot_object.add_node(cur_node)

            # add node's inputs
            for i in range(len(node.inputs)):
                input = node.inputs[i]

                from cntk import cntk_py
                if node.is_block and isinstance (input, cntk_py.Variable) and input.is_constant:
                    continue

                model += input.uid
                if (i != len(node.inputs) - 1):
                    model += ", "

                if (write_to_file):
                    if isinstance (input, cntk_py.Variable) and not input.is_output:
                        name = 'Parameter' if input.is_parameter else 'Constant' if input.is_constant else 'Input' if input.is_input else 'Placeholder'
                        if input.name:
                            if name == 'Parameter':  # don't say Parameter for parameters, it's clear from the box
                                    name = input.name
                            else:
                                name = name + '\n' + input.name
                        name += '\n' + str(input.shape)
                        if input.is_input or input.is_placeholder:
                            child_node = pydot.Node(input.uid, shape='egg', label=name, fixedsize='true', height=1, width=1.3, penwidth=4) # wish it had an oval
                        else:
                            child_node = pydot.Node(input.uid, shape='box', label=name, height=0.6, width=1)
                    else:
                        #child_node = pydot.Node(input.uid, shape='ellipse')
                        child_node = pydot.Node(input.uid, shape='point', height=0.1, width=0.1, label='')
                    dot_object.add_node(child_node)
                    label = str(input.shape)
                    if input.name != '':
                        label = input.name + '\n' +  label
                    dot_object.add_edge(pydot.Edge(child_node, cur_node, label=label))

            # add node's output
            # BUGBUG: multiple outputs not handled correctly in returned text string
            model += ") -> " + node.outputs[0].uid +'\n'

            if (write_to_file):
                for output in node.outputs:
                    out_node = pydot.Node(output.uid)
                    dot_object.add_node(out_node)
                    dot_object.add_edge(pydot.Edge(cur_node,out_node,label=str(output.shape)))
                    if is_root: # network outputs are drawn like an egg instead
                        final_node = pydot.Node(output.uid + "_final", shape='egg', label=output.name + '\n' + str(output.shape),
                                                fixedsize='true', height=1, width=1.3, penwidth=4)
                        dot_object.add_node(final_node)
                        dot_object.add_edge(pydot.Edge(out_node, final_node, label=str(output.shape)))

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

    if (svg_file_path):
        dot_object.write_svg(svg_file_path, prog='dot')
    if (pdf_file_path):
        dot_object.write_pdf(pdf_file_path, prog='dot')
    if (png_file_path):
        dot_object.write_png(png_file_path, prog='dot')
    if (dot_file_path):
        dot_object.write_raw(dot_file_path)

    # return lines in reversed order
    return "\n".join(model.split("\n")[::-1])
