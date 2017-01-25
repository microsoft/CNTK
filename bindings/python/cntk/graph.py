# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os


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
                         'a %s' % type(node_name))

    result = depth_first_search(node, lambda x: x.name == node_name, max_depth)

    if len(result) > 1:
        raise ValueError('found multiple functions matching "%s". '
                         'If that was expected call find_all_with_name' % node_name)

    if not result: # TODO: a better name would be try_find_by_name()
        return None

    return result[0]

# TODO: this is no longer needed, delete
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


def plot(root, filename=None):
    '''
    Walks through every node of the graph starting at ``root``,
    creates a network graph, and returns a network description. It `filename` is
    specified, it outputs a DOT, PNG, PDF, or SVD file depending on the file name's suffix.

    Requirements:

     * for DOT output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`_
     * for PNG, PDF, and SVG output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`_ 
       and `graphviz <http://graphviz.org>`_

    Args:
        node (graph node): the node to start the journey from
        filename (`str`, default None): file with extension '.dot', 'png', 'pdf', or 'svd'
        to denote what format should be written. If `None` then nothing
        will be plotted. Instead, and the returned string can be used to debug the graph.

    Returns:
        `str` describing the graph
    '''

    if filename:
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in ('.svd', '.pdf', '.png', '.dot'):
            raise ValueError('only file extensions ".svd", ".pdf", ".png", and ".dot" are supported')
    else:
        suffix = None

    if filename:
        try:
            import pydot_ng as pydot
        except ImportError:
            raise ImportError("SVG, PDF, PNG, and DOT format requires pydot_ng package. Unable to import pydot_ng.")

        # initialize a dot object to store vertices and edges
        dot_object = pydot.Dot(graph_name="network_graph", rankdir='TB')
        dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                     style='filled',    # FROM WILLI
                                     #color='lightgray',  # BUGBUG: This will kill the borders??
                                     height=.85, width=.85, fontsize=12)
        dot_object.set_edge_defaults(fontsize=10)

    # string to store model
    model = []

    stack = [root.root_function]
    visited = set() # [uid] instead of node object itself, as this gives us duplicate entries for nodes with multiple outputs

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
    function_nodes = {}  # [uid] -> dot node

    def node_desc(node):
        name = "<font point-size=\"10\" face=\"sans\">'%s'</font> <br/>"%node.name
        try:
            name += "<b><font point-size=\"14\" face=\"sans\">%s</font></b> <br/>"%node.op_name
        except AttributeError:
            pass

        name += "<font point-size=\"8\" face=\"sans\">%s</font>"%node.uid

        return '<' + name + '>'

    def shape_desc(node):
        dyn_axes = node.dynamic_axes
        dyn = '[*' + ',*' * (len(dyn_axes) - 1) + ']' if len(dyn_axes) > 0 else ''
        return dyn + str(node.shape)
        static_shape = str(node.shape)
        return '"#dyn: %i\nstatic: %s"'%(num_dyn_axes, static_shape)

    while stack:
        node = stack.pop()
        
        if node.uid in visited:
            continue

        try:
            # Function node
            is_root = node is root
            node = node.root_function

            stack.extend(node.inputs)

            # add current Function node
            def lazy_create_node(node):
                if node.uid in function_nodes: # dot node already exists
                    return function_nodes[node.uid]
                if node.is_primitive and not node.is_block and len(node.outputs) == 1 and node.output.name == node.name:     # skip the node name if redundant
                    op_name = map_primitive_op(node.op_name)
                    render_as_primitive = len(op_name) <= 4
                    size = 0.4 if render_as_primitive else 0.6
                    cur_node = pydot.Node(node.uid, label='"' + op_name + '"',
                                          shape='ellipse'  if render_as_primitive else 'box',
                                          fixedsize='true' if render_as_primitive else 'false', height=size, width=size,
                                          fontsize=20  if render_as_primitive and len(op_name) == 1 else 12 ,
                                          penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
                else:
                    f_name = '\n' + node.name + '()' if node.name else ''
                    cur_node = pydot.Node(node.uid, label='"' + node.op_name + f_name + '"',
                                          fixedsize='true', height=1, width=1.3,
                                          penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
                dot_object.add_node(cur_node)
                function_nodes[node.uid] = cur_node
                return cur_node

            # add current node
            line = [node.op_name]
            line.append('(')

            if filename:
                cur_node = lazy_create_node(node)
                # FROM WILLI:
                #cur_node = pydot.Node(node.uid, label=node_desc(node),
                #        shape='circle')
                dot_object.add_node(cur_node)

            # add node's inputs
            for i in range(len(node.inputs)):
                input = node.inputs[i]

                # Suppress Constants inside BlockFunctions, since those are really private to the BlockFunction.
                # Still show Parameters, so users know what parameters it learns, e.g. a layer.
                from cntk import cntk_py
                if node.is_block and isinstance (input, cntk_py.Variable) and input.is_constant:
                    continue

                line.append(input.uid)
                if i != len(node.inputs) - 1:
                    line.append(', ')

                if filename:
                    # TODO: further merge this
                    if input.is_input:  # does this include placeholder?
                        shape = 'invhouse'
                        color = 'yellow'
                    elif input.is_parameter:
                        shape = 'diamond'
                        color = 'green'
                    elif input.is_constant:
                        shape = 'rectangle'
                        color = 'lightblue'
                    else: # is_output
                        shape = 'invhouse'
                        color = 'grey'
                    if isinstance (input, cntk_py.Variable) and not input.is_output:
                        name = 'Parameter' if input.is_parameter else 'Constant' if input.is_constant else 'Input' if input.is_input else 'Placeholder'
                        if input.name:
                            if name == 'Parameter':  # don't say Parameter for parameters, it's clear from the box
                                    name = input.name
                            else:
                                name = name + '\n' + input.name
                        name += '\n' + shape_desc(input)
                        if input.is_input or input.is_placeholder: # graph inputs are eggs (since dot has no oval)
                            input_node = pydot.Node(input.uid, shape='egg', label=name, fixedsize='true', height=1, width=1.3, penwidth=4) # wish it had an oval
                        else:                                      # parameters and constants are boxes
                            input_node = pydot.Node(input.uid, shape='box', label=name, height=0.6, width=1)
                    else: # output variables never get drawn except the final output
                        assert(isinstance (input, cntk_py.Variable))
                        # BUGBUG: The Output variables all have no names...?
                        input_node = lazy_create_node(input.owner)  # connect to where the output comes from directly, no need to draw it
                    dot_object.add_node(input_node)
                    label = input.name if input.name else input.uid # TODO: not optimal; why is the .name field not fillled for Outputs?
                    label += '\n' + shape_desc(input)
                    dot_object.add_edge(pydot.Edge(input_node, cur_node, label=label))
                    # FROM WILLI--also move above to here before merge
                    #child_node = pydot.Node(input.uid, label=node_desc(input),
                    #        shape=shape, color=color)
                    #dot_object.add_node(child_node)
                    #dot_object.add_edge(pydot.Edge(
                    #    child_node, cur_node, label=shape_desc(input)))

            # add node's output
            line.append(') -> ')
            line = ''.join(line)

            for n in node.outputs:
                model.append(line + n.uid + ';\n')

            if (filename):
                if is_root: # only final network outputs are drawn
                    for output in node.outputs:
                        final_node = pydot.Node(output.uid, shape='egg', label=output.name + '\n' + shape_desc(output),
                                                fixedsize='true', height=1, width=1.3, penwidth=4)
                        dot_object.add_node(final_node)
                        dot_object.add_edge(pydot.Edge(cur_node, final_node, label=shape_desc(output)))
                # FROM WILLI --these are Output nodes, keep them out
                #if filename:
                #    out_node = pydot.Node(n.uid, label=node_desc(n))
                #    dot_object.add_node(out_node)
                #    dot_object.add_edge(pydot.Edge(
                #        cur_node, out_node, label=shape_desc(node)))

        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

        visited.add(node.uid)

    if filename:
        if suffix == '.svd':
            dot_object.write_svg(filename, prog='dot')
        elif suffix == '.pdf':
            dot_object.write_pdf(filename, prog='dot')
        elif suffix == '.png':
            dot_object.write_png(filename, prog='dot')
        else:
            dot_object.write_raw(filename)

    model = "\n".join(reversed(model))

    return model

def output_function_graph(node, dot_file_path=None, png_file_path=None):
    import warnings
    warnings.warn('This will be removed in future versions. Please use '
            'plot(...) instead', DeprecationWarning)

    result = plot(node, dot_file_path)
    if png_file_path:
        result2 = plot(node, dot_file_path)
        if not result:
            result = result2

    return result

