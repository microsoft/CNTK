# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
from cntk.variables import Variable

def depth_first_search(root, visitor, depth=0):
    '''
    Generic function that walks through the graph starting at ``root`` and
    uses function ``visitor`` on each node to check whether it should be
    returned.

    Args:
        root (:class:`~cntk.ops.functions.Function` or :class:`~cntk.variables.Variable`): the root to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns ``True`` if that node should be returned.
        depth (int, default 0): how deep into the block hierarchy the DFS
         algorithm should go into. Set to -1 for infinite depth.
    Returns:
        List of functions, for which ``visitor`` was ``True``
    '''
    if depth == -1:
        depth = sys.maxsize

    stack = [(root.root_function, depth)] # node
    accum = []         # final result (list of all unique nodes)
    visited = set()    # [node.uid]
    
    while stack:
        node, depth = stack.pop(0)
        if node.uid in visited:
            continue
        from cntk import cntk_py
        dive_into_blocks = 0 < depth
        if isinstance(node, cntk_py.Function) and node.is_block and dive_into_blocks:
            composite = node.block_root
            # BlockFunction node
            mapping = node.block_arguments_mapping
            # redirect the composite's inputs to the true inputs
            stack.extend([(actual_input, depth-1) for _, actual_input in mapping]) # traverse into actual composite inputs
            visited |= {comp_input.uid for comp_input, _ in mapping}    # don't traverse into the mapped-away inputs
            stack.append((composite, depth-1))
            visited.add(node.uid)
            if visitor(node):
                accum.append(node)
            continue
            # BlockFunctions are short-circuited, and not added to accum[]
        try:
            # Function node
            stack = list((i, depth) for i in node.root_function.inputs) + stack
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.insert(0, (node.owner, depth))
                    visited.add(node.uid)
                    continue
            except AttributeError:
                pass

        if visitor(node):
            if isinstance(node, Variable):
                if node.is_parameter:
                    node = node.as_parameter()
                elif node.is_constant:
                    node = node.as_constant()

            accum.append(node)

        visited.add(node.uid)

    return accum

def find_all_with_name(node, node_name, depth=0):
    '''
    Finds functions in the graph starting from ``node`` and doing a depth-first
    search.

    Args:
        node (:class:`~cntk.ops.functions.Function` or :class:`~cntk.variables.Variable`): the node to start the journey from
        node_name (`str`): name for which we are search nodes
        depth (int, default 0): how deep into the block hierarchy the DFS
         algorithm should go into. Set to -1 for infinite depth.

    Returns:
        List of primitive (or block) functions having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_all_with_name` in class
        :class:`~cntk.ops.functions.Function`.
    '''
    return depth_first_search(node, lambda x: x.name == node_name,
                              depth)

def find_by_name(node, node_name, depth=0):
    '''
    Finds a function in the graph starting from ``node`` and doing a depth-first
    search. It assumes that the name occurs only once.

    Args:
        node (:class:`~cntk.ops.functions.Function` or :class:`~cntk.variables.Variable`): the node to start the journey from
        node_name (`str`): name for which we are search nodes
        depth (int, default 0): how deep into the block hierarchy the DFS
         algorithm should go into. Set to -1 for infinite depth.

    Returns:
        Primitive (or block) function having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_by_name` in class
        :class:`~cntk.ops.functions.Function`.

    '''
    if not isinstance(node_name, str):
        raise ValueError('node name has to be a string. You gave '
                         'a %s' % type(node_name))

    result = depth_first_search(node, lambda x: x.name == node_name,
                                depth)

    if len(result) > 1:
        raise ValueError('found multiple functions matching "%s". '
                         'If that was expected call find_all_with_name' % node_name)

    if not result:
        return None

    return result[0]

def find_by_uid(node, node_uid, depth=0):
    '''
    Finds a function in the graph based on its UID starting from ``node`` and doing a depth-first
    search. It assumes that the name occurs only once.

    Args:
        node (:class:`~cntk.ops.functions.Function` or :class:`~cntk.variables.Variable`): the node to start the journey from
        node_uid (`str` or `unicode` (in Python 2)): uid for which we are search nodes.
        depth (int, default 0): how deep into the block hierarchy the DFS
         algorithm should go into. Set to -1 for infinite depth.

    Returns:
        Primitive (or block) function having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_by_uid` in class
        :class:`~cntk.ops.functions.Function`.
    '''
    # The try-except block below is in place to allow working in Python 2, where
    # the input argument node_uid could be of type 'unicode' instead of 'str'. But
    # Python 3 does not have type 'unicode', hence the check.
    try:
        uid_is_type_unicode = isinstance(node_uid, unicode)
    except NameError:
        uid_is_type_unicode = False

    if not (isinstance(node_uid, str) or uid_is_type_unicode):
        raise ValueError('node_uid must be string of type str or unicode (Python 2.7). You gave '
                         'a %s' % type(node_uid))

    if uid_is_type_unicode:
        node_uid = node_uid.encode('ascii')

    result = depth_first_search(node, lambda x: x.uid == node_uid,
                                depth)

    if len(result) > 1:
        raise ValueError('found multiple functions matching "%s". '
                         'This should not happen as UIDs are unique.' % node_uid)

    if not result:
        return None

    return result[0]

def plot(root, filename=None):
    '''
    Walks through every node of the graph starting at ``root``,
    creates a network graph, and returns a network description. If ``filename`` is
    specified, it outputs a DOT, PNG, PDF, or SVG file depending on the file name's suffix.

    Requirements:

     * for DOT output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`__
     * for PNG, PDF, and SVG output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`__
       and `graphviz <http://graphviz.org>`__ (GraphViz executable has to be in the system's PATH).

    Args:
        node (graph node): the node to start the journey from
        filename (`str`, default None): file with extension '.dot', 'png', 'pdf', or 'svg'
         to denote what format should be written. If `None` then nothing
         will be plotted, and the returned string can be used to debug the graph.

    Returns:
        `str` describing the graph
    '''

    if filename:
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in ('.svg', '.pdf', '.png', '.dot'):
            raise ValueError('only file extensions ".svg", ".pdf", ".png", and ".dot" are supported')
    else:
        suffix = None

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

    # string to store model
    model = []

    root = root.root_function
    root_uid = root.uid
    stack = [root]
    visited = set() # [uid] instead of node object itself, as this gives us duplicate entries for nodes with multiple outputs

    primitive_op_map = {
        'Plus': '+',
        'Minus': '-',
        'ElementTimes': '*',
        'Times': '@',
    }
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
        dyn = '[#' + ',*' * (len(dyn_axes) - 1) + ']' if len(dyn_axes) > 0 else ''
        # the '#' indicates the batch axis, while * indicate dynamic axes (which can be sequences)
        return dyn + str(node.shape)
        static_shape = str(node.shape)
        return '"#dyn: %i\nstatic: %s"'%(num_dyn_axes, static_shape)

    while stack:
        node = stack.pop(0)

        if node.uid in visited:
            continue

        try:
            # Function node
            node = node.root_function

            stack = list(node.root_function.inputs) + stack

            # add current Function node
            def lazy_create_node(node):
                if node.uid in function_nodes: # dot node already exists
                    return function_nodes[node.uid]
                if node.is_primitive and not node.is_block and len(node.outputs) == 1 and node.output.name == node.name:     # skip the node name if redundant
                    op_name = primitive_op_map.get(node.op_name, node.op_name)
                    render_as_primitive = len(op_name) <= 4
                    size = 0.4 if render_as_primitive else 0.6
                    cur_node = pydot.Node(node.uid, label='"' + op_name + '"',
                                          shape='ellipse'  if render_as_primitive else 'box',
                                          fixedsize='true' if render_as_primitive else 'false', height=size, width=size,
                                          fontsize=20  if render_as_primitive and len(op_name) == 1 else 12 ,
                                          penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
                    # TODO: Would be cool, if the user could pass a dictionary with overrides. But maybe for a later version.
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
                dot_object.add_node(cur_node)

            # add node's inputs
            for i, input in enumerate(node.inputs):
                # Suppress Constants inside BlockFunctions, since those are really private to the BlockFunction.
                # Still show Parameters, so users know what parameters it learns, e.g. a layer.
                from cntk import cntk_py
                if node.is_block and isinstance (input, cntk_py.Variable) and input.is_constant:
                    continue

                line.append(input.uid)
                if i != len(node.inputs) - 1:
                    line.append(', ')

                if filename:
                    if input.is_input:
                        shape = 'invhouse'
                        color = 'yellow'
                    elif input.is_placeholder:
                        shape = 'invhouse'
                        color = 'grey'
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
                            if name == 'Parameter':  # don't say 'Parameter' for named parameters, it's already indicated by being a box
                                name = input.name
                            else:
                                name = name + '\n' + input.name
                        name += '\n' + shape_desc(input)
                        if input.is_input or input.is_placeholder: # graph inputs are eggs (since dot has no oval)
                            input_node = pydot.Node(input.uid, shape='egg', label=name, fixedsize='true', height=1, width=1.3, penwidth=4) # wish it had an oval
                        elif not input.name and input.is_constant and (input.shape == () or input.shape == (1,)): # unnamed scalar constants are just shown as values
                            input_node = pydot.Node(input.uid, shape='box', label=str(input.as_constant().value), color='white', fillcolor='white', height=0.3, width=0.4)
                        else:                                      # parameters and constants are boxes
                            input_node = pydot.Node(input.uid, shape='box', label=name, height=0.6, width=1)
                    else: # output variables never get drawn except the final output
                        assert(isinstance (input, cntk_py.Variable))
                        input_node = lazy_create_node(input.owner)  # connect to where the output comes from directly, no need to draw it
                    dot_object.add_node(input_node)
                    label = input.name if input.name else input.uid # the Output variables have no name if the function has none
                    label += '\n' + shape_desc(input)
                    dot_object.add_edge(pydot.Edge(input_node, cur_node, label=label))

            # add node's output
            line.append(') -> ')
            line = ''.join(line)

            for n in node.outputs:
                model.append(line + n.uid + ';\n')

            if (filename):
                if node.uid == root_uid: # only final network outputs are drawn
                    for output in node.outputs:
                        final_node = pydot.Node(output.uid, shape='egg', label=output.name + '\n' + shape_desc(output),
                                                fixedsize='true', height=1, width=1.3, penwidth=4)
                        dot_object.add_node(final_node)
                        dot_object.add_edge(pydot.Edge(cur_node, final_node, label=shape_desc(output)))

        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.insert(0, node.owner)
            except AttributeError:
                pass

        visited.add(node.uid)

    if filename:
        if suffix == '.svg':
            dot_object.write_svg(filename, prog='dot')
        elif suffix == '.pdf':
            dot_object.write_pdf(filename, prog='dot')
        elif suffix == '.png':
            dot_object.write_png(filename, prog='dot')
        else:
            dot_object.write_raw(filename)

    model = "\n".join(reversed(model))

    return model


def get_node_outputs(node, depth=0):
    '''
    Walks through every node of the graph starting at ``node``
    and returns a list of all node outputs.

    Args:
        node (graph node): the node to start the journey from

    Returns:
        A list of all node outputs
    '''
    node_list = depth_first_search(node, lambda x: True, depth)
    node_outputs = []
    for node in node_list:
        try:
            for out in node.outputs:
                node_outputs.append(out)
        except AttributeError:
            pass

    return node_outputs
