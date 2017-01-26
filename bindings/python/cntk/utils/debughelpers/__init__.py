# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def dump_signature(root, tag=None):
    '''
    Debug helper that prints the signature of a Function.
    '''
    f_name = root.name if root.name else tag if tag else 'Function'
    args = root.signature
    arg_names = [param.name for param in args]
    output_names = [output.name if output.name else '_' for output in root.outputs]
    if len(output_names) > 1:
        output_signature = 'Tuple[' + ', '.join(output_names) + ']'
    else:
        output_signature = output_names[0]
    # attempting Python type hint syntax, although we use variable names instead of their types here
    print(f_name + ': Callable[[' + ", ".join(arg_names) + '], ' + output_signature + ']')

def dump_function(root, tag=None):
    from ...graph import depth_first_search
    from cntk import cntk_py
    graph = depth_first_search(root.root_function, lambda x: not isinstance(x, cntk_py.Variable) or not x.is_output)
    names = dict()
    def make_name(n): # come up with a letter sequence
        if n < 26:
            return chr(n + 97)
        else:
            return make_name(n // 26) + make_name(n % 26)
    def name_it(item):
        if item.name != '':
            return item.name
        if item in names:
            name = names[item]
        else:
            name = make_name(len(names))
            names[item] = name
        return name
    axis_names = dict()
    def name_axis(axis):
        actual_name = axis.name
        if actual_name in axis_names:
            return axis_names[actual_name]
        if axis.name == "staticAxis_2147483645":  # TODO: what is the correct way of testing this?
            name = "?"
        elif axis.name == "defaultBatchAxis":
            name = "b*"
        else:
            name = make_name(len(axis_names)+12) + "*"
            print("  Axis", actual_name, "==", name)
        axis_names[actual_name] = name
        return name
    def type_spec(var):
        s = "[" + ",".join([name_axis(axis) for axis in var.dynamic_axes]) + "]" if var.dynamic_axes else ''
        s += str(var.shape)
        return s
    def print_item(item):
        name = name_it(item)
        if isinstance(item, cntk_py.Function):
            op_name = item.op_name
            shape = '(' +  ', '.join([name_it(output) + ':' + type_spec(output) for output in item.root_function.outputs]) + ')'
            inputs = '(' +  ', '.join([name_it(input) + ':' + type_spec( input) for input in item.root_function.inputs]) + ')'
            sep = '-> '
        elif isinstance(item, cntk_py.Constant):
            op_name = "Constant"
            shape = type_spec(item)
            inputs = ''
            sep = ''
        elif isinstance(item, cntk_py.Parameter):
            op_name = "Parameter"
            shape = type_spec(item)
            inputs = ''
            sep = ''
        elif isinstance(item, cntk_py.Variable):
            if item.is_parameter:
                op_name = "Parameter"
            elif item.is_placeholder:
                op_name = "Placeholder"
            elif item.is_input:
                op_name = "Input"
            elif item.is_constant:
                op_name = "Constant"
            else:
                op_name = "Variable"
            shape = type_spec(item)
            name = name + " " + item.uid
            sep = ''
            inputs = ''
        print('  {:20} {:30} {} {}{}'.format(op_name, name, inputs, sep, shape))
        pass
    dump_signature(root, tag)
    for item in graph:
        print_item(item)



# TODO: All below should no longer be used and be deleted.

# helper to name nodes for printf debugging
_auto_node_names = dict()
_auto_name_count = dict()
def _name_node(n, name):
    if not n in _auto_node_names:     # only name a node once
        # strip _.*
        #name = name.split('[')[0]
        if not name in _auto_name_count: # count each type separately
            _auto_name_count[name] = 1
        else:
            _auto_name_count[name] += 1
        #name = name + "[{}]".format(_auto_name_count[name])
        name = name + ".{}".format(_auto_name_count[name])
        _auto_node_names[n] = name
    return n

# this gives a name to anything not yet named
def _node_name(n):
    global _auto_node_names, _auto_name_count
    if n in _auto_node_names:
        return _auto_node_names[n]
    try:
        name = n.what()
    except:
        name = n.name
    # internal node names (not explicitly named)
    if name == '':
        if hasattr(n, 'is_placeholder') and n.is_placeholder:
            name = '_'
        else:
            name = '_f'
    _name_node(n, name)
    return _node_name(n)

# -> node name (names of function args if any)
def _node_description(n):
    desc = _node_name(n)
    if hasattr(n, 'inputs'):
        inputs = n.inputs
        #desc = "{} [{}]".format(desc, ", ".join([_node_name(p) for p in inputs]))
        func_params = [input for input in inputs if input.is_parameter]
        func_args   = [input for input in inputs if input.is_placeholder]
        if func_params:
            desc = "{} {{{}}}".format(desc, ", ".join([_node_name(p) for p in func_params]))
        desc = "{} <{}>".format(desc, ", ".join([_node_name(func_arg) for func_arg in func_args]))
    return desc

def _log_node(n):
    print (_node_description(n))
