# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
from __future__ import print_function

from .debug import *
from .profiler import *

'''
Helper functions for debugging graphs.
'''

def dump_signature(root, tag=None):
    '''
    Debug helper that prints the signature of a Function.
    '''
    f_name = root.name if root.name else tag if tag else 'Function'
    args = root.signature
    def format_arg_spec(v):
        s = v.name + ': ' if v.name else ''
        return s + str(v.type)
    outputs = root.outputs
    if len(outputs) > 1:
        output_signature = 'Tuple[' + ', '.join(format_arg_spec(output) for output in outputs) + ']'
    else:
        output_signature = format_arg_spec(outputs[0])
    print(f_name + '(' + ", ".join([format_arg_spec(param) for param in args]) + ') -> ' + output_signature)

def dump_function(root, tag=None):
    from cntk.logging.graph import depth_first_search
    from cntk import cntk_py
    graph = depth_first_search(root.root_function,
                               lambda x: not isinstance(x, cntk_py.Variable)\
                                         or not x.is_output,
                               depth=-1)
    names = dict()
    def make_name(n): # come up with a letter sequence
        if n < 26:
            return chr(n + ord('a'))
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
        if axis.name == "UnknownAxes":  # TODO: what is the correct way of testing this?
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
