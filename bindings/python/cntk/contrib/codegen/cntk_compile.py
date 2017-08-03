# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Generator of the eval graph for a given CNTK model.
"""

from cntk import *
from util import *
from quantizer import *
from model_transforms import *
from expression_generator import *
import networkx as nx
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the CNTK model file',
                        required=True, default=None)
    parser.add_argument('-p', '--plot', help='Path to the output file with resulting DAG of the model. Should have one of supported suffixes (i.e. pdf)',
                        required=False, default=None)
    parser.add_argument('-c', '--classname', help='Name of the resulting class',
                        required=False, default='Evaluator')
    parser.add_argument('-o', '--output', help='File name for the output',
                        required=False, default=None)
    parser.add_argument('-w', '--weights', help='File name for the serialized weights/constants',
                        required=False, default=None)
    parser.add_argument('-q', '--quantization', help='Type of quantization. Currently only symmetric quantization of weights is supported.',
                        required=False, default=None)
    parser.add_argument('-b', '--reserved_bits', help='Number of reserved bits for quantization.',
                        required=False, default=0)
    parser.add_argument('-t', '--total_bits', help='Number of total bits for quantization.',
                        required=False, default=16)
    args = vars(parser.parse_args())

    if args['output'] is None:
        args['output'] = args['classname'] + '.h'

    if args['weights'] is None:
        args['weights'] = args['classname'] + '.json'

    # Create the graph and perform some transforms on it
    model = Function.load(args['model'])

    graph = ModelToGraphConverter().convert(model)
    remove_intermediate_output_nodes(graph)
    split_past_values(graph)

    if not nx.is_directed_acyclic_graph(graph):
        if args['plot']:
            nx_plot(graph, args['plot'])
        raise ValueError('Unsupported type of graph: please make sure there are no loops or non connected components')

    # Perform topological sort for evaluation
    nodes_sorted_for_eval = nx.topological_sort(graph)

    # Attribute nodes with quantized values if required.
    if args['quantization']:
        q = OperationQuantizer(graph, args['quantization'], args['reserved_bits'], args['total_bits'])
        q.quantize(list(reversed(nodes_sorted_for_eval)))

    # Ok, let's plot the resulting graph if asked.
    if args['plot']:
        nx_plot(graph, args['plot'])

    nodes_sorted_for_eval = nx.topological_sort(graph)

    # Now generate the actual Halide/C++ file with the evaluator inside
    listing = HalideExpressionGenerator(graph).generate(nodes_sorted_for_eval, args['classname'])

    with open(args['output'], 'w') as f:
        f.write(listing)

    print('Successfully finished generation of halide expression')

    # Also make sure we generate the json weights/constants for now.
    # These should be taken by C++ directly from the model though.
    # Or better built-in inside the source file. Not yet clear how though
    # because long arrays break the C++ linker.
    # TODO: We should enabled reading of weights directly from CNTK model
    # Also contact VS team to ask where to put huge arrays of weights.
    print('Start dumping the weights...')
    WeightsExtractor(graph).dump(args['weights'])
    print('Successfully dumped the weights')
