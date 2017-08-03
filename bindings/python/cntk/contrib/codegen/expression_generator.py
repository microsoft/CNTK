# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Classes and functions for expression generation from a CNTK model.
TODO: Switch to Jinja templates for C++ code generation.
TODO: All headers and code should go there, the expression generator
TODO: should only provide the context. 
"""
from model_transforms import *
from node_visitor import *
from quantizer import *
from cntk import *
from cntk import cntk_py
from pdb import set_trace
import cntk.variables
import networkx as nx
import itertools
import functools
import json

class WeightsExtractor(EmptyNodeVisitor):
    '''
    Extracts weights and constants into a separate json file.
    TODO: We should take dependency on protobuf and extract 
    this values directly from the model.
    '''
    def __init__(self, graph):
        super(EmptyNodeVisitor, self).__init__(graph)

    def dump(self, filepath):
        self.weights = {}
        self.visit(self.graph.nodes())
        json.encoder.FLOAT_REPR = lambda o: format(o, '.9f')
        with open(filepath, "w") as f:
            json.dump(self.weights, f)

    def visit_parameter(self, node):
        self.weights[node.id] = [float(f) for f in np.transpose(node.model.as_parameter().value).flatten()]

    def visit_constant(self, node):
        self.weights[node.id] = [float(f) for f in np.transpose(node.model.as_constant().value).flatten()]

class CppNamespaceGen:
    '''
    Helper class for generation of C++ namespace.
    '''
    def __init__(self, name):
        '''
        Constructor.
        Args:
            name(str): name of the namespace.
        '''
        self.name = name
        self.members = []


    def add_member(self, member_definition):
        '''
        Adds a member with the provided definition.
        Args:
            member_definition(str): member definition/body.
        ''' 
        self.members.append(member_definition)

    def __str__(self):
        result = []
        result.append('namespace %s' % self.name)
        result.append('{')
        result.extend(self.members)
        result.append('};')
        return '\n'.join(result)

class CppClassGen:
    '''
    Helper class for generation of C++ class.
    '''
    def __init__(self, name):
        '''
        Constructor.
        Args:
            name(str): name of the class.
        '''
        self.public = []
        self.private = []
        self.name = name

    def add_private_member(self, member_definition):
        '''
        Adds a private member with the provided definition.
        Args:
            member_definition(str): member definition/body.
        ''' 
        self.private.append(member_definition)

    def add_public_member(self, member_definition):
        '''
        Adds a public member with the provided definition.
        Args:
            member_definition(str): member definition/body.
        ''' 
        self.public.append(member_definition)

    def __str__(self):
        result = []
        result.append('class %s final' % self.name)
        result.append('{')
        if len(self.public) > 0:
            result.append('public:')
            for m in self.public:
                result.append(str(m))
        if len(self.private) > 0:
            result.append('private:')
            for m in self.private:
                result.append(str(m))
        result.append('};')
        return '\n'.join(result)

class HalideExpressionGenerator(NodeVisitor):
    '''
    Generator of halide graph from the NX model graph.
    TODO: Switch to Jinja templates for C++ code generation.
    TODO: All headers and code should go there, the expression generator
    TODO: should only provide the context. 
    '''
    def __init__(self, graph, quantize=False, total_bits=16, reserved_bits=3):
        super(HalideExpressionGenerator, self).__init__(graph)
        self.id_to_exp = {}
        self.listing = ''
        self.inputs = []
        self.outputs = []
        self.values = []
        self.quantize = quantize
        self.total_bits = total_bits
        self.reserved_bits = reserved_bits

    def generate(self, nodes, class_name):
        self.visit(nodes)

        past_inputs = ['%s' % i[1].name for i in self.inputs if i[0]]
        actual_inputs = ['%s' % i[1].name if i[1].name else i[1].id for i in self.inputs if not i[0]] 

        all_params = list(sorted(actual_inputs))
        all_params.extend(sorted(past_inputs))

        all_params = ', '.join(['const Halide::ImageParam& %s' % p for p in all_params])

        # Generating the class with setters for weights and constants. 
        evaluator = CppClassGen(class_name)
        for node in self.values:
            original_type = node.type
            quantized_type = self.data_type(node)
            evaluator.add_private_member('std::vector<%s> m_%s;' % (quantized_type, node.id))       
            evaluator.add_public_member('const std::vector<%s> get_%s() const { return m_%s; }' % (quantized_type, node.id.lower(), node.id))

            if original_type == quantized_type:
                evaluator.add_public_member('void set_%s(const std::vector<%s>&& v) { m_%s = std::move(v); };' % (node.id.lower(), original_type, node.id))
            else:
                evaluator.add_private_member('%s m_step_%s;' % (original_type, node.id))       
                evaluator.add_public_member('void set_%s(const std::vector<%s>&& v) { auto r = Quantize<%s, %s>(v, %d); m_%s = r.first; m_step_%s = r.second; };' % (node.id.lower(),
                                             original_type, original_type, quantized_type, self.reserved_bits, node.id, node.id))

        # Actually generating the function that will create the computation graph.
        create_eval_graph_method = 'Halide::Pipeline create_eval_graph(%s)\n {\n %s \n %s \n %s \n }\n' % (all_params, 'Halide::Var var1, var2;', self.listing, self.generate_return_value())
        evaluator.add_public_member(create_eval_graph_method)

        init_method = self.generate_init_method()
        evaluator.add_public_member(init_method)

        evaluator.add_private_member('Halide::Pipeline m_graph;')       
        evaluator.add_private_member('bool m_graphInitialized {false};')       
        evaluator.add_private_member('Halide::Buffer<int> m_bufferTimestamp { 1 };')       
        evaluator.add_private_member('Halide::ImageParam m_timestamp { Halide::type_of<int>(), 1 };')       

        self.generate_eval_method(evaluator)

        nspace = CppNamespaceGen('CNTK')
        nspace.add_member(str(evaluator))
        return self.generate_file_header() + str(nspace)

    def generate_eval_method(self, evaluator):
        past_inputs = [i[1] for i in self.inputs if i[0]]
        past_inputs = list(sorted(past_inputs, key=lambda n: n.name))

        for node in past_inputs:
            evaluator.add_private_member('std::vector<Halide::Buffer<%s>> m_buffer%s;' % (self.data_type(node), node.name))
            evaluator.add_private_member('Halide::ImageParam m_%s { Halide::type_of<%s>(), %d };' % (node.name, self.data_type(node), len(node.shape)))

        content = ''
        past_outputs = [o[1] for o in self.outputs if o[0]]
        past_outputs = list(sorted(past_outputs, key=lambda n: n.name))
        
        actual_inputs = ['%s' % i[1].name if i[1].name else i[1].id for i in self.inputs if not i[0]]
        actual_inputs = list(sorted(actual_inputs))

        input_params = actual_inputs + ['m_%s' % p.name for p in past_inputs]

        actual_outputs = ['%s' % o[1].name if o[1].name else o[1].id for o in self.outputs if not o[0]]
        actual_outputs = list(sorted(actual_outputs))

        output_params = actual_outputs + ['m_buffer%s[timestamp %% m_buffer%s.size()]' % (p.name, p.name) for p in past_inputs]

        content += 'void Evaluate(%s %s, %s)\n' % ('' if len(past_inputs) == 0 else 'int timestamp,', 
                                                   ', '.join(['const Halide::ImageParam& ' + i for i in actual_inputs]),
                                                   ', '.join(['Halide::Buffer<float>& ' + o for o in actual_outputs]))
        content += '{\n'
        content += '    if(!m_graphInitialized)\n'
        content += '    {\n'
        content += '        m_timestamp.set(m_bufferTimestamp);\n'
        content += '        m_graph = create_eval_graph(%s);\n' % (', '.join(input_params))
        content += '        m_graphInitialized = true;\n'
        content += '    }\n'

        for p in past_inputs:
            content += 'm_%s.set(m_buffer%s[((timestamp - %d) %% m_buffer%s.size())]);\n' % (p.name, p.name, p.original.attributes['offset'], p.name)

        if len(past_inputs) != 0:
            content += 'm_bufferTimestamp(0) = timestamp;\n'

        content += '    m_graph.realize({%s});\n' % (', '.join(output_params))
        content += '}\n'
        evaluator.add_public_member(content)

    def generate_init_method(self):
        content = '''
        public:
        void init(const std::string& weightFilePath)
        {
            boost::property_tree::ptree root;
            boost::property_tree::read_json(weightFilePath.c_str(), root);

            auto get_value = [&](const std::string& name)
            {
                std::vector<float> result;
                for (auto& v : root.get_child(name))
                    result.push_back(v.second.get_value<float>());
                return result;
            };
        '''
        for node in self.values:
            content += 'set_%s(get_value("%s"));\n' % (node.id.lower(), node.id)

        # Initializing the past value input buffers.
        past_inputs = [i[1] for i in self.inputs if i[0]]
        for node in past_inputs:
            past_node = node.original
            content += 'm_buffer%s.resize(%d, Halide::Buffer<%s>(%s));\n' % (node.name, past_node.attributes['offset'] + 1, self.data_type(node), ','.join(node.cm_shape))
        content += '}\n'
        return content

    def visit_parameter(self, node):
        exp = self.generate_value(node)
        self.listing += exp + '\n\n'
        self.id_to_exp[node.id] = '%s' % node.id

    def visit_constant(self, node):
        exp = self.generate_value(node)
        self.listing += exp + '\n\n'
        self.id_to_exp[node.id] = '%s' % node.id

    def visit_input(self, node):
        is_past_value = node.original is not None
        name = '%s' % (node.name if node.name else node.id)
        self.id_to_exp[node.id] = '%s' % name
        self.inputs.append((is_past_value, node))

    def visit_output(self, node):
        operands = node.predecessors
        is_past_value = node.original is not None
        if len(operands) != 1:
            raise ValueError('Output value is expected to have a single operand, given %s' % str(operands))
        self.id_to_exp[node.id] = '%s' % operands[0].id
        self.outputs.append((is_past_value, node))

    def index_vars(self, node):
        if len(node.shape) == 1 or len(node.shape) == 0:
            return 'var1'
        elif len(node.shape) == 2:
            return 'var1, var2'
        else:
            set_trace()
            raise ValueError("Shape is not supported %s" % str(node.shape))

    def visit_primitive_function(self, node):
        op_name = node.op_name
        if op_name == 'Times':
            self.generate_times(node)
        elif op_name == 'Plus':
            self.generate_plus(node)
        elif op_name == 'Minus':
            self.generate_minus(node)
        elif op_name == 'Log':
            self.generate_log(node)
        elif op_name == 'Slice':
            self.generate_slice(node)
        elif op_name == 'Splice':
            self.generate_splice(node)
        elif op_name == 'StableSigmoid' or op_name == 'Sigmoid':
            self.generate_stable_sigmoid(node)
        elif op_name == 'Tanh':
            self.generate_tanh(node)
        elif op_name == 'ElementTimes':
            self.generate_element_times(node)
        elif op_name == 'PastValue':
            self.generate_past_state_selector(node)
        else:
            set_trace()
            raise ValueError('Not implemented function %s' % node.op_name)
        self.id_to_exp[node.id] = '%s' % node.id

    def visit_node(self, node):
        if node.op_name == 'Quantize':
            self.generate_quantization(node)
        else:
            raise ValueError('Unexpected node' % node)
        self.id_to_exp[node.id] = '%s' % node.id

    def generate_quantization(self, node):
        operands = node.predecessors
        if len(operands) != 1:
            raise ValueError('Operation "quantize" expects 1 operand, given %s', str(operands))
        shape = node.shape if len(node.shape) > 0 else (1,)
        shape_arg = '%d, %d' % (shape[0], shape[1]) if len(shape) == 2 else '%d' % shape[0]
        exp = 'std::vector<Halide::Func> %s; %s = Quantize<%s, %s>(%s, %s, %d)' % tuple([node.id, node.id, node.type, self.data_type(node), 
                                                                                        self.id_to_exp[operands[0].id], shape_arg, self.reserved_bits])
        self.listing += exp + ';\n'

    def generate_binary_call(self, node, op_name):
        operands = node.predecessors
        if len(operands) != 2:
            raise ValueError('Operation "%s" expects 2 operands, given %s', op_name, str(operands))
        exp = 'Halide::Func %s("%s"); %s = %s(%s, %s, %d)' % tuple([node.id, node.id, node.id, op_name] + [self.id_to_exp[o.id] for o in operands] + [self.total_num_elements(node.shape)])
        if len(node.successors) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.id
        self.listing += exp + ';\n'

    def generate_call(self, node, op_name, operands):
        str_operands = ','.join(operands)
        exp = 'Halide::Func %s("%s"); %s = %s(%s)' % tuple([node.id, node.id, node.id, op_name, str_operands])
        if len(node.successors) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.id
        self.listing += exp + ';\n'

    def generate_unary_call(self, node, op_name):
        operands = node.predecessors
        if len(operands) != 1:
            raise ValueError('Operation "%s" expects 1 operand, given %s', op_name, str(operands))
        exp = 'Halide::Func %s("%s"); %s = %s(%s)' % tuple([node.id, node.id, node.id, op_name, self.id_to_exp[operands[0].id]])
        if len(node.successors) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.id
        self.listing += exp + ';\n'

    def generate_times(self, node):
        operands = node.predecessors
        if len(operands) != 2:
            raise ValueError('Expecting 2 operands')
        vector = operands[0]
        if len(vector.shape) != 0 and len(vector.shape) != 1:
            set_trace()
            raise ValueError("Times is currently supported only for 1D * 2D, given %s" % str(vector.shape))

        matrix = operands[1]
        shape = matrix.shape if len(matrix.shape) > 0 else (1,)        

        op_name = 'MatrixByVectorTimes'
        if node.quantize:
            op_name += 'Quantized'
        self.generate_call(node, op_name, [self.id_to_exp[o.id] for o in reversed(operands)] + [str(shape[1]), str(shape[0])])

    def generate_element_times(self, node):
        self.generate_binary_call(node, 'ElementTimes')

    def generate_plus(self, node):
        self.generate_binary_call(node, 'Plus')

    def generate_minus(self, node):
        self.generate_binary_call(node, 'Minus')

    def generate_stable_sigmoid(self, node):
        self.generate_unary_call(node, 'Sigmoid<%s>' % self.data_type(node))

    def generate_tanh(self, node):
        self.generate_unary_call(node, 'Tanh')

    def generate_log(self, node):
        self.generate_unary_call(node, 'Log')

    def generate_slice(self, node):
        if len(node.predecessors) != 1:
            set_trace()
            raise ValueError('Operation "slice" expects 1 operand')
        operand = node.predecessors[0]
        if len(operand.shape) == 1:
            begin = node.model.attributes['beginIndex']
            end = node.model.attributes['endIndex']
            if 'sliceStrides' in node.model.attributes and node.model.attributes['sliceStrides'] != 1:
                set_trace()
                raise ValueError('Unexpected stride "%s", only stride of 1 is currently supported' % str(node.model.attributes['sliceStrides']))
            exp = 'Halide::Func %s("%s"); %s = Slice(%s, %d, %d)' % (node.id, node.id, node.id, self.id_to_exp[operand.id], begin, end)
        else:
            raise ValueError('Slice is not supported on node of shape %s' % str(node.shape)) 
        if len(node.successors) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.id
        self.listing += exp + ';\n'

    def generate_splice(self, node):
        operands = node.predecessors
        if len(operands) != 2:
            raise ValueError('Operation "splice" expects 2 operands')
        operand1 = operands[0]
        operand2 = operands[1]
        if len(operand1.shape) != 1 or len(operand2.shape) != 1:
            raise ValueError('Currently splice only supports vectors as operands')

        exp = 'Halide::Func %s("%s"); %s = Splice(%s, %s, %d, %d)' % (node.id, node.id, node.id, self.id_to_exp[operand1.id],
                                                                      self.id_to_exp[operand2.id], operand1.shape[0], operand2.shape[0])
        if len(node.successors) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.id
        self.listing += exp + ';\n'

    def generate_file_header(self):
        header  = '#pragma once\n'
        header += '#include <vector>\n'
        header += '#include <string>\n'
        header += '#include "HalideDNNLib.h"\n'
        header += '#pragma warning(push)\n'
        header += '#pragma warning(disable : 4715)\n'
        header += '#include <boost/property_tree/ptree.hpp>\n'
        header += '#include <boost/property_tree/json_parser.hpp>\n'
        header += '#pragma warning(pop)\n'
        header += '\n'
        return header;

    def generate_return_value(self):
        past_outputs = [o[1] for o in self.outputs if o[0]]
        actual_outputs = [o[1] for o in self.outputs if not o[0]]
        # Sorting by name
        all_outputs = list(sorted(actual_outputs, key = lambda n: n.name if n.name else n.id))
        all_outputs.extend(sorted(past_outputs, key = lambda n: n.name))
        return 'return Halide::Pipeline({ %s });' % ', '.join(['%s/*%s*/' % (self.id_to_exp[o.id], o.name if o.name else o.id) for o in all_outputs])

    def data_type(self, node):
        if node.quantize:
            if self.total_bits == 16:
                return 'short'
            elif self.total_bits == 8:
                return 'char'
            else:
                raise ValueError('Unsupported number of total quantized bits.')
        else:
            return node.type

    def total_num_elements(self, shape):
        return shape[0] if len(shape) == 1 else 1 if len(shape) == 0 else functools.reduce(lambda x, y: x*y, shape)

    def generate_value(self, node):
        type = self.data_type(node)
        if len(node.shape) == 2:
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, %s, "%s");\n' % (node.id, type, node.id, node.shape[0], node.shape[1], node.id)
        elif len(node.shape) == 1:
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, "%s");\n' % (node.id, type, node.id, node.shape[0], node.id)
        elif len(node.shape) == 0: # Scalar represent as array
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, "%s");\n' % (node.id, type, node.id, 1, node.id)
        else:
            set_trace()
            raise ValueError('Unexpected shape encountered, only 1 and 2D are currently supported %s' % node)
        
        if not node.quantize:
            expression += 'Halide::Func %s("%s"); %s(%s) = b_%s(%s);' % (node.id, node.id, node.id, self.index_vars(node), node.id, self.index_vars(node))
        else:
            expression += 'Halide::Func f_%s("f_%s"); f_%s(%s) = b_%s(%s);\n' % (node.id, node.id, node.id, self.index_vars(node), node.id, self.index_vars(node))
            expression += 'Halide::Func f_step_%s("f_step_%s"); f_step_%s() = m_step_%s;\n' % (node.id, node.id, node.id, node.id)
            expression += 'std::vector<Halide::Func> %s { f_%s, f_step_%s };\n' % (node.id, node.id, node.id)
        self.values.append(node)
        return expression

    def generate_past_state_selector(self, node):
        operands = node.predecessors
        if len(operands) != 2:
            raise ValueError('Selection of past state expects 2 operands, given %s', str(operands))
        input = operands[0].model
        state = operands[1].model
        is_same_shape = input.shape == state.shape
        if not is_same_shape and len(state.shape) != 0:
            raise ValueError('Shape of the past value node does not match the shape of the state. Implicit broadcasting is currently not supported.')

        if len(input.shape) != 1:
            raise ValueError('Selection of past value state is currenlty supported only for vectors.')

        shape = node.shape if len(node.shape) > 0 else (1,)
        exp = 'Halide::Func %s("%s"); %s(var1) = Halide::select(m_timestamp(0) < %d, %s(%s), %s(var1));' % (node.id, node.id, node.id, 
                                                                                                            node.model.attributes['offset'], 
                                                                                                            self.id_to_exp[operands[1].id], 
                                                                                                            'var1' if is_same_shape else '0',
                                                                                                            self.id_to_exp[operands[0].id])
        self.listing += exp + ';\n'
