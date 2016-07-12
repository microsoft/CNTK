# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# This is a tool that semi-automatically creates operator classes from
# CNTK.Core.bs and puts them into ops.py.
# It is just a temporary solution until we have CNTK as a library.

import os
import re
import sys

REGEX_STANDARD = re.compile(r'(?P<operator>\w+)\((?P<operands>.*?)\) = .*')
REGEX_COMPNODE = re.compile(
    r'(?P<operator>\w+) ?\((?P<operands>.*?)\)\s*=\s*new\s*ComputationNode\s*\[\s*(?P<inputs>.*?inputs\s*=.*?[;|\/])?')
REGEX_ALIAS = re.compile(r'(?P<operator>[A-Z]\w*)\s*=\s*(?P<alias>\w+)\s*(//.*|#.*|)$')
# ElementDivide(aMatrix, anotherMatrix, tag='') = ElementTimes(aMatrix,
# Reciprocal(anotherMatrix))
REGEX_INSTANTIATION = re.compile(
    r'(?P<operator>\w+)\((?P<operands>.*?)\)\s*=\s*(?P<inst_operator>\w+)\s*\((?P<inst_operands>.*?)\)\s*(//.*|#.*|)')

REGEX_COMMENT = re.compile(r'/\*.*\*/')

REGEX_CNTK2_START = re.compile(r'^\s*CNTK2\s*=\s*\[', re.IGNORECASE)

OPERANDS_TO_IGNORE = {"tag=''"}
OPERATORS_TO_IGNORE = {'Print', 'Fail', 'Format', 'Replace',
                       'Substr', 'Chr', 'Length', 'ConstantFromString',
                       'ElementDivide', 'Ceil', 'Round', 'Constant'}

INPUT_NODES = ['Input', 'SparseInput']
IMAGE_INPUT_NODES = ['ImageInput', 'SparseImageInput']


class Operand(object):

    def __init__(self, name, init_value):
        if '/*' in name:
            # Example:
            # Pooling(input, poolKind/*'max'|'average'*/, ....
            name = name[:name.index('/*')]

        self.name = name

        if init_value is None:
            self.init_value = None
        else:
            init_value = REGEX_COMMENT.sub('', init_value)

            if init_value[0] == "'" and init_value[-1] == "'":
                self.init_value = init_value[1:-1]
            else:
                if init_value.lower() == 'false':
                    self.init_value = False
                elif init_value.lower() == 'true':
                    self.init_value = True
                elif '.' in init_value:
                    self.init_value = float(init_value)
                else:
                    self.init_value = int(init_value)


# BrainScript parameter names are not always consisten. Here, we fix the
# obvious misnamings.
SMOOTH_NAMING = {
    'val': 'value',
    'from': 'from_'
}


class CompNodeOperator(object):
    COMP_NODE_TEMPLATE = """\
class %(name)s(%(parentclass)s):
    def __init__(self, %(signature)sop_name='%(namespace)s%(name)s', name=None):
        super(%(name)s, self).__init__(params=[%(paramlist)s], op_name=op_name, name=name)
%(initialization)s
        self.params_with_defaults = [%(params_with_defaults)s]
        self.inputs = [%(inputs_string)s]
"""

    def _smooth(self, name):
        if name in SMOOTH_NAMING:
            return SMOOTH_NAMING[name]
        return name

    def __init__(self, comp_match, namespace=''):
        self.namespace = namespace
        self.name = comp_match.group('operator')

        if self.name in INPUT_NODES:
            self.parentclass = '_InputComputationNodeBase'
        elif self.name in IMAGE_INPUT_NODES:
            self.parentclass = '_ImageInputComputationNodeBase'
        else:
            self.parentclass = 'ComputationNode'

        self.raw_operands = comp_match.group('operands')
        
        self.raw_inputs = []
        self.inputs = []
        try:
            self.raw_inputs = comp_match.group('inputs')
            if (self.raw_inputs):
                self.raw_inputs = self.raw_inputs.split("inputs =")[1]                
                self. inputs = ["'%s'" % i for i in re.split("[\/|:| |\)|\(|;]", 
                                                    self.raw_inputs.strip()) if i]                    
        except IndexError:            
            pass
            
        self.inputs_string = ', '.join(self.inputs)            
        
        self.operands = []                
        for op in self.raw_operands.split(','):
            if op.strip() in OPERANDS_TO_IGNORE:
                continue

            parts = op.split('=')
            if len(parts) == 1:
                self.operands.append(
                    Operand(self._smooth(parts[0].strip()), None))
            elif len(parts) == 2:
                self.operands.append(
                    Operand(self._smooth(parts[0].strip()), parts[1].strip()))
            else:
                raise ValueError('Did not expect this format')

        self.signature = ", ".join(self.sig(op) for op in self.operands)
        if self.signature:
            self.signature += ", "

        self.initialization = "\n".join(
            (" " * 8 + "self.%s = %s" % (op.name, op.name) for op in self.operands))

        self.paramlist = ", ".join(("'%s'" % op.name for op in self.operands))                
        
        default_init_started = False
        params_with_defaults = []
        for op in self.operands:
            if op.init_value is not None:
                params_with_defaults.append("'%s'" % op.name)
                default_init_started = True
            # Ensure that arguments with default values are not followed by
            # arguments without default values.
            assert op.init_value is None or default_init_started

        self.params_with_defaults = ', '.join(params_with_defaults)

    def __str__(self):
        return self.COMP_NODE_TEMPLATE % self.__dict__

    def sig(self, op):
        name, init = op.name, op.init_value
        if init is None:
            return name

        if type(init) == str:
            return "%s='%s'" % (op.name, op.init_value)
        else:
            return "%s=%s" % (op.name, op.init_value)


class AliasOperator(object):

    def __init__(self, alias_match):
        self.name = alias_match.group('operator')
        self.alias = alias_match.group('alias')

    def __str__(self):
        return "%(name)s = %(alias)s" % self.__dict__


class InstantiationOperator(CompNodeOperator):
    INST_NODE_TEMPLATE = """\
class %(name)s(%(inst_operator)s):
    def __init__(self, %(signature)s op_name='%(namespace)s%(name)s', name=None):
        super(%(name)s, self).__init__(%(inst_operands)s, op_name=op_name, name=name)
        self.params=[%(paramlist)s]
        self.params_with_defaults = [%(params_with_defaults)s]
"""

    def __init__(self, match, namespace=''):
        super(InstantiationOperator, self).__init__(match, namespace)
        self.inst_operator = match.group('inst_operator')
        raw_inst_operands = match.group('inst_operands').split(',')
        inst_operands = []
        for operand in raw_inst_operands:

            parts = operand.split('=')
            if len(parts) == 1:
                elem = parts[0].strip()
                if ':' in elem:
                    elem = "'<not yet supported>'"
                inst_operands.append(elem)
            elif len(parts) == 2:
                init = parts[1].strip()
                if ':' in init:
                    init = "'<not yet supported>'"
                inst_operands.append(
                    '%s=%s' % (parts[0].strip(), self._smooth(init)))
            else:
                raise ValueError(
                    'Did not expect more than 1 equal sign: %s' % operand)

        self.inst_operands = ', '.join(inst_operands)

    def __str__(self):
        return self.INST_NODE_TEMPLATE % self.__dict__

CNTK1_MANUAL_PREFIX = """\
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

# This file is auto-generated by _fetch_ops.py.

from cntk.graph import ComputationNode, _InputComputationNodeBase, _ImageInputComputationNodeBase

class Slice(ComputationNode):
    def __init__(self, beginIndex, endIndex, input, axis=1, op_name='Slice',
            name=None):
        super(Slice, self).__init__(params=['beginIndex', 'endIndex', 'input', 'axis'], op_name=op_name, name=name)
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        self.input = input
        self.axis = axis
        self.inputs = ['input']
        self.params_with_defaults = []

class Splice(ComputationNode):
    def __init__(self, beginIndex, endIndex, input, axis=1, op_name='Splice',
            name=None):
        super(Splice, self).__init__(params=['beginIndex', 'endIndex', 'input', 'axis'], op_name=op_name, name=name)
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        self.input = input
        self.axis = axis
        self.inputs = ['input']
        self.params_with_defaults = []
    
class ElementDivide(ComputationNode):
    def __init__(self, aMatrix, anotherMatrix, op_name='ElementDivide', name=None):
        super(ElementDivide, self).__init__(params=['aMatrix', 'anotherMatrix'], op_name=op_name, name=name)
        self.aMatrix = aMatrix
        self.anotherMatrix = anotherMatrix
        self.inputs = ['aMatrix', 'anotherMatrix']
        self.params_with_defaults = []
        
class Round(ComputationNode):
    def __init__(self, x, op_name='Round', name=None):
        super(Round, self).__init__(params=['x'], op_name=op_name, name=name)
        self.x = x
        self.inputs = ['x']
        self.params_with_defaults = []
        
class Ceil(ComputationNode):
    def __init__(self, x, op_name='Ceil', name=None):
        super(Ceil, self).__init__(params=['x'], op_name=op_name, name=name)
        self.x = x
        self.inputs = ['x']
        self.params_with_defaults = []

class If(ComputationNode):
    def __init__(self, cond, thenVal, elseVal, op_name='BS.Boolean.If', name=None):
        super(If, self).__init__(
            params=['cond', 'thenVal', 'elseVal'], op_name=op_name, name=name)
        self.cond = cond
        self.thenVal = thenVal
        self.elseVal = elseVal
        self.params_with_defaults = []
        self.inputs = ['cond', 'thenVal', 'elseVal']

"""

CNTK2_MANUAL_PREFIX = """\
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

# This file is auto-generated by _fetch_ops.py.

from cntk.graph import ComputationNode, _InputComputationNodeBase, _ImageInputComputationNodeBase

class Slice(ComputationNode):
    def __init__(self, _, beginIndex, endIndex, axis=1, op_name='CNTK2.Slice',
            name=None):
        super(Slice, self).__init__(params=['_', 'beginIndex', 'endIndex', 'axis'], op_name=op_name, name=name)
        self._ = _
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        self.axis = axis
        self.inputs = ['_']
        self.params_with_defaults = ['axis']

class Splice(ComputationNode):
    def __init__(self, _, axis=1, op_name='CNTK2.Splice',
            name=None):
        super(Splice, self).__init__(params=['_', 'axis'], op_name=op_name, name=name)
        self._ = _
        self.axis = axis
        self.inputs = ['_']
        self.params_with_defaults = ['axis']

class Ceil(ComputationNode):
    def __init__(self, _, op_name='CNTK2.Ceil', name=None):
        super(Ceil, self).__init__(params=['_'], op_name=op_name, name=name)
        self._ = _
        self.inputs = ['_']
        self.params_with_defaults = []

class ElementDivide(ComputationNode):
    def __init__(self, _, y,  op_name='CNTK2.ElementDivide', name=None):
        super(ElementDivide, self).__init__(params=['_', 'y'], op_name=op_name, name=name)
        self._ = _
        self.y = y
        self.inputs = ['_', 'y']
        self.params_with_defaults = []

class Round(ComputationNode):
    def __init__(self, _, op_name='CNTK2.Round', name=None):
        super(Round, self).__init__(params=['_'], op_name=op_name, name=name)
        self._ = _
        self.inputs = ['_']
        self.params_with_defaults = []
        
class ReduceLogSum(ComputationNode):
    def __init__(self, _, axis=0, op_name='CNTK2.ReduceLogSum',
            name=None):
        super(ReduceLogSum, self).__init__(params=['_', 'axis'], op_name=op_name, name=name)
        self._ = _
        self.axis = axis
        self.inputs = ['_']
        self.params_with_defaults = ['axis']        

"""

def convert_bs_to_python(bs_fn, out_dir):
    # We have to append these at the end to make sure, because the
    # BrainScript file does not keep order.
    alias_ops_cntk1 = []
    alias_ops_cntk2 = []

    inst_ops_cntk1 = []
    inst_ops_cntk2 = []

    IGNORE_SECT, COMP_NODE_SECT, STAND_NODE_SECT, CNTK2_SECT = 0, 1, 2, 3

    with open(os.path.join(out_dir, 'cntk1.py'), 'w') as cntk1f, \
        open(os.path.join(out_dir, 'cntk2.py'), 'w') as cntk2f:

        cntk1f.write(CNTK1_MANUAL_PREFIX)
        cntk2f.write(CNTK2_MANUAL_PREFIX)

        pyf = cntk1f

        in_computation_node_section = False
        in_standard_node_section = False

        part_of_file = IGNORE_SECT
        for line in open(bs_fn, 'r'):
            line = line.strip()

            if line.lower().startswith('# computationnodes'):
                part_of_file = COMP_NODE_SECT
            elif line.lower().startswith('# standard functions'):
                part_of_file = STAND_NODE_SECT
            elif REGEX_CNTK2_START.match(line):
                part_of_file = CNTK2_SECT
                pyf = cntk2f
            elif part_of_file == CNTK2_SECT and line == ']':
                part_of_file = COMP_NODE_SECT
                pyf = cntk1f
            elif line.lower().startswith('# common macros'):
                break

            if part_of_file==STAND_NODE_SECT:
                standard_match = REGEX_STANDARD.match(line)
                if standard_match:
                    po = CompNodeOperator(standard_match)
                    if po.name in OPERATORS_TO_IGNORE:
                        continue
                    pyf.write(str(po) + '\n')
                    continue

            if part_of_file in [COMP_NODE_SECT, CNTK2_SECT]:
                comp_match = REGEX_COMPNODE.match(line)
                if comp_match:
                    ns = 'CNTK2.' if part_of_file==CNTK2_SECT else ''
                    try:
                        op = CompNodeOperator(comp_match, ns)
                    except ValueError:
                        print('ERROR while parsing: %s'%line)
                        continue
                    if op.name in OPERATORS_TO_IGNORE and part_of_file==COMP_NODE_SECT:
                        continue
                    pyf.write(str(op) + '\n')
                    continue

                alias_match = REGEX_ALIAS.match(line)
                if alias_match:
                    alias_ops = alias_ops_cntk1 if part_of_file==COMP_NODE_SECT else alias_ops_cntk2
                    
                    alias_ops.append(alias_match)
                    continue

                instantiation_match = REGEX_INSTANTIATION.match(line)
                if instantiation_match:
                    inst_ops = inst_ops_cntk1 if part_of_file==COMP_NODE_SECT else inst_ops_cntk2
                    inst_ops.append(instantiation_match)
                    continue

        for alias_ops, pyf in [(alias_ops_cntk1, cntk1f), (alias_ops_cntk2, cntk2f)]:
            for match in alias_ops:
                op = AliasOperator(match)
                if op.name in OPERATORS_TO_IGNORE:
                    continue
                pyf.write(str(op) + '\n')

        for inst_ops, pyf, ns in [(inst_ops_cntk1, cntk1f, ns),
                (inst_ops_cntk2, cntk2f, ns)]:
            for match in inst_ops:
                op = InstantiationOperator(match, ns)
                if op.name in OPERATORS_TO_IGNORE:
                    continue
                pyf.write(str(op) + '\n')

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-c", "--cntk", dest="cntkcore_defs",
                      help="CNTK.core.bs file", metavar="FILE")
    parser.add_option("-o", "--output", dest="output",
                      help="output directory where cntk1.py and cntk2.py " +
                      "will be placed", default='.')

    (opts, args) = parser.parse_args()

    if opts.cntkcore_defs:
        CNTKCORE_DEFS = opts.cntkcore_defs
    else:
        CUR_DIR = os.path.dirname(__file__)
        CNTKCORE_DEFS = os.path.join(CUR_DIR, '..', '..', '..', '..', 'Source',
                                     'CNTK', 'BrainScript', 'CNTKCoreLib', 'CNTK.core.bs')

    print('Using %s' % CNTKCORE_DEFS)

    outdir = opts.output

    convert_bs_to_python(CNTKCORE_DEFS, outdir)
