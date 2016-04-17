# This is a tool that semi-automatically creates operator classes from
# CNTK.Core.bs and puts them into ops.py.
# It is just a temporary solution until we have CNTK as a library.

import os
import re
import sys

REGEX_STANDARD = re.compile(r'(?P<operator>\w+)\((?P<operands>.*?)\) = .*')
REGEX_COMPNODE = re.compile(
    r'(?P<operator>\w+)\((?P<operands>.*?)\) = new ComputationNode \[')
REGEX_ALIAS = re.compile(r'(?P<operator>\w+) = (?P<alias>\w+)\s*(//.*|)')
# ElementDivide(aMatrix, anotherMatrix, tag='') = ElementTimes(aMatrix, Reciprocal(anotherMatrix))
REGEX_INSTANTIATION = re.compile( r'(?P<operator>\w+)\((?P<operands>.*?)\) = (?P<inst_operator>\w+)\s*\((?P<inst_operands>.*?)\)\s*(//.*|)')

REGEX_COMMENT = re.compile(r'/\*.*\*/')

OPERANDS_TO_IGNORE = {"tag=''"}
OPERATORS_TO_IGNORE = {'Print', 'Fail', 'Format', 'Replace', 'Substr', 'Chr', 'Length', 'ConstantFromString', 'ElementDivide'}

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
    def __init__(self, %(signature)s, name='%(name)s', var_name=None):
        super(%(name)s, self).__init__(params=[%(paramlist)s], name=name, var_name=var_name)
%(initialization)s
        self.params_with_defaults = [%(params_with_defaults)s]
"""

    def _smooth(self, name):
        if name in SMOOTH_NAMING:
            return SMOOTH_NAMING[name]
        return name

    def __init__(self, comp_match):
        self.name = comp_match.group('operator')

        if self.name in INPUT_NODES:
            self.parentclass = 'InputComputationNodeBase'
        elif self.name in IMAGE_INPUT_NODES:
            self.parentclass = 'ImageInputComputationNodeBase'
        else:
            self.parentclass = 'ComputationNode'

        self.raw_operands = comp_match.group('operands')

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
    def __init__(self, %(signature)s, name='%(name)s', var_name=None):
        super(%(name)s, self).__init__(%(inst_operands)s, name=name, var_name=var_name)
        self.params=[%(paramlist)s]
        self.params_with_defaults = [%(params_with_defaults)s]
"""

    def __init__(self, match):
        super(InstantiationOperator, self).__init__(match)
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

OPS_PREAMBLE = """\
# This file is auto-generated by _fetch_ops.py.

from cntk.graph import ComputationNode, InputComputationNodeBase, ImageInputComputationNodeBase

class Slice(ComputationNode):
    def __init__(self, beginIndex, endIndex, input, axis=1, name='Slice',
            var_name=None):
        super(Slice, self).__init__(params=['value', 'format'], name=name, var_name=var_name)
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        self.input = input
        self.axis = axis

class Splice(ComputationNode):
    def __init__(self, beginIndex, endIndex, input, axis=1, name='Splice',
            var_name=None):
        super(Splice, self).__init__(params=['value', 'format'], name=name, var_name=var_name)
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        self.input = input
        self.axis = axis

class ElementDivide(ComputationNode):
    def __init__(self, aMatrix, anotherMatrix, name='ElementDivide', var_name=None):
        super(ElementDivide, self).__init__(params=['aMatrix', 'anotherMatrix'], name=name, var_name=var_name)
        self.aMatrix = aMatrix
        self.anotherMatrix = anotherMatrix
       

"""


def convert_bs_to_python(bs_fn, pyf):
    # We have to append these at the end to make sure, because the
    # BrainScript file does not keep order.
    alias_ops = []
    inst_ops = []

    with pyf:
        pyf.write(OPS_PREAMBLE)

        in_computation_node_section = False
        in_standard_node_section = False

        for line in open(bs_fn, 'r'):

            if line.lower().startswith('# computationnodes'):
                in_computation_node_section = True
                in_standard_node_section = False
            elif line.lower().startswith('# standard functions'):
                in_computation_node_section = False
                in_standard_node_section = True
            elif line.lower().startswith('# common macros'):
                break

            if in_standard_node_section:
                standard_match = REGEX_STANDARD.match(line)
                if standard_match:
                    po = CompNodeOperator(standard_match)
                    pyf.write(str(po) + '\n')
                    continue

            if in_computation_node_section:
                comp_match = REGEX_COMPNODE.match(line)
                if comp_match:
                    op = CompNodeOperator(comp_match)
                    if op.name in OPERATORS_TO_IGNORE:
                        continue
                    pyf.write(str(op) + '\n')
                    continue

                alias_match = REGEX_ALIAS.match(line)
                if alias_match:
                    alias_ops.append(alias_match)
                    continue

                instantiation_match = REGEX_INSTANTIATION.match(line)
                if instantiation_match:
                    inst_ops.append(instantiation_match)
                    continue

        for match in alias_ops:
            op = AliasOperator(match)
            if op.name in OPERATORS_TO_IGNORE:
                continue
            pyf.write(str(op) + '\n')

        for match in inst_ops:
            op = InstantiationOperator(match)
            if op.name in OPERATORS_TO_IGNORE:
                continue
            pyf.write(str(op) + '\n')

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-c", "--cntk", dest="cntkcore_defs",
            help="CNTK.core.bs file", metavar="FILE")
    parser.add_option("-o", "--output", dest="output",
            help="output file, or '-' for stdout", default='-')

    (opts, args) = parser.parse_args()

    if opts.cntkcore_defs:
        CNTKCORE_DEFS = opts.cntkcore_defs
    else:
        CUR_DIR = os.path.dirname(__file__)
        CNTKCORE_DEFS = os.path.join(CUR_DIR, '..', '..', '..', '..', 'Source',
                'CNTK', 'BrainScript', 'CNTKCoreLib', 'CNTK.core.bs')

    print('Using %s'%CNTKCORE_DEFS)

    if opts.output == '-':
        outfile = sys.stdout
    else:
        outfile = open(opts.output, "w")
        
    convert_bs_to_python(CNTKCORE_DEFS, outfile)
