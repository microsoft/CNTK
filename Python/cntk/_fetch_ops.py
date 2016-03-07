# This is a tool that semi-automatically creates operator classes from
# CNTK.Core.bs and puts them into _ops.py.
# It is just a temporary solution until we have CNTK as a library.
# 
# Ignores deprecated symbols (e.g. Parameter)

import os
import re

DIRNAME_OF_THIS_FILE = os.path.abspath(os.path.dirname(__file__))

# BrainSCript's node definitions
CNTKCORE_DEFS = os.path.abspath(os.path.join(DIRNAME_OF_THIS_FILE, 'CNTK.core.bs'))

#REGEX = re.compile('ComputationNodePtr\s+(.*);', re.MULTILINE)
REGEX_COMPNODE = re.compile(r'(?P<operator>\w+)\((?P<operands>.*?)\) = new ComputationNode \[')
REGEX_COMMENT = re.compile(r'/\*.*\*/')

OPERANDS_TO_IGNORE = {"tag=''"}

class Operand(object):
    def __init__(self, name, init_value):
        self.name = name

        if init_value is None:
            self.init_value = None
        else:
            init_value = REGEX_COMMENT.sub('', init_value)

            if init_value[0]=="'" and init_value[-1]=="'":
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

COMP_NODE_TEMPLATE = """\
class %(name)s(ComputationNode):
    def __init__(self, %(signature)s):
        super(%(name)s, self).__init__('%(name)s')
%(initialization)s
"""

class CompNodeOperator(object):
    def __init__(self, comp_node):
        self.name = comp_node.group('operator')
        self.raw_operands = comp_node.group('operands')

        self.operands = []
        for op in self.raw_operands.split(','):
            if op.strip() in OPERANDS_TO_IGNORE:
                continue

            parts = op.split('=')
            if len(parts)==1:
                self.operands.append(Operand(parts[0].strip(), None))
            elif len(parts)==2:
                self.operands.append(Operand(parts[0].strip(), parts[1].strip()))
            else:
                raise ValueError('Did not expect this format')

        self.signature = ", ".join(self.sig(op) for op in self.operands )

        self.initialization = "\n".join((" "*8 + "self.%s = %s"%(op.name, op.name) for op in self.operands)) 

        # Ensure that arguments with default values are not followed by
        # arguments without default values.
        default_init_started = False
        for op in self.operands:
            if op.init_value is not None:
                default_init_started = True
            assert op.init_value is None or default_init_started


    def __str__(self):
        return COMP_NODE_TEMPLATE%self.__dict__

    def sig(self, op):
        name, init = op.name, op.init_value
        if init is None:
            return name

        if type(init) == str:
            return "%s='%s'"%(op.name, op.init_value) 
        else:

            return "%s=%s"%(op.name, op.init_value) 

def convert_bs_to_python(bs_fn, py_fn):
    with open(py_fn, 'w') as pyf:
        pyf.write('from graph import ComputationNode\n')
        
        for line in open(bs_fn, 'r'):
            comp_node = REGEX_COMPNODE.match(line)
            if comp_node:
                po = CompNodeOperator(comp_node)
                pyf.write(str(po)+'\n')

if __name__=='__main__':
    convert_bs_to_python(CNTKCORE_DEFS, "_ops.py")

