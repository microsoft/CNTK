from ..utils._fetch_ops import *

import pytest


@pytest.mark.parametrize("input_line, expected", [
    # Format of expected: [OperatorName, [(OperandName_1, OperandInitValue_1),
    # ...]]
    (r"Times(A, B, outputRank=1, tag='') = new ComputationNode [ operation = 'Times' ; inputs = ( A : B ) /*plus the function args*/ ]",
     ['Times', [('A', None), ('B', None), ('outputRank', 1)]]),


    (r"Convolution(weightNode, inputValueNode, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0, imageLayout='CHW', tag='') = new ComputationNode [ operation = 'Convolution' ; inputs = (weightNode : inputValueNode) /*plus the function args*/ ]",
     ['Convolution', [('weightNode', None), ('inputValueNode', None), ('kernelWidth', None), ('kernelHeight', None), ('outputChannels', None), ('horizontalSubsample', None), ('verticalSubsample', None), ('zeroPadding', False), ('maxTempMemSizeInSamples', 0), ('imageLayout', 'CHW')]]),

    (r"LearnableParameter(rows, cols, learningRateMultiplier = 1.0, init = 'uniform'/*|fixedValue|gaussian|fromFile*/, initValueScale = 1, value = 0, initFromFilePath = '', initOnCPUOnly=true, randomSeed=-1, tag='') = new ComputationNode [ operation = 'LearnableParameter' ; shape = new TensorShape [ dims = (rows : cols) ] /*plus the function args*/ ]",
     ['LearnableParameter', [('rows', None), ('cols', None), ('learningRateMultiplier', 1.0), ('init', 'uniform'), ('initValueScale', 1), ('value', 0), ('initFromFilePath', ''), ('initOnCPUOnly', True), ('randomSeed', -1)]]),
])
def test_parsing_comp_node(input_line, expected):
    match = REGEX_COMPNODE.match(input_line)
    po = CompNodeOperator(match)

    assert po.name == expected[0]
    assert len(po.operands) == len(expected[1])

    for po_op, (exp_op, exp_init) in zip(po.operands, expected[1]):
        assert po_op.name == exp_op
        assert po_op.init_value == exp_init


@pytest.mark.parametrize("input_line, expected", [
    (r"Constant(val, rows = 1, cols = 1, tag='') = Parameter(rows, cols, learningRateMultiplier = 0, init = 'fixedValue', value = val)",
     ['Constant', [('value', None), ('rows', 1), ('cols', 1)]]),  # note that we changed 'val' to 'value'
])
def test_parsing_inst_node(input_line, expected):
    match = REGEX_INSTANTIATION.match(input_line)
    po = InstantiationOperator(match)

    assert po.name == expected[0]
    assert len(po.operands) == len(expected[1])

    for po_op, (exp_op, exp_init) in zip(po.operands, expected[1]):
        assert po_op.name == exp_op
        assert po_op.init_value == exp_init


@pytest.mark.parametrize("input_line, expected", [
    (r"Length(x) = new NumericFunction [ what = 'Length' ; arg = x ]",
     ['Length', [('x', None)]]),

    (r"Ceil(x) = -Floor(-x)",
     ['Ceil', [('x', None)]]),

    (r"Round(x) = Floor(x+0.5)",
     ['Round', [('x', None)]]),

    (r"Abs(x) = if x >= 0 then x else -x",
     ['Abs', [('x', None)]]),
])
def test_parsing_standard_node(input_line, expected):
    match = REGEX_STANDARD.match(input_line)
    po = CompNodeOperator(match)

    assert po.name == expected[0]
    assert len(po.operands) == len(expected[1])

    for po_op, (exp_op, exp_init) in zip(po.operands, expected[1]):
        assert po_op.name == exp_op
        assert po_op.init_value == exp_init
