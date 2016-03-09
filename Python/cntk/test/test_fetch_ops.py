from .._fetch_ops import REGEX_COMPNODE, CompNodeOperator

import pytest
@pytest.mark.parametrize("input_line, expected", [
    # Format of expected: [OperatorName, [(OperandName_1, OperandInitValue_1), ...]]
(r"Times(A, B, outputRank=1, tag='') = new ComputationNode [ operation = 'Times' ; inputs = ( A : B ) /*plus the function args*/ ]",
    ['Times', [('A', None), ('B', None), ('outputRank', 1)]]),


(r"Convolution(weightNode, inputValueNode, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0, imageLayout='CHW', tag='') = new ComputationNode [ operation = 'Convolution' ; inputs = (weightNode : inputValueNode) /*plus the function args*/ ]",
    ['Convolution', [('weightNode', None), ('inputValueNode', None), ('kernelWidth', None), ('kernelHeight', None), ('outputChannels', None), ('horizontalSubsample', None), ('verticalSubsample', None), ('zeroPadding', False), ('maxTempMemSizeInSamples', 0), ('imageLayout', 'CHW')]]),

(r"LearnableParameter(rows, cols, learningRateMultiplier = 1.0, init = 'uniform'/*|fixedValue|gaussian|fromFile*/, initValueScale = 1, value = 0, initFromFilePath = '', initOnCPUOnly=true, randomSeed=-1, tag='') = new ComputationNode [ operation = 'LearnableParameter' ; shape = new TensorShape [ dims = (rows : cols) ] /*plus the function args*/ ]",
    ['LearnableParameter', [('rows', None), ('cols', None), ('learningRateMultiplier', 1.0), ('init', 'uniform'), ('initValueScale', 1), ('value', 0), ('initFromFilePath', ''), ('initOnCPUOnly', True), ('randomSeed', -1)]]),
])
def test_parsing_comp_nodes(input_line, expected):
    comp_node = REGEX_COMPNODE.match(input_line)
    po = CompNodeOperator(comp_node)

    assert po.name == expected[0]
    assert len(po.operands) == len(expected[1])

    for po_op, (exp_op, exp_init) in zip(po.operands, expected[1]):
        assert po_op.name == exp_op
        assert po_op.init_value == exp_init
