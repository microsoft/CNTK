//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "UserMatrixMultiplicationOp.h"

#pragma warning(push)
#pragma warning(disable:  4459)
void UserTimesFunctionExample()
{
    auto device = DeviceDescriptor::CPUDevice();
    size_t outDim = 15;
    size_t inDim = 10;
    auto W = Parameter(NDShape({ outDim, inDim }), DataType::Float, GlorotUniformInitializer(), device);
    auto x = InputVariable(NDShape({ inDim }), DataType::Float, { Axis::DefaultBatchAxis() });
    auto userDefinedTimes = UserTimesFunction::Create(W, x, L"UserDefinedTimes");

    size_t batchSize = 3;
    std::vector<float> inputData(inDim * batchSize);
    for (size_t i = 0; i < inputData.size(); ++i)
        inputData[i] = (float)rand() / RAND_MAX;

    auto inputDataValue = Value::CreateBatch(x.Shape(), inputData, device);

    std::vector<float> rootGradientData(outDim * batchSize, 1);
    auto rootGradientValue = Value::CreateBatch(userDefinedTimes->Output().Shape(), rootGradientData, device);

    std::unordered_map<Variable, ValuePtr> outputValues = { { userDefinedTimes->Output(), nullptr } };
    auto backPropState = userDefinedTimes->Forward({ { x, inputDataValue } }, outputValues, device, { userDefinedTimes->Output() });

    std::unordered_map<Variable, ValuePtr> inputGradientValues = { { W, nullptr } };
    userDefinedTimes->Backward(backPropState, { { userDefinedTimes->Output(), rootGradientValue } }, inputGradientValues);
    auto userDefinedTimesOutputValue = outputValues[userDefinedTimes->Output()];
    auto userDefinedTimesInputGradientValue = inputGradientValues[W];

    // Compare against the CNTK built-in implementation
    auto builtInTimes = Times(W, x, L"BuiltInTimes");
    outputValues = { { builtInTimes->Output(), nullptr } };
    backPropState = builtInTimes->Forward({ { x, inputDataValue } }, outputValues, device, { builtInTimes->Output() });
    inputGradientValues = { { W, nullptr } };
    builtInTimes->Backward(backPropState, { { builtInTimes->Output(), rootGradientValue } }, inputGradientValues);
    auto builtInTimesOutputValue = outputValues[builtInTimes->Output()];
    auto builtInTimesInputGradientValue = inputGradientValues[W];

    const double relativeTolerance = 0.001f;
    const double absoluteTolerance = 0.000001f;

    if (!Internal::AreEqual(*userDefinedTimesOutputValue, *builtInTimesOutputValue, relativeTolerance, absoluteTolerance))
        std::runtime_error("UserTimesOp's Forward result does not match built-in result");

    if (!Internal::AreEqual(*userDefinedTimesInputGradientValue, *builtInTimesInputGradientValue, relativeTolerance, absoluteTolerance))
        std::runtime_error("UserTimesOp's Forward result does not match built-in result");
}
#pragma warning(pop)

void main()
{
    UserTimesFunctionExample();
}
