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


    auto compareWithBuiltInTimes = [device, outDim, inDim](FunctionPtr times) {
        size_t batchSize = 3;
        std::vector<float> inputData(inDim * batchSize);
        for (size_t i = 0; i < inputData.size(); ++i)
            inputData[i] = (float)rand() / RAND_MAX;

        auto input = times->Arguments()[0];
        auto inputDataValue = Value::CreateBatch(input.Shape(), inputData, device);

        std::vector<float> rootGradientData(outDim * batchSize, 1);
        auto rootGradientValue = Value::CreateBatch(times->Output().Shape(), rootGradientData, device);

        std::unordered_map<Variable, ValuePtr> outputValues = { { times->Output(), nullptr } };
        auto backPropState = times->Forward({ { input, inputDataValue } }, outputValues, device, { times->Output() });


        auto parameter = times->Parameters()[0];

        std::unordered_map<Variable, ValuePtr> inputGradientValues = { { parameter, nullptr } };
        times->Backward(backPropState, { { times->Output(), rootGradientValue } }, inputGradientValues);
        auto userDefinedTimesOutputValue = outputValues[times->Output()];
        auto userDefinedTimesInputGradientValue = inputGradientValues[parameter];

        // Compare against the CNTK built-in implementation
        auto builtInTimes = Times(parameter, input, L"BuiltInTimes");
        outputValues = { { builtInTimes->Output(), nullptr } };
        backPropState = builtInTimes->Forward({ { input, inputDataValue } }, outputValues, device, { builtInTimes->Output() });
        inputGradientValues = { { parameter, nullptr } };
        builtInTimes->Backward(backPropState, { { builtInTimes->Output(), rootGradientValue } }, inputGradientValues);
        auto builtInTimesOutputValue = outputValues[builtInTimes->Output()];
        auto builtInTimesInputGradientValue = inputGradientValues[parameter];

        const double relativeTolerance = 0.001f;
        const double absoluteTolerance = 0.000001f;

        if (!Internal::AreEqual(*userDefinedTimesOutputValue, *builtInTimesOutputValue, relativeTolerance, absoluteTolerance))
            std::runtime_error("UserTimesOp's Forward result does not match built-in result");

        if (!Internal::AreEqual(*userDefinedTimesInputGradientValue, *builtInTimesInputGradientValue, relativeTolerance, absoluteTolerance))
            std::runtime_error("UserTimesOp's Forward result does not match built-in result");

    };

    compareWithBuiltInTimes(userDefinedTimes);

    auto version = std::string(CNTK_COMPONENT_VERSION);
    std::wstring wversion(version.begin(), version.end());
    Function::RegisterNativeUserFunction(L"NativeUserTimesOp", L"Cntk.ExtensibilityExamples-" + wversion, L"CreateUserTimesFunction");

    userDefinedTimes->Save(L"UserTimesFunctionExample.model");

    auto userDefinedTimes_reloaded_1 = Function::Load(L"UserTimesFunctionExample.model", device);

    compareWithBuiltInTimes(userDefinedTimes_reloaded_1);

    Function::RegisterUDFDeserializeCallback(L"NativeUserTimesOp", [](const std::vector<Variable>& inputs,
        const std::wstring& name,
        const Dictionary& state) {
        return UserTimesFunction::Create(inputs[0], inputs[1], state, name);
    });

    auto userDefinedTimes_reloaded_2 = Function::Load(L"UserTimesFunctionExample.model", device);

    compareWithBuiltInTimes(userDefinedTimes_reloaded_2);
}
#pragma warning(pop)

void main()
{
    UserTimesFunctionExample();
}
