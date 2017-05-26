//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

#include <iostream>
#include <cstdio>

using namespace CNTK;

template <typename ElementType>
void TestTensorPlus(size_t numAxesLeftOperand, size_t numAxesRightOperand, const DeviceDescriptor& device, bool useConstantInputsOnly)
{
    srand(1);

    size_t maxDimSize = 15;
    NDShape leftInputShape(numAxesLeftOperand);
    for (size_t i = 0; i < numAxesLeftOperand; ++i)
        leftInputShape[i] = (rand() % maxDimSize) + 1;

    NDShape rightInputShape(numAxesRightOperand);
    for (size_t i = 0; i < std::min(numAxesLeftOperand, numAxesRightOperand); ++i)
        rightInputShape[i] = leftInputShape[i];

    for (size_t i = std::min(numAxesLeftOperand, numAxesRightOperand); i < numAxesRightOperand; ++i)
        rightInputShape[i] = (rand() % maxDimSize) + 1;

    std::vector<ElementType> leftInputData(leftInputShape.TotalSize());
    for (size_t i = 0; i < leftInputData.size(); ++i)
        leftInputData[i] = ((ElementType)rand()) / RAND_MAX;

    auto leftInputValueShape = leftInputShape.AppendShape({ 1, 1 });
    auto leftInputValue = MakeSharedObject<NDArrayView>(leftInputValueShape, leftInputData, true);

    std::vector<ElementType> rightInputData(rightInputShape.TotalSize());
    for (size_t i = 0; i < rightInputData.size(); ++i)
        rightInputData[i] = ((ElementType)rand()) / RAND_MAX;

    auto rightInputValueShape = rightInputShape.AppendShape({ 1, 1 });
    auto rightInputValue = MakeSharedObject<NDArrayView>(rightInputValueShape, rightInputData, true);

    Variable leftInputVar, rightInputVar;
    if (useConstantInputsOnly)
    {
        leftInputValue = leftInputValue->DeepClone(device, false);
        rightInputValue = rightInputValue->DeepClone(device, false);

        leftInputVar = Parameter(leftInputValue, L"leftInput");
        rightInputVar = Parameter(rightInputValue, L"rightInput");
    }
    else
    {
        leftInputVar = InputVariable(leftInputShape, AsDataType<ElementType>(), true, L"leftInput");
        rightInputVar = InputVariable(rightInputShape, AsDataType<ElementType>(), true, L"rightInput");
    }

    auto plusFunc = Plus(leftInputVar, rightInputVar);

    NDShape outputShape = plusFunc->Output().Shape();
    if (!useConstantInputsOnly)
        outputShape = outputShape.AppendShape({ 1, 1 });

    std::vector<ElementType> outputData(outputShape.TotalSize());
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));

    std::unordered_map<Variable, ValuePtr> outputs = { { plusFunc->Output(), outputValue } };
    BackPropStatePtr backPropState;
    if (useConstantInputsOnly)
        backPropState = plusFunc->Forward(std::unordered_map<Variable, ValuePtr>({}), outputs, device, { plusFunc->Output() });
    else
        backPropState = plusFunc->Forward({ { leftInputVar, MakeSharedObject<Value>(leftInputValue) }, { rightInputVar, MakeSharedObject<Value>(rightInputValue) } }, outputs, device, { plusFunc->Output() });

    // Perform backprop
    std::vector<ElementType> rootGradientsData(outputShape.TotalSize(), 1);
    ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData, true));

    std::vector<ElementType> leftInputGradientsData(leftInputValueShape.TotalSize());
    ValuePtr leftInputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(leftInputValueShape, leftInputGradientsData, false));
    std::vector<ElementType> rightInputGradientsData(rightInputValueShape.TotalSize());
    ValuePtr rightInputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(rightInputValueShape, rightInputGradientsData, false));

    std::unordered_map<Variable, ValuePtr> gradients = { { leftInputVar, leftInputGradientValue }, { rightInputVar, rightInputGradientValue } };
    plusFunc->Backward(backPropState, { { plusFunc->Output(), rootGradientValue } }, gradients);

    // Verify forward prop results
    auto& smallerInput = (numAxesLeftOperand < numAxesRightOperand) ? leftInputData : rightInputData;
    auto& largerInput = (numAxesLeftOperand < numAxesRightOperand) ? rightInputData : leftInputData;
    std::vector<ElementType> expectedOutputValues = largerInput;
    for (size_t i = 0; i < (expectedOutputValues.size() / smallerInput.size()); ++i)
    {
        for (size_t j = 0; j < smallerInput.size(); ++j)
            expectedOutputValues[(i * smallerInput.size()) + j] += smallerInput[j];
    }

    FloatingPointVectorCompare(outputData, expectedOutputValues, "Forward prop results do not match expected results");

    auto& smallerInputGradients = (numAxesLeftOperand < numAxesRightOperand) ? leftInputGradientsData : rightInputGradientsData;
    auto& largerInputGradients = (numAxesLeftOperand < numAxesRightOperand) ? rightInputGradientsData : leftInputGradientsData;
    std::vector<ElementType> expectedLargerInputGradientValues(largerInputGradients.size(), (ElementType)1);
    std::vector<ElementType> expectedSmallerInputGradientValues(smallerInputGradients.size(), (ElementType)(largerInputGradients.size() / smallerInputGradients.size()));
    FloatingPointVectorCompare(smallerInputGradients, expectedSmallerInputGradientValues, "TestTimesAndPlus: Backward prop results do not match expected results");
    FloatingPointVectorCompare(largerInputGradients, expectedLargerInputGradientValues, "TestTimesAndPlus: Backward prop results do not match expected results");
}

/*
template <typename ElementType>
void TestCenter(const DeviceDescriptor& device, bool useConstantInputsOnly)
{
    size_t num_minibatch = 2;
    NDShape leftInputShape(1);
    leftInputShape[0] = 3;

    NDShape rightInputShape(1);
    rightInputShape[0] = 2;

    std::vector<ElementType> leftInputData = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };

    auto leftInputValueShape = leftInputShape.AppendShape({ 1, num_minibatch });
    auto leftInputValue = MakeSharedObject<NDArrayView>(leftInputValueShape, leftInputData, true);

    std::vector<ElementType> rightInputData = {1.0f, 2.0f, 3.0f, 4.0f};

    auto rightInputValueShape = rightInputShape.AppendShape({ 1, num_minibatch });
    auto rightInputValue = MakeSharedObject<NDArrayView>(rightInputValueShape, rightInputData, true);

    Variable leftInputVar, rightInputVar;
    if (useConstantInputsOnly)
    {
        leftInputValue = leftInputValue->DeepClone(device, false);
        rightInputValue = rightInputValue->DeepClone(device, false);

        leftInputVar = Parameter(leftInputValue, L"leftInput");
        rightInputVar = Parameter(rightInputValue, L"rightInput");
    }
    else
    {
        leftInputVar = InputVariable(leftInputShape, AsDataType<ElementType>(), false, L"leftInput");
        rightInputVar = InputVariable(rightInputShape, AsDataType<ElementType>(), true, L"rightInput");
    }

    auto plusFunc = CenterLoss(rightInputVar, leftInputVar, 0.1, 2, 3);

    NDShape outputShape = plusFunc->Output().Shape();
  //  if (!useConstantInputsOnly)
  //      outputShape = outputShape.AppendShape({ 1, num_minibatch });

    std::vector<ElementType> outputData(outputShape.TotalSize());
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));

    std::unordered_map<Variable, ValuePtr> outputs = { { plusFunc->Output(), outputValue } };
    BackPropStatePtr backPropState;
    if (useConstantInputsOnly)
        backPropState = plusFunc->Forward(std::unordered_map<Variable, ValuePtr>({}), outputs, device, { plusFunc->Output() });
    else
        backPropState = plusFunc->Forward({ { leftInputVar, MakeSharedObject<Value>(leftInputValue) }, { rightInputVar, MakeSharedObject<Value>(rightInputValue) } }, outputs, device, { plusFunc->Output() });

    // Perform backprop
    std::vector<ElementType> rootGradientsData(outputShape.TotalSize(), 1);
    ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData, true));

    std::vector<ElementType> leftInputGradientsData(leftInputValueShape.TotalSize());
    ValuePtr leftInputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(leftInputValueShape, leftInputGradientsData, false));
    std::vector<ElementType> rightInputGradientsData(rightInputValueShape.TotalSize());
    ValuePtr rightInputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(rightInputValueShape, rightInputGradientsData, false));

    std::unordered_map<Variable, ValuePtr> gradients = { { leftInputVar, leftInputGradientValue }, { rightInputVar, rightInputGradientValue } };
    plusFunc->Backward(backPropState, { { plusFunc->Output(), rootGradientValue } }, gradients);
}
*/
int main()
{
    //TestCenter<float>(DeviceDescriptor::CPUDevice(), false);

    NDShape leftShape(1);
    leftShape[0] = 2;

    NDShape rightShape(1);
    rightShape[0] = 2;

    std::vector<float> leftInputData = { 1.0f, 2.0f };
    std::vector<float> rightInputData = { 3.0f, 4.0f };

    Variable leftInputVar, rightInputVar;

    leftInputVar = InputVariable(leftShape, AsDataType<float>(), true, L"leftInput");
    rightInputVar = InputVariable(rightShape, AsDataType<float>(), true, L"rightInput");

    auto leftInputValueShape = leftShape.AppendShape({ 1, 1 });
    auto leftInputValue = MakeSharedObject<NDArrayView>(leftShape, leftInputData, true);

    auto rightInputValueShape = rightShape.AppendShape({ 1, 1 });
    auto rightInputValue = MakeSharedObject<NDArrayView>(rightShape, rightInputData, true);

    auto squareLossFunc = SquaredError(leftInputVar, rightInputVar);

    NDShape outputShape = squareLossFunc->Output().Shape();

    std::vector<float> outputData(outputShape.TotalSize());
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));

    std::unordered_map<Variable, ValuePtr> outputs = { { squareLossFunc->Output(), outputValue } };
    std::unordered_map<Variable, ValuePtr> arguments = { { leftInputVar, MakeSharedObject<Value>(leftInputValue) }, { rightInputVar, MakeSharedObject<Value>(rightInputValue) } };
    std::unordered_set<Variable> funcy = { squareLossFunc->Output() };

    const DeviceDescriptor& device = DeviceDescriptor::CPUDevice();
    BackPropStatePtr backPropState = squareLossFunc->Forward(arguments, outputs, device, funcy);

/*
#x = cntk.input(2)
#y = cntk.input(2)
#x0 = np.asarray([[2., 1.]], dtype=np.float32)
#y0 = np.asarray([[4., 6.]], dtype=np.float32)
#print(cntk.squared_error(x, y).eval({x:x0, y:y0}))
*/
}
