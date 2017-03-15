//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

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

void TestInfAndNans()
{
    auto device = DeviceDescriptor::CPUDevice();

    // Test 1/0 == INF
    auto divideFunc = ElementDivide(Constant::Scalar(1.0f, device), Constant::Scalar(0.0f, device));
    std::vector<float> outputData(1, 0.2f);
    auto outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(NDShape(0), outputData));

    std::unordered_map<Variable, ValuePtr> outputs = { { divideFunc->Output(), outputValue } };
    divideFunc->Forward(std::unordered_map<Variable, ValuePtr>({}), outputs, device);

    if (outputData[0] != std::numeric_limits<float>::infinity())
        BOOST_ERROR("1/0 != Infinity");
}

// TODO: Enable after the core engine reciprocal bug of 1/0 not being INF is fixed
//TestInfAndNans();

BOOST_AUTO_TEST_SUITE(TensorSuite)

BOOST_AUTO_TEST_CASE(TensorPlusInCPU)
{
    if (ShouldRunOnCpu())
        TestTensorPlus<float>(0, 3, DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(TensorPlusRightOperandWithAxes)
{
    if (ShouldRunOnGpu())
    {
        TestTensorPlus<double>(4, 1, DeviceDescriptor::GPUDevice(0), true);
        TestTensorPlus<float>(1, 3, DeviceDescriptor::GPUDevice(0), false);
    }
}

BOOST_AUTO_TEST_CASE(TensorPlusRightOperandWithoutAxes)
{
    if (ShouldRunOnGpu())
    {
        TestTensorPlus<double>(2, 0, DeviceDescriptor::GPUDevice(0), false);
        TestTensorPlus<float>(0, 0, DeviceDescriptor::GPUDevice(0), false);
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}
