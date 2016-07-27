#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

template <typename ElementType>
void TestTensorPlus(size_t numAxesLeftOperand, size_t numAxesRightOperand, const DeviceDescriptor& device)
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

    Variable leftInputVar(leftInputShape, AsDataType<ElementType>(), true, L"leftInput");
    Variable rightInputVar(rightInputShape, AsDataType<ElementType>(), true, L"rightInput");

    auto plusFunc = Plus(leftInputVar, rightInputVar);

    std::vector<ElementType> leftInputData(leftInputShape.TotalSize());
    for (size_t i = 0; i < leftInputData.size(); ++i)
        leftInputData[i] = ((ElementType)rand()) / RAND_MAX;

    auto leftInputValueShape = leftInputShape.AppendShape({ 1, 1 });
    ValuePtr leftInputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(leftInputValueShape, leftInputData, true));

    std::vector<ElementType> rightInputData(rightInputShape.TotalSize());
    for (size_t i = 0; i < rightInputData.size(); ++i)
        rightInputData[i] = ((ElementType)rand()) / RAND_MAX;

    auto rightInputValueShape = rightInputShape.AppendShape({ 1, 1 });
    ValuePtr rightInputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(rightInputValueShape, rightInputData, true));

    NDShape outputShape = plusFunc->Output().Shape().AppendShape({ 1, 1 });
    std::vector<ElementType> outputData(outputShape.TotalSize());
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));

    std::unordered_map<Variable, ValuePtr> outputs = { { plusFunc->Output(), outputValue } };
    auto backPropState = plusFunc->Forward({ { leftInputVar, leftInputValue }, { rightInputVar, rightInputValue } }, outputs, device, { plusFunc->Output() });

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

    FloatingPointVectorCompare(outputData, expectedOutputValues, "TestTimesAndPlus: Forward prop results do not match expected results");

    auto& smallerInputGradients = (numAxesLeftOperand < numAxesRightOperand) ? leftInputGradientsData : rightInputGradientsData;
    auto& largerInputGradients = (numAxesLeftOperand < numAxesRightOperand) ? rightInputGradientsData : leftInputGradientsData;
    std::vector<ElementType> expectedLargerInputGradientValues(largerInputGradients.size(), (ElementType)1);
    std::vector<ElementType> expectedSmallerInputGradientValues(smallerInputGradients.size(), (ElementType)(largerInputGradients.size() / smallerInputGradients.size()));
    FloatingPointVectorCompare(smallerInputGradients, expectedSmallerInputGradientValues, "TestTimesAndPlus: Backward prop results do not match expected results");
    FloatingPointVectorCompare(largerInputGradients, expectedLargerInputGradientValues, "TestTimesAndPlus: Backward prop results do not match expected results");
}

void TensorTests()
{
    TestTensorPlus<float>(0, 3, DeviceDescriptor::CPUDevice());
#ifndef CPUONLY
    TestTensorPlus<double>(4, 1, DeviceDescriptor::GPUDevice(0));
    TestTensorPlus<float>(1, 3, DeviceDescriptor::GPUDevice(0));
    TestTensorPlus<double>(2, 0, DeviceDescriptor::GPUDevice(0));
    TestTensorPlus<float>(0, 0, DeviceDescriptor::GPUDevice(0));
#endif
}
