//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

// TODO: Need to further cleanup/simplify definition of user defined functions
class UserDefinedTimesFunction final : public Function
{
    template <typename T, typename ...CtorArgTypes>
    friend inline std::shared_ptr<T> CNTK::MakeSharedObject(CtorArgTypes&& ...ctorArgs);

public:
    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"")
    {
        auto userDefinedTimesFunc = MakeSharedObject<UserDefinedTimesFunction>(leftOperand, rightOperand, name);
        return Combine({ userDefinedTimesFunc->Output() });
    }

public:

    BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice,
                             const std::unordered_set<Variable>& outputsToRetainBackwardStateFor) override
    {
        std::unordered_map<Variable, ValuePtr> outputValues = { { m_timesFunc->Output(), nullptr } };
        std::unordered_set<Variable> retainBackwardStateFor;
        if (!outputsToRetainBackwardStateFor.empty())
            retainBackwardStateFor = { m_timesFunc->Output() };
        auto retVal = m_timesFunc->Forward(arguments, outputValues, computeDevice, retainBackwardStateFor);
        outputs[Output()] = outputValues[m_timesFunc->Output()];

        return retVal;
    }

    void Backward(const BackPropStatePtr& state,
                  const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                  std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override
    {
        std::unordered_map<Variable, ValuePtr> gradientValues = { { m_timesFunc->Output(), rootGradientValues.begin()->second } };
        return m_timesFunc->Backward(state, gradientValues, backPropagatedGradientValuesForInputs);
    }

    const std::wstring& OpName() const override
    {
        static std::wstring opName = L"UserDefinedTimesOp";
        return opName;
    }

    Dictionary Serialize() const override { NOT_IMPLEMENTED; }
    size_t CurrentVersion() const override { NOT_IMPLEMENTED; }

private:
    std::vector<Variable> GetOutputVariables(const Variable& leftOperand, const Variable& rightOperand)
    {
        auto tempFunc = Times(leftOperand, rightOperand);
        auto tempFuncOutputs = tempFunc->Outputs();

        std::vector<Variable> outputs;
        for (auto tempFuncOutput : tempFuncOutputs)
            outputs.push_back(OutputVariable(tempFuncOutput.Shape(), tempFuncOutput.GetDataType(), this, tempFuncOutput.DynamicAxes()));

        return outputs;
    }

    UserDefinedTimesFunction(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
        : Function({ leftOperand, rightOperand }, GetOutputVariables(leftOperand, rightOperand), Dictionary(), name)
    {
        m_timesFunc = Times(leftOperand, rightOperand);
    }

private:
    FunctionPtr m_timesFunc;
};

template <typename ElementType>
void TestTimesAndPlus(size_t inputDim,
                      size_t outputDim,
                      size_t numSamples,
                      const DeviceDescriptor& device,
                      size_t numIterations,
                      bool usePreAllocatedOutputs,
                      bool outputOnSpecifiedDevice)
{
    auto timesParamName = L"timesParameters";
    auto plusParamName = L"plusParameters";
    Parameter timesParam(MakeSharedObject<NDArrayView>((ElementType)0.5, NDShape({ outputDim, inputDim }), device), timesParamName);
    Parameter plusParam(MakeSharedObject<NDArrayView>((ElementType)1.2, std::initializer_list<size_t>({ outputDim }), device), plusParamName);

    auto inputVarName = L"input";
    auto inputVar = InputVariable({ inputDim }, AsDataType<ElementType>(), inputVarName);
    auto timesAndPlusFunc = Plus(plusParam, UserDefinedTimesFunction::Create(timesParam, inputVar));

    srand(1);
    for (size_t iterIdx = 0; iterIdx < numIterations; ++iterIdx)
    {
        std::vector<ElementType> inputData(inputDim * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
            inputData[i] = ((ElementType)rand()) / RAND_MAX;

        NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        NDShape outputShape = timesAndPlusFunc->Output().Shape().AppendShape({ 1, numSamples });
        std::vector<ElementType> outputData(outputShape.TotalSize());
        ValuePtr outputValue;
        if (usePreAllocatedOutputs)
        {
            auto outputAllocationDevice = outputOnSpecifiedDevice ? device : DeviceDescriptor::CPUDevice();
            if (outputAllocationDevice.Type() == DeviceKind::CPU)
                outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData.data(), outputData.size(), outputAllocationDevice, false));
            else
                outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), outputShape, outputAllocationDevice));
        }

        std::unordered_map<Variable, ValuePtr> outputs = { { timesAndPlusFunc->Output(), outputValue } };
        auto backpropState = timesAndPlusFunc->Forward({ { inputVar, inputValue } }, outputs, device, { timesAndPlusFunc->Output() });

        if (!usePreAllocatedOutputs)
            outputValue = outputs[timesAndPlusFunc->Output()];

        // Perform backprop
        std::vector<ElementType> rootGradientsData(outputShape.TotalSize(), 1);
        ValuePtr rootGradientValue;
        if (device.Type() == DeviceKind::CPU)
            rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), device, true));
        else
        {
            NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true);
            NDArrayViewPtr gpuArrayView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), outputShape, device);
            gpuArrayView->CopyFrom(*cpuArrayView);
            rootGradientValue = MakeSharedObject<Value>(gpuArrayView);
        }

        std::vector<ElementType> plusParameterGradientData(plusParam.Shape().TotalSize());
        std::vector<ElementType> timesParameterGradientData(timesParam.Shape().TotalSize());
        ValuePtr plusParameterGradientValue, timesParameterGradientValue;
        if (usePreAllocatedOutputs)
        {
            auto outputAllocationDevice = outputOnSpecifiedDevice ? device : DeviceDescriptor::CPUDevice();
            if (outputAllocationDevice.Type() == DeviceKind::CPU)
            {
                plusParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusParam.Shape(), plusParameterGradientData.data(), plusParameterGradientData.size(), outputAllocationDevice, false));
                timesParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(timesParam.Shape(), timesParameterGradientData.data(), timesParameterGradientData.size(), outputAllocationDevice, false));
            }
            else
            {
                plusParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), plusParam.Shape(), outputAllocationDevice));
                timesParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), timesParam.Shape(), outputAllocationDevice));
            }
        }

        std::unordered_map<Variable, ValuePtr> paramGradients = { { plusParam, plusParameterGradientValue }, { timesParam, timesParameterGradientValue } };
        timesAndPlusFunc->Backward(backpropState, { { timesAndPlusFunc->Output(), rootGradientValue } }, paramGradients);

        if (!usePreAllocatedOutputs)
        {
            plusParameterGradientValue = paramGradients[plusParam];
            timesParameterGradientValue = paramGradients[timesParam];
        }

        // Verify forward prop results
        if (!usePreAllocatedOutputs || (outputOnSpecifiedDevice && (device.Type() != DeviceKind::CPU)))
        {
            NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(outputShape, outputData.data(), outputData.size(), DeviceDescriptor::CPUDevice(), false);
            cpuArrayView->CopyFrom(*outputValue->Data());
        }

        std::vector<ElementType> expectedOutputValues(outputShape.TotalSize());
        for (size_t i = 0; i < numSamples; ++i)
        {
            ElementType expectedVal = (ElementType)1.2;
            for (size_t j = 0; j < inputDim; ++j)
                expectedVal += (ElementType)(inputData[i * inputDim + j] * 0.5);

            for (size_t j = 0; j < outputDim; ++j)
                expectedOutputValues[i * outputDim + j] = expectedVal;
        }

        FloatingPointVectorCompare(outputData, expectedOutputValues, "TestTimesAndPlus: Forward prop results do not match expected results");

        // Verify backward prop results
        if (device.Type() != DeviceKind::CPU)
        {
            NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), plusParam.Shape(), DeviceDescriptor::CPUDevice());
            cpuArrayView->CopyFrom(*plusParameterGradientValue->Data());
            const ElementType* cpuArrayViewBuffer = cpuArrayView->DataBuffer<ElementType>();
            memcpy(plusParameterGradientData.data(), cpuArrayViewBuffer, plusParam.Shape().TotalSize() * sizeof(ElementType));

            cpuArrayView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), timesParam.Shape(), DeviceDescriptor::CPUDevice());
            cpuArrayView->CopyFrom(*timesParameterGradientValue->Data());
            cpuArrayViewBuffer = cpuArrayView->DataBuffer<ElementType>();
            memcpy(timesParameterGradientData.data(), cpuArrayViewBuffer, timesParam.Shape().TotalSize() * sizeof(ElementType));
        }

        for (size_t i = 0; i < outputDim; ++i)
            if (plusParameterGradientData[i] != numSamples)
                throw std::runtime_error("TestTimesAndPlus: Backprop prop results do not match expected results for Plus params gradients");

        std::vector<ElementType> expectedTimesParamsGradientValues(timesParam.Shape().TotalSize());
        for (size_t i = 0; i < inputDim; ++i)
        {
            ElementType expectedVal = 0;
            for (size_t j = 0; j < numSamples; ++j)
                expectedVal += inputData[j * inputDim + i];

            for (size_t j = 0; j < outputDim; ++j)
                expectedTimesParamsGradientValues[i * outputDim + j] = expectedVal;
        }

        FloatingPointVectorCompare(timesParameterGradientData, expectedTimesParamsGradientValues, "TestTimesAndPlus: Backprop prop results do not match expected results for Times params gradients");
    }
}

void UserDefinedFunctionTests()
{
    fprintf(stderr, "\nUserDefinedFunctionTests..\n");

    TestTimesAndPlus<double>(4, 2, 5, DeviceDescriptor::CPUDevice(), 3, true, true);
    if (IsGPUAvailable())
    {
        TestTimesAndPlus<float>(145, 32, 2, DeviceDescriptor::GPUDevice(0), 10, true, false);
        TestTimesAndPlus<double>(145, 15, 200, DeviceDescriptor::GPUDevice(0), 21, false, false);
    }
}
