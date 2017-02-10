//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;
// TODO: Need to further cleanup/simplify definition of user defined functions
class UserDefinedTimesOrPlusFunction final : public Function
{
    template <typename T, typename ...CtorArgTypes>
    friend inline std::shared_ptr<T> CNTK::MakeSharedObject(CtorArgTypes&& ...ctorArgs);

public:
    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, bool isTimes, const std::wstring& name = L"")
    {
        auto userDefinedTimesFunc = MakeSharedObject<UserDefinedTimesOrPlusFunction>(leftOperand, rightOperand, isTimes, name);
        return Combine({ userDefinedTimesFunc->Output() });
    }

public:

    BackPropStatePtr Forward(const std::vector<ValuePtr>& inputValues,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice,
                             const std::unordered_set<Variable>& outputsToRetainBackwardStateFor) override
    {
        std::unordered_map<Variable, ValuePtr> outputValues = { { m_timesOrPlusFunc->Output(), nullptr } };
        std::unordered_set<Variable> retainBackwardStateFor;
        if (!outputsToRetainBackwardStateFor.empty())
            retainBackwardStateFor = { m_timesOrPlusFunc->Output() };

        auto inputs = Inputs();
        auto GetInputIndex = [&inputs](const Variable& input) -> size_t {
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                if (inputs[i] == input)
                    return i;
            }

            BOOST_ERROR("GetInputIndex: Specified variable is not an input of this Function");
            return 0;
        };

        std::unordered_map<Variable, ValuePtr> argumentValues;
        for (auto argumentMapping : m_timesOrPlusFuncArgumentMap)
        {
            ValuePtr argValue = inputValues[GetInputIndex(argumentMapping.second)];

            if (argumentMapping.first.IsParameter())
                Parameter(argumentMapping.first).SetValue(argValue->Data());
            else
                argumentValues.insert({ argumentMapping.first, argValue });
        }

        auto retVal = m_timesOrPlusFunc->Forward(argumentValues, outputValues, computeDevice, retainBackwardStateFor);
        outputs[Output()] = outputValues[m_timesOrPlusFunc->Output()];

        return retVal;
    }

    void Backward(const BackPropStatePtr& state,
        const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
        std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override
    {
        std::unordered_map<Variable, ValuePtr> gradientValues = { { m_timesOrPlusFunc->Output(), rootGradientValues.begin()->second } };
        std::unordered_map<Variable, ValuePtr> timesFuncBackPropagatedGradientValuesForInputs;
        for (auto argumentMapping : m_timesOrPlusFuncArgumentMap)
        {
            if (backPropagatedGradientValuesForInputs.find(argumentMapping.second) != backPropagatedGradientValuesForInputs.end())
                timesFuncBackPropagatedGradientValuesForInputs.insert({ argumentMapping.first, backPropagatedGradientValuesForInputs.at(argumentMapping.second) });
        }

        m_timesOrPlusFunc->Backward(state, gradientValues, timesFuncBackPropagatedGradientValuesForInputs);

        auto origBackPropagatedGradientValuesForInputs = backPropagatedGradientValuesForInputs;
        for (auto argumentMapping : m_timesOrPlusFuncArgumentMap)
        {
            if (timesFuncBackPropagatedGradientValuesForInputs.find(argumentMapping.first) != timesFuncBackPropagatedGradientValuesForInputs.end())
            {
                if (backPropagatedGradientValuesForInputs[argumentMapping.second] == nullptr)
                    backPropagatedGradientValuesForInputs[argumentMapping.second] = timesFuncBackPropagatedGradientValuesForInputs.at(argumentMapping.first);
                else
                {
                    if (origBackPropagatedGradientValuesForInputs[argumentMapping.second] == nullptr)
                    {
                        // We need to aggregate
                        auto inVar1 = InputVariable(argumentMapping.second.Shape(), argumentMapping.second.GetDataType(), argumentMapping.second.DynamicAxes());
                        auto inVar2 = InputVariable(argumentMapping.second.Shape(), argumentMapping.second.GetDataType(), argumentMapping.second.DynamicAxes());
                        auto aggregationFunc = Plus(inVar1, inVar2);
                        std::unordered_map<Variable, ValuePtr> outputValues = { { aggregationFunc->Output(), nullptr } };
                        aggregationFunc->Forward({ { inVar1, backPropagatedGradientValuesForInputs[argumentMapping.second] }, { inVar2, timesFuncBackPropagatedGradientValuesForInputs.at(argumentMapping.first) } }, outputValues, state->Device());
                        backPropagatedGradientValuesForInputs[argumentMapping.second] = outputValues[aggregationFunc->Output()];
                    }
                }
            }
        }
    }

    const std::wstring& OpName() const override
    {
        static std::wstring opName = L"UserDefinedTimesOp";
        return opName;
    }

    Dictionary Serialize() const override { NOT_IMPLEMENTED; }
    size_t CurrentVersion() const override { NOT_IMPLEMENTED; }

private:
    void InferOutputs(std::vector<Variable>& outputs) override
    {
        auto leftOperand = Inputs()[0];
        auto rightOperand = Inputs()[1];
        auto tempFunc = m_isTimes ? Times(leftOperand, rightOperand) : Plus(leftOperand, rightOperand);
        auto tempFuncOutputs = tempFunc->Outputs();

        for (auto tempFuncOutput : tempFuncOutputs)
            outputs.push_back(OutputVariable(tempFuncOutput.Shape(), tempFuncOutput.GetDataType(), tempFuncOutput.DynamicAxes()));
    }

    UserDefinedTimesOrPlusFunction(const Variable& leftOperand, const Variable& rightOperand, bool isTimes, const std::wstring& name)
        : Function({ leftOperand, rightOperand }, Dictionary(), name), m_isTimes(isTimes)
    {
        auto createTimesOperandVar = [this](const Variable& operand, const std::wstring& operandName) {
            Variable var;

            if (operand.DynamicAxes().empty())
            {
                if (Combine({ operand })->Parameters().empty())
                    BOOST_ERROR("Cannot determine device to place Parameter on!");

                var = Parameter(operand.Shape(), operand.GetDataType(), 0, Combine({ operand })->Parameters()[0].Value()->Device());
            }
            else
                var = InputVariable(operand.Shape(), operand.IsSparse(), operand.GetDataType(), operand.NeedsGradient(), operandName, operand.DynamicAxes());

            m_timesOrPlusFuncArgumentMap.insert({ var, operand });
            return var;
        };

        auto timesLeftOperandInputVar = createTimesOperandVar(leftOperand, L"leftOperand");
        auto timesRightOperandInputVar = createTimesOperandVar(rightOperand, L"rightOperand");
        m_timesOrPlusFunc = isTimes ? Times(timesLeftOperandInputVar, timesRightOperandInputVar) : Plus(timesLeftOperandInputVar, timesRightOperandInputVar);
    }

private:
    bool m_isTimes;
    FunctionPtr m_timesOrPlusFunc;
    std::unordered_map<Variable, Variable> m_timesOrPlusFuncArgumentMap;
};

namespace CNTK { namespace Test {

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
    auto timesAndPlusFunc = Plus(plusParam, UserDefinedTimesOrPlusFunction::Create(timesParam, inputVar, /* isTimes = */ true));

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
                BOOST_ERROR("Backprop prop results do not match expected results for Plus params gradients");

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

void TestDuplicateVariablesInInputs(size_t dim, const DeviceDescriptor& device)
{
    auto inputVar = InputVariable({ dim }, DataType::Float, /* needsGradient = */ true, L"input");
    auto plusFunc = UserDefinedTimesOrPlusFunction::Create(inputVar, inputVar, /* isTimes = */ false);

    srand(1);
    size_t numSamples = 7;
    std::vector<float> inputData(dim * numSamples);
    for (size_t i = 0; i < inputData.size(); ++i)
        inputData[i] = ((float)rand()) / RAND_MAX;

    NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
    ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

    NDShape outputShape = plusFunc->Output().Shape().AppendShape({ 1, numSamples });
    std::unordered_map<Variable, ValuePtr> outputs = { { plusFunc->Output(), nullptr } };
    auto backpropState = plusFunc->Forward({ { inputVar, inputValue } }, outputs, device, { plusFunc->Output() });

    // Perform backprop
    std::vector<float> rootGradientsData(outputShape.TotalSize(), 1);
    ValuePtr rootGradientValue;
    if (device.Type() == DeviceKind::CPU)
        rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), device, true));
    else
    {
        NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true);
        NDArrayViewPtr gpuArrayView = MakeSharedObject<NDArrayView>(DataType::Float, outputShape, device);
        gpuArrayView->CopyFrom(*cpuArrayView);
        rootGradientValue = MakeSharedObject<Value>(gpuArrayView);
    }

    std::vector<float> inputGradientData(inputShape.TotalSize());
    ValuePtr inputGradientValue;
    if (device.Type() == DeviceKind::CPU)
        inputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputGradientData.data(), inputGradientData.size(), device, false));
    else
        inputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, inputShape, device));

    std::unordered_map<Variable, ValuePtr> inputGradients = { { inputVar, inputGradientValue } };
    plusFunc->Backward(backpropState, { { plusFunc->Output(), rootGradientValue } }, inputGradients);

    // Verify forward prop results
    std::vector<float> outputData(outputShape.TotalSize());
    auto outputValue = outputs[plusFunc->Output()];
    NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(outputShape, outputData.data(), outputData.size(), DeviceDescriptor::CPUDevice(), false);
    cpuArrayView->CopyFrom(*outputValue->Data());

    std::vector<float> expectedOutputValues(outputShape.TotalSize());
    for (size_t i = 0; i < numSamples; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
            expectedOutputValues[(i * dim) + j] = inputData[(i * dim) + j] * 2;
    }

    FloatingPointVectorCompare(outputData, expectedOutputValues, "TestTimesAndPlus: Forward prop results do not match expected results");

    // Verify backward prop results
    if (device.Type() != DeviceKind::CPU)
    {
        NDArrayViewPtr cpuArrayViewBack = MakeSharedObject<NDArrayView>(DataType::Float, inputShape, DeviceDescriptor::CPUDevice());
        cpuArrayViewBack->CopyFrom(*inputGradientValue->Data());
        const float* cpuArrayViewBuffer = cpuArrayViewBack->DataBuffer<float>();
        memcpy(inputGradientData.data(), cpuArrayViewBuffer, inputShape.TotalSize() * sizeof(float));
    }

    for (size_t i = 0; i < dim; ++i)
        if (inputGradientData[i] != 2)
            BOOST_ERROR("TestTimesAndPlus: Backprop prop results do not match expected results for Plus params gradients");
}

BOOST_AUTO_TEST_SUITE(UserDefinedFunctionSuite)

BOOST_AUTO_TEST_CASE(DuplicateVariablesInCPU)
{
    TestDuplicateVariablesInInputs(11, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(DuplicateVariablesInGPU)
{
    if (IsGPUAvailable()) {
        TestDuplicateVariablesInInputs(117, DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(TimesAndPlusInCPU)
{
    TestTimesAndPlus<double>(4, 2, 5, DeviceDescriptor::CPUDevice(), 3, true, true);
}

BOOST_AUTO_TEST_CASE(TimesAndPlusInGPU)
{
    if (IsGPUAvailable())
    {
        TestTimesAndPlus<float>(145, 32, 2, DeviceDescriptor::GPUDevice(0), 10, true, false);
        TestTimesAndPlus<double>(145, 15, 200, DeviceDescriptor::GPUDevice(0), 21, false, false);
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}
