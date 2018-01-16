//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

std::wstring s_tempModelPath = L"feedForward.net";

void TestFeedForwardNetworkCreation(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    using namespace std::placeholders;

    const size_t inputDim = 937;
    const size_t numOutputClasses = 9304;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    auto inputVarName = L"features";
    auto inputVar = InputVariable({ inputDim }, DataType::Float, inputVarName);
    auto classifierOutput = FullyConnectedFeedForwardClassifierNet(inputVar, numOutputClasses, hiddenLayersDim, numHiddenLayers, device, std::bind(Sigmoid, _1, L""), L"classifierOutput");

    auto labelsVarName = L"Labels";
    auto labelsVar = InputVariable({ numOutputClasses }, DataType::Float, labelsVarName);
    auto trainingLoss = ReduceSum(CrossEntropyWithSoftmax(classifierOutput, labelsVar), Axis::AllAxes(), L"LossFunction");
    auto prediction = ReduceSum(ClassificationError(classifierOutput, labelsVar), Axis::AllAxes(), L"ClassificationError");

    auto ffNet = Combine({ trainingLoss, prediction, classifierOutput }, L"ClassifierModel");

    // Now test the structure
    BOOST_TEST((ffNet->Parameters().size() == ((numHiddenLayers * 2) + 1)), "Function does not have expected Parameter count");

    BOOST_TEST((ffNet->Arguments().size() == 2), "Function does not have expected Argument count");

    BOOST_TEST(ffNet->Outputs().size() == 3, "Function does not have expected Output count");

    if (testSaveAndReLoad)
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;

        SaveAndReloadModel<float>(ffNet, { &inputVar, &labelsVar, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        // Make sure that the names of the input variables were properly restored
        BOOST_TEST(!((inputVar.Name() != inputVarName) || (labelsVar.Name() != labelsVarName)), "One or more input variable names were not properly restored after save and load");

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    // Run Forward and backward a few times
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSamples = 3;
    for (size_t i = 0; i < iterationCount; ++i)
    {
        std::vector<float> inputData(inputDim * numSamples);
        for (size_t i2 = 0; i2 < inputData.size(); ++i2)
            inputData[i2] = ((float)rand()) / RAND_MAX;

        NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        std::vector<float> labelData(numOutputClasses * numSamples, 0);
        for (size_t i3 = 0; i3 < numSamples; ++i3)
            labelData[(i3*numOutputClasses) + (rand() % numOutputClasses)] = 1;

        NDShape labelShape = labelsVar.Shape().AppendShape({ 1, numSamples });
        ValuePtr labelValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(labelShape, labelData.data(), labelData.size(), DeviceDescriptor::CPUDevice(), true));

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = { { classifierOutput, outputValue }, { prediction, predictionErrorValue } };
        auto backpropState = ffNet->Forward({ { inputVar, inputValue }, { labelsVar, labelValue } }, outputs, device, { trainingLoss });

        // Perform backprop
        NDShape outputShape = trainingLoss->Output().Shape();
        std::vector<float> rootGradientsData(outputShape.TotalSize(), 1);
        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true));
        std::unordered_map<Variable, ValuePtr> paramGradients;
        auto allParams = ffNet->Parameters();
        for (auto iter = allParams.begin(); iter != allParams.end(); ++iter)
            paramGradients[*iter] = nullptr;
        ffNet->Backward(backpropState, { { trainingLoss, rootGradientValue } }, paramGradients);
    }
}

template <typename ElementType>
void TestTimesAndPlus(size_t inputDim,
                      size_t outputDim,
                      size_t numSamples,
                      const DeviceDescriptor& device,
                      size_t numIterations,
                      bool usePreAllocatedOutputs,
                      bool outputOnSpecifiedDevice,
                      bool testSaveAndReLoad,
                      unsigned int seed = 1)
{
    auto timesParamName = L"timesParameters";
    auto plusParamName = L"plusParameters";
    Parameter timesParam(MakeSharedObject<NDArrayView>((ElementType)0.5, NDShape({ outputDim, inputDim }), device), timesParamName);
    Parameter plusParam(MakeSharedObject<NDArrayView>((ElementType)1.2, std::initializer_list<size_t>({ outputDim }), device), plusParamName);

    auto inputVarName = L"input";
    auto inputVar = InputVariable({ inputDim }, AsDataType<ElementType>(), inputVarName);
    auto timesAndPlusFunc = Plus(plusParam, Times(timesParam, inputVar));

    if (testSaveAndReLoad)
    {
        SaveAndReloadModel<ElementType>(timesAndPlusFunc, { &inputVar, &timesParam, &plusParam }, device);

        // Make sure that the names of the input variables were properly restored
        BOOST_TEST(!((inputVar.Name() != inputVarName) || (timesParam.Name() != timesParamName) || (plusParam.Name() != plusParamName)), "One or more input variable names were not properly restored after save and load");
    }

    srand(seed);
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

        FloatingPointVectorCompare(outputData, expectedOutputValues, "Forward prop results do not match expected results");

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

        FloatingPointVectorCompare(timesParameterGradientData, expectedTimesParamsGradientValues, "Backprop prop results do not match expected results for Times params gradients");
    }
}

template <typename ElementType>
void TestReduceableTransposeTimes(size_t inputDim,
    size_t numSamples,
    const DeviceDescriptor& device,
    size_t numIterations,
    unsigned int seed = 1)
{
    auto timesParamName = L"timesParameters";
    Parameter timesParam(MakeSharedObject<NDArrayView>((ElementType)0.5, NDShape({inputDim}), device), timesParamName);

    auto inputVarName = L"input";
    auto inputVar = InputVariable({ inputDim }, AsDataType<ElementType>(), inputVarName);
    auto dotFunc = TransposeTimes(ElementTimes(timesParam, inputVar), inputVar + Constant({}, 0.0f, device));

    srand(seed);
    for (size_t iterIdx = 0; iterIdx < numIterations; ++iterIdx)
    {
        std::vector<ElementType> inputData(inputDim * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
            inputData[i] = ((ElementType)rand()) / RAND_MAX;

        NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        NDShape outputShape = dotFunc->Output().Shape().AppendShape({ 1, numSamples });
        std::vector<ElementType> outputData(outputShape.TotalSize());
        ValuePtr outputValue;

        std::unordered_map<Variable, ValuePtr> outputs = { { dotFunc->Output(), outputValue } };
        auto backpropState = dotFunc->Forward({ { inputVar, inputValue } }, outputs, device, { dotFunc->Output() });

        outputValue = outputs[dotFunc->Output()];

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

        ValuePtr timesParamGradientValue;
        std::vector<ElementType> timesParamGradientData(inputVar.Shape().TotalSize(), std::numeric_limits<ElementType>::quiet_NaN());
        if (device.Type() == DeviceKind::CPU)
        {
            timesParamGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputVar.Shape(), timesParamGradientData.data(), timesParamGradientData.size(), device));
        }
        else
        {
            NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(inputVar.Shape(), timesParamGradientData.data(), timesParamGradientData.size(), DeviceDescriptor::CPUDevice());
            NDArrayViewPtr gpuArrayView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), inputVar.Shape(), device);
            gpuArrayView->CopyFrom(*cpuArrayView);
            timesParamGradientValue = MakeSharedObject<Value>(gpuArrayView);
        }
        std::unordered_map<Variable, ValuePtr> paramGradients = { { timesParam, timesParamGradientValue } };
        dotFunc->Backward(backpropState, { { dotFunc->Output(), rootGradientValue } }, paramGradients);

        if (device.Type() == DeviceKind::CPU)
        {
            const ElementType* p = timesParamGradientValue->Data()->DataBuffer<ElementType>();
            for (int i = 0; i < inputDim; i++)
            {
                if (std::isnan(p[i])) ReportFailure("Found NaN in gradient!");
            }
        }
        else
        {
            NDArrayViewPtr cpuView = timesParamGradientValue->Data()->DeepClone(DeviceDescriptor::CPUDevice());
            const ElementType* p = cpuView->DataBuffer<ElementType>();
            for (int i = 0; i < inputDim; i++)
            {
                if (std::isnan(p[i])) ReportFailure("Found NaN in gradient!");
            }
        }
    }
}

template <typename ElementType>
void TestTimesReduceSequenceAxis(
    size_t inputDimM,
    size_t inputDimK,
    bool isLeftSparse,
    bool isRightSparse,
    const std::vector<size_t>& sequencesLength,
    const DeviceDescriptor& device,
    unsigned int seed = 1)
{
    const int NumInputs = 2;

    enum class FuncType : int
    {
        Times_ReduceSequenceAxis = 0,
        ReduceSumTimes,
        TotalTypes,
    };

    Variable inputVar[NumInputs] = 
    {
        inputDimK > 1 ?
            InputVariable({ inputDimM, inputDimK }, isLeftSparse, AsDataType<ElementType>(), /*needsGradient*/ !isLeftSparse, L"inputLeft") :
            InputVariable({ inputDimM }, isLeftSparse, AsDataType<ElementType>(), /*needsGradient*/ !isLeftSparse, L"inputLeft"),
        InputVariable({ inputDimK }, isRightSparse, AsDataType<ElementType>(), /*needsGradient*/ !isRightSparse, L"inputRight")
    };

    FunctionPtr funcs[(int)FuncType::TotalTypes]=
    {
        Times(inputVar[0], inputVar[1], 1, TimesReduceSequenceAxisWithoutInferredInputRank),
        Sequence::ReduceSum(Times(inputVar[0], inputVar[1]))
    };

    size_t maxTimestepsPerSequence = 0;
    for (auto s : sequencesLength)
    {
        maxTimestepsPerSequence = std::max(maxTimestepsPerSequence, s);
    }
    size_t numSequences = sequencesLength.size();

    ValuePtr inputValue[NumInputs];
    NDShape  inputShape[NumInputs];
    bool     inputSparse[NumInputs] = {isLeftSparse, isRightSparse};
    srand(seed);
    for(int inputIndex = 0; inputIndex < NumInputs; ++inputIndex)
    {
        auto inputVarShape = inputVar[inputIndex].Shape();
        inputShape[inputIndex] = inputVarShape.AppendShape({maxTimestepsPerSequence, numSequences});
        inputValue[inputIndex] = GenerateSequences<ElementType>(sequencesLength, inputVarShape, DeviceDescriptor::CPUDevice(), inputSparse[inputIndex]);
    }

    std::unordered_map<Variable, ValuePtr> inputMap = { { inputVar[0], inputValue[0] },{ inputVar[1], inputValue[1] } };

    std::vector<ElementType> outputData[(int)FuncType::TotalTypes];
    ValuePtr outputValue[(int)FuncType::TotalTypes];
    NDShape outputShape = funcs[0]->Output().Shape().AppendShape({ numSequences });

    std::vector<ElementType> inputGradientData[(int)FuncType::TotalTypes][NumInputs];
    ValuePtr inputGradientValue[(int)FuncType::TotalTypes][NumInputs];

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

    for(int f = 0; f < (int)FuncType::TotalTypes; ++f)
    {
        outputData[f] = std::vector<ElementType>(outputShape.TotalSize());
        outputValue[f] = 
            MakeSharedObject<Value>(
                MakeSharedObject<NDArrayView>(
                    outputShape, outputData[f].data(),
                    outputShape.TotalSize(),
                    DeviceDescriptor::CPUDevice()));

        // forward
        std::unordered_map<Variable, ValuePtr> outputMap = { { funcs[f]->Output(), outputValue[f] } };
        auto backpropState =
            funcs[f]->Forward(
                inputMap,
                outputMap,
                device,
                { funcs[f]->Output() });

        // backward
        for(int inputIndex = 0; inputIndex < NumInputs; ++inputIndex)
        {
            inputGradientData[f][inputIndex] = std::vector<ElementType>(inputVar[inputIndex].Shape().TotalSize() * maxTimestepsPerSequence * numSequences, std::numeric_limits<ElementType>::quiet_NaN());
            if (device.Type() == DeviceKind::CPU)
            {
                inputGradientValue[f][inputIndex] = MakeSharedObject<Value>(
                    MakeSharedObject<NDArrayView>(inputShape[inputIndex], inputGradientData[f][inputIndex].data(), inputGradientData[f][inputIndex].size(), device),
                    inputValue[inputIndex]->Mask());
            }
            else
            {
                NDArrayViewPtr cpuArrayView = MakeSharedObject<NDArrayView>(inputShape[inputIndex], inputGradientData[f][inputIndex].data(), inputGradientData[f][inputIndex].size(), DeviceDescriptor::CPUDevice());
                NDArrayViewPtr gpuArrayView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), inputShape[inputIndex], device);
                gpuArrayView->CopyFrom(*cpuArrayView);
                inputGradientValue[f][inputIndex] = MakeSharedObject<Value>(
                    gpuArrayView,
                    inputValue[inputIndex]->Mask());
            }
        }
        std::unordered_map<Variable, ValuePtr> gradientMap;
        if (!isLeftSparse) gradientMap.insert(std::make_pair(inputVar[0], inputGradientValue[f][0]));
        if (!isRightSparse) gradientMap.insert(std::make_pair(inputVar[1], inputGradientValue[f][1]));
        funcs[f]->Backward(backpropState, { { funcs[f]->Output(), rootGradientValue } }, gradientMap);
    }

    FloatingPointVectorCompare(outputData[(int)FuncType::Times_ReduceSequenceAxis], outputData[(int)FuncType::ReduceSumTimes], "Forward results do not match expected results for Sequence::ReduceSum(Times())");

    for (int inputIndex = 0; inputIndex < NumInputs; ++inputIndex)
    {
        if (inputSparse[inputIndex]) continue;
        FloatingPointVectorCompare(inputGradientData[(int)FuncType::Times_ReduceSequenceAxis][inputIndex], inputGradientData[(int)FuncType::ReduceSumTimes][inputIndex], "Backprop results do not match expected results for Sequence::ReduceSum(Times())");
    }
}

BOOST_AUTO_TEST_SUITE(FeedForwardSuite)

BOOST_AUTO_TEST_CASE(FFTimesAndPlusInCPU)
{
    TestTimesAndPlus<double>(4, 2, 5, DeviceDescriptor::CPUDevice(), 3, true, true, true);
}

BOOST_AUTO_TEST_CASE(ReduceableTransposeTimesInCPU)
{
    TestReduceableTransposeTimes<double>(4, 5, DeviceDescriptor::CPUDevice(), 3);
}

BOOST_AUTO_TEST_CASE(TimesReduceSequenceAxis)
{
    if (ShouldRunOnGpu())
    {
        TestTimesReduceSequenceAxis<double>(153, 21, false, false, { 20, 7, 8 }, DeviceDescriptor::GPUDevice(0));
        TestTimesReduceSequenceAxis<double>(153, 21, false, false, { 20 }, DeviceDescriptor::GPUDevice(0));
        TestTimesReduceSequenceAxis<double>(345, 1, false, false, { 7, 8 }, DeviceDescriptor::GPUDevice(0));
        TestTimesReduceSequenceAxis<double>(345, 1, true, false, { 7, 8 }, DeviceDescriptor::GPUDevice(0));
        TestTimesReduceSequenceAxis<double>(345, 1, true, false, { 7 }, DeviceDescriptor::GPUDevice(0));
    }

    if (ShouldRunOnCpu())
    {
        TestTimesReduceSequenceAxis<double>(153, 21, false, false, { 20, 7, 8 }, DeviceDescriptor::CPUDevice());
        TestTimesReduceSequenceAxis<double>(153, 21, false, false, { 20 }, DeviceDescriptor::CPUDevice());
        TestTimesReduceSequenceAxis<double>(345, 1, false, false, { 7, 8 }, DeviceDescriptor::CPUDevice());
        TestTimesReduceSequenceAxis<double>(345, 1, true, false, { 7, 8 }, DeviceDescriptor::CPUDevice());
        TestTimesReduceSequenceAxis<double>(345, 1, true, false, { 7 }, DeviceDescriptor::CPUDevice());
    }
}

BOOST_AUTO_TEST_CASE(FFTimesAndPlusInGPU)
{
    if (ShouldRunOnGpu())
    {
        TestTimesAndPlus<float>(145, 32, 2, DeviceDescriptor::GPUDevice(0), 10, true, false, true);
        TestTimesAndPlus<double>(145, 15, 200, DeviceDescriptor::GPUDevice(0), 21, false, false, false);
    }
}

BOOST_AUTO_TEST_CASE(FFNetworkCreationInGPU)
{
    if (ShouldRunOnGpu())
    {
        TestFeedForwardNetworkCreation(DeviceDescriptor::GPUDevice(0), true);
        TestFeedForwardNetworkCreation(DeviceDescriptor::GPUDevice(0), false);
    }
}

BOOST_AUTO_TEST_CASE(FFNetworkCreationInCPU)
{
    if (ShouldRunOnCpu())
    {
        TestFeedForwardNetworkCreation(DeviceDescriptor::CPUDevice(), false);
        TestFeedForwardNetworkCreation(DeviceDescriptor::CPUDevice(), true);
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}
