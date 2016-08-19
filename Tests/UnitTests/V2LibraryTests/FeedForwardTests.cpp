#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

FunctionPtr FullyConnectedFeedForwardClassifierNet(Variable input,
                                                   size_t numOutputClasses,
                                                   size_t hiddenLayerDim,
                                                   size_t numHiddenLayers,
                                                   const DeviceDescriptor& device,
                                                   const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity,
                                                   const std::wstring& outputName)
{
    assert(numHiddenLayers >= 1);
    auto classifierRoot = FullyConnectedDNNLayer(input, hiddenLayerDim, device, nonLinearity);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierRoot = FullyConnectedDNNLayer(classifierRoot, hiddenLayerDim, device, nonLinearity);

    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.5, 0.5, 1, device));
    return Times(outputTimesParam, classifierRoot, 1, outputName);
}

std::wstring s_tempModelPath = L"feedForward.net";

void TestFeedForwardNetworkCreation(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    using namespace std::placeholders;

    const size_t inputDim = 937;
    const size_t numOutputClasses = 9304;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    Variable inputVar({ inputDim }, DataType::Float, L"features");
    auto classifierOutputFunction = FullyConnectedFeedForwardClassifierNet(inputVar, numOutputClasses, hiddenLayersDim, numHiddenLayers, device, std::bind(Sigmoid, _1, L""), L"classifierOutput");
    Variable classifierOutput = classifierOutputFunction;

    Variable labelsVar({ numOutputClasses }, DataType::Float, L"Labels");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutput, labelsVar, L"LossFunction");
    Variable trainingLoss = trainingLossFunction;
    auto predictionFunction = CNTK::ClassificationError(classifierOutput, labelsVar, L"ClassificationError");
    Variable prediction = predictionFunction;

    auto ffNet = CNTK::Combine({ trainingLoss.Owner(), prediction.Owner(), classifierOutput.Owner() }, L"ClassifierModel");

    // Now test the structure
    if (ffNet->Parameters().size() != ((numHiddenLayers * 2) + 1))
        throw std::runtime_error("TestFeedForwardNetworkCreation: Function does not have expected Parameter count");

    if (ffNet->Arguments().size() != 2)
        throw std::runtime_error("TestFeedForwardNetworkCreation: Function does not have expected Argument count");

    if (ffNet->Outputs().size() != 3)
        throw std::runtime_error("TestFeedForwardNetworkCreation: Function does not have expected Output count");

    if (testSaveAndReLoad)
        SaveAndReloadModel<float>(ffNet, { &inputVar, &labelsVar, &trainingLoss, &prediction, &classifierOutput }, device);

    // Run Forward and backward a few times
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSamples = 3;
    for (size_t i = 0; i < iterationCount; ++i)
    {
        std::vector<float> inputData(inputDim * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
            inputData[i] = ((float)rand()) / RAND_MAX;

        NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        std::vector<float> labelData(numOutputClasses * numSamples, 0);
        for (size_t i = 0; i < numSamples; ++i)
            labelData[(i*numOutputClasses) + (rand() % numOutputClasses)] = 1;

        NDShape labelShape = labelsVar.Shape().AppendShape({ 1, numSamples });
        ValuePtr labelValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(labelShape, labelData.data(), labelData.size(), DeviceDescriptor::CPUDevice(), true));

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = { { classifierOutput, outputValue }, { prediction, predictionErrorValue } };
        auto backpropState = ffNet->Forward({ { inputVar, inputValue }, { labelsVar, labelValue } }, outputs, device, { trainingLoss });

        // Perform backprop
        NDShape outputShape = trainingLoss.Shape();
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
    Parameter timesParam(MakeSharedObject<NDArrayView>((ElementType)0.5, NDShape({ outputDim, inputDim }), device), L"timesParameters");
    Parameter plusParam(MakeSharedObject<NDArrayView>((ElementType)1.2, std::initializer_list<size_t>({ outputDim }), device), L"plusParameters");

    Variable inputVar({ inputDim }, AsDataType<ElementType>(), L"input");
    auto timesAndPlusFunc = Plus(plusParam, Times(timesParam, inputVar));

    if (testSaveAndReLoad)
        SaveAndReloadModel<ElementType>(timesAndPlusFunc, { &inputVar, &timesParam, &plusParam }, device);

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

void FeedForwardTests()
{
    TestTimesAndPlus<double>(4, 2, 5, DeviceDescriptor::CPUDevice(), 3, true, true, true);
#ifndef CPUONLY
    TestTimesAndPlus<float>(145, 32, 2, DeviceDescriptor::GPUDevice(0), 10, true, false, true);
    TestTimesAndPlus<double>(145, 15, 200, DeviceDescriptor::GPUDevice(0), 21, false, false, false);

    TestFeedForwardNetworkCreation(DeviceDescriptor::GPUDevice(0), true);
    TestFeedForwardNetworkCreation(DeviceDescriptor::GPUDevice(0), false);
#endif
    TestFeedForwardNetworkCreation(DeviceDescriptor::CPUDevice(), false);
    TestFeedForwardNetworkCreation(DeviceDescriptor::CPUDevice(), true);
}
