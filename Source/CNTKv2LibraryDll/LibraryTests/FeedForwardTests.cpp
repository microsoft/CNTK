#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

FunctionPtr FullyConnectedDNNLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
    assert(input.Shape().NumAxes() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = Parameter(NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.5, 0.5, 1, device));
    auto timesFunction = Times(timesParam, input);

    auto plusParam = Parameter({ outputDim }, 0.0f, device);
    auto plusFunction = Plus(plusParam, timesFunction);

    return nonLinearity(plusFunction);
}

FunctionPtr FullyConnectedFeedForwardClassifierNet(Variable input, size_t numOutputClasses, size_t hiddenLayerDim, size_t numHiddenLayers, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
    assert(numHiddenLayers >= 1);
    auto classifierRoot = FullyConnectedDNNLayer(input, hiddenLayerDim, device, nonLinearity);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierRoot = FullyConnectedDNNLayer(classifierRoot, hiddenLayerDim, device, nonLinearity);

    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.5, 0.5, 1, device));
    classifierRoot = Times(outputTimesParam, classifierRoot);
    return classifierRoot;
}

void TestFeedForwardNetworkCreation(const DeviceDescriptor& device)
{
    using namespace std::placeholders;

    const size_t inputDim = 937;
    const size_t numOutputClasses = 9304;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    Variable inputVar({ inputDim }, DataType::Float, L"Features");
    auto classifierOutputFunction = FullyConnectedFeedForwardClassifierNet(inputVar, numOutputClasses, hiddenLayersDim, numHiddenLayers, device, std::bind(Sigmoid, _1, L""));

    Variable labelsVar({ numOutputClasses }, DataType::Float, L"Labels");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"LossFunction");
    auto predictionFunction = CNTK::PredictionError(classifierOutputFunction, labelsVar, L"PredictionError");

    auto ffNet = CNTK::Combine({ trainingLossFunction, predictionFunction, classifierOutputFunction }, L"ClassifierModel");

    // Now test the structure
    if (ffNet->Parameters().size() != ((numHiddenLayers * 2) + 1))
        throw std::exception("TestFeedForwardNetworkCreation: Function does not have expected Parameter count");

    if (ffNet->Arguments().size() != 2)
        throw std::exception("TestFeedForwardNetworkCreation: Function does not have expected Argument count");

    if (ffNet->Outputs().size() != 3)
        throw std::exception("TestFeedForwardNetworkCreation: Function does not have expected Output count");

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

        NDShape inputShape = { inputDim, 1, numSamples };
        ValuePtr inputValue = new Value(new NDArrayView(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        std::vector<float> labelData(numOutputClasses * numSamples, 0);
        for (size_t i = 0; i < numSamples; ++i)
            labelData[(i*numOutputClasses) + (rand() % numOutputClasses)] = 1;

        NDShape labelShape = { numOutputClasses, 1, numSamples };
        ValuePtr labelValue = new Value(new NDArrayView(labelShape, labelData.data(), labelData.size(), DeviceDescriptor::CPUDevice(), true));

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = { { classifierOutputFunction->Output(), outputValue }, { predictionFunction->Output(), predictionErrorValue } };
        auto backpropState = ffNet->Forward({ { inputVar, inputValue }, { labelsVar, labelValue } }, outputs, device, { trainingLossFunction->Output() });

        // Perform backprop
        NDShape outputShape = trainingLossFunction->Output().Shape();
        std::vector<float> rootGradientsData(outputShape.TotalSize(), 1);
        ValuePtr rootGradientValue = new Value(new NDArrayView(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true));
        std::unordered_map<Variable, ValuePtr> paramGradients;
        auto allParams = ffNet->Parameters();
        for (auto iter = allParams.begin(); iter != allParams.end(); ++iter)
            paramGradients[*iter] = nullptr;
        
        ffNet->Backward(backpropState, { { trainingLossFunction->Output(), rootGradientValue } }, paramGradients);
    }
}

template <typename ElementType>
void TestTimesAndPlus(size_t inputDim,
                      size_t outputDim,
                      size_t numSamples,
                      const DeviceDescriptor& device,
                      size_t numIterations,
                      bool usePreAllocatedOutputs,
                      bool outputOnSpecifiedDevice = false,
                      unsigned int seed = 1)
{
    Parameter timesParam(new NDArrayView((ElementType)0.5, { outputDim, inputDim }, device));
    Parameter plusParam(new NDArrayView((ElementType)1.2, { outputDim }, device));

    Variable inputVar({ inputDim }, GetDataType<ElementType>(), L"input");
    auto timesAndPlusFunc = Plus(plusParam, Times(timesParam, inputVar));

    srand(seed);
    for (size_t iterIdx = 0; iterIdx < numIterations; ++iterIdx)
    {
        std::vector<ElementType> inputData(inputDim * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
            inputData[i] = ((ElementType)rand()) / RAND_MAX;

        NDShape inputShape = { inputDim, 1, numSamples };
        ValuePtr inputValue = new Value(new NDArrayView(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        NDShape outputShape = { outputDim, 1, numSamples };
        std::vector<ElementType> outputData(outputShape.TotalSize());
        ValuePtr outputValue;
        if (usePreAllocatedOutputs)
        {
            auto outputAllocationDevice = outputOnSpecifiedDevice ? device : DeviceDescriptor::CPUDevice();
            if (outputAllocationDevice.Type() == DeviceType::CPU)
                outputValue = new Value(new NDArrayView(outputShape, outputData.data(), outputData.size(), outputAllocationDevice, false));
            else
                outputValue = new Value(new NDArrayView(GetDataType<ElementType>(), outputShape, nullptr, 0, outputAllocationDevice, false));
        }

        std::unordered_map<Variable, ValuePtr> outputs = { { timesAndPlusFunc->Output(), outputValue } };
        auto backpropState = timesAndPlusFunc->Forward({ { inputVar, inputValue } }, outputs, device, { timesAndPlusFunc->Output() });

        if (!usePreAllocatedOutputs)
            outputValue = outputs[timesAndPlusFunc->Output()];

        // Perform backprop
        std::vector<ElementType> rootGradientsData(outputShape.TotalSize(), 1);
        ValuePtr rootGradientValue;
        if (device.Type() == DeviceType::CPU)
            rootGradientValue = new Value(new NDArrayView(outputShape, rootGradientsData.data(), rootGradientsData.size(), device, true));
        else
        {
            NDArrayViewPtr cpuArrayView = new NDArrayView(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true);
            NDArrayViewPtr gpuArrayView = new NDArrayView(GetDataType<ElementType>(), outputShape, nullptr, 0, device, false);
            gpuArrayView->CopyFrom(*cpuArrayView);
            rootGradientValue = new Value(gpuArrayView);
        }

        std::vector<ElementType> plusParameterGradientData(plusParam.Shape().TotalSize());
        std::vector<ElementType> timesParameterGradientData(timesParam.Shape().TotalSize());
        ValuePtr plusParameterGradientValue, timesParameterGradientValue;
        if (usePreAllocatedOutputs)
        {
            auto outputAllocationDevice = outputOnSpecifiedDevice ? device : DeviceDescriptor::CPUDevice();
            if (outputAllocationDevice.Type() == DeviceType::CPU)
            {
                plusParameterGradientValue = new Value(new NDArrayView(plusParam.Shape(), plusParameterGradientData.data(), plusParameterGradientData.size(), outputAllocationDevice, false));
                timesParameterGradientValue = new Value(new NDArrayView(timesParam.Shape(), timesParameterGradientData.data(), timesParameterGradientData.size(), outputAllocationDevice, false));
            }
            else
            {
                plusParameterGradientValue = new Value(new NDArrayView(GetDataType<ElementType>(), plusParam.Shape(), nullptr, 0, outputAllocationDevice, false));
                timesParameterGradientValue = new Value(new NDArrayView(GetDataType<ElementType>(), timesParam.Shape(), nullptr, 0, outputAllocationDevice, false));
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
        if (!usePreAllocatedOutputs || (outputOnSpecifiedDevice && (device.Type() != DeviceType::CPU)))
        {
            NDArrayViewPtr cpuArrayView = new NDArrayView(outputShape, outputData.data(), outputData.size(), DeviceDescriptor::CPUDevice(), false);
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
        if (device.Type() != DeviceType::CPU)
        {
            NDArrayViewPtr cpuArrayView = new NDArrayView(GetDataType<ElementType>(), plusParam.Shape(), nullptr, 0, DeviceDescriptor::CPUDevice(), false);
            cpuArrayView->CopyFrom(*plusParameterGradientValue->Data());
            const ElementType* cpuArrayViewBuffer = cpuArrayView->DataBuffer<ElementType>();
            memcpy(plusParameterGradientData.data(), cpuArrayViewBuffer, plusParam.Shape().TotalSize() * sizeof(ElementType));

            cpuArrayView = new NDArrayView(GetDataType<ElementType>(), timesParam.Shape(), nullptr, 0, DeviceDescriptor::CPUDevice(), false);
            cpuArrayView->CopyFrom(*timesParameterGradientValue->Data());
            cpuArrayViewBuffer = cpuArrayView->DataBuffer<ElementType>();
            memcpy(timesParameterGradientData.data(), cpuArrayViewBuffer, timesParam.Shape().TotalSize() * sizeof(ElementType));
        }

        for (size_t i = 0; i < outputDim; ++i)
            if (plusParameterGradientData[i] != numSamples)
                throw std::exception("TestTimesAndPlus: Backprop prop results do not match expected results for Plus params gradients");

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
    TestTimesAndPlus<double>(4, 2, 5, DeviceDescriptor::CPUDevice(), 3, true, true);
    TestTimesAndPlus<float>(145, 32, 2, DeviceDescriptor::GPUDevice(0), 10, true, false);
    TestTimesAndPlus<double>(145, 15, 200, DeviceDescriptor::GPUDevice(0), 21, false);

    TestFeedForwardNetworkCreation(DeviceDescriptor::GPUDevice(0));
    TestFeedForwardNetworkCreation(DeviceDescriptor::CPUDevice());
}
