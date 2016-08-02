#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include <numeric>

using namespace CNTK;

static unsigned long seed = 1;

template <typename ElementType>
std::pair<FunctionPtr, FunctionPtr> LSTMPCellWithSelfStabilization(Variable input, Variable prevOutput, Variable prevCellState, const DeviceDescriptor& device)
{
    assert(input.Shape().NumAxes() == 1);
    size_t inputDim = input.Shape()[0];

    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto Wxo = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));
    auto Wxi = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));
    auto Wxf = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));
    auto Wxc = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));

    auto Bo = Parameter({ cellDim }, (ElementType)0.0, device);
    auto Bc = Parameter({ cellDim }, (ElementType)0.0, device);
    auto Bi = Parameter({ cellDim }, (ElementType)0.0, device);
    auto Bf = Parameter({ cellDim }, (ElementType)0.0, device);

    auto Whi = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));
    auto Wci = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim }, -0.5, 0.5, seed++, device));

    auto Whf = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));
    auto Wcf = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim }, -0.5, 0.5, seed++, device));

    auto Who = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));
    auto Wco = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim }, -0.5, 0.5, seed++, device));

    auto Whc = Parameter(NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));

    auto Wmr = Parameter(NDArrayView::RandomUniform<ElementType>({ outputDim, cellDim }, -0.5, 0.5, seed++, device));

    // Stabilization by routing input through an extra scalar parameter
    auto sWxo = Parameter({}, (ElementType)0.0, device);
    auto sWxi = Parameter({}, (ElementType)0.0, device);
    auto sWxf = Parameter({}, (ElementType)0.0, device);
    auto sWxc = Parameter({}, (ElementType)0.0, device);

    auto sWhi = Parameter({}, (ElementType)0.0, device);
    auto sWci = Parameter({}, (ElementType)0.0, device);

    auto sWhf = Parameter({}, (ElementType)0.0, device);
    auto sWcf = Parameter({}, (ElementType)0.0, device);
    auto sWho = Parameter({}, (ElementType)0.0, device);
    auto sWco = Parameter({}, (ElementType)0.0, device);
    auto sWhc = Parameter({}, (ElementType)0.0, device);

    auto sWmr = Parameter({}, (ElementType)0.0, device);

    auto expsWxo = Exp(sWxo);
    auto expsWxi = Exp(sWxi);
    auto expsWxf = Exp(sWxf);
    auto expsWxc = Exp(sWxc);

    auto expsWhi = Exp(sWhi);
    auto expsWci = Exp(sWci);

    auto expsWhf = Exp(sWhf);
    auto expsWcf = Exp(sWcf);
    auto expsWho = Exp(sWho);
    auto expsWco = Exp(sWco);
    auto expsWhc = Exp(sWhc);

    auto expsWmr = Exp(sWmr);

    auto Wxix = Times(Wxi, ElementTimes(expsWxi, input));
    auto Whidh = Times(Whi, ElementTimes(expsWhi, prevOutput));
    auto Wcidc = ElementTimes(Wci, ElementTimes(expsWci, prevCellState));

    auto it = Sigmoid(Plus(Plus(Plus(Wxix, Bi), Whidh), Wcidc));

    auto Wxcx = Times(Wxc, ElementTimes(expsWxc, input));
    auto Whcdh = Times(Whc, ElementTimes(expsWhc, prevOutput));
    auto bit = ElementTimes(it, Tanh(Plus(Wxcx, Plus(Whcdh, Bc))));

    auto Wxfx = Times(Wxf, ElementTimes(expsWxf, input));
    auto Whfdh = Times(Whf, ElementTimes(expsWhf, prevOutput));
    auto Wcfdc = ElementTimes(Wcf, ElementTimes(expsWcf, prevCellState));

    auto ft = Sigmoid(Plus(Plus(Plus(Wxfx, Bf), Whfdh), Wcfdc));

    auto bft = ElementTimes(ft, prevCellState);

    auto ct = Plus(bft, bit);

    auto Wxox = Times(Wxo, ElementTimes(expsWxo, input));
    auto Whodh = Times(Who, ElementTimes(expsWho, prevOutput));
    auto Wcoct = ElementTimes(Wco, ElementTimes(expsWco, ct));

    auto ot = Sigmoid(Plus(Plus(Plus(Wxox, Bo), Whodh), Wcoct));

    auto mt = ElementTimes(ot, Tanh(ct));

    return{ Times(Wmr, ElementTimes(expsWmr, mt)), ct };
}

template <typename ElementType>
FunctionPtr LSTMPComponentWithSelfStabilization(Variable input, size_t outputDim, size_t cellDim, const DeviceDescriptor& device)
{
    auto dh = Placeholder({ outputDim });
    auto dc = Placeholder({ cellDim });

    auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

    auto actualDh = PastValue(Constant({}, (ElementType)0.0, device), LSTMCell.first, 1);
    auto actualDc = PastValue(Constant({}, (ElementType)0.0, device), LSTMCell.second, 1);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    return LSTMCell.first->ReplacePlaceholders({ { dh, actualDh }, { dc, actualDc } });
}

template <typename ElementType>
FunctionPtr LSTMNet(Variable features, size_t cellDim, size_t hiddenDim, size_t numOutputClasses, size_t numLSTMLayers, const DeviceDescriptor& device, const std::wstring& outputName)
{
    assert(numLSTMLayers >= 1);
    auto classifierRoot = LSTMPComponentWithSelfStabilization<ElementType>(features, hiddenDim, cellDim, device);
    for (size_t i = 1; i < numLSTMLayers; ++i) {
        classifierRoot = LSTMPComponentWithSelfStabilization<ElementType>(classifierRoot, hiddenDim, cellDim, device);
    }

    auto W = Parameter(NDArrayView::RandomUniform<ElementType>({ numOutputClasses, hiddenDim }, -0.5, 0.5, seed++, device));
    auto b = Parameter({ numOutputClasses }, (ElementType)0.0, device);

    auto sW = Parameter({}, (ElementType)0.0, device);
    auto expsW = Exp(sW);

    return Plus(Times(W, ElementTimes(expsW, classifierRoot)), b, outputName);
}

template <typename ElementType>
void TestRecurrentNetworkCreation(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t inputDim = 937;
    const size_t numLSTMLayers = 3;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;
    const size_t numOutputClasses = 9304;

    Variable features({ inputDim }, AsDataType<ElementType>(), L"features");
    auto classifierOutputFunction = LSTMNet<ElementType>(features, cellDim, hiddenDim, numOutputClasses, numLSTMLayers, device, L"classifierOutput");
    Variable classifierOutput = classifierOutputFunction;

    Variable labelsVar = Variable({ numOutputClasses }, AsDataType<ElementType>(), L"labels");
    auto trainingLossFunction = CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"lossFunction");
    Variable trainingLoss = trainingLossFunction;
    auto predictionFunction = ClassificationError(classifierOutputFunction, labelsVar, L"classificationError");
    Variable prediction = predictionFunction;

    auto LSTMClassifier = Combine({ trainingLossFunction, predictionFunction, classifierOutputFunction }, L"LSTMClassifier");

    // Now test the structure
    if (LSTMClassifier->Arguments().size() != 2)
        throw std::runtime_error("TestFeedForwardNetworkCreation: Function does not have expected Argument count");

    if (LSTMClassifier->Outputs().size() != 3)
        throw std::runtime_error("TestFeedForwardNetworkCreation: Function does not have expected Output count");

    if (LSTMClassifier->Parameters().size() != ((numLSTMLayers * 28) + 3))
        throw std::runtime_error("TestFeedForwardNetworkCreation: Function does not have expected Parameter count");

    if (testSaveAndReLoad)
        SaveAndReloadModel<ElementType>(LSTMClassifier, { &features, &labelsVar, &trainingLoss, &prediction, &classifierOutput }, device);

    // Run Forward and backward a few times
    size_t iterationCount = 3;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSequences = 7;
    size_t maxAllowedSequenceLength = 11;
    for (size_t i = 0; i < iterationCount; ++i)
    {
        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxActualSequenceLength = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            sequenceLengths[i] = (rand() % maxAllowedSequenceLength) + 1;
            if (sequenceLengths[i] > maxActualSequenceLength)
                maxActualSequenceLength = sequenceLengths[i];
        }

        std::vector<std::vector<ElementType>> inputSequences;
        for (size_t i = 0; i < numSequences; ++i)
        {
            std::vector<ElementType> currentSequence(inputDim * sequenceLengths[i]);
            for (size_t j = 0; j < currentSequence.size(); ++j)
                currentSequence[j] = ((ElementType)rand()) / RAND_MAX;

            inputSequences.push_back(std::move(currentSequence));
        }

        ValuePtr inputValue = Value::Create({ inputDim }, inputSequences, device, true);

        std::vector<std::vector<ElementType>> labelsData;
        for (size_t i = 0; i < numSequences; ++i)
        {
            std::vector<ElementType> currentSequence(numOutputClasses * sequenceLengths[i]);
            for (size_t j = 0; j < sequenceLengths[i]; ++j)
                currentSequence[(j * numOutputClasses) + (rand() % numOutputClasses)] = 1;

            labelsData.push_back(std::move(currentSequence));
        }

        ValuePtr labelValue = Value::Create({ numOutputClasses }, labelsData, device, true);

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = { { classifierOutput, outputValue }, { prediction, predictionErrorValue } };
        auto backpropState = LSTMClassifier->Forward({ { features, inputValue }, { labelsVar, labelValue } }, outputs, device, { trainingLoss });

        // Perform backprop
        NDShape outputShape = trainingLoss.Shape();
        std::vector<ElementType> rootGradientsData(outputShape.TotalSize(), 1);
        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true));
        std::unordered_map<Variable, ValuePtr> paramGradients;
        auto allParams = LSTMClassifier->Parameters();
        for (auto iter = allParams.begin(); iter != allParams.end(); ++iter)
            paramGradients[*iter] = nullptr;

        LSTMClassifier->Backward(backpropState, { { trainingLoss, rootGradientValue } }, paramGradients);
    }
}

template <typename ElementType>
void TestSimpleRecurrence(size_t inputDim,
                          size_t outputDim,
                          size_t maxAllowedSequenceLength,
                          size_t numSequences,
                          const DeviceDescriptor& device,
                          bool testSaveAndReLoad,
                          size_t numIterations,
                          bool useFutureValue,
                          bool useSparseInputs,
                          bool useOneHotSparseInputs = false,
                          unsigned int seed = 1)
{
    if (useOneHotSparseInputs && !useSparseInputs)
        throw std::runtime_error("useOneHotSparseInputs option can only be true when useSparseInputs is true");

    Parameter timesParam(MakeSharedObject<NDArrayView>((ElementType)0.5, NDShape({ outputDim, inputDim }), device), L"timesParameters");
    Parameter plusParam(MakeSharedObject<NDArrayView>((ElementType)0.1, std::initializer_list<size_t>({ outputDim }), device), L"plusParameters");

    Variable inputVar({ inputDim }, useSparseInputs, AsDataType<ElementType>(), true, L"input");

    auto placeholder = Placeholder({ outputDim });
    auto plusOutputFunction = Plus(plusParam, Plus(placeholder, Times(timesParam, inputVar)), L"plusOutput");
    FunctionPtr placeholderReplacement;
    if (useFutureValue)
        placeholderReplacement = FutureValue(Constant({}, (ElementType)0.0, device), plusOutputFunction, 1);
    else
        placeholderReplacement = PastValue(Constant({}, (ElementType)0.0, device), plusOutputFunction, 1);

    plusOutputFunction = plusOutputFunction->ReplacePlaceholders({ { placeholder, placeholderReplacement } });
    Variable plusOutput = plusOutputFunction;

    auto reducedOutputFunction = ReduceSum(plusOutput, L"sum");
    Variable reducedOutput = reducedOutputFunction;

    auto rootFunc = Combine({ reducedOutputFunction, plusOutputFunction });

    if (testSaveAndReLoad)
        SaveAndReloadModel<ElementType>(rootFunc, { &inputVar, &timesParam, &plusParam, &plusOutput, &reducedOutput }, device);

    srand(seed);
    for (size_t iterIdx = 0; iterIdx < numIterations; ++iterIdx)
    {
        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxActualSequenceLength = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            sequenceLengths[i] = (rand() % maxAllowedSequenceLength) + 1;
            if (sequenceLengths[i] > maxActualSequenceLength)
                maxActualSequenceLength = sequenceLengths[i];
        }

        NDShape inputShape = inputVar.Shape().AppendShape({ maxActualSequenceLength, numSequences });
        ValuePtr inputValue;
        size_t totalNumInputSamples = maxActualSequenceLength * numSequences;
        std::vector<ElementType> inputData(inputDim * totalNumInputSamples, useSparseInputs ? 0 : std::numeric_limits<ElementType>::quiet_NaN());
        if (useOneHotSparseInputs)
        {
            std::vector<std::vector<size_t>> oneHotSequences;
            for (size_t i = 0; i < numSequences; ++i)
            {
                std::vector<size_t> currentSequence(sequenceLengths[i]);
                for (size_t j = 0; j < sequenceLengths[i]; ++j)
                {
                    size_t hotRowIndex = rand() % inputDim;
                    currentSequence[j] = hotRowIndex;
                    size_t sampleIdx = (i * maxActualSequenceLength) + j;
                    inputData[(sampleIdx * inputDim) + hotRowIndex] = 1;
                }

                oneHotSequences.push_back(std::move(currentSequence));
            }

            inputValue = Value::Create<ElementType>({ inputDim }, oneHotSequences, DeviceDescriptor::CPUDevice(), true);
        }
        else
        {
            for (size_t i = 0; i < numSequences; ++i)
            {
                for (size_t j = 0; j < maxActualSequenceLength; ++j)
                {
                    size_t sampleIdx = (i * maxActualSequenceLength) + j;
                    size_t maxNumberOfNonZeroValuesPerSparseInputSample = std::max<size_t>(inputDim / 200, 1);
                    size_t numActualValuesWritten = 0;
                    for (size_t k = 0; k < inputDim; ++k)
                    {
                        if ((j < sequenceLengths[i]) && (!useSparseInputs || ((numActualValuesWritten < maxNumberOfNonZeroValuesPerSparseInputSample) && ((rand() % inputDim) < maxNumberOfNonZeroValuesPerSparseInputSample))))
                        {
                            numActualValuesWritten++;
                            inputData[(sampleIdx * inputDim) + k] = ((ElementType)rand()) / RAND_MAX;
                        }
                    }
                }
            }

            NDArrayViewPtr inputValueData = MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true);
            if (useSparseInputs)
            {
                NDArrayViewPtr sparseInputValueData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), StorageFormat::SparseCSC, inputShape, DeviceDescriptor::CPUDevice());
                sparseInputValueData->CopyFrom(*inputValueData);
                inputValueData = sparseInputValueData->Alias(true);
            }

            NDMaskPtr inputMask = MakeSharedObject<NDMask>(NDShape({ maxActualSequenceLength, numSequences }), DeviceDescriptor::CPUDevice());
            for (size_t i = 0; i < numSequences; ++i)
                inputMask->MaskSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });

            inputValue = MakeSharedObject<Value>(inputValueData, inputMask);
        }

        NDShape reducedOutputShape = {};
        std::vector<ElementType> reducedOutputData(reducedOutputShape.TotalSize());
        ValuePtr reducedOutputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(reducedOutputShape, reducedOutputData.data(), reducedOutputData.size(), DeviceDescriptor::CPUDevice(), false));

        NDShape plusOutputShape = plusOutput.Shape().AppendShape({ maxActualSequenceLength, numSequences });
        std::vector<ElementType> plusOutputData(plusOutputShape.TotalSize());
        ValuePtr plusOutputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusOutputShape, plusOutputData.data(), plusOutputData.size(), DeviceDescriptor::CPUDevice(), false), MakeSharedObject<NDMask>(inputValue->Mask()->Shape(), inputValue->Mask()->Device()));

        std::unordered_map<Variable, ValuePtr> outputs = { { reducedOutput, reducedOutputValue }, { plusOutput, plusOutputValue } };
        auto backpropState = rootFunc->Forward({ { inputVar, inputValue } }, outputs, device, { plusOutput });

        // Perform backprop
        std::vector<ElementType> rootGradientsData(plusOutputShape.TotalSize(), std::numeric_limits<ElementType>::quiet_NaN());
        for (size_t i = 0; i < numSequences; ++i)
        {
            for (size_t j = 0; j < maxActualSequenceLength; ++j)
            {
                size_t sampleIdx = (i * maxActualSequenceLength) + j;
                for (size_t k = 0; k < outputDim; ++k)
                {
                    if (j < sequenceLengths[i])
                        rootGradientsData[(sampleIdx * outputDim) + k] = 1;
                }
            }
        }

        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusOutputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true), inputValue->Mask()->DeepClone());

        std::vector<ElementType> plusParameterGradientData(plusParam.Shape().TotalSize());
        std::vector<ElementType> timesParameterGradientData(timesParam.Shape().TotalSize());
        std::vector<ElementType> inputGradientData(inputShape.TotalSize());
        ValuePtr plusParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusParam.Shape(), plusParameterGradientData.data(), plusParameterGradientData.size(), DeviceDescriptor::CPUDevice(), false));
        ValuePtr timesParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(timesParam.Shape(), timesParameterGradientData.data(), timesParameterGradientData.size(), DeviceDescriptor::CPUDevice(), false));
        ValuePtr inputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputGradientData.data(), inputGradientData.size(), DeviceDescriptor::CPUDevice(), false), inputValue->Mask()->DeepClone());

        std::unordered_map<Variable, ValuePtr> outGradients = { { inputVar, inputGradientValue }, { plusParam, plusParameterGradientValue }, { timesParam, timesParameterGradientValue } };
        rootFunc->Backward(backpropState, { { plusOutput, rootGradientValue } }, outGradients);

        // Verify forward prop results
        std::vector<ElementType> expectedPlusOutputData(plusOutputShape.TotalSize(), 0);
        ElementType expectedReducedValue = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            if (useFutureValue)
            {
                for (int j = (int)(currentSequenceLength - 1); j >= 0; j--)
                {
                    ElementType value = 0;
                    for (size_t k = 0; k < inputDim; ++k)
                        value += (ElementType)(0.5 * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k]);

                    for (size_t k = 0; k < outputDim; ++k)
                    {
                        expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] = (ElementType)(value + 0.1);

                        if (j != (currentSequenceLength - 1))
                            expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] += expectedPlusOutputData[(((i * maxActualSequenceLength) + (j + 1)) * outputDim) + k];
                    }

                    expectedReducedValue += (outputDim * (ElementType)((value + 0.1) * (j + 1)));
                }
            }
            else
            {
                for (size_t j = 0; j < currentSequenceLength; j++)
                {
                    ElementType value = 0;
                    for (size_t k = 0; k < inputDim; ++k)
                        value += (ElementType)(0.5 * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k]);

                    for (size_t k = 0; k < outputDim; ++k)
                    {
                        expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] = (ElementType)(value + 0.1);

                        if (j != 0)
                            expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] += expectedPlusOutputData[(((i * maxActualSequenceLength) + (j - 1)) * outputDim) + k];
                    }

                    expectedReducedValue += (outputDim * (ElementType)((value + 0.1) * (currentSequenceLength - j)));
                }
            }
        }

        FloatingPointVectorCompare(reducedOutputData, std::vector<ElementType>({ expectedReducedValue }), "TestTimesAndPlus: Forward prop results do not match expected results");
        FloatingPointVectorCompare(plusOutputData, expectedPlusOutputData, "TestTimesAndPlus: Forward prop results do not match expected results");

        // Verify backward prop results
        ElementType expectedPlusParameterGradientValue = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            expectedPlusParameterGradientValue += (currentSequenceLength * (currentSequenceLength + 1)) / 2;
        }

        for (size_t k = 0; k < plusParam.Shape().TotalSize(); ++k)
            if (plusParameterGradientData[k] != expectedPlusParameterGradientValue)
                throw std::runtime_error("TestSimpleRecurrence: Backprop prop results do not match expected results for Plus params gradients");

        std::vector<ElementType> expectedTimesParamsGradientValues(timesParam.Shape().TotalSize(), 0);
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            for (size_t k = 0; k < inputDim; ++k)
            {
                ElementType gradVal = 0;
                for (size_t j = 0; j < currentSequenceLength; j++)
                {
                    if (useFutureValue)
                        gradVal += (j + 1) * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k];
                    else
                        gradVal += (currentSequenceLength - j) * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k];
                }

                for (size_t j = 0; j < outputDim; ++j)
                    expectedTimesParamsGradientValues[(k * outputDim) + j] += gradVal;
            }
        }

        FloatingPointVectorCompare(timesParameterGradientData, expectedTimesParamsGradientValues, "TestSimpleRecurrence: Backprop prop results do not match expected results for Times params gradients");

        std::vector<ElementType> expectedInputGradientValues(inputShape.TotalSize(), 0);
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            for (size_t j = 0; j < currentSequenceLength; j++)
            {
                ElementType gradVal = 0;
                for (size_t k = 0; k < outputDim; ++k)
                {
                    if (useFutureValue)
                        gradVal += (ElementType)((j + 1) * 0.5);
                    else
                        gradVal += (ElementType)((currentSequenceLength - j) * 0.5);
                }

                for (size_t k = 0; k < inputDim; ++k)
                    expectedInputGradientValues[(((i * maxActualSequenceLength) + j) * inputDim) + k] = gradVal;
            }
        }

        FloatingPointVectorCompare(inputGradientData, expectedInputGradientValues, "TestSimpleRecurrence: Backprop prop results do not match expected results for Times params gradients");
    }
}

void RecurrentFunctionTests()
{
    TestSimpleRecurrence<float>(2, 1, 4, 1, DeviceDescriptor::CPUDevice(), true, 3, false, false);
#ifndef CPUONLY
    TestSimpleRecurrence<double>(11, 9, 16, 7, DeviceDescriptor::GPUDevice(0), true, 5, true, false);
#endif
    TestSimpleRecurrence<double>(1000, 9, 16, 3, DeviceDescriptor::CPUDevice(), false, 2, true, true);
#ifndef CPUONLY
    TestSimpleRecurrence<float>(5000, 200, 19, 6, DeviceDescriptor::GPUDevice(0), false, 3, false, true);
    TestSimpleRecurrence<double>(1000, 9, 16, 3, DeviceDescriptor::GPUDevice(0), true, 3, true, true, true);
#endif
    TestSimpleRecurrence<float>(5000, 200, 19, 6, DeviceDescriptor::CPUDevice(), true, 2, false, true, true);

#ifndef CPUONLY
    TestRecurrentNetworkCreation<float>(DeviceDescriptor::GPUDevice(0), true);
#endif
    TestRecurrentNetworkCreation<double>(DeviceDescriptor::CPUDevice(), false);
}
