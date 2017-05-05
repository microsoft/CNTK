//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <boost/test/unit_test.hpp>

#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>
#include <fstream>
#include <random>

// enable assert in Release mode.
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#endif

using namespace CNTK;

#ifdef _MSC_VER
// In case of asserts in debug mode, print the message into stderr and throw exception
int HandleDebugAssert(int /* reportType */,
                      char *message, // fully assembled debug user message
                      int *returnValue); // returnValue - retVal value of zero continues execution
#endif

struct V2LibraryTestFixture
{
    V2LibraryTestFixture()
    {
#if defined(_MSC_VER)
        // in case of asserts in debug mode, print the message into stderr and throw exception
        if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
            fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        }
#endif

        // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking
        // which will have a silent performance degradation otherwise
        Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

        // Turn on gap nan tracking
        SetCheckedMode(true);
    }
};

BOOST_GLOBAL_FIXTURE(V2LibraryTestFixture);

#pragma warning(push)
#pragma warning(disable : 4996)
#ifndef _MSC_VER // TODO: what is the correct trigger for gcc?
inline void ReportFailure(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

inline void ReportFailure(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    char buffer[1024] = { 0 };
    vsnprintf(buffer, _countof(buffer) - 1, format, args);
    if (strlen(buffer)/*written*/ >= (int)_countof(buffer) - 2)
        sprintf(buffer + _countof(buffer) - 4, "...");
    BOOST_ERROR(buffer);
}
#pragma warning(pop)

static const double relativeTolerance = 0.001f;
static const double absoluteTolerance = 0.000001f;

bool ShouldRunOnCpu();
bool ShouldRunOnGpu();

template <typename ElementType>
inline void FloatingPointCompare(ElementType actual, ElementType expected, const char* message)
{
    ElementType allowedTolerance = (std::max<ElementType>)((ElementType)absoluteTolerance, std::abs(((ElementType)relativeTolerance) * actual));
    if (std::abs(actual - expected) > allowedTolerance)
    {
        ReportFailure((message + std::string("; Expected=%g, Actual=%g")).c_str(), expected, actual);
    }
}

template <typename ElementType>
inline void FloatingPointVectorCompare(const std::vector<ElementType>& actual, const std::vector<ElementType>& expected, const char* message)
{
    if (actual.size() != expected.size())
    {
        ReportFailure((message + std::string("; actual data vector size (%d) and expected data vector size (%d) are not equal")).c_str(), (int)actual.size(), (int)expected.size());
    }

    for (size_t i = 0; i < actual.size(); ++i)
        FloatingPointCompare(actual[i], expected[i], message);
}

inline void VerifyException(const std::function<void()>& functionToTest, std::string errorMessage) {
    bool exceptionWasThrown = false;
    try
    {
        functionToTest();
    }
    catch (const std::exception&)
    {
        exceptionWasThrown = true;
    }

    BOOST_TEST(exceptionWasThrown, errorMessage);
};

static std::mt19937_64 rng(0);

#pragma warning(push)
#pragma warning(disable: 4996)

#ifndef _MSC_VER
#include <unistd.h>
static inline std::string wtocharpath(const wchar_t *p)
{
    size_t len = wcslen(p);
    std::string buf;
    buf.resize(2 * len + 1);            // max: 1 wchar => 2 mb chars
    ::wcstombs(&buf[0], p, buf.size()); // note: technically it is forbidden to stomp over std::strings 0 terminator, but it is known to work in all implementations
    buf.resize(strlen(&buf[0]));        // set size correctly for shorter strings
    return buf;
}

static inline int _wunlink(const wchar_t *p)
{
    return unlink(wtocharpath(p).c_str());
}

static inline FILE *_wfopen(const wchar_t *path, const wchar_t *mode)
{
    return fopen(wtocharpath(path).c_str(), wtocharpath(mode).c_str());
}

#endif

template <typename ElementType>
inline void SaveAndReloadModel(FunctionPtr& functionPtr, const std::vector<Variable*>& variables, const DeviceDescriptor& device, size_t rank = 0)
{
    const std::wstring tempModelPath = L"feedForward.net" + std::to_wstring((int)rank);

    if ((_wunlink(tempModelPath.c_str()) != 0) && (errno != ENOENT))
        BOOST_ERROR("Error deleting temp model file 'feedForward.net'");

    std::unordered_map<std::wstring, Variable*> inputVarUids;
    std::unordered_map<std::wstring, Variable*> outputVarNames;

    for (auto varPtr : variables)
    {
        auto retVal = varPtr->IsOutput() ? outputVarNames.insert({ varPtr->Name(), varPtr }) : inputVarUids.insert({ varPtr->Uid(), varPtr });
        if (!retVal.second)
           BOOST_ERROR("SaveAndReloadModel: Multiple variables having same name cannot be restored after save and reload");
    }

    functionPtr->Save(tempModelPath);
    functionPtr = Function::Load(tempModelPath, device);

    if (_wunlink(tempModelPath.c_str()) != 0)
         BOOST_ERROR("Error deleting temp model file 'feedForward.net'");

    auto inputs = functionPtr->Inputs();
    for (auto inputVarInfo : inputVarUids)
    {
        auto newInputVar = *(std::find_if(inputs.begin(), inputs.end(), [inputVarInfo](const Variable& var)
        {
            return (var.Uid() == inputVarInfo.first);
        }));

        *(inputVarInfo.second) = newInputVar;
    }

    auto outputs = functionPtr->Outputs();
    for (auto outputVarInfo : outputVarNames)
    {
        auto newOutputVar = *(std::find_if(outputs.begin(), outputs.end(), [outputVarInfo](const Variable& var) {
            return (var.Name() == outputVarInfo.first);
        }));

        *(outputVarInfo.second) = newOutputVar;
    }
}

inline FunctionPtr FullyConnectedLinearLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"", unsigned long seed = 1)
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = Parameter({ outputDim, inputDim }, DataType::Float, GlorotUniformInitializer(DefaultParamInitScale,
                                SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, seed), device, L"timesParam");
    auto timesFunction = Times(timesParam, input, L"times");

    auto plusParam = Parameter({ outputDim }, 0.0f, device, L"plusParam");
    return Plus(plusParam, timesFunction, outputName);
}

inline FunctionPtr FullyConnectedDNNLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity, const std::wstring& outputName = L"", unsigned long seed = 1)
{
    return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device, outputName, seed));
}

inline FunctionPtr FullyConnectedFeedForwardClassifierNet(Variable input,
                                                   size_t numOutputClasses,
                                                   size_t hiddenLayerDim,
                                                   size_t numHiddenLayers,
                                                   const DeviceDescriptor& device,
                                                   const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity,
                                                   const std::wstring& outputName,
                                                   unsigned long seed = 1)
{
    assert(numHiddenLayers >= 1);
    auto classifierRoot = FullyConnectedDNNLayer(input, hiddenLayerDim, device, nonLinearity, L"", seed);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierRoot = FullyConnectedDNNLayer(classifierRoot, hiddenLayerDim, device, nonLinearity, L"", seed);

    auto outputTimesParam = Parameter({ numOutputClasses, hiddenLayerDim }, DataType::Float, UniformInitializer(0.5, seed), device);
    return Times(outputTimesParam, classifierRoot, 1, outputName);
}

template <typename ElementType>
inline FunctionPtr Stabilize(const Variable& x, const DeviceDescriptor& device)
{
    ElementType scalarConstant = 4.0f;
    auto f = Constant::Scalar(scalarConstant);
    auto fInv = Constant::Scalar(f.GetDataType(), 1.0 / scalarConstant);

    auto beta = ElementTimes(fInv, Log(Constant::Scalar(f.GetDataType(), 1.0) + Exp(ElementTimes(f, Parameter({}, f.GetDataType(), 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
    return ElementTimes(beta, x);
}

template <typename ElementType>
std::pair<FunctionPtr, FunctionPtr> LSTMPCellWithSelfStabilization(Variable input, Variable prevOutput, Variable prevCellState, const DeviceDescriptor& device)
{
    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto createBiasParam = [device](size_t dim) {
        return Parameter({ dim }, (ElementType)0.0, device);
    };

    unsigned long seed2 = 1;
    auto createProjectionParam = [device, &seed2](size_t outputDim) {
        return Parameter({ outputDim, NDShape::InferredDimension }, AsDataType<ElementType>(), GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto createDiagWeightParam = [device, &seed2](size_t dim) {
        return Parameter({ dim }, AsDataType<ElementType>(), GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
    auto stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

    auto projectInput = [input, cellDim, createBiasParam, createProjectionParam]() {
        return createBiasParam(cellDim) + Times(createProjectionParam(cellDim), input);
    };

    // Input gate
    auto it = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bit = ElementTimes(it, Tanh(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput)));

    // Forget-me-not gate
    auto ft = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bft = ElementTimes(ft, prevCellState);

    auto ct = bft + bit;

    // Output gate
    auto ot = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
    auto ht = ElementTimes(ot, Tanh(ct));

    auto c = ct;
    auto h = (outputDim != cellDim) ? Times(createProjectionParam(outputDim), Stabilize<ElementType>(ht, device)) : ht;

    return{ h, c };
}

template <typename ElementType>
std::pair<FunctionPtr, FunctionPtr> LSTMPComponentWithSelfStabilization(Variable input,
                                                                                    const NDShape& outputShape,
                                                                                    const NDShape& cellShape,
                                                                                    const std::function<FunctionPtr(const Variable&)>& recurrenceHookH,
                                                                                    const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
                                                                                    const DeviceDescriptor& device)
{
    auto dh = PlaceholderVariable(outputShape, input.DynamicAxes());
    auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

    auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

    auto actualDh = recurrenceHookH(LSTMCell.first);
    auto actualDc = recurrenceHookC(LSTMCell.second);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    LSTMCell.first->ReplacePlaceholders({ { dh, actualDh }, { dc, actualDc } });

    return { LSTMCell.first, LSTMCell.second };
}

// This is currently unused
inline FunctionPtr SimpleRecurrentLayer(const  Variable& input, const NDShape& outputDim, const std::function<FunctionPtr(const Variable&)>& recurrenceHook, const DeviceDescriptor& device)
{
    auto dh = PlaceholderVariable(outputDim, input.DynamicAxes());

    unsigned long seed = 1;
    auto createProjectionParam = [device, &seed](size_t outputDim, size_t inputDim) {
        return Parameter({ outputDim, inputDim }, DataType::Float, UniformInitializer(0.5, seed), device);
    };

    auto hProjWeights = createProjectionParam(outputDim[0], outputDim[0]);
    auto inputProjWeights = createProjectionParam(outputDim[0], input.Shape()[0]);

    auto output = Times(hProjWeights, recurrenceHook(dh)) + Times(inputProjWeights, input);
    return output->ReplacePlaceholders({ { dh, output } });
}

inline std::vector<bool> GenerateSequenceStartFlags(size_t numSequences)
{
    std::vector<bool> sequenceStartFlags(numSequences);
    for (size_t i = 0; i < numSequences; ++i)
    {
        sequenceStartFlags[i] = static_cast<int>(rand()) % 2 == 0 ? true : false;
    }
    return sequenceStartFlags;
}

inline std::vector<size_t> GenerateSequenceLengths(size_t numSequences, size_t maxAllowedSequenceLength)
{
    std::vector<size_t> sequenceLengths(numSequences);
    size_t maxActualSequenceLength = 0;
    size_t minActualSequenceLength = 3;
    for (size_t i = 0; i < numSequences; ++i)
    {
        sequenceLengths[i] = (rand() % maxAllowedSequenceLength) + minActualSequenceLength;
        if (sequenceLengths[i] > maxActualSequenceLength)
            maxActualSequenceLength = sequenceLengths[i];
    }

    return sequenceLengths;
}

template <typename ElementType>
inline std::vector<std::vector<ElementType>> GenerateSequences(const std::vector<size_t>& sequenceLengths, const NDShape& sampleShape)
{
    size_t numSequences = sequenceLengths.size();
    std::vector<std::vector<ElementType>> sequences;
    for (size_t i = 0; i < numSequences; ++i)
    {
        std::vector<ElementType> currentSequence(sampleShape.TotalSize() * sequenceLengths[i]);
        for (size_t j = 0; j < currentSequence.size(); ++j)
            currentSequence[j] = ((ElementType)rand()) / RAND_MAX;

        sequences.push_back(std::move(currentSequence));
    }

    return sequences;
}

inline std::vector<std::vector<size_t>> GenerateOneHotSequences(const std::vector<size_t>& sequenceLengths, size_t dim)
{
    size_t numSequences = sequenceLengths.size();
    std::vector<std::vector<size_t>> oneHotSequences;
    for (size_t i = 0; i < numSequences; ++i)
    {
        std::vector<size_t> currentSequence(sequenceLengths[i]);
        for (size_t j = 0; j < sequenceLengths[i]; ++j)
        {
            size_t hotRowIndex = rand() % dim;
            currentSequence[j] = hotRowIndex;
        }

        oneHotSequences.push_back(std::move(currentSequence));
    }

    return oneHotSequences;
}

template <typename ElementType>
inline ValuePtr GenerateSequences(const std::vector<size_t>& sequenceLengths, const NDShape& sampleShape, const DeviceDescriptor& device, bool oneHot)
{
    if (!oneHot)
    {
        std::vector<std::vector<ElementType>> sequences = GenerateSequences<ElementType>(sequenceLengths, sampleShape);
        return Value::Create(sampleShape, sequences, device, true);
    }
    else
    {
        size_t numSequences = sequenceLengths.size();
        size_t vocabularySize = sampleShape[0];
        size_t numColumnsPerSample = sampleShape.SubShape(1).TotalSize();
        std::vector<size_t> columnLengths = sequenceLengths;
        for (size_t i = 0; i < numSequences; ++i)
            columnLengths[i] *= numColumnsPerSample;

        std::vector<std::vector<size_t>> oneHotSequences = GenerateOneHotSequences(columnLengths, vocabularySize);
        return Value::Create<ElementType>(sampleShape, oneHotSequences, {}, device, true);
    }
}

template <typename ElementType>
inline std::pair<NDArrayViewPtr, NDArrayViewPtr> GenerateSparseSequence(size_t vocabSize, size_t sequenceLength, size_t maxNumberOfNonZeroValuesPerSparseInputSample)
{
    std::vector<ElementType> inputData(vocabSize * sequenceLength, 0);
    for (size_t j = 0; j < sequenceLength; ++j)
    {
        size_t numActualValuesWritten = 0;
        for (size_t k = 0; k < vocabSize; ++k)
        {
            if ((numActualValuesWritten < maxNumberOfNonZeroValuesPerSparseInputSample) && ((rand() % vocabSize) < maxNumberOfNonZeroValuesPerSparseInputSample))
            {
                numActualValuesWritten++;
                inputData[(j * vocabSize) + k] = ((ElementType)rand()) / RAND_MAX;
            }
        }
    }

    NDShape inputDataShape = NDShape({ vocabSize, sequenceLength });
    NDArrayViewPtr inputValueData = MakeSharedObject<NDArrayView>(inputDataShape, inputData);
    NDArrayViewPtr sparseData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), StorageFormat::SparseCSC, inputDataShape, DeviceDescriptor::CPUDevice());
    sparseData->CopyFrom(*inputValueData);
    return{ inputValueData->DeepClone(), sparseData };
}

template <typename ElementType>
std::tuple<std::vector<ElementType>, std::vector<SparseIndexType>, std::vector<SparseIndexType>, std::vector<ElementType>, size_t> GenerateSequenceInCSC(size_t dimension, size_t sequenceLength)
{
    auto numMatrixRows = dimension;
    auto numMatrixCols = sequenceLength;
    std::vector<SparseIndexType> colsStarts(numMatrixCols + 1);

    std::default_random_engine randomG;
    std::uniform_int_distribution<int> numValuesDistribution(0, static_cast<int>(numMatrixRows));
    colsStarts[0] = 0;
    int numNonZeroValues = 0;
    for (size_t i = 1; i <= numMatrixCols; ++i)
    {
        int numValuesInCurrentCol = numValuesDistribution(randomG);
        numNonZeroValues += numValuesInCurrentCol;
        colsStarts[i] = colsStarts[i - 1] + numValuesInCurrentCol;
    }
    if (numNonZeroValues == 0)
    {
        // The uniform distribution does not generate any non-zero values, force to have non-zero values at 1 column.
        int colHavingNonZeroValue = rand() % numMatrixCols;
        std::uniform_int_distribution<int> uniformDistribution(1, static_cast<int>(numMatrixRows));

        colsStarts[0] = 0;
        numNonZeroValues = 0;
        for (size_t i = 1; i <= numMatrixCols; ++i)
        {
            int numValuesInCurrentCol = 0;
            if (i == colHavingNonZeroValue + 1)
            {
                numValuesInCurrentCol = uniformDistribution(randomG);
            }
            numNonZeroValues += numValuesInCurrentCol;
            colsStarts[i] = colsStarts[i - 1] + numValuesInCurrentCol;
        }
    }

    // Now fill the actual values
    std::vector<ElementType> nonZeroValues(numNonZeroValues);
    std::vector<SparseIndexType> rowIndices(numNonZeroValues);
    size_t nnzIndex = 0;
    std::vector<ElementType> referenceDenseData(dimension * sequenceLength);
    for (size_t j = 0; j < numMatrixCols; ++j)
    {
        size_t numRowsWithValuesInCurrentCol = colsStarts[j + 1] - colsStarts[j];
        size_t numValuesWritten = 0;
        std::unordered_set<int> rowsWrittenTo;
        while (numValuesWritten < numRowsWithValuesInCurrentCol)
        {
            int rowIndex = rand() % numMatrixRows;
            if (rowsWrittenTo.insert(rowIndex).second)
            {
                ElementType value = ((ElementType)rand()) / RAND_MAX;
                nonZeroValues[nnzIndex] = value;
                referenceDenseData[(j * numMatrixRows) + rowIndex] = value;
                rowIndices[nnzIndex] = rowIndex;
                numValuesWritten++;
                nnzIndex++;
            }
        }
    }

    return std::make_tuple(referenceDenseData, colsStarts, rowIndices, nonZeroValues, numNonZeroValues);
}

#pragma warning(pop)

inline NDShape CreateShape(size_t numAxes, size_t maxDimSize)
{
    NDShape shape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
    {
        shape[i] = (rng() % maxDimSize) + 1;
    }

    return shape;
}

inline void OpenStream(std::fstream& stream, const std::wstring& filename, bool readonly)
{
    if (filename.empty())
       throw std::runtime_error("File: filename is empty");

    std::ios_base::openmode mode = std::ios_base::binary;
    mode = mode | (readonly ? std::ios_base::in : std::ios_base::out);

    #ifdef _MSC_VER
    stream.open(filename.c_str(), mode);
    #else
    stream.open(wtocharpath(filename.c_str()).c_str(), mode);
    #endif
    stream.exceptions(std::ios_base::badbit);  
}

inline void PrintTrainingProgress(const TrainerPtr trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
{
    if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer->PreviousMinibatchSampleCount() != 0)
    {
        double trainLossValue = trainer->PreviousMinibatchLossAverage();
        double evaluationValue = trainer->PreviousMinibatchEvaluationAverage();
        printf("Minibatch %d: CrossEntropy loss = %.8g, Evaluation criterion = %.8g\n", (int)minibatchIdx, trainLossValue, evaluationValue);
    }
}
inline std::vector<size_t> GetStrides(const NDShape& shape)
{
    if (shape.Rank() == 0)
        return std::vector<size_t>();

    std::vector<size_t> strides(shape.Rank() - 1);
    size_t totalSize = 1;
    for (size_t i = 0; i < shape.Rank() - 1; ++i)
    {
        totalSize *= shape[i];
        strides[i] = totalSize;
    }

    return strides;
}

inline NDShape UnflattenedShape(size_t flatennedIdx, const std::vector<size_t>& strides)
{
    NDShape unflattenedShape(strides.size() + 1);
    size_t remainder = flatennedIdx;
    for (int i = (int)strides.size() - 1; i >= 0; --i)
    {
        unflattenedShape[i + 1] = remainder / strides[i];
        remainder = remainder % strides[i];
    }
    unflattenedShape[0] = remainder;

    return unflattenedShape;
}

inline size_t FlattenedIndex(const NDShape& shape, const std::vector<size_t>& strides)
{
    if (shape.Rank() == 0)
        return 0;

    size_t flattenedIdx = shape[0];
    for (int i = 0; i < strides.size(); ++i)
        flattenedIdx += shape[i + 1] * strides[i];

    return flattenedIdx;
};

inline FunctionPtr Embedding(const Variable& input, size_t embeddingDim, const DeviceDescriptor& device)
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];
    auto embeddingParameters = Parameter({ embeddingDim, inputDim }, DataType::Float, GlorotUniformInitializer(), device);
    return Times(embeddingParameters, input);
}

inline FunctionPtr LSTMSequenceClassifierNet(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim, const DeviceDescriptor& device, const std::wstring& outputName, unsigned long seed = 1)
{
    auto embeddingFunction = Embedding(input, embeddingDim, device);
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
    auto thoughtVectorFunction = Sequence::Last(LSTMFunction);

    return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName, seed);
}

inline bool AreEqual(const NDArrayViewPtr& view1, const NDArrayViewPtr& view2)
{
    return Internal::AreEqual(*view1, *view2);
}

inline bool AreEqual(const Variable& var1, const Variable& var2)
{

    if (!Internal::AreEquivalent(var1, var2))
    {
        return false;
    }

    if (!(var1.IsParameter() || var1.IsConstant()))
    {
        return true;
    }

    NDArrayViewPtr ptr1, ptr2;
       
    if (var1.IsParameter()) 
    {
        ptr1 = reinterpret_cast<const Parameter&>(var1).Value();
        ptr2 = reinterpret_cast<const Parameter&>(var2).Value();
    }


    if (var1.IsConstant()) 
    {
        ptr1 = reinterpret_cast<const Constant&>(var1).Value();
        ptr2 = reinterpret_cast<const Constant&>(var2).Value();
    }

    if (!Internal::AreEqual(*ptr1, *ptr2, relativeTolerance, absoluteTolerance))
    {
        return false;
    }

    return true;
}

inline bool AreEqual(const FunctionPtr& f1, const FunctionPtr& f2)
{
    if (f1 == f2)
    { 
        return true;
    }

    if (!Internal::AreEquivalent(f1, f2))
    {
        return false;
    }

    auto inputs1 = f1->Inputs();
    auto inputs2 = f2->Inputs();

    if (inputs1.size() != inputs2.size())
    {
        return false;
    }

    for (int i = 0; i < inputs1.size(); i++)
    {
        if (!AreEqual(inputs1[i], inputs2[i]))
        { 
            return false;
        }
    }

    return true;
}

inline void CompareFunctions(const FunctionPtr& first, const FunctionPtr& second, ParameterCloningMethod parameterCloningMethod,
    const std::unordered_map<Variable, Variable>& replacements, std::unordered_set<FunctionPtr>& visitedFunctions)
{
    // TODO: try to refactor this some more, using AreEqual functions above.
    if (first->Name() != second->Name())
        throw std::runtime_error("CompareFunctions: Both functions' names should match");

    if (first->Attributes() != second->Attributes())
        throw std::runtime_error("CompareFunctions: Both functions' attributes should match");

    auto firstPrimitive = first->RootFunction();
    auto secondPrimitive = second->RootFunction();

    if (firstPrimitive->Name() != secondPrimitive->Name())
        throw std::runtime_error("CompareFunctions: Both functions' names should match");

    // All the outputs must be equivalent
    if (firstPrimitive->Outputs().size() != secondPrimitive->Outputs().size())
        throw std::runtime_error("CompareFunctions: Both functions' should have same number of outputs");

    visitedFunctions.insert(firstPrimitive);

    for (size_t i = 0; i < firstPrimitive->Outputs().size(); ++i)
    {
        auto firstFunctionOutput = firstPrimitive->Outputs()[i];
        auto secondFunctionOutput = secondPrimitive->Outputs()[i];

        if (!AreEqual(firstFunctionOutput, secondFunctionOutput))
        {
            throw std::runtime_error("CompareFunctions: Both functions' outputs should match");
        }
    }

    // All of the inputs must be identical
    if (firstPrimitive->Inputs().size() != secondPrimitive->Inputs().size())
        throw std::runtime_error("CompareFunctions: Both functions' should have same number of inputs");

    for (size_t i = 0; i < firstPrimitive->Inputs().size(); ++i)
    {
        auto firstFunctionInput = firstPrimitive->Inputs()[i];
        auto secondFunctionInput = secondPrimitive->Inputs()[i];

        if (replacements.find(firstFunctionInput) != replacements.end())
        {
            if (replacements.at(firstFunctionInput) != secondFunctionInput)
                throw std::runtime_error("CompareFunctions: The 2nd function does not have the expected replacement");
        }
       else
        {
            if (!Internal::AreEquivalent(firstFunctionInput, secondFunctionInput, true))
            {
                throw std::runtime_error("CompareFunctions: The leaves of the functions are not equivalent");
            }

            if ((firstFunctionInput.Kind() != VariableKind::Parameter) && ((firstFunctionInput.Kind() != secondFunctionInput.Kind()) || (firstFunctionInput.NeedsGradient() != secondFunctionInput.NeedsGradient())))
                throw std::runtime_error("CompareFunctions: The leaves of the functions are not equivalent");

            switch (firstFunctionInput.Kind())
            {
            case VariableKind::Parameter:
            case VariableKind::Constant:
                if ((parameterCloningMethod == ParameterCloningMethod::Share) && (firstFunctionInput != secondFunctionInput))
                    throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");

                NDArrayViewPtr firstFunctionInputValue = firstFunctionInput.IsConstant() ? Constant(firstFunctionInput).Value() : Parameter(firstFunctionInput).Value();
                NDArrayViewPtr secondFunctionInputValue = secondFunctionInput.IsConstant() ? Constant(secondFunctionInput).Value() : Parameter(secondFunctionInput).Value();
                if ((parameterCloningMethod == ParameterCloningMethod::Clone) &&
                    ((firstFunctionInput == secondFunctionInput) || (!Internal::AreEqual(*firstFunctionInputValue, *secondFunctionInputValue))))
                {
                    throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");
                }

                if ((parameterCloningMethod == ParameterCloningMethod::Freeze) &&
                    ((firstFunctionInput == secondFunctionInput) || !secondFunctionInput.IsConstant() || (!Internal::AreEqual(*firstFunctionInputValue, *secondFunctionInputValue))))
                {
                    throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");
                }

                break;
            }
        }
    }
}

MinibatchSourceConfig GetHTKMinibatchSourceConfig(size_t featureDim, size_t numOutputClasses, size_t epochSize = MinibatchSource::InfinitelyRepeat, bool randomize = true);
