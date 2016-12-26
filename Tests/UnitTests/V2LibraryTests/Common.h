#pragma once
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

#ifdef _MSC_VER
// In case of asserts in debug mode, print the message into stderr and throw exception
int HandleDebugAssert(int /* reportType */,
                      char *message, // fully assembled debug user message
                      int *returnValue); // returnValue - retVal value of zero continues execution
#endif

#pragma warning(push)
#pragma warning(disable : 4996)
#ifndef _MSC_VER // TODO: what is the correct trigger for gcc?
__declspec_noreturn inline void ReportFailure(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

__declspec_noreturn inline void ReportFailure(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    char buffer[1024] = { 0 }; // Note: pre-VS2015 vsnprintf() is not standards-compliant and may not add a terminator
    vsnprintf(buffer, _countof(buffer) - 1, format, args); // -1 because pre-VS2015 vsnprintf() does not always write a 0-terminator
    if (strlen(buffer)/*written*/ >= (int)_countof(buffer) - 2)
        sprintf(buffer + _countof(buffer) - 4, "...");

    throw std::runtime_error(buffer);
}
#pragma warning(pop)

static const double relativeTolerance = 0.001f;
static const double absoluteTolerance = 0.000001f;

bool IsGPUAvailable();

template <typename ElementType>
inline void FloatingPointCompare(ElementType actual, ElementType expected, const char* message)
{
    ElementType allowedTolerance = (std::max<ElementType>)((ElementType)absoluteTolerance, std::abs(((ElementType)relativeTolerance) * actual));
    if (std::abs(actual - expected) > allowedTolerance)
        ReportFailure((message + std::string("; Expected=%g, Actual=%g")).c_str(), expected, actual);
}

template <typename ElementType>
inline void FloatingPointVectorCompare(const std::vector<ElementType>& actual, const std::vector<ElementType>& expected, const char* message)
{
    if (actual.size() != expected.size())
        ReportFailure((message + std::string("; actual data vector size (%d) and expected data vector size (%d) are not equal")).c_str(), (int)actual.size(), (int)expected.size());

    for (size_t i = 0; i < actual.size(); ++i)
        FloatingPointCompare(actual[i], expected[i], message);
}

inline void VerifyException(const std::function<void()>& functionToTest, std::string errorMessage) {
    bool error = false;
    try
    {
        functionToTest();
    }
    catch (const std::exception&)
    {
        error = true;
    }

    if (!error)
        throw std::runtime_error(errorMessage);
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
inline void SaveAndReloadModel(CNTK::FunctionPtr& functionPtr, const std::vector<CNTK::Variable*>& variables, const CNTK::DeviceDescriptor& device, size_t rank = 0)
{
    const std::wstring tempModelPath = L"feedForward.net" + std::to_wstring((int)rank);

    if ((_wunlink(tempModelPath.c_str()) != 0) && (errno != ENOENT))
       throw std::runtime_error("Error deleting temp model file 'feedForward.net'");

    std::unordered_map<std::wstring, CNTK::Variable*> inputVarUids;
    std::unordered_map<std::wstring, CNTK::Variable*> outputVarNames;

    for (auto varPtr : variables)
    {
        auto retVal = varPtr->IsOutput() ? outputVarNames.insert({ varPtr->Owner()->Name(), varPtr }) : inputVarUids.insert({ varPtr->Uid(), varPtr });
        if (!retVal.second)
           throw std::runtime_error("SaveAndReloadModel: Multiple variables having same name cannot be restored after save and reload");
    }

    functionPtr->SaveModel(tempModelPath);
    functionPtr = CNTK::Function::LoadModel(tempModelPath, device);

    if (_wunlink(tempModelPath.c_str()) != 0)
         throw std::runtime_error("Error deleting temp model file 'feedForward.net'");

    auto inputs = functionPtr->Inputs();
    for (auto inputVarInfo : inputVarUids)
    {
        auto newInputVar = *(std::find_if(inputs.begin(), inputs.end(), [inputVarInfo](const CNTK::Variable& var) {
            return (var.Uid() == inputVarInfo.first);
        }));

        *(inputVarInfo.second) = newInputVar;
    }

    auto outputs = functionPtr->Outputs();
    for (auto outputVarInfo : outputVarNames)
    {
        auto newOutputVar = *(std::find_if(outputs.begin(), outputs.end(), [outputVarInfo](const CNTK::Variable& var) {
            return (var.Owner()->Name() == outputVarInfo.first);
        }));

        *(outputVarInfo.second) = newOutputVar;
    }
}

inline CNTK::FunctionPtr FullyConnectedLinearLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device, const std::wstring& outputName = L"")
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = CNTK::Parameter({ outputDim, inputDim }, CNTK::DataType::Float, CNTK::GlorotUniformInitializer(CNTK::DefaultParamInitScale, CNTK::SentinelValueForInferParamInitRank, CNTK::SentinelValueForInferParamInitRank, 1), device, L"timesParam");
    auto timesFunction = CNTK::Times(timesParam, input, L"times");

    auto plusParam = CNTK::Parameter({ outputDim }, 0.0f, device, L"plusParam");
    return CNTK::Plus(plusParam, timesFunction, outputName);
}

inline CNTK::FunctionPtr FullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device, const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity, const std::wstring& outputName = L"")
{
    return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device, outputName));
}

inline CNTK::FunctionPtr FullyConnectedFeedForwardClassifierNet(CNTK::Variable input,
                                                   size_t numOutputClasses,
                                                   size_t hiddenLayerDim,
                                                   size_t numHiddenLayers,
                                                   const CNTK::DeviceDescriptor& device,
                                                   const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity,
                                                   const std::wstring& outputName)
{
    assert(numHiddenLayers >= 1);
    auto classifierRoot = FullyConnectedDNNLayer(input, hiddenLayerDim, device, nonLinearity);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierRoot = FullyConnectedDNNLayer(classifierRoot, hiddenLayerDim, device, nonLinearity);

    auto outputTimesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.5, 0.5, 1, device));
    return Times(outputTimesParam, classifierRoot, 1, outputName);
}

template <typename ElementType>
inline CNTK::FunctionPtr Stabilize(const CNTK::Variable& x, const CNTK::DeviceDescriptor& device)
{
    ElementType scalarConstant = 4.0f;
    auto f = CNTK::Constant::Scalar(scalarConstant);
    auto fInv = CNTK::Constant::Scalar(f.GetDataType(), 1.0 / scalarConstant);

    auto beta = CNTK::ElementTimes(fInv, CNTK::Log(CNTK::Constant::Scalar(f.GetDataType(), 1.0) + CNTK::Exp(CNTK::ElementTimes(f, CNTK::Parameter({}, f.GetDataType(), 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
    return CNTK::ElementTimes(beta, x);
}

template <typename ElementType>
std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPCellWithSelfStabilization(CNTK::Variable input, CNTK::Variable prevOutput, CNTK::Variable prevCellState, const CNTK::DeviceDescriptor& device)
{
    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto createBiasParam = [device](size_t dim) {
        return CNTK::Parameter({ dim }, (ElementType)0.0, device);
    };

    unsigned long seed2 = 1;
    auto createProjectionParam = [device, &seed2](size_t outputDim) {
        return CNTK::Parameter({ outputDim, CNTK::NDShape::InferredDimension }, CNTK::AsDataType<ElementType>(), CNTK::GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto createDiagWeightParam = [device, &seed2](size_t dim) {
        return CNTK::Parameter({ dim }, CNTK::AsDataType<ElementType>(), CNTK::GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
    auto stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

    auto projectInput = [input, cellDim, createBiasParam, createProjectionParam]() {
        return createBiasParam(cellDim) + CNTK::Times(createProjectionParam(cellDim), input);
    };

    // Input gate
    auto it = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput) + CNTK::ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bit = CNTK::ElementTimes(it, CNTK::Tanh(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput)));

    // Forget-me-not gate
    auto ft = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput) + CNTK::ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bft = CNTK::ElementTimes(ft, prevCellState);

    auto ct = bft + bit;

    // Output gate
    auto ot = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim), stabilizedPrevOutput) + CNTK::ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
    auto ht = CNTK::ElementTimes(ot, CNTK::Tanh(ct));

    auto c = ct;
    auto h = (outputDim != cellDim) ? CNTK::Times(createProjectionParam(outputDim), Stabilize<ElementType>(ht, device)) : ht;

    return{ h, c };
}

template <typename ElementType>
std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPComponentWithSelfStabilization(CNTK::Variable input,
                                                                                    const CNTK::NDShape& outputShape,
                                                                                    const CNTK::NDShape& cellShape,
                                                                                    const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookH,
                                                                                    const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookC,
                                                                                    const CNTK::DeviceDescriptor& device)
{
    auto dh = CNTK::PlaceholderVariable(outputShape, input.DynamicAxes());
    auto dc = CNTK::PlaceholderVariable(cellShape, input.DynamicAxes());

    auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

    auto actualDh = recurrenceHookH(LSTMCell.first);
    auto actualDc = recurrenceHookC(LSTMCell.second);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    LSTMCell.first->ReplacePlaceholders({ { dh, actualDh }, { dc, actualDc } });

    return { LSTMCell.first, LSTMCell.second };
}

// This is currently unused
inline CNTK::FunctionPtr SimpleRecurrentLayer(const  CNTK::Variable& input, const  CNTK::NDShape& outputDim, const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHook, const CNTK::DeviceDescriptor& device)
{
    auto dh = CNTK::PlaceholderVariable(outputDim, input.DynamicAxes());

    unsigned long seed = 1;
    auto createProjectionParam = [device, &seed](size_t outputDim, size_t inputDim) {
        return CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.5, 0.5, seed++, device));
    };

    auto hProjWeights = createProjectionParam(outputDim[0], outputDim[0]);
    auto inputProjWeights = createProjectionParam(outputDim[0], input.Shape()[0]);

    auto output = Times(hProjWeights, recurrenceHook(dh)) + Times(inputProjWeights, input);
    return output->ReplacePlaceholders({ { dh, output } });
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
inline std::vector<std::vector<ElementType>> GenerateSequences(const std::vector<size_t>& sequenceLengths, const CNTK::NDShape& sampleShape)
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
inline CNTK::ValuePtr GenerateSequences(const std::vector<size_t>& sequenceLengths, const CNTK::NDShape& sampleShape, const CNTK::DeviceDescriptor& device, bool oneHot)
{
    if (!oneHot)
    {
        std::vector<std::vector<ElementType>> sequences = GenerateSequences<ElementType>(sequenceLengths, sampleShape);
        return CNTK::Value::Create(sampleShape, sequences, device, true);
    }
    else
    {
        if (sampleShape.Rank() != 1)
            throw std::runtime_error("GenerateSequences can generate one hot sequences only for 1D sample shapes");

        size_t vocabularySize = sampleShape[0];
        std::vector<std::vector<size_t>> oneHotSequences = GenerateOneHotSequences(sequenceLengths, vocabularySize);
        return CNTK::Value::Create<ElementType>(vocabularySize, oneHotSequences, device, true);
    }
}

template <typename ElementType>
inline std::pair<CNTK::NDArrayViewPtr, CNTK::NDArrayViewPtr> GenerateSparseSequence(size_t vocabSize, size_t sequenceLength, size_t maxNumberOfNonZeroValuesPerSparseInputSample)
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

    CNTK::NDShape inputDataShape = CNTK::NDShape({ vocabSize, sequenceLength });
    CNTK::NDArrayViewPtr inputValueData = CNTK::MakeSharedObject<CNTK::NDArrayView>(inputDataShape, inputData);
    CNTK::NDArrayViewPtr sparseData = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::AsDataType<ElementType>(), CNTK::StorageFormat::SparseCSC, inputDataShape, CNTK::DeviceDescriptor::CPUDevice());
    sparseData->CopyFrom(*inputValueData);
    return{ inputValueData->DeepClone(), sparseData };
}

#pragma warning(pop)

inline CNTK::NDShape CreateShape(size_t numAxes, size_t maxDimSize)
{
    CNTK::NDShape shape(numAxes);
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

inline void PrintTrainingProgress(const CNTK::Trainer& trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
{
    if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
    {
        double trainLossValue = trainer.PreviousMinibatchLossAverage();
        double evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
        printf("Minibatch %d: CrossEntropy loss = %.8g, Evaluation criterion = %.8g\n", (int)minibatchIdx, trainLossValue, evaluationValue);
    }
}
inline std::vector<size_t> GetStrides(const CNTK::NDShape& shape)
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

inline CNTK::NDShape UnflattenedShape(size_t flatennedIdx, const std::vector<size_t>& strides)
{
    CNTK::NDShape unflattenedShape(strides.size() + 1);
    size_t remainder = flatennedIdx;
    for (int i = (int)strides.size() - 1; i >= 0; --i)
    {
        unflattenedShape[i + 1] = remainder / strides[i];
        remainder = remainder % strides[i];
    }
    unflattenedShape[0] = remainder;

    return unflattenedShape;
}

inline size_t FlattenedIndex(const CNTK::NDShape& shape, const std::vector<size_t>& strides)
{
    if (shape.Rank() == 0)
        return 0;

    size_t flattenedIdx = shape[0];
    for (int i = 0; i < strides.size(); ++i)
        flattenedIdx += shape[i + 1] * strides[i];

    return flattenedIdx;
};

inline CNTK::FunctionPtr Embedding(const CNTK::Variable& input, size_t embeddingDim, const CNTK::DeviceDescriptor& device)
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];
    auto embeddingParameters = CNTK::Parameter({ embeddingDim, inputDim }, CNTK::DataType::Float, CNTK::GlorotUniformInitializer(), device);
    return Times(embeddingParameters, input);
}

inline CNTK::FunctionPtr LSTMSequenceClassiferNet(const CNTK::Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim, const CNTK::DeviceDescriptor& device, const std::wstring& outputName)
{
    auto embeddingFunction = Embedding(input, embeddingDim, device);
    auto pastValueRecurrenceHook = [](const CNTK::Variable& x) { return PastValue(x); };
    auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
    auto thoughtVectorFunction = CNTK::Sequence::Last(LSTMFunction);

    return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
}

inline bool AreEqual(const CNTK::NDArrayViewPtr& view1, const CNTK::NDArrayViewPtr& view2)
{
    return CNTK::Internal::AreEqual(*view1, *view2);
}

inline bool AreEqual(const CNTK::Variable& var1, const CNTK::Variable& var2)
{

    if (!CNTK::Internal::AreEquivalent(var1, var2))
    {
        return false;
    }

    if (!(var1.IsParameter() || var1.IsConstant()))
    {
        return true;
    }

    CNTK::NDArrayViewPtr ptr1, ptr2;
       
    if (var1.IsParameter()) 
    {
        ptr1 = reinterpret_cast<const CNTK::Parameter&>(var1).Value();
        ptr2 = reinterpret_cast<const CNTK::Parameter&>(var2).Value();
    }


    if (var1.IsConstant()) 
    {
        ptr1 = reinterpret_cast<const CNTK::Constant&>(var1).Value();
        ptr2 = reinterpret_cast<const CNTK::Constant&>(var2).Value();
    }

    if (!CNTK::Internal::AreEqual(*ptr1, *ptr2, relativeTolerance, absoluteTolerance))
    {
        return false;
    }

    return true;
}

inline bool AreEqual(const CNTK::FunctionPtr& f1, const CNTK::FunctionPtr& f2)
{
    if (f1 == f2)
    { 
        return true;
    }

    if (!CNTK::Internal::AreEquivalent(f1, f2))
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

using namespace CNTK;

inline void CompareFunctions(const FunctionPtr& first, const FunctionPtr& second, ParameterCloningMethod parameterCloningMethod, const std::unordered_map<Variable, Variable>& replacements, std::unordered_set<FunctionPtr>& visitedFunctions)
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
                    ((firstFunctionInput == secondFunctionInput) || (!CNTK::Internal::AreEqual(*firstFunctionInputValue, *secondFunctionInputValue))))
                {
                    throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");
                }

                if ((parameterCloningMethod == ParameterCloningMethod::Freeze) &&
                    ((firstFunctionInput == secondFunctionInput) || !secondFunctionInput.IsConstant() || (!CNTK::Internal::AreEqual(*firstFunctionInputValue, *secondFunctionInputValue))))
                {
                    throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");
                }

                break;
            }
        }
    }
}
