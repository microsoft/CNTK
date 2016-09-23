#pragma once
#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>
#include <fstream>
#include <random>

static const double relativeTolerance = 0.001f;
static const double absoluteTolerance = 0.000001f;

template <typename ElementType>
inline void FloatingPointVectorCompare(const std::vector<ElementType>& first, const std::vector<ElementType>& second, const char* message)
{
    for (size_t i = 0; i < first.size(); ++i)
    {
        ElementType leftVal = first[i];
        ElementType rightVal = second[i];
        ElementType allowedTolerance = (std::max<ElementType>)((ElementType)absoluteTolerance, ((ElementType)relativeTolerance) * leftVal);
        if (std::abs(leftVal - rightVal) > allowedTolerance)
            throw std::runtime_error(message);
    }
}

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
inline void SaveAndReloadModel(CNTK::FunctionPtr& functionPtr, const std::vector<CNTK::Variable*>& variables, const CNTK::DeviceDescriptor& device)
{
    static std::wstring s_tempModelPath = L"feedForward.net";

    if ((_wunlink(s_tempModelPath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temp model file 'feedForward.net'");

    std::unordered_map<std::wstring, CNTK::Variable*> inputVarUids;
    std::unordered_map<std::wstring, CNTK::Variable*> outputVarNames;

    for (auto varPtr : variables)
    {
        auto retVal = varPtr->IsOutput() ? outputVarNames.insert({ varPtr->Owner()->Name(), varPtr }) : inputVarUids.insert({ varPtr->Uid(), varPtr });
        if (!retVal.second)
            std::runtime_error("SaveAndReloadModel: Multiple variables having same name cannot be restored after save and reload");
    }

    CNTK::SaveAsLegacyModel(functionPtr, s_tempModelPath);
    functionPtr = CNTK::LoadLegacyModel(functionPtr->Outputs()[0].GetDataType(), s_tempModelPath, device);

    if (_wunlink(s_tempModelPath.c_str()) != 0)
         std::runtime_error("Error deleting temp model file 'feedForward.net'");

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

    auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
    auto timesFunction = CNTK::Times(timesParam, input);

    auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim }, -0.05, 0.05, 1, device));
    return CNTK::Plus(plusParam, timesFunction, outputName);
}

inline CNTK::FunctionPtr FullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device, const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity)
{
    return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device));
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
    size_t inputDim = input.Shape()[0];
    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto createBiasParam = [device](size_t dim) {
        return CNTK::Parameter({ dim }, (ElementType)0.0, device);
    };

    unsigned long seed = 1;
    auto createProjectionParam = [device, &seed](size_t outputDim, size_t inputDim) {
        return CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ outputDim, inputDim }, -0.5, 0.5, seed++, device));
    };

    auto createDiagWeightParam = [device, &seed](size_t dim) {
        return CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ dim }, -0.5, 0.5, seed++, device));
    };

    auto stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
    auto stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

    auto projectInput = [input, cellDim, inputDim, createBiasParam, createProjectionParam]() {
        return createBiasParam(cellDim) + CNTK::Times(createProjectionParam(cellDim, inputDim), input);
    };

    // Input gate
    auto it = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim, outputDim), stabilizedPrevOutput) + CNTK::ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bit = CNTK::ElementTimes(it, CNTK::Tanh(projectInput() + CNTK::Times(createProjectionParam(cellDim, outputDim), stabilizedPrevOutput)));

    // Forget-me-not gate
    auto ft = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim, outputDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bft = CNTK::ElementTimes(ft, prevCellState);

    auto ct = bft + bit;

    // Output gate
    auto ot = CNTK::Sigmoid(projectInput() + CNTK::Times(createProjectionParam(cellDim, outputDim), stabilizedPrevOutput) + CNTK::ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
    auto ht = CNTK::ElementTimes(ot, CNTK::Tanh(ct));

    auto c = ct;
    auto h = (outputDim != cellDim) ? CNTK::Times(createProjectionParam(outputDim, cellDim), Stabilize<ElementType>(ht, device)) : ht;

    return{ h, c };
}

template <typename ElementType>
std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPComponentWithSelfStabilization(CNTK::Variable input,
                                                                                    const CNTK::NDShape& outputDim,
                                                                                    const CNTK::NDShape& cellDim,
                                                                                    const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookH,
                                                                                    const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookC,
                                                                                    const CNTK::DeviceDescriptor& device)
{
    auto dh = CNTK::PlaceholderVariable(outputDim, input.DynamicAxes());
    auto dc = CNTK::PlaceholderVariable(cellDim, input.DynamicAxes());

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
    for (size_t i = 0; i < numSequences; ++i)
    {
        sequenceLengths[i] = (rand() % maxAllowedSequenceLength) + 1;
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
        std::runtime_error("File: filename is empty");

    std::ios_base::openmode mode = std::ios_base::binary;
    mode = mode | (readonly ? std::ios_base::in : std::ios_base::out);

    #ifdef _MSC_VER
    stream.open(filename.c_str(), mode);
    #else
    stream.open(wtocharpath(filename.c_str()).c_str(), mode);
    #endif
    stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);  
}

inline void PrintTrainingProgress(const CNTK::Trainer& trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
{
    if ((minibatchIdx % outputFrequencyInMinibatches) == 0)
    {
        double trainLossValue = trainer.PreviousMinibatchLossAverage();
        double evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
        printf("Minibatch %d: CrossEntropy loss = %.8g, Evaluation criterion = %.8g\n", (int)minibatchIdx, trainLossValue, evaluationValue);
    }
}

inline std::vector<size_t> GetStrides(const CNTK::NDShape& shape)
{
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
    size_t flattenedIdx = shape[0];
    for (int i = 0; i < strides.size(); ++i)
        flattenedIdx += shape[i + 1] * strides[i];

    return flattenedIdx;
};
