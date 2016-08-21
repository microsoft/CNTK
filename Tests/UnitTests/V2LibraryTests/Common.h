#pragma once

#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>

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
#endif

template <typename ElementType>
inline void SaveAndReloadModel(CNTK::FunctionPtr& functionPtr, const std::vector<CNTK::Variable*>& variables, const CNTK::DeviceDescriptor& device)
{
    static std::wstring s_tempModelPath = L"feedForward.net";

    if ((_wunlink(s_tempModelPath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temp model file 'feedForward.net'");

    std::unordered_map<std::wstring, CNTK::Variable*> inputVarNames;
    std::unordered_map<std::wstring, CNTK::Variable*> outputVarNames;

    for (auto varPtr : variables)
    {
        auto retVal = varPtr->IsOutput() ? outputVarNames.insert({ varPtr->Owner()->Name(), varPtr }) : inputVarNames.insert({ varPtr->Name(), varPtr });
        if (!retVal.second)
            std::runtime_error("SaveAndReloadModel: Multiple variables having same name cannot be restored after save and reload");
    }

    CNTK::SaveAsLegacyModel<ElementType>(functionPtr, s_tempModelPath);
    functionPtr = CNTK::LoadLegacyModel<ElementType>(s_tempModelPath, device);

    if (_wunlink(s_tempModelPath.c_str()) != 0)
         std::runtime_error("Error deleting temp model file 'feedForward.net'");

    auto inputs = functionPtr->Inputs();
    for (auto inputVarInfo : inputVarNames)
    {
        auto newInputVar = *(std::find_if(inputs.begin(), inputs.end(), [inputVarInfo](const CNTK::Variable& var) {
            return (var.Name() == inputVarInfo.first);
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
    assert(input.Shape().NumAxes() == 1);
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
std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPCellWithSelfStabilization(CNTK::Variable input, CNTK::Variable prevOutput, CNTK::Variable prevCellState, const CNTK::DeviceDescriptor& device)
{
    assert(input.Shape().NumAxes() == 1);
    size_t inputDim = input.Shape()[0];

    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    unsigned long seed = 1;

    auto Wxo = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));
    auto Wxi = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));
    auto Wxf = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));
    auto Wxc = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, inputDim }, -0.5, 0.5, seed++, device));

    auto Bo = CNTK::Parameter({ cellDim }, (ElementType)0.0, device);
    auto Bc = CNTK::Parameter({ cellDim }, (ElementType)0.0, device);
    auto Bi = CNTK::Parameter({ cellDim }, (ElementType)0.0, device);
    auto Bf = CNTK::Parameter({ cellDim }, (ElementType)0.0, device);

    auto Whi = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));
    auto Wci = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim }, -0.5, 0.5, seed++, device));

    auto Whf = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));
    auto Wcf = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim }, -0.5, 0.5, seed++, device));

    auto Who = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));
    auto Wco = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim }, -0.5, 0.5, seed++, device));

    auto Whc = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ cellDim, outputDim }, -0.5, 0.5, seed++, device));

    auto Wmr = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<ElementType>({ outputDim, cellDim }, -0.5, 0.5, seed++, device));

    // Stabilization by routing input through an extra scalar parameter
    auto sWxo = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWxi = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWxf = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWxc = CNTK::Parameter({}, (ElementType)0.0, device);

    auto sWhi = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWci = CNTK::Parameter({}, (ElementType)0.0, device);

    auto sWhf = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWcf = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWho = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWco = CNTK::Parameter({}, (ElementType)0.0, device);
    auto sWhc = CNTK::Parameter({}, (ElementType)0.0, device);

    auto sWmr = CNTK::Parameter({}, (ElementType)0.0, device);

    auto expsWxo = CNTK::Exp(sWxo);
    auto expsWxi = CNTK::Exp(sWxi);
    auto expsWxf = CNTK::Exp(sWxf);
    auto expsWxc = CNTK::Exp(sWxc);

    auto expsWhi = CNTK::Exp(sWhi);
    auto expsWci = CNTK::Exp(sWci);

    auto expsWhf = CNTK::Exp(sWhf);
    auto expsWcf = CNTK::Exp(sWcf);
    auto expsWho = CNTK::Exp(sWho);
    auto expsWco = CNTK::Exp(sWco);
    auto expsWhc = CNTK::Exp(sWhc);

    auto expsWmr = CNTK::Exp(sWmr);

    auto Wxix = CNTK::Times(Wxi, CNTK::ElementTimes(expsWxi, input));
    auto Whidh = CNTK::Times(Whi, CNTK::ElementTimes(expsWhi, prevOutput));
    auto Wcidc = CNTK::ElementTimes(Wci, CNTK::ElementTimes(expsWci, prevCellState));

    auto it = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxix, Bi), Whidh), Wcidc));

    auto Wxcx = CNTK::Times(Wxc, CNTK::ElementTimes(expsWxc, input));
    auto Whcdh = CNTK::Times(Whc, CNTK::ElementTimes(expsWhc, prevOutput));
    auto bit = CNTK::ElementTimes(it, CNTK::Tanh(CNTK::Plus(Wxcx, CNTK::Plus(Whcdh, Bc))));

    auto Wxfx = CNTK::Times(Wxf, CNTK::ElementTimes(expsWxf, input));
    auto Whfdh = CNTK::Times(Whf, CNTK::ElementTimes(expsWhf, prevOutput));
    auto Wcfdc = CNTK::ElementTimes(Wcf, CNTK::ElementTimes(expsWcf, prevCellState));

    auto ft = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxfx, Bf), Whfdh), Wcfdc));

    auto bft = CNTK::ElementTimes(ft, prevCellState);

    auto ct = CNTK::Plus(bft, bit);

    auto Wxox = CNTK::Times(Wxo, CNTK::ElementTimes(expsWxo, input));
    auto Whodh = CNTK::Times(Who, CNTK::ElementTimes(expsWho, prevOutput));
    auto Wcoct = CNTK::ElementTimes(Wco, CNTK::ElementTimes(expsWco, ct));

    auto ot = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxox, Bo), Whodh), Wcoct));

    auto mt = CNTK::ElementTimes(ot, Tanh(ct));

    return{ CNTK::Times(Wmr, CNTK::ElementTimes(expsWmr, mt)), ct };
}

template <typename ElementType>
CNTK::FunctionPtr LSTMPComponentWithSelfStabilization(CNTK::Variable input, size_t outputDim, size_t cellDim, const CNTK::DeviceDescriptor& device)
{
    auto dh = CNTK::Placeholder({ outputDim });
    auto dc = CNTK::Placeholder({ cellDim });

    auto LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

    auto actualDh = CNTK::PastValue(CNTK::Constant({}, (ElementType)0.0, device), LSTMCell.first, 1);
    auto actualDc = CNTK::PastValue(CNTK::Constant({}, (ElementType)0.0, device), LSTMCell.second, 1);

    // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
    return LSTMCell.first->ReplacePlaceholders({ { dh, actualDh }, { dc, actualDc } });
}

inline float PrevMinibatchTrainingLossValue(const CNTK::Trainer& trainer)
{
    float trainLossValue = 0.0;
    auto prevMBTrainingLossValue = trainer.PreviousMinibatchTrainingLossValue()->Data();
    CNTK::NDArrayView cpuTrainLossValue(prevMBTrainingLossValue->Shape(), &trainLossValue, 1, CNTK::DeviceDescriptor::CPUDevice());
    cpuTrainLossValue.CopyFrom(*prevMBTrainingLossValue);

    return trainLossValue;
}

inline CNTK::MinibatchSourcePtr CreateTextMinibatchSource(const std::wstring& filePath,
                                                          size_t featureDim,
                                                          size_t labelDim,
                                                          size_t epochSize,
                                                          bool isFeatureSparse = false,
                                                          bool isLabelSparse = false,
                                                          const std::wstring& featureAlias = L"",
                                                          const std::wstring& labelAlias = L"")
{
    CNTK::Dictionary featuresStreamConfig;
    featuresStreamConfig[L"dim"] = featureDim;
    featuresStreamConfig[L"format"] = isFeatureSparse ? L"sparse" : L"dense";
    if (!featureAlias.empty())
        featuresStreamConfig[L"alias"] = featureAlias;

    CNTK::Dictionary labelsStreamConfig;
    labelsStreamConfig[L"dim"] = labelDim;
    labelsStreamConfig[L"format"] = isLabelSparse ? L"sparse" : L"dense";
    if (!labelAlias.empty())
        labelsStreamConfig[L"alias"] = labelAlias;

    CNTK::Dictionary inputStreamsConfig;
    inputStreamsConfig[L"features"] = featuresStreamConfig;
    inputStreamsConfig[L"labels"] = labelsStreamConfig;

    CNTK::Dictionary deserializerConfiguration;
    deserializerConfiguration[L"type"] = L"CNTKTextFormatDeserializer";
    deserializerConfiguration[L"module"] = L"CNTKTextFormatReader";
    deserializerConfiguration[L"file"] = filePath;
    deserializerConfiguration[L"input"] = inputStreamsConfig;

    CNTK::Dictionary minibatchSourceConfiguration;
    minibatchSourceConfiguration[L"epochSize"] = epochSize;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<CNTK::DictionaryValue>({ deserializerConfiguration });

    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
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
inline std::vector<std::vector<ElementType>> GenerateSequences(const std::vector<size_t>& sequenceLengths, size_t dim)
{
    size_t numSequences = sequenceLengths.size();
    std::vector<std::vector<ElementType>> sequences;
    for (size_t i = 0; i < numSequences; ++i)
    {
        std::vector<ElementType> currentSequence(dim * sequenceLengths[i]);
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
inline CNTK::ValuePtr GenerateSequences(const std::vector<size_t>& sequenceLengths, size_t dim, const CNTK::DeviceDescriptor& device, bool oneHot)
{
    if (!oneHot)
    {
        std::vector<std::vector<ElementType>> sequences = GenerateSequences<ElementType>(sequenceLengths, dim);
        return CNTK::Value::Create({ dim }, sequences, device, true);
    }
    else
    {
        std::vector<std::vector<size_t>> oneHotSequences = GenerateOneHotSequences(sequenceLengths, dim);
        return CNTK::Value::Create<ElementType>({ dim }, oneHotSequences, device, true);
    }
}

#pragma warning(pop)
