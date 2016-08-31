#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

inline CNTK::MinibatchSourcePtr CreateSeq2SeqMinibatchSource(const std::wstring& filePath, size_t inputVocabSize, size_t labelsVocabSize)
{
    CNTK::Dictionary inputStreamConfig;
    inputStreamConfig[L"dim"] = inputVocabSize;
    inputStreamConfig[L"format"] = L"sparse";
    inputStreamConfig[L"alias"] = L"S0";

    CNTK::Dictionary labelsStreamConfig;
    labelsStreamConfig[L"dim"] = labelsVocabSize;
    labelsStreamConfig[L"format"] = L"sparse";
    labelsStreamConfig[L"alias"] = L"S1";

    CNTK::Dictionary inputStreamsConfig;
    inputStreamsConfig[L"rawInput"] = inputStreamConfig;
    inputStreamsConfig[L"rawLabels"] = labelsStreamConfig;

    CNTK::Dictionary deserializerConfiguration;
    deserializerConfiguration[L"type"] = L"CNTKTextFormatDeserializer";
    deserializerConfiguration[L"file"] = filePath;
    deserializerConfiguration[L"input"] = inputStreamsConfig;
    deserializerConfiguration[L"skipSequenceIds"] = L"false";
    deserializerConfiguration[L"maxErrors"] = (size_t)100;
    deserializerConfiguration[L"traceLevel"] = (size_t)1;
    deserializerConfiguration[L"chunkSizeInBytes"] = (size_t)30000000;

    CNTK::Dictionary minibatchSourceConfiguration;
    minibatchSourceConfiguration[L"epochSize"] = (size_t)2000;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<CNTK::DictionaryValue>({ deserializerConfiguration });

    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
}

void TrainSequenceToSequenceTranslator(const DeviceDescriptor& device, bool useSparseInputs, bool testSaveAndReLoad)
{
    using namespace std::placeholders;

    const size_t inputVocabDim = 69;
    const size_t labelVocabDim = 69;

    const size_t hiddenDim = 512;
    const size_t numLayers = 2;

    const size_t embeddingDim = 300;
    const size_t inputEmbeddingDim = std::min(inputVocabDim, embeddingDim);
    const size_t labelEmbeddingDim = std::min(labelVocabDim, embeddingDim);

    /* Inputs */
    std::vector<Axis> inputDynamicAxes = { Axis(L"inputAxis"), Axis::DefaultBatchAxis() };
    auto rawInput = Variable({ inputVocabDim }, useSparseInputs /*isSparse*/, DataType::Float, L"rawInput", inputDynamicAxes);

    std::vector<Axis> labelDynamicAxes = { Axis(L"labelAxis"), Axis::DefaultBatchAxis() };
    auto rawLabels = Variable({ labelVocabDim }, useSparseInputs /*isSparse*/, DataType::Float, L"rawLabels", labelDynamicAxes);

    FunctionPtr inputSequence = rawInput;

    // Drop the sentence start token from the label, for decoder training
    auto labelSequence = Slice(rawLabels, labelDynamicAxes[0], 1, 0);
    auto labelSentenceStart = Sequence::First(rawLabels);

    auto isFirstLabel = Sequence::IsFirst(labelSequence);

    bool forceEmbedding = useSparseInputs;

    /* Embeddings */
    auto inputEmbeddingWeights = Parameter(NDArrayView::RandomUniform<float>({ inputEmbeddingDim, inputVocabDim }, -0.05, 0.05, 1, device));
    auto labelEmbeddingWeights = Parameter(NDArrayView::RandomUniform<float>({ labelEmbeddingDim, labelVocabDim }, -0.05, 0.05, 1, device));

    auto inputEmbedding = (!forceEmbedding && (inputVocabDim <= inputEmbeddingDim)) ? inputSequence : Times(inputEmbeddingWeights, inputSequence);
    auto labelEmbedding = (!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSequence : Times(labelEmbeddingWeights, labelSequence);
    auto labelSentenceStartEmbedding = (!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSentenceStart : Times(labelEmbeddingWeights, labelSentenceStart);
    auto labelSentenceStartEmbeddedScattered = Sequence::Scatter(labelSentenceStartEmbedding, isFirstLabel);

    auto stabilize = [](const Variable& x) {
        float scalarConstant = 4.0f;
        auto f = Constant({}, scalarConstant);
        auto fInv = Constant({}, 1.0f/scalarConstant);

        auto beta = ElementTimes(fInv, Log(Constant({}, 1.0f) + Exp(ElementTimes(f, Parameter({}, 0.99537863f /* 1/f*ln (e^f-1) */)))));
        return ElementTimes(beta, x);
    };

    /* Encoder */
    auto encoderOutputH = stabilize(inputEmbedding);
    FunctionPtr encoderOutputC;
    auto futureValueRecurrenceHook = std::bind(FutureValue, _1, CNTK::Constant({}, 0.0f), 1, L"");
    for (size_t i = 0; i < numLayers; ++i)
        std::tie(encoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(encoderOutputH, hiddenDim, hiddenDim, futureValueRecurrenceHook, futureValueRecurrenceHook, device);

    auto thoughtVectorH = Sequence::First(encoderOutputH);
    auto thoughtVectorC = Sequence::First(encoderOutputC);

    auto thoughtVectorBroadcastH = Sequence::BroadcastAs(thoughtVectorH, labelEmbedding);
    auto thoughtVectorBroadcastC = Sequence::BroadcastAs(thoughtVectorC, labelEmbedding);

    /* Decoder */
    auto decoderHistoryFromGroundTruth = labelEmbedding;
    auto decoderInput = ElementSelect(isFirstLabel, labelSentenceStartEmbeddedScattered, PastValue(decoderHistoryFromGroundTruth, Constant({}, 0.0f), 1));

    auto decoderOutputH = stabilize(decoderInput);
    FunctionPtr decoderOutputC;
    auto pastValueRecurrenceHook = std::bind(PastValue, _1, CNTK::Constant({}, 0.0f), 1, L"");
    for (size_t i = 0; i < numLayers; ++i)
    {
        std::function<FunctionPtr(const Variable&)> recurrenceHookH, recurrenceHookC;
        if (i == 0)
        {
            recurrenceHookH = pastValueRecurrenceHook;
            recurrenceHookC = pastValueRecurrenceHook;
        }
        else
        {
            auto isFirst = Sequence::IsFirst(labelEmbedding);
            recurrenceHookH = [labelEmbedding, thoughtVectorBroadcastH, isFirst](const Variable& operand) {
                return ElementSelect(isFirst, thoughtVectorBroadcastH, PastValue(operand, CNTK::Constant({}, 0.0f), 1, L""));
            };

            recurrenceHookC = [labelEmbedding, thoughtVectorBroadcastC, isFirst](const Variable& operand) {
                return ElementSelect(isFirst, thoughtVectorBroadcastC, PastValue(operand, CNTK::Constant({}, 0.0f), 1, L""));
            };
        }

        std::tie(decoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(decoderOutputH, hiddenDim, hiddenDim, recurrenceHookH, recurrenceHookC, device);
    }

    auto decoderOutput = decoderOutputH;
    auto decoderDim = hiddenDim;

    /* Softmax output layer */
    auto outputLayerProjWeights = Parameter(NDArrayView::RandomUniform<float>({ labelVocabDim, decoderDim }, -0.05, 0.05, 1, device));
    auto biasWeights = Parameter({ labelVocabDim }, 0.0f, device);

    auto z = Plus(Times(outputLayerProjWeights, stabilize(decoderOutput)), biasWeights, L"classifierOutput");
    auto ce = CrossEntropyWithSoftmax(z, labelSequence, L"lossFunction");
    auto errs = ClassificationError(z, labelSequence, L"classificationError");

    if (testSaveAndReLoad)
    {
        Variable zVar = z;
        Variable ceVar = ce;
        Variable errsVar = errs;
        auto seq2seqModel = Combine({ ce, errs, z }, L"seq2seqModel");
        SaveAndReloadModel<float>(seq2seqModel, { &rawInput, &rawLabels, &ceVar, &errsVar, &zVar }, device);

        z = zVar;
        ce = ceVar;
        errs = errsVar;
    }

    auto minibatchSource = CreateSeq2SeqMinibatchSource(L"cmudict-0.7b.train-dev-20-21.bsf.ctf.2", inputVocabDim, labelVocabDim);
    auto rawInputStreamInfo = minibatchSource->StreamInfo(L"rawInput");
    auto rawLabelsStreamInfo = minibatchSource->StreamInfo(L"rawLabels");

    double learningRatePerSample = 0.007;
    size_t momentumTimeConstant = 1100;
    double momentumPerSample = std::exp(-1.0 / momentumTimeConstant);
    double clippingThresholdPerSample = 2.3;
    bool gradientClippingWithTruncation = true;
    Trainer trainer(z, ce, errs, { MomentumSGDLearner(z->Parameters(), learningRatePerSample, momentumPerSample, clippingThresholdPerSample, gradientClippingWithTruncation) });

    size_t outputFrequencyInMinibatches = 1;
    size_t minibatchSize = 72;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer.TrainMinibatch({ { rawInput, minibatchData[rawInputStreamInfo].m_data }, { rawLabels, minibatchData[rawLabelsStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TrainSequenceToSequenceTranslator()
{
    // TODO: Also test with sparse input variables in the graph

    TrainSequenceToSequenceTranslator(DeviceDescriptor::GPUDevice(0), false, true);
    TrainSequenceToSequenceTranslator(DeviceDescriptor::CPUDevice(), false, false);
}
