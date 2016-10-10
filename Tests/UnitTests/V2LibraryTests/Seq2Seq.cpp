#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

void TrainSequenceToSequenceTranslator(const DeviceDescriptor& device, bool useSparseInputs, bool testSaveAndReLoad, bool testCheckpointing, bool addBeamSearchReorderingHook)
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
    auto rawInput = InputVariable({ inputVocabDim }, useSparseInputs /*isSparse*/, DataType::Float, L"rawInput", inputDynamicAxes);

    std::vector<Axis> labelDynamicAxes = { Axis(L"labelAxis"), Axis::DefaultBatchAxis() };
    auto rawLabels = InputVariable({ labelVocabDim }, useSparseInputs /*isSparse*/, DataType::Float, L"rawLabels", labelDynamicAxes);

    FunctionPtr inputSequence = rawInput;

    // Drop the sentence start token from the label, for decoder training
    auto labelSequence = Slice(rawLabels, labelDynamicAxes[0], 1, 0);
    auto labelSentenceStart = Sequence::First(rawLabels);

    auto isFirstLabel = Sequence::IsFirst(labelSequence);

    bool forceEmbedding = useSparseInputs;

    /* Embeddings */
    auto inputEmbeddingWeights = Parameter({ inputEmbeddingDim, inputVocabDim }, DataType::Float, GlorotUniformInitializer(), device);
    auto labelEmbeddingWeights = Parameter({ labelEmbeddingDim, labelVocabDim }, DataType::Float, GlorotUniformInitializer(), device);

    auto inputEmbedding = (!forceEmbedding && (inputVocabDim <= inputEmbeddingDim)) ? inputSequence : Times(inputEmbeddingWeights, inputSequence);
    auto labelEmbedding = (!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSequence : Times(labelEmbeddingWeights, labelSequence);
    auto labelSentenceStartEmbedding = (!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSentenceStart : Times(labelEmbeddingWeights, labelSentenceStart);
    auto labelSentenceStartEmbeddedScattered = Sequence::Scatter(labelSentenceStartEmbedding, isFirstLabel);

    /* Encoder */
    auto encoderOutputH = Stabilize<float>(inputEmbedding, device);
    FunctionPtr encoderOutputC;
    auto futureValueRecurrenceHook = [](const Variable& x) { return FutureValue(x); };
    for (size_t i = 0; i < numLayers; ++i)
        std::tie(encoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(encoderOutputH, { hiddenDim }, { hiddenDim }, futureValueRecurrenceHook, futureValueRecurrenceHook, device);

    auto thoughtVectorH = Sequence::First(encoderOutputH);
    auto thoughtVectorC = Sequence::First(encoderOutputC);
    if (addBeamSearchReorderingHook)
    {
        thoughtVectorH = Reshape(thoughtVectorH, thoughtVectorH->Output().Shape().AppendShape({ 1 }));
        thoughtVectorC = Reshape(thoughtVectorC, thoughtVectorC->Output().Shape().AppendShape({ 1 }));
        labelEmbedding = Reshape(labelEmbedding, labelEmbedding->Output().Shape().AppendShape({ 1 }));
        labelSentenceStartEmbeddedScattered = Reshape(labelSentenceStartEmbeddedScattered, labelSentenceStartEmbeddedScattered->Output().Shape().AppendShape({ 1 }));
    }

    auto thoughtVectorBroadcastH = Sequence::BroadcastAs(thoughtVectorH, labelEmbedding);
    auto thoughtVectorBroadcastC = Sequence::BroadcastAs(thoughtVectorC, labelEmbedding);

    /* Decoder */
    auto beamSearchReorderHook = Constant({ 1, 1 }, 1.0f, device);
    auto decoderHistoryFromGroundTruth = labelEmbedding;
    auto decoderInput = ElementSelect(isFirstLabel, labelSentenceStartEmbeddedScattered, PastValue(decoderHistoryFromGroundTruth));

    auto decoderOutputH = Stabilize<float>(decoderInput, device);
    FunctionPtr decoderOutputC;
    auto pastValueRecurrenceHookWithBeamSearchReordering = [addBeamSearchReorderingHook, beamSearchReorderHook](const FunctionPtr& operand) {
        return PastValue(addBeamSearchReorderingHook ? Times(operand, beamSearchReorderHook) : operand);
    };

    for (size_t i = 0; i < numLayers; ++i)
    {
        std::function<FunctionPtr(const Variable&)> recurrenceHookH, recurrenceHookC;
        if (i > 0)
        {
            recurrenceHookH = pastValueRecurrenceHookWithBeamSearchReordering;
            recurrenceHookC = pastValueRecurrenceHookWithBeamSearchReordering;
        }
        else
        {
            auto isFirst = Sequence::IsFirst(labelEmbedding);
            recurrenceHookH = [labelEmbedding, thoughtVectorBroadcastH, isFirst, addBeamSearchReorderingHook, beamSearchReorderHook](const FunctionPtr& operand) {
                return ElementSelect(isFirst, thoughtVectorBroadcastH, PastValue(addBeamSearchReorderingHook ? Times(operand, beamSearchReorderHook) : operand));
            };

            recurrenceHookC = [labelEmbedding, thoughtVectorBroadcastC, isFirst, addBeamSearchReorderingHook, beamSearchReorderHook](const FunctionPtr& operand) {
                return ElementSelect(isFirst, thoughtVectorBroadcastC, PastValue(addBeamSearchReorderingHook ? Times(operand, beamSearchReorderHook) : operand));
            };
        }

        NDShape outDims = { hiddenDim };
        NDShape cellDims = { hiddenDim };
        if (addBeamSearchReorderingHook)
        {
            outDims = outDims.AppendShape({ 1 });
            cellDims = cellDims.AppendShape({ 1 });
        }
        std::tie(decoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(decoderOutputH, outDims, cellDims, recurrenceHookH, recurrenceHookC, device);
    }

    auto decoderOutput = decoderOutputH;
    auto decoderDim = hiddenDim;

    /* Softmax output layer */
    auto outputLayerProjWeights = Parameter({ labelVocabDim, decoderDim }, DataType::Float, GlorotUniformInitializer(), device);
    auto biasWeights = Parameter({ labelVocabDim }, 0.0f, device);

    auto z = Plus(Times(outputLayerProjWeights, Stabilize<float>(decoderOutput, device)), biasWeights, L"classifierOutput");
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

    auto featureStreamName = L"rawInput";
    auto labelStreamName = L"rawLabels";
    auto minibatchSource = TextFormatMinibatchSource(L"cmudict-0.7b.train-dev-20-21.ctf",
                                                     { { featureStreamName, inputVocabDim, true, L"S0" }, {labelStreamName, labelVocabDim, true, L"S1" } },
                                                     5000);

    auto rawInputStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto rawLabelsStreamInfo = minibatchSource->StreamInfo(labelStreamName);

    double learningRatePerSample = 0.007;
    size_t momentumTimeConstant = 1100;
    double momentumPerSample = std::exp(-1.0 / momentumTimeConstant);
    AdditionalLearningOptions additionalOptions;
    additionalOptions.gradientClippingThresholdPerSample = 2.3;
    additionalOptions.gradientClippingWithTruncation = true;
    Trainer trainer(z, ce, errs, { MomentumSGDLearner(z->Parameters(), learningRatePerSample, momentumPerSample, additionalOptions) });

    size_t outputFrequencyInMinibatches = 1;
    size_t minibatchSize = 72;
    size_t numMinibatchesToCheckpointAfter = testCheckpointing ? 3 : SIZE_MAX;
    size_t numMinibatchesToRestoreFromCheckpointAfter = testCheckpointing ? 20 : SIZE_MAX;
    bool restorationDone = false;
    const wchar_t* modelFile = L"seq2seq.model";
    for (size_t i = 0; true; i++)
    {
        if (!restorationDone && (i == numMinibatchesToRestoreFromCheckpointAfter))
        {
            printf("Trainer restoring from checkpoint at path %S\n", modelFile);
            trainer.RestoreFromCheckpoint(modelFile);
            i = numMinibatchesToCheckpointAfter;
            restorationDone = true;
        }

        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer.TrainMinibatch({ { rawInput, minibatchData[rawInputStreamInfo].m_data }, { rawLabels, minibatchData[rawLabelsStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

        if ((i + 1) == numMinibatchesToCheckpointAfter)
        {
            printf("Trainer checkpointing to path %S\n", modelFile);
            trainer.SaveCheckpoint(modelFile);
        }
    }
}

void TrainSequenceToSequenceTranslator()
{
    // TODO: Also test with sparse input variables in the graph
    if (IsGPUAvailable())
    {
        TrainSequenceToSequenceTranslator(DeviceDescriptor::GPUDevice(0), false, false, true, false);
    }
    TrainSequenceToSequenceTranslator(DeviceDescriptor::CPUDevice(), false, true, false, true);
}
