#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

void TrainSequenceToSequenceTranslator(const DeviceDescriptor& device, bool useSparseInputs, bool testSaveAndReLoad, bool testCheckpointing, bool addBeamSearchReorderingHook, bool testCloning, bool usePlaceholders)
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

    FunctionPtr inputSequence = Alias(rawInput, L"inputSequence");

    // Drop the sentence start token from the label, for decoder training
    auto labelSequence = Sequence::Slice(rawLabels, 1, 0, L"labelSequenceWithStartTrimmed");
    auto labelSentenceStart = Sequence::First(rawLabels, L"labelSequenceStart");

    auto isFirstLabel = Sequence::IsFirst(labelSequence, L"isFirstLabel");

    bool forceEmbedding = useSparseInputs;

    /* Embeddings */
    auto inputEmbeddingWeights = Parameter({ inputEmbeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"inputEmbeddingWeights");
    auto labelEmbeddingWeights = Parameter({ labelEmbeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"labelEmbeddingWeights");

    auto inputEmbedding = Alias((!forceEmbedding && (inputVocabDim <= inputEmbeddingDim)) ? inputSequence : Times(inputEmbeddingWeights, inputSequence), L"inputEmbedding");
    auto labelEmbedding = Alias((!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSequence : Times(labelEmbeddingWeights, labelSequence), L"labelEmbedding");
    auto labelSentenceStartEmbedding = Alias((!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSentenceStart : Times(labelEmbeddingWeights, labelSentenceStart), L"labelSentenceStartEmbedding");
    auto labelSentenceStartEmbeddedScattered = Sequence::Scatter(labelSentenceStartEmbedding, isFirstLabel, L"labelSentenceStartEmbeddedScattered");

    /* Encoder */
    auto encoderOutputH = Stabilize<float>(inputEmbedding, device);
    FunctionPtr encoderOutputC;
    auto futureValueRecurrenceHook = [](const Variable& x) { return FutureValue(x); };
    for (size_t i = 0; i < numLayers; ++i)
        std::tie(encoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(encoderOutputH, { hiddenDim }, { hiddenDim }, futureValueRecurrenceHook, futureValueRecurrenceHook, device);

    auto thoughtVectorH = Sequence::First(encoderOutputH, L"thoughtVectorH");
    auto thoughtVectorC = Sequence::First(encoderOutputC, L"thoughtVectorC");
    if (addBeamSearchReorderingHook)
    {
        thoughtVectorH = Reshape(thoughtVectorH, thoughtVectorH->Output().Shape().AppendShape({ 1 }), L"thoughtVectorH");
        thoughtVectorC = Reshape(thoughtVectorC, thoughtVectorC->Output().Shape().AppendShape({ 1 }), L"thoughtVectorC");
        labelEmbedding = Reshape(labelEmbedding, labelEmbedding->Output().Shape().AppendShape({ 1 }), L"labelEmbedding");
        labelSentenceStartEmbeddedScattered = Reshape(labelSentenceStartEmbeddedScattered, labelSentenceStartEmbeddedScattered->Output().Shape().AppendShape({ 1 }), L"labelSentenceStartEmbeddedScattered");
    }

    auto actualThoughtVectorBroadcastH = Sequence::BroadcastAs(thoughtVectorH, labelEmbedding, L"thoughtVectorBroadcastH");
    auto actualThoughtVectorBroadcastC = Sequence::BroadcastAs(thoughtVectorC, labelEmbedding, L"thoughtVectorBroadcastC");

    Variable thoughtVectorBroadcastH, thoughtVectorBroadcastC;
    if (usePlaceholders)
    {
        thoughtVectorBroadcastH = PlaceholderVariable();
        thoughtVectorBroadcastC = PlaceholderVariable();
    }
    else
    {
        thoughtVectorBroadcastH = actualThoughtVectorBroadcastH;
        thoughtVectorBroadcastC = actualThoughtVectorBroadcastC;
    }

    /* Decoder */
    auto beamSearchReorderHook = Constant({ 1, 1 }, 1.0f, device);
    auto decoderHistoryFromGroundTruth = labelEmbedding;
    auto decoderHistoryHook = Alias(decoderHistoryFromGroundTruth);
    auto decoderInput = ElementSelect(isFirstLabel, labelSentenceStartEmbeddedScattered, PastValue(decoderHistoryHook));

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

    if (usePlaceholders)
        z->ReplacePlaceholders({ { thoughtVectorBroadcastH, actualThoughtVectorBroadcastH }, { thoughtVectorBroadcastC, actualThoughtVectorBroadcastC } });

    auto ce = CrossEntropyWithSoftmax(z, labelSequence, L"lossFunction");
    auto errs = ClassificationError(z, labelSequence, L"classificationError");

    if (testCloning)
    {
        std::unordered_set<FunctionPtr> visitedFunctions;

        auto combinedFunc = Combine({ z, ce, errs });
        auto clonedFunctionWithParametersCloned = combinedFunc->Clone();
        CompareFunctions(combinedFunc, clonedFunctionWithParametersCloned, ParameterCloningMethod::Clone, {}, visitedFunctions);

        visitedFunctions.clear();
        auto clonedFunctionWithParametersShared = clonedFunctionWithParametersCloned->Clone(ParameterCloningMethod::Share);
        CompareFunctions(clonedFunctionWithParametersCloned, clonedFunctionWithParametersShared, ParameterCloningMethod::Share, {}, visitedFunctions);
    }

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

    // Decoder history for greedy decoding
    auto decoderHistoryFromOutput = Hardmax(z);
    auto decodingFunction = decoderHistoryFromOutput->Clone(ParameterCloningMethod::Share, { {decoderHistoryHook, decoderHistoryFromOutput} });
    decodingFunction = Combine({ decodingFunction->RootFunction()->Arguments()[0] });

    auto featureStreamName = L"rawInput";
    auto labelStreamName = L"rawLabels";
    auto minibatchSource = TextFormatMinibatchSource(L"cmudict-0.7b.train-dev-20-21.ctf",
                                                     { { featureStreamName, inputVocabDim, true, L"S0" }, {labelStreamName, labelVocabDim, true, L"S1" } },
                                                     5000);

    auto rawInputStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto rawLabelsStreamInfo = minibatchSource->StreamInfo(labelStreamName);

    LearningRatePerSampleSchedule learningRatePerSample = 0.007;
    MomentumAsTimeConstantSchedule momentumTimeConstant = 1100;
    AdditionalLearningOptions additionalOptions;
    additionalOptions.gradientClippingThresholdPerSample = 2.3;
    additionalOptions.gradientClippingWithTruncation = true;

    auto trainer = CreateTrainer(z, ce, errs, { MomentumSGDLearner(z->Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true, additionalOptions) });

    size_t outputFrequencyInMinibatches = 1;
    size_t minibatchSize1 = 72;
    size_t minibatchSize2 = 144;
    size_t numMinbatchesToChangeMinibatchSizeAfter = 30;
    size_t numMinibatchesToCheckpointAfter = testCheckpointing ? 3 : SIZE_MAX;
    size_t numMinibatchesToRestoreFromCheckpointAfter = testCheckpointing ? 20 : SIZE_MAX;
    bool restorationDone = false;
    const wchar_t* modelFile = L"seq2seq.model";
    size_t decodingFrequency = 10;
    Dictionary minibatchSourceCheckpoint;
    for (size_t i = 0; true; i++)
    {
        if (!restorationDone && (i == numMinibatchesToRestoreFromCheckpointAfter))
        {
            printf("Trainer restoring from checkpoint at path %S\n", modelFile);
            trainer->RestoreFromCheckpoint(modelFile);
            minibatchSource->RestoreFromCheckpoint(minibatchSourceCheckpoint);
            i = numMinibatchesToCheckpointAfter;
            restorationDone = true;
        }

        auto minibatchSize = (i >= numMinbatchesToChangeMinibatchSizeAfter) ? minibatchSize2 : minibatchSize1;
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer->TrainMinibatch({ { rawInput, minibatchData[rawInputStreamInfo] }, { rawLabels, minibatchData[rawLabelsStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

        if ((i + 1) == numMinibatchesToCheckpointAfter)
        {
            printf("Trainer checkpointing to path %S\n", modelFile);
            trainer->SaveCheckpoint(modelFile);
            minibatchSourceCheckpoint = minibatchSource->GetCheckpointState();
        }

        if ((i % decodingFrequency) == 0)
        {
            std::unordered_map<Variable, ValuePtr> outputs = { { decodingFunction, nullptr }};
            decodingFunction->Forward({ { decodingFunction->Arguments()[0], minibatchData[rawInputStreamInfo].data }, { decodingFunction->Arguments()[1], minibatchData[rawLabelsStreamInfo].data } },
                                      outputs,
                                      device);
        }
    }
}

void TrainSequenceToSequenceTranslator()
{
    fprintf(stderr, "\nTrainSequenceToSequenceTranslator..\n");

    // TODO: Also test with sparse input variables in the graph
    TrainSequenceToSequenceTranslator(DeviceDescriptor::CPUDevice(), false, true, false, false, true, true);

    if (IsGPUAvailable())
        TrainSequenceToSequenceTranslator(DeviceDescriptor::GPUDevice(0), false, false, true, true, false, false);
}
