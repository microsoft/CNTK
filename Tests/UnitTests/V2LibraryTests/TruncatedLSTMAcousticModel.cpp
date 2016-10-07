#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

MinibatchSourcePtr CreateMinibatchSource(size_t featureDim, size_t numOutputClasses, const Dictionary& readModeConfig, size_t epochSize, bool randomize = true)
{
    auto featuresFilePath = L"glob_0000.scp";
    auto labelsFilePath = L"glob_0000.mlf";
    auto labelMappingFile = L"state.list";

    Dictionary featuresStreamConfig;
    featuresStreamConfig[L"dim"] = featureDim;
    featuresStreamConfig[L"scpFile"] = featuresFilePath;

    CNTK::Dictionary featInputStreamsConfig;
    featInputStreamsConfig[L"features"] = featuresStreamConfig;

    CNTK::Dictionary featDeserializerConfiguration;
    featDeserializerConfiguration[L"type"] = L"HTKFeatureDeserializer";
    featDeserializerConfiguration[L"input"] = featInputStreamsConfig;

    Dictionary labelsStreamConfig;
    labelsStreamConfig[L"dim"] = numOutputClasses;
    labelsStreamConfig[L"mlfFile"] = labelsFilePath;
    labelsStreamConfig[L"labelMappingFile"] = labelMappingFile;
    labelsStreamConfig[L"scpFile"] = featuresFilePath;

    CNTK::Dictionary labelsInputStreamsConfig;
    labelsInputStreamsConfig[L"labels"] = labelsStreamConfig;

    CNTK::Dictionary labelsDeserializerConfiguration;
    labelsDeserializerConfiguration[L"type"] = L"HTKMLFDeserializer";
    labelsDeserializerConfiguration[L"input"] = labelsInputStreamsConfig;

    Dictionary minibatchSourceConfiguration;
    if (randomize)
        minibatchSourceConfiguration[L"randomize"] = true;

    minibatchSourceConfiguration[L"epochSize"] = epochSize;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<DictionaryValue>({ featDeserializerConfiguration, labelsDeserializerConfiguration });
    minibatchSourceConfiguration.Add(readModeConfig);

    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
}

static FunctionPtr LSTMAcousticSequenceClassiferNet(const Variable& input, size_t numOutputClasses, size_t LSTMDim, size_t cellDim, size_t numLSTMs, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    FunctionPtr r = input;
    for (size_t i = 0; i < numLSTMs; ++i)
        r = LSTMPComponentWithSelfStabilization<float>(r, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;

    return FullyConnectedLinearLayer(r, numOutputClasses, device, outputName);
}

void TrainTruncatedLSTMAcousticModelClassifer(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t baseFeaturesDim = 33;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 256;
    const size_t numOutputClasses = 132;
    const size_t numLSTMLayers = 3;

    auto features = InputVariable({ baseFeaturesDim }, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");

    const size_t numSamplesForFeatureStatistics = 0;
    Dictionary frameModeConfig;
    frameModeConfig[L"frameMode"] = true;
    auto minibatchSource = CreateMinibatchSource(baseFeaturesDim, numOutputClasses, frameModeConfig, numSamplesForFeatureStatistics, false);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>> featureMeansAndInvStdDevs = { { featureStreamInfo, { nullptr, nullptr } } };
    ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, featureMeansAndInvStdDevs);

    auto normalizedFeatures = PerDimMeanVarianceNormalize(features, featureMeansAndInvStdDevs[featureStreamInfo].first, featureMeansAndInvStdDevs[featureStreamInfo].second);
    auto classifierOutput = LSTMAcousticSequenceClassiferNet(normalizedFeatures, numOutputClasses, hiddenDim, cellDim, numLSTMLayers, device, L"classifierOutput");

    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    if (testSaveAndReLoad)
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto oneHiddenLayerClassifier = CNTK::Combine({ trainingLoss, prediction, classifierOutput }, L"classifierModel");
        SaveAndReloadModel<float>(oneHiddenLayerClassifier, { &features, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    const size_t numTrainingSamples = 81920;
    const size_t truncationLength = 20;
    Dictionary truncatedModeConfig;
    truncatedModeConfig[L"truncated"] = true;
    truncatedModeConfig[L"truncationLength"] = truncationLength;
    minibatchSource = CreateMinibatchSource(baseFeaturesDim, numOutputClasses, truncatedModeConfig, numTrainingSamples);

    const size_t numberParallelSequencesPerMB = 32;
    const size_t minibatchSize = truncationLength * numberParallelSequencesPerMB;

    featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    double learningRatePerSample = 0.000781;
    size_t momentumTimeConstant = 6074;
    double momentumPerSample = std::exp(-1.0 / momentumTimeConstant);
    auto learner = MomentumSGDLearner(classifierOutput->Parameters(), learningRatePerSample, momentumPerSample);
    Trainer trainer(classifierOutput, trainingLoss, prediction, {learner});

    size_t outputFrequencyInMinibatches = 1;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TrainTruncatedLSTMAcousticModelClassifer()
{
    if (IsGPUAvailable())
    {
        TrainTruncatedLSTMAcousticModelClassifer(DeviceDescriptor::GPUDevice(0), true);
    }
    TrainTruncatedLSTMAcousticModelClassifer(DeviceDescriptor::CPUDevice(), false);
}
