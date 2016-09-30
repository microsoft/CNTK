#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

FunctionPtr Embedding(const Variable& input, size_t embeddingDim, const DeviceDescriptor& device)
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];

    auto embeddingParameters = Parameter({ embeddingDim, inputDim }, DataType::Float, GlorotUniformInitializer(), device);
    return Times(embeddingParameters, input);
}

FunctionPtr LSTMSequenceClassiferNet(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto embeddingFunction = Embedding(input, embeddingDim, device);
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
    auto thoughtVectorFunction = Sequence::Last(LSTMFunction);

    return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
}

void TrainLSTMSequenceClassifer(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
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

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    const size_t minibatchSize = 200;
    
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    double learningRatePerSample = 0.0005;
    size_t momentumTimeConstant = 256;
    double momentumPerSample = std::exp(-1.0 / momentumTimeConstant);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { MomentumSGDLearner(classifierOutput->Parameters(), learningRatePerSample, momentumPerSample) });

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

void TestLearningRateControl(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { learner });

    if (learner->LearningRate() != 0.0005)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    if (learner->LearningRate() != 0.0005)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    const wchar_t* modelFile = L"seq2seq.model";
    trainer.SaveCheckpoint(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto MB2Loss = trainer.PreviousMinibatchLossAverage();
    if (learner->LearningRate() != 0.00025)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto MB3Loss = trainer.PreviousMinibatchLossAverage();
    if (learner->LearningRate() != 0.00025)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.RestoreFromCheckpoint(modelFile);
    if (learner->LearningRate() != 0.0005)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    if (postRestoreMB2Loss != MB2Loss)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

    if (learner->LearningRate() != 0.00025)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB3Loss = trainer.PreviousMinibatchLossAverage();
    if (postRestoreMB3Loss != MB3Loss)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

    trainer.RestoreFromCheckpoint(modelFile);
    if (learner->LearningRate() != 0.0005)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    learner->ResetLearningRate(0.0004);
    if (learner->LearningRate() != 0.0004)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.SaveCheckpoint(modelFile);
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    if (postRestoreMB2Loss != MB2Loss)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

    if (learner->LearningRate() != 0.0004)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB3Loss = trainer.PreviousMinibatchLossAverage();
    if (postRestoreMB3Loss == MB3Loss)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

    if (learner->LearningRate() != 0.0004)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.RestoreFromCheckpoint(modelFile);
    if (learner->LearningRate() != 0.0004)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    if (postRestoreMB2Loss != MB2Loss)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

    if (learner->LearningRate() != 0.0004)
        throw std::runtime_error("Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB3Loss = trainer.PreviousMinibatchLossAverage();
    if (postRestoreMB3Loss == MB3Loss)
        throw std::runtime_error("Post checkpoint restoration training loss does not match expectation");

    if (learner->LearningRate() != 0.0004)
        throw std::runtime_error("Learner::LearningRate does not match expectation");
}

void TrainLSTMSequenceClassifer()
{
#ifndef CPUONLY
    TestLearningRateControl(DeviceDescriptor::GPUDevice(0));
#endif

#ifndef CPUONLY
    TrainLSTMSequenceClassifer(DeviceDescriptor::GPUDevice(0), true);
#endif
    TrainLSTMSequenceClassifer(DeviceDescriptor::CPUDevice(), false);
}
