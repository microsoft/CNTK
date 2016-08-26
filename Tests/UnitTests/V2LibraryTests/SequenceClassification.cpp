#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

FunctionPtr Embedding(const Variable& input, size_t embeddingDim, const DeviceDescriptor& device)
{
    assert(input.Shape().NumAxes() == 1);
    size_t inputDim = input.Shape()[0];

    auto embeddingParameters = Parameter(CNTK::NDArrayView::RandomUniform<float>({ embeddingDim, inputDim }, -0.05, 0.05, 1, device));
    return Times(embeddingParameters, input);
}

FunctionPtr SelectLast(const Variable& operand)
{
    return Slice(operand, Axis::DefaultDynamicAxis(), -1, 0);
}

FunctionPtr LSTMSequenceClassiferNet(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto embeddingFunction = Embedding(input, embeddingDim, device);
    auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, LSTMDim, cellDim, device);
    auto thoughtVectorFunction = SelectLast(LSTMFunction);

    return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
}

void TrainLSTMSequenceClassifer(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    Variable features({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto classifierOutputFunction = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
    Variable classifierOutput = classifierOutputFunction;

    Variable labels({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    Variable trainingLoss = trainingLossFunction;
    auto predictionFunction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");
    Variable prediction = predictionFunction;

    auto oneHiddenLayerClassifier = CNTK::Combine({ trainingLoss.Owner(), prediction.Owner(), classifierOutput.Owner() }, L"classifierModel");

    if (testSaveAndReLoad)
        SaveAndReloadModel<float>(oneHiddenLayerClassifier, { &features, &labels, &trainingLoss, &prediction, &classifierOutput }, device);

    auto minibatchSource = CreateTextMinibatchSource(L"Train.ctf", inputDim, numOutputClasses, 0, true, false, L"x", L"y");
    const size_t minibatchSize = 200;
    
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);
    double learningRatePerSample = 0.0005;
    Trainer trainer(oneHiddenLayerClassifier, trainingLoss, { SGDLearner(oneHiddenLayerClassifier->Parameters(), learningRatePerSample) });
    size_t outputFrequencyInMinibatches = 1;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

        if ((i % outputFrequencyInMinibatches) == 0)
        {
            double trainLossValue = trainer.PreviousMinibatchAverageTrainingLoss();
            printf("Minibatch %d: CrossEntropy loss = %.8g\n", (int)i, trainLossValue);
        }
    }
}

void TrainLSTMSequenceClassifer()
{
    TrainLSTMSequenceClassifer(DeviceDescriptor::GPUDevice(0), true);
    TrainLSTMSequenceClassifer(DeviceDescriptor::CPUDevice(), false);
}
