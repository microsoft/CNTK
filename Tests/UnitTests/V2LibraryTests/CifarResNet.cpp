#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include "Image.h"

using namespace CNTK;

MinibatchSourcePtr CreateCifarMinibatchSource(size_t epochSize)
{
    size_t imageHeight = 32;
    size_t imageWidth = 32;
    size_t numChannels = 3;
    size_t numClasses = 10;
    auto mapFilePath = L"cifar-10-batches-py/train_map.txt";
    auto meanFilePath = L"cifar-10-batches-py/CIFAR-10_mean.xml";

    Dictionary cropTransformConfig;
    cropTransformConfig[L"type"] = L"Crop";
    cropTransformConfig[L"cropType"] = L"Random";
    cropTransformConfig[L"cropRatio"] = L"0.8";
    cropTransformConfig[L"jitterType"] = L"uniRatio";

    Dictionary scaleTransformConfig;
    scaleTransformConfig[L"type"] = L"Scale";
    scaleTransformConfig[L"width"] = imageWidth;
    scaleTransformConfig[L"height"] = imageHeight;
    scaleTransformConfig[L"channels"] = numChannels;
    scaleTransformConfig[L"interpolations"] = L"linear";

    Dictionary meanTransformConfig;
    meanTransformConfig[L"type"] = L"Mean";
    meanTransformConfig[L"meanFile"] = meanFilePath;

    std::vector<DictionaryValue> allTransforms = { cropTransformConfig, scaleTransformConfig, meanTransformConfig };

    Dictionary featuresStreamConfig;
    featuresStreamConfig[L"transforms"] = allTransforms;

    Dictionary labelsStreamConfig;
    labelsStreamConfig[L"labelDim"] = numClasses;

    Dictionary inputStreamsConfig;
    inputStreamsConfig[L"features"] = featuresStreamConfig;
    inputStreamsConfig[L"labels"] = labelsStreamConfig;

    Dictionary deserializerConfiguration;
    deserializerConfiguration[L"type"] = L"ImageDeserializer";
    deserializerConfiguration[L"file"] = mapFilePath;
    deserializerConfiguration[L"input"] = inputStreamsConfig;

    Dictionary minibatchSourceConfiguration;
    minibatchSourceConfiguration[L"epochSize"] = epochSize;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<DictionaryValue>({ deserializerConfiguration });

    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
}

Constant GetProjectionMap(size_t outputDim, size_t inputDim, const DeviceDescriptor& device)
{
    if (inputDim > outputDim)
        throw std::runtime_error("Can only project from lower to higher dimensionality");

    std::vector<float> projectionMapValues(inputDim * outputDim);
    for (size_t i = 0; i < inputDim; ++i)
        projectionMapValues[(i * outputDim) + i] = 1.0f;

    auto projectionMap = MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ outputDim, 1, 1, inputDim }), device);
    projectionMap->CopyFrom(NDArrayView(NDShape({ outputDim, 1, 1, inputDim }), projectionMapValues));

    return Constant(projectionMap);
}

FunctionPtr ResNetClassifier(Variable input, size_t numOutputClasses, const DeviceDescriptor& device, const std::wstring& outputName)
{
    double convWScale = 7.07;
    double convBValue = 0;

    double fc1WScale = 0.4;
    double fc1BValue = 0;

    double scValue = 1;
    size_t bnTimeConst = 4096;

    size_t kernelWidth = 3;
    size_t kernelHeight = 3;

    double conv1WScale = 0.26;
    size_t cMap1 = 16;
    auto conv1 = ConvBNReLULayer(input, cMap1, kernelWidth, kernelHeight, 1, 1, conv1WScale, convBValue, scValue, bnTimeConst, device);

    auto rn1_1 = ResNetNode2(conv1, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);
    auto rn1_2 = ResNetNode2(rn1_1, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);
    auto rn1_3 = ResNetNode2(rn1_2, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);

    size_t cMap2 = 32;
    auto rn2_1_wProj = GetProjectionMap(cMap2, cMap1, device);
    auto rn2_1 = ResNetNode2Inc(rn1_3, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, rn2_1_wProj, device);
    auto rn2_2 = ResNetNode2(rn2_1, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);
    auto rn2_3 = ResNetNode2(rn2_2, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);

    size_t cMap3 = 64;
    auto rn3_1_wProj = GetProjectionMap(cMap3, cMap2, device);
    auto rn3_1 = ResNetNode2Inc(rn2_3, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, rn3_1_wProj, device);
    auto rn3_2 = ResNetNode2(rn3_1, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);
    auto rn3_3 = ResNetNode2(rn3_2, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, device);

    // Global average pooling
    size_t poolW = 8;
    size_t poolH = 8;
    size_t poolhStride = 1;
    size_t poolvStride = 1;
    auto pool = Pooling(rn3_3, PoolingType::Average, { poolW, poolH, 1 }, { poolhStride, poolvStride, 1 });

    // Output DNN layer
    auto outTimesParams = Parameter(NDArrayView::RandomNormal<float>({ numOutputClasses, 1, 1, cMap3 }, 0.0, fc1WScale, 1, device));
    auto outBiasParams = Parameter({ numOutputClasses }, (float)fc1BValue, device);

    return Plus(Times(outTimesParams, pool), outBiasParams, outputName);
}

void TrainResNetCifarClassifer(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    auto minibatchSource = CreateCifarMinibatchSource(SIZE_MAX);
    auto imageStreamInfo = minibatchSource->StreamInfo(L"features");
    auto labelStreamInfo = minibatchSource->StreamInfo(L"labels");

    auto inputImageShape = imageStreamInfo.m_sampleLayout;
    const size_t numOutputClasses = labelStreamInfo.m_sampleLayout[0];

    auto imageInput = InputVariable(inputImageShape, imageStreamInfo.m_elementType, L"Images");
    auto classifierOutput = ResNetClassifier(imageInput, numOutputClasses, device, L"classifierOutput");

    auto labelsVar = InputVariable({ numOutputClasses }, labelStreamInfo.m_elementType, L"Labels");
    auto trainingLoss = CrossEntropyWithSoftmax(classifierOutput, labelsVar, L"lossFunction");
    auto prediction = ClassificationError(classifierOutput, labelsVar, L"predictionError");

    if (testSaveAndReLoad)
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto imageClassifier = Combine({ trainingLoss, prediction, classifierOutput }, L"ImageClassifier");
        SaveAndReloadModel<float>(imageClassifier, { &imageInput, &labelsVar, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        trainingLoss = trainingLossVar;
        prediction = predictionVar;
        classifierOutput = classifierOutputVar;
    }

    double learningRatePerSample = 0.0078125;
    Trainer trainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) });

    const size_t minibatchSize = 32;
    size_t numMinibatchesToTrain = 2000;
    size_t outputFrequencyInMinibatches = 20;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer.TrainMinibatch({ { imageInput, minibatchData[imageStreamInfo].m_data }, { labelsVar, minibatchData[labelStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TestCifarResnet()
{
#ifndef CPUONLY
    TrainResNetCifarClassifer(DeviceDescriptor::GPUDevice(0), true /*testSaveAndReLoad*/);
#endif
}
