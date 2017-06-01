//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
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
    cropTransformConfig[L"cropType"] = L"RandomSide";
    cropTransformConfig[L"sideRatio"] = L"0.8";
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

    MinibatchSourceConfig config({ deserializerConfiguration });
    config.maxSamples = epochSize;

    return CreateCompositeMinibatchSource(config);
}

Constant GetProjectionMap(size_t outputDim, size_t inputDim, const DeviceDescriptor& device)
{
    if (inputDim > outputDim)
        throw std::runtime_error("Can only project from lower to higher dimensionality");

    std::vector<float> projectionMapValues(inputDim * outputDim, 0);
    for (size_t i = 0; i < inputDim; ++i)
        projectionMapValues[(i * inputDim) + i] = 1.0f;

    auto projectionMap = MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ 1, 1, inputDim, outputDim }), device);
    projectionMap->CopyFrom(NDArrayView(NDShape({ 1, 1, inputDim, outputDim }), projectionMapValues));

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
    auto conv1 = ConvBNReLULayer(input, cMap1, kernelWidth, kernelHeight, 1, 1, conv1WScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

    auto rn1_1 = ResNetNode2(conv1, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
    auto rn1_2 = ResNetNode2(rn1_1, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);
    auto rn1_3 = ResNetNode2(rn1_2, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

    size_t cMap2 = 32;
    auto rn2_1_wProj = GetProjectionMap(cMap2, cMap1, device);
    auto rn2_1 = ResNetNode2Inc(rn1_3, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn2_1_wProj, device);
    auto rn2_2 = ResNetNode2(rn2_1, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
    auto rn2_3 = ResNetNode2(rn2_2, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

    size_t cMap3 = 64;
    auto rn3_1_wProj = GetProjectionMap(cMap3, cMap2, device);
    auto rn3_1 = ResNetNode2Inc(rn2_3, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn3_1_wProj, device);
    auto rn3_2 = ResNetNode2(rn3_1, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
    auto rn3_3 = ResNetNode2(rn3_2, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

    // Global average pooling
    size_t poolW = 8;
    size_t poolH = 8;
    size_t poolhStride = 1;
    size_t poolvStride = 1;
    auto pool = Pooling(rn3_3, PoolingType::Average, { poolW, poolH, 1 }, { poolhStride, poolvStride, 1 });

    // Output DNN layer
    auto outTimesParams = Parameter({ numOutputClasses, 1, 1, cMap3 }, DataType::Float, GlorotUniformInitializer(fc1WScale, 1, 0), device);
    auto outBiasParams = Parameter({ numOutputClasses }, (float)fc1BValue, device);

    return Plus(Times(outTimesParams, pool), outBiasParams, outputName);
}

void TrainResNetCifarClassifier(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    auto minibatchSource = CreateCifarMinibatchSource(SIZE_MAX);
    auto imageStreamInfo = minibatchSource->StreamInfo(L"features");
    auto labelStreamInfo = minibatchSource->StreamInfo(L"labels");

    auto inputImageShape = imageStreamInfo.m_sampleLayout;
    const size_t numOutputClasses = labelStreamInfo.m_sampleLayout[0];

    auto imageInputName = L"Images";
    auto imageInput = InputVariable(inputImageShape, imageStreamInfo.m_elementType, imageInputName);
    auto classifierOutput = ResNetClassifier(imageInput, numOutputClasses, device, L"classifierOutput");

    auto labelsInputName = L"Labels";
    auto labelsVar = InputVariable({ numOutputClasses }, labelStreamInfo.m_elementType, labelsInputName);
    auto trainingLoss = CrossEntropyWithSoftmax(classifierOutput, labelsVar, L"lossFunction");
    auto prediction = ClassificationError(classifierOutput, labelsVar, 5, L"predictionError");

    if (testSaveAndReLoad)
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto imageClassifier = Combine({ trainingLoss, prediction, classifierOutput }, L"ImageClassifier");
        SaveAndReloadModel<float>(imageClassifier, { &imageInput, &labelsVar, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        // Make sure that the names of the input variables were properly restored
        if ((imageInput.Name() != imageInputName) || (labelsVar.Name() != labelsInputName))
            throw std::runtime_error("One or more input variable names were not properly restored after save and load");

        trainingLoss = trainingLossVar;
        prediction = predictionVar;
        classifierOutput = classifierOutputVar;
    }

    LearningRatePerSampleSchedule learningRatePerSample = 0.0078125;
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) });

    const size_t minibatchSize = 32;
    size_t numMinibatchesToTrain = 2000;
    size_t outputFrequencyInMinibatches = 20;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer->TrainMinibatch({ { imageInput, minibatchData[imageStreamInfo] }, { labelsVar, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TrainCifarResnet()
{
    fprintf(stderr, "\nTrainCifarResnet..\n");

    TrainResNetCifarClassifier(DeviceDescriptor::GPUDevice(0), true /*testSaveAndReLoad*/);
}
