//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <boost/random/uniform_real_distribution.hpp>
#include "CNTKLibrary.h"
#include "PrimitiveOpType.h"
#include "Common.h"
#include <string>
#include <random>
#include <vector>
#include <functional>
#include <iostream>

using namespace CNTK;
using namespace std;

static const size_t maxNDShapeSize = 10;
static const size_t maxNumAxes = 3;
static const size_t maxDimSize = 5;

namespace CNTK { namespace Test {

static size_t keyCounter = 0;
static boost::random::uniform_real_distribution<double> double_dist = boost::random::uniform_real_distribution<double>();
static boost::random::uniform_real_distribution<float> float_dist = boost::random::uniform_real_distribution<float>();
static std::wstring tempFilePath = L"serialization.tmp";

DictionaryValue CreateDictionaryValue(DictionaryValue::Type, size_t, size_t);

DictionaryValue::Type GetType()
{
    return DictionaryValue::Type(rng() % (unsigned int) DictionaryValue::Type::NDArrayView + 1);
}

void AddKeyValuePair(Dictionary& dict, size_t maxSize, size_t maxDepth)
{
    auto type = GetType();
    if (maxDepth <= 0)
    {
        while (type == DictionaryValue::Type::Vector || type == DictionaryValue::Type::Dictionary)
        {
            type = GetType();
        }
    }

    dict[L"key" + to_wstring(keyCounter++)] = CreateDictionaryValue(type, maxSize, maxDepth);
}

Dictionary CreateDictionary(size_t size, size_t depth)
{
    Dictionary dict;
    for (auto i = 0; i < size; ++i)
    {
        AddKeyValuePair(dict, size-1, depth-1);
    }

    return dict;
}

template <typename ElementType>
NDArrayViewPtr CreateNDArrayView(size_t numAxes, const DeviceDescriptor& device)
{
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rng() % maxDimSize) + 1;

    return NDArrayView::RandomUniform<ElementType>(viewShape, ElementType(-4.0), ElementType(19.0), 1, device);
}

NDArrayViewPtr CreateNDArrayView()
{
    auto numAxes = (rng() % maxNumAxes) + 1;
    auto device = DeviceDescriptor::CPUDevice();

    if (ShouldRunOnGpu())
    {
        if (rng() % 2 == 0)
        {
            device = DeviceDescriptor::GPUDevice(0);
        }
    }

    return (rng() % 2 == 0) ?
        CreateNDArrayView<float>(numAxes, device) : CreateNDArrayView<double>(numAxes, device);
}

DictionaryValue CreateDictionaryValue(DictionaryValue::Type type, size_t maxSize, size_t maxDepth)
{
    if (maxSize == 0) maxSize = 1;
    switch (type)
    {
    case DictionaryValue::Type::Bool:
        return DictionaryValue(!!(rng() % 2));
    case DictionaryValue::Type::Int:
        return DictionaryValue(rng());
    case DictionaryValue::Type::SizeT:
        return DictionaryValue(rng());
    case DictionaryValue::Type::Float:
        return DictionaryValue(float_dist(rng));
    case DictionaryValue::Type::Double:
        return DictionaryValue(double_dist(rng));
    case DictionaryValue::Type::String:
        return DictionaryValue(((rng() % 2 == 0) ?L"string_" : L"\u0441\u0442\u0440\u043E\u043A\u0430_") + to_wstring(rng()));
    case DictionaryValue::Type::Axis:
        return ((rng() % 2) == 0) ? DictionaryValue(Axis(0)) : DictionaryValue(Axis(L"newDynamicAxis_" + to_wstring(rng())));
    case DictionaryValue::Type::NDShape:
    {
        size_t size = rng() % maxNDShapeSize + 1;
        NDShape shape(size);
        for (auto i = 0; i < size; i++)
        {
            shape[i] = rng();
        }
        return DictionaryValue(shape);
    }
    case DictionaryValue::Type::Vector:
    {
        auto type2 = GetType();
        size_t size = rng() % maxSize + 1;
        vector<DictionaryValue> vector(size);
        for (auto i = 0; i < size; i++)
        {
            vector[i] = CreateDictionaryValue(type2, maxSize-1, maxDepth-1);
        }
        return DictionaryValue(vector);
    }
    case DictionaryValue::Type::Dictionary:
    {
        size_t size = rng() % maxSize + 1;
        return DictionaryValue(CreateDictionary(size, maxDepth));
    }
    case DictionaryValue::Type::NDArrayView:
        return DictionaryValue(*(CreateNDArrayView()));
    default:
        ReportFailure("Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__);
        return 0;
    }
}

void TestDictionarySerialization(size_t dictSize)
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
       BOOST_ERROR("Error deleting temporary test file 'serialization.tmp'.");

    Dictionary originalDict = CreateDictionary(dictSize, dictSize);
    {
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << originalDict;
        stream.flush();
    }

    Dictionary deserializedDict1;
    {
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> deserializedDict1;
    }

    if (originalDict != deserializedDict1)
        BOOST_ERROR("TestDictionarySerialization: original and deserialized dictionaries are not identical.");

    originalDict.Save(tempFilePath);
    Dictionary deserializedDict2 = Dictionary::Load(tempFilePath);

     if (originalDict != deserializedDict2)
        BOOST_ERROR("TestDictionarySerialization: original and deserialized dictionaries are not identical.");
}

template <typename ElementType>
void TestLargeValueSerialization(size_t numElements)
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
      BOOST_ERROR("Error deleting temporary test file 'serialization.tmp'.");

    DictionaryValue originalValue(*NDArrayView::RandomUniform<ElementType>({ numElements }, -0.5, 0.5, SentinelValueForAutoSelectRandomSeed, DeviceDescriptor::CPUDevice()));
    originalValue.Save(tempFilePath);

    DictionaryValue deserializedValue = DictionaryValue::Load(tempFilePath);

    if (originalValue != deserializedValue)
        BOOST_ERROR("TestLargeValueSerialization: original and deserialized values are not identical.");
}

template <typename ElementType>
void TestLearnerSerialization(int numParameters, const DeviceDescriptor& device)
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
       BOOST_ERROR("Error deleting temporary test file 'serialization.tmp'.");

    NDShape shape = CreateShape(5, maxDimSize);

    vector<Parameter> parameters;
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (int i = 0; i < numParameters; i++)
    {
        Parameter parameter(NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, i, device), L"parameter_" + to_wstring(i));
        parameters.push_back(parameter);
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, numParameters + i, device);
    }

    auto learner1 = SGDLearner(parameters, LearningRatePerSampleSchedule(0.05));

    learner1->Update(gradientValues, 1);

    {
        auto checkpoint = learner1->CreateCheckpoint();
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << checkpoint;
        stream.flush();
    }

    auto learner2 = SGDLearner(parameters, LearningRatePerSampleSchedule( 0.05));

    {
        Dictionary checkpoint;
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> checkpoint;
        learner2->RestoreFromCheckpoint(checkpoint);
    }

    int i = 0;
    for (auto parameter : parameters)
    {
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, 2*numParameters + i, device);
        i++;
    }

    learner1->Update(gradientValues, 1);
    learner2->Update(gradientValues, 1);

     auto checkpoint1 = learner1->CreateCheckpoint();
     auto checkpoint2 = learner2->CreateCheckpoint();

    if (checkpoint1 != checkpoint2)
        BOOST_ERROR("TestLearnerSerialization: original and restored from a checkpoint learners diverge.");
}


void CheckEnumValuesNotModified() {
    // During the model and checkpoint serialization, for all enum values we save corresponding 
    // integer values. For this reason, we need to make sure that enum values never change 
    // corresponding integer values (new enum values can only be appended to the end of the value
    // list and never inserted in the middle). 

    // The following list of asserts is APPEND ONLY. DO NOT CHANGE existing assert statements.

    static_assert(static_cast<size_t>(DataType::Unknown) == 0 &&
                  static_cast<size_t>(DataType::Float) == 1 &&
                  static_cast<size_t>(DataType::Double) == 2,
                  "DataType enum value was modified.");

    static_assert(static_cast<size_t>(VariableKind::Input) == 0 &&
                  static_cast<size_t>(VariableKind::Output) == 1 &&
                  static_cast<size_t>(VariableKind::Parameter) == 2 &&
                  static_cast<size_t>(VariableKind::Constant) == 3 &&
                  static_cast<size_t>(VariableKind::Placeholder) == 4,
                  "VariableKind enum value was modified.");

    static_assert(static_cast<size_t>(PrimitiveOpType::Negate) == 0 &&
                  static_cast<size_t>(PrimitiveOpType::Sigmoid) == 1 &&
                  static_cast<size_t>(PrimitiveOpType::Tanh) == 2 &&
                  static_cast<size_t>(PrimitiveOpType::ReLU) == 3 &&
                  static_cast<size_t>(PrimitiveOpType::Exp) == 4 &&
                  static_cast<size_t>(PrimitiveOpType::Log) == 5 &&
                  static_cast<size_t>(PrimitiveOpType::Sqrt) == 6 &&
                  static_cast<size_t>(PrimitiveOpType::Floor) == 7 &&
                  static_cast<size_t>(PrimitiveOpType::Abs) == 8 &&
                  static_cast<size_t>(PrimitiveOpType::Reciprocal) == 9 &&
                  static_cast<size_t>(PrimitiveOpType::Softmax) == 10 &&
                  static_cast<size_t>(PrimitiveOpType::Hardmax) == 11 &&
                  static_cast<size_t>(PrimitiveOpType::TransposeAxes) == 12 &&
                  static_cast<size_t>(PrimitiveOpType::Where) == 13 &&
                  static_cast<size_t>(PrimitiveOpType::Slice) == 14 &&
                  static_cast<size_t>(PrimitiveOpType::Dropout) == 15 &&
                  static_cast<size_t>(PrimitiveOpType::Reshape) == 16 &&
                  static_cast<size_t>(PrimitiveOpType::Pooling) == 17 &&
                  static_cast<size_t>(PrimitiveOpType::SumAll) == 18 &&
                  static_cast<size_t>(PrimitiveOpType::Plus) == 19  &&
                  static_cast<size_t>(PrimitiveOpType::Minus) == 20 &&
                  static_cast<size_t>(PrimitiveOpType::ElementTimes) == 21 &&
                  static_cast<size_t>(PrimitiveOpType::Equal) == 22 &&
                  static_cast<size_t>(PrimitiveOpType::NotEqual) == 23 &&
                  static_cast<size_t>(PrimitiveOpType::Less) == 24 &&
                  static_cast<size_t>(PrimitiveOpType::LessEqual) == 25 &&
                  static_cast<size_t>(PrimitiveOpType::Greater) == 26 &&
                  static_cast<size_t>(PrimitiveOpType::GreaterEqual) == 27 &&
                  static_cast<size_t>(PrimitiveOpType::PackedIndex) == 28 &&
                  static_cast<size_t>(PrimitiveOpType::GatherPacked) == 29 &&
                  static_cast<size_t>(PrimitiveOpType::ScatterPacked) == 30 &&
                  static_cast<size_t>(PrimitiveOpType::Times) == 31 &&
                  static_cast<size_t>(PrimitiveOpType::TransposeTimes) == 32 &&
                  static_cast<size_t>(PrimitiveOpType::Convolution) == 33 &&
                  static_cast<size_t>(PrimitiveOpType::SquaredError) == 34 &&
                  static_cast<size_t>(PrimitiveOpType::CrossEntropyWithSoftmax) == 35 &&
                  static_cast<size_t>(PrimitiveOpType::ClassificationError) == 36 &&
                  static_cast<size_t>(PrimitiveOpType::PastValue) == 37 &&
                  static_cast<size_t>(PrimitiveOpType::FutureValue) == 38 &&
                  static_cast<size_t>(PrimitiveOpType::ReduceElements) == 39 &&
                  static_cast<size_t>(PrimitiveOpType::BatchNormalization) == 40 &&
                  static_cast<size_t>(PrimitiveOpType::Clip) == 41 &&
                  static_cast<size_t>(PrimitiveOpType::Select) == 42 &&
                  static_cast<size_t>(PrimitiveOpType::Splice) == 43 &&
                  static_cast<size_t>(PrimitiveOpType::Combine) == 44 &&
                  static_cast<size_t>(PrimitiveOpType::RandomSample) == 45 &&
                  static_cast<size_t>(PrimitiveOpType::RandomSampleInclusionFrequency) == 46 &&
                  static_cast<size_t>(PrimitiveOpType::ROIPooling) == 47 &&
                  static_cast<size_t>(PrimitiveOpType::Logistic) == 48 &&
                  static_cast<size_t>(PrimitiveOpType::OptimizedRNNStack) == 49 &&
                  static_cast<size_t>(PrimitiveOpType::ReconcileDynamicAxis) == 50 &&
                  static_cast<size_t>(PrimitiveOpType::LogSoftmax) == 51 &&
                  static_cast<size_t>(PrimitiveOpType::LogPlus) == 52 &&
                  static_cast<size_t>(PrimitiveOpType::CosDistance) == 53 &&
                  static_cast<size_t>(PrimitiveOpType::Sin) == 54 &&
                  static_cast<size_t>(PrimitiveOpType::Cos) == 55 &&
                  static_cast<size_t>(PrimitiveOpType::Pass) == 56 &&
                  static_cast<size_t>(PrimitiveOpType::Block) == 57 &&
                  static_cast<size_t>(PrimitiveOpType::Unpooling) == 58 &&
                  static_cast<size_t>(PrimitiveOpType::LambdaRank) == 59 &&
                  static_cast<size_t>(PrimitiveOpType::NDCG) == 60 &&
                  static_cast<size_t>(PrimitiveOpType::EditDistanceError) == 61 &&
                  static_cast<size_t>(PrimitiveOpType::NoOp) == 62 &&
                  static_cast<size_t>(PrimitiveOpType::LabelsToGraph) == 63 &&
                  static_cast<size_t>(PrimitiveOpType::StopGradient) == 64 &&
                  static_cast<size_t>(PrimitiveOpType::ELU) == 65 &&
                  static_cast<size_t>(PrimitiveOpType::ForwardBackward) == 66 &&
                  static_cast<size_t>(PrimitiveOpType::CosDistanceWithNegativeSamples) == 67 &&
                  static_cast<size_t>(PrimitiveOpType::OneHot) == 68 &&
                  static_cast<size_t>(PrimitiveOpType::Pow) == 69 &&
                  static_cast<size_t>(PrimitiveOpType::ToSequence) == 70 &&
                  static_cast<size_t>(PrimitiveOpType::ToSequenceLike) == 71 &&
                  static_cast<size_t>(PrimitiveOpType::UnpackSequence) == 72 &&
                  static_cast<size_t>(PrimitiveOpType::Assign) == 73,
                  "PrimitiveOpType enum value was modified.");
}

std::shared_ptr<std::fstream> GetFstream(const std::wstring& filePath, bool readOnly)
{
        std::ios_base::openmode mode = std::ios_base::binary | (readOnly ? std::ios_base::in : std::ios_base::out);
#ifdef _MSC_VER
        return std::make_shared<std::fstream>(filePath, mode);
#else
        return std::make_shared<std::fstream>(wtocharpath(filePath.c_str()).c_str(), mode);
#endif
}

void ForceInitParameters(FunctionPtr f) 
{
    for (const auto& p : f->Parameters()) 
        UNUSED(p.Value()); 
}

FunctionPtr BuildFFClassifierNet(const Variable& inputVar, size_t numOutputClasses, const DeviceDescriptor& device, unsigned long seed = 1)
{
    Internal::ResetRandomSeed(seed);
    const size_t numHiddenLayers = 2;
    const size_t hiddenLayersDim = 32;
    auto nonLinearity = std::bind(Sigmoid, std::placeholders::_1, L"");
    auto f = FullyConnectedFeedForwardClassifierNet(inputVar, numOutputClasses, hiddenLayersDim, numHiddenLayers, device, nonLinearity,
        L"classifierOutput", SentinelValueForAutoSelectRandomSeed);
    // initialize the function parameters right away, using the current seed value.
    ForceInitParameters(f);
    return f;
}

FunctionPtr BuildLSTMClassifierNet(const Variable& inputVar, const size_t numOutputClasses, const DeviceDescriptor& device, unsigned long seed = 1)
{
    Internal::ResetRandomSeed(seed);
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    auto f = LSTMSequenceClassifierNet(inputVar, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, 
        L"classifierOutput", SentinelValueForAutoSelectRandomSeed);
    // initialize the function parameters right away, using the current seed value.
    ForceInitParameters(f);
    return f;
}

void TestFunctionSaveAndLoad(const FunctionPtr& function, const DeviceDescriptor& device)
{
    auto file = L"TestFunctionSaveAndLoad.out";

    {
        Dictionary model = function->Serialize();
        auto stream = GetFstream(file, false);
        // todo : as text.
        *stream << model;
        stream->flush();
    }

    Dictionary model;
    {
        auto stream = GetFstream(file, true);
        *stream >> model;
    }

    auto reloadedFunction = Function::Deserialize(model, device);

    if (!AreEqual(function, reloadedFunction))
    {
        BOOST_ERROR("TestFunctionSaveAndLoad: original and reloaded functions are not identical.");
    }
}

void TestFunctionsForEquality(const DeviceDescriptor& device)
{
    auto inputVar = InputVariable({ 2 }, false, DataType::Float, L"features");

    auto f1 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 1);
    auto f2 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 1);
    
    if (!AreEqual(f1, f2))
    {
        BOOST_ERROR("TestFunctionsForEquality: two functions built with the same seed values are not identical.");
    }

    auto f3 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 2);
    auto f4 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 3);

    if (AreEqual(f3, f4))
    {
        BOOST_ERROR("TestFunctionsForEquality: two functions built with different seed values are identical.");
    }
}

void TestFunctionSerialization(const DeviceDescriptor& device)
{
    const size_t inputDim = 20;
    auto inputVar = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"input_variable");

    TestFunctionSaveAndLoad(FullyConnectedLinearLayer(inputVar, 30, device), device);

    TestFunctionSaveAndLoad(BuildFFClassifierNet(inputVar, 5, device), device);

    TestFunctionSaveAndLoad(BuildLSTMClassifierNet(inputVar, 5, device), device);
}

TrainerPtr BuildTrainer(const FunctionPtr& function, const Variable& labels,
                     LearningRateSchedule lr = LearningRatePerSampleSchedule(0.005),
                     MomentumSchedule m = MomentumAsTimeConstantSchedule(0.0))
{
    auto trainingLoss = CrossEntropyWithSoftmax(function, labels, L"lossFunction");
    auto prediction = ClassificationError(function, labels, L"classificationError");
    auto learner = MomentumSGDLearner(function->Parameters(), lr, m, /*unitGainMomentum = */true);
    return CreateTrainer(function, trainingLoss, prediction, { learner });
}

void TestFunctionSerializationDuringTraining(const FunctionPtr& function, const Variable& labels, const MinibatchSourcePtr& minibatchSource, const DeviceDescriptor& device)
{
    auto classifierOutput1 = function;

    auto featureStreamInfo = minibatchSource->StreamInfo(classifierOutput1->Arguments()[0]);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);

    auto trainer1 = BuildTrainer(classifierOutput1, labels);

    Dictionary model = classifierOutput1->Serialize();

    trainer1->TrainMinibatch({ { classifierOutput1->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    auto classifierOutput2 = Function::Deserialize(model, device);

    if (AreEqual(classifierOutput1, classifierOutput2))
    {
        BOOST_ERROR("TestModelSerialization: reloaded function is still identical to the original after it was trained.");
    }

    for (int i = 0; i < 3; ++i)
    {
        Dictionary model2 = classifierOutput1->Serialize();

        auto classifierOutput3 = Function::Deserialize(model2, device);

        if (!AreEqual(classifierOutput1, classifierOutput3))
        {
            BOOST_ERROR("TestModelSerialization: original and reloaded functions are not identical.");
        }

        auto trainer2 = BuildTrainer(classifierOutput3, labels);

        for (int j = 0; j < 3; ++j)
        {
            trainer1->TrainMinibatch({ { classifierOutput1->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
            trainer2->TrainMinibatch({ { classifierOutput3->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

            double mbLoss1 = trainer1->PreviousMinibatchLossAverage();
            double mbLoss2 = trainer2->PreviousMinibatchLossAverage();
            FloatingPointCompare(mbLoss1, mbLoss2, "Post checkpoint restoration training loss does not match expectation");
        }
    }
}

void TestModelSerializationDuringTraining(const DeviceDescriptor& device)
{
    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";

    size_t inputDim = 784;
    size_t numOutputClasses = 10;
    auto features1 = InputVariable({ inputDim }, false /*isSparse*/, DataType::Float, featureStreamName);
    auto labels1 = InputVariable({ numOutputClasses }, DataType::Float, labelsStreamName);
    auto net1 = BuildFFClassifierNet(features1, numOutputClasses, device);
    auto minibatchSource1 = TextFormatMinibatchSource(L"Train-28x28_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } }, 1000, false);

    TestFunctionSerializationDuringTraining(net1, labels1, minibatchSource1, device);

    //TODO: find out why the test below fails and fix it.
    return;

    inputDim = 2000;
    numOutputClasses = 5;
    auto features2 = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featureStreamName);
    auto labels2 = InputVariable({ numOutputClasses }, DataType::Float, labelsStreamName, { Axis::DefaultBatchAxis() });
    auto net2 = BuildLSTMClassifierNet(features2, numOutputClasses, device);
    auto minibatchSource2 = TextFormatMinibatchSource(L"Train.ctf", { { featureStreamName, inputDim, true, L"x" }, {  labelsStreamName, numOutputClasses, false, L"y" } },  1000, false);

    TestFunctionSerializationDuringTraining(net2, labels2, minibatchSource2, device);
}


void TestTrainingWithCheckpointing(const FunctionPtr& function1, const FunctionPtr& function2, const Variable& labels, const MinibatchSourcePtr& minibatchSource, const DeviceDescriptor& device)
{
    auto featureStreamInfo = minibatchSource->StreamInfo(function1->Arguments()[0]);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 50;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].numberOfSamples;

    LearningRatePerSampleSchedule learningRateSchedule({ { 2, 0.005 }, { 2, 0.0025 }, { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);
    MomentumAsTimeConstantSchedule momentumValues({ { 2, 100 }, { 2, 200 }, { 2, 400 }, { 2, 800 } }, actualMBSize);


    auto trainer1 = BuildTrainer(function1, labels, learningRateSchedule, momentumValues);
    auto trainer2 = BuildTrainer(function2, labels, learningRateSchedule, momentumValues);

    assert(AreEqual(function1, function2));

    trainer2->SaveCheckpoint(L"trainer.v2.checkpoint");
    trainer2->RestoreFromCheckpoint(L"trainer.v2.checkpoint");

    if (!AreEqual(function1, function2))
    {
        BOOST_ERROR("TestModelSerialization: reloaded function is not identical to the original.");
    }

    trainer1->TrainMinibatch({ { function1->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    if (AreEqual(function1, function2))
    {
        BOOST_ERROR("TestModelSerialization: reloaded function is still identical to the original after it was trained.");
    }

    trainer2->TrainMinibatch({ { function2->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    if (!AreEqual(function1, function2))
    {
        BOOST_ERROR("TestModelSerialization: reloaded function is not identical to the original.");
    }

    for (int i = 0; i < 3; ++i)
    {
        trainer2->SaveCheckpoint(L"trainer.v2.checkpoint");
        trainer2->RestoreFromCheckpoint(L"trainer.v2.checkpoint");

        if (!AreEqual(function1, function2))
        {
            BOOST_ERROR("TestModelSerialization: original and reloaded functions are not identical.");
        }

        for (int j = 0; j < 3; ++j)
        {
            trainer1->TrainMinibatch({ { function1->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
            trainer2->TrainMinibatch({ { function2->Arguments()[0], minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

            double mbLoss1 = trainer1->PreviousMinibatchLossAverage();
            double mbLoss2 = trainer2->PreviousMinibatchLossAverage();
            FloatingPointCompare(mbLoss1, mbLoss2, "Post checkpoint restoration training loss does not match expectation");
        }
    }
}

void TestCheckpointing(const DeviceDescriptor& device)
{
    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";

    size_t inputDim = 784;
    size_t numOutputClasses = 10;
    auto features1 = InputVariable({ inputDim }, false /*isSparse*/, DataType::Float, featureStreamName);
    auto labels1 = InputVariable({ numOutputClasses }, DataType::Float, labelsStreamName);
    auto net1_1 = BuildFFClassifierNet(features1, numOutputClasses, device, 1);
    FunctionPtr net1_2;

    if (device.Type() == DeviceKind::GPU)
    {
        // TODO: instead of cloning here, reset curand generator to make sure that parameters are initialized to the same state.
        for (auto& p : net1_1->Parameters())
        {
            // make sure all parameters are initialized
            assert(p.Value() != nullptr);
        }
        net1_2 = net1_1->Clone();
    }
    else
    {
        net1_2 = BuildFFClassifierNet(features1, numOutputClasses, device, 1);
    }

    auto minibatchSource1 = TextFormatMinibatchSource(L"Train-28x28_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },  1000, false);

    TestTrainingWithCheckpointing(net1_1, net1_2, labels1, minibatchSource1, device);

    inputDim = 2000;
    numOutputClasses = 5;
    auto features2 = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featureStreamName);
    auto labels2 = InputVariable({ numOutputClasses }, DataType::Float, labelsStreamName, { Axis::DefaultBatchAxis() });
    auto net2_1 = BuildLSTMClassifierNet(features2, numOutputClasses, device, 1);
    FunctionPtr net2_2;

    if (device.Type() == DeviceKind::GPU)
    {
        // TODO: instead of cloning here, reset curand generator to make sure that parameters are initialized to the same state.
        for (auto& p : net2_1->Parameters())
        {
            // make sure all parameters are initialized
            assert(p.Value() != nullptr);
        }
        net2_2 = net2_1->Clone();
    }
    else
    {
        net2_2 = BuildLSTMClassifierNet(features2, numOutputClasses, device, 1);
    }

    auto minibatchSource2 = TextFormatMinibatchSource(L"Train.ctf", { { featureStreamName, inputDim, true, L"x" }, {  labelsStreamName, numOutputClasses, false, L"y" } }, 1000, false);

    TestTrainingWithCheckpointing(net2_1, net2_2, labels2, minibatchSource2, device);
}


void TestLegacyModelSaving(const DeviceDescriptor& device)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto classifierOutput = LSTMSequenceClassifierNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
    auto trainingLoss = CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = ClassificationError(classifierOutput, labels, L"classificationError");

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, MinibatchSource::FullDataSweep);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 50;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].numberOfSamples;

    LearningRatePerSampleSchedule learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { learner });

    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    const wchar_t* modelFile = L"seq2seq.legacy.model";
    Internal::SaveAsLegacyModel(classifierOutput, modelFile);

    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
    auto MB2Loss = trainer->PreviousMinibatchLossAverage();
    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    classifierOutput->Restore(modelFile);

    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
    auto postRestoreMB2Loss = trainer->PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");

    classifierOutput->Restore(modelFile);
    Internal::SaveAsLegacyModel(classifierOutput, modelFile);

    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    classifierOutput->Restore(modelFile);

    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
    postRestoreMB2Loss = trainer->PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");


    LearningRatePerSampleSchedule learningRateSchedule2({ { 0.04, 0.02, 0.01, 0.008, 0.004, 0.002, 0.001 } }, actualMBSize);
    MomentumAsTimeConstantSchedule momentumSchedule({ { 900, 800, 700, 600, 500 } }, actualMBSize);
    auto learner2 = AdamLearner(classifierOutput->Parameters(), learningRateSchedule, momentumSchedule, /*unitGainMomentum = */true);
    auto trainer2 = CreateTrainer(classifierOutput, trainingLoss, prediction, { learner });

    classifierOutput->Restore(modelFile);

    vector<double> expectedLoss;
    for (int i = 0; i < 10; i++)
    {
        trainer->SaveCheckpoint(L"trainer.checkpoint" + std::to_wstring(i));
        Internal::SaveAsLegacyModel(classifierOutput, modelFile + std::to_wstring(i));
        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        expectedLoss.push_back(trainer->PreviousMinibatchLossAverage());
}

    for (int i = 0; i < 10; i++)
    {
        trainer->RestoreFromCheckpoint(L"trainer.checkpoint" + std::to_wstring(i));
        classifierOutput->Restore(modelFile + std::to_wstring(i));
        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        double loss = trainer->PreviousMinibatchLossAverage();
        FloatingPointCompare(loss, expectedLoss[i], "Post checkpoint restoration training loss does not match expectation");
    }
}

void TestThatExceptionsAreRaisedForNonExistentPaths()
{
    VerifyException([]() {
        Function::Load(L"This.File.Does.Not.Exist");
    }, "Was able to open file 'This.File.Does.Not.Exist' for reading.");

    VerifyException([]() {
        Dictionary::Load(L"This.File.Does.Not.Exist");
    }, "Was able to open file 'This.File.Does.Not.Exist' for reading.");

    VerifyException([]() {
        Function::Load(L"This_Path_Does_Not_Exist/Models/model.file");
    }, "Was able to open file 'This_Path_Does_Not_Exist/Models/model.file' for reading.");


    VerifyException([]() {
        Dictionary::Load(L"This_Path_Does_Not_Exist/Dictionaries/dict.file");
    }, "Was able to open file 'This_Path_Does_Not_Exist/Dictionaries/dict.file' for reading.");

}

void TestLoadingAModelWithALoadBatchNormFunction() {
    {
        auto model = Function::Load(L"batch.norm.no.sample.count.v2.bin");
        if (model == nullptr) {
            ReportFailure("Failed to load a V2 model with a BatchNorm node that has only 5 inputs.");
        }
    }
    
    {
        // make sure, we can load legacy V1 model.
        auto model = Function::Load(L"batch.norm.no.sample.count.v1.bin");
        if (model == nullptr) {
            ReportFailure("Failed to load a legacy V1 model with a BatchNorm node.");
        }
    } 
}

void TestLoadingDictionariesGeneratedFromPresentPastAndFutureProtos()
{
    Dictionary presentDict, pastDict, futureDict;
    // load dictionaries from binary protobuf format and make sure we don't barf.
    {
        auto stream = GetFstream(L"v2.0.beta1.0.dictionary.proto.bin", true);
        *stream >> presentDict;
    }
    {
        // this file was generated with a proto that does not define NDArrayView message
        // and for Axis message only defines static axis index (no name and dynamic flag)
        auto stream = GetFstream(L"past.dictionary.proto.bin", true);
        *stream >> pastDict;

    }
    {
        // this file was generated with a proto that defines a new message,
        // adds a corresponding type to DictinaryValue, as well as a value to the oneof.
        // Additionally, the proto extends NDShape message with an additional string field.
        *GetFstream(L"future.dictionary.proto.bin", true) >> futureDict;
    }
    assert(presentDict.Size() > 0);
    assert(pastDict.Size() > 0);
    assert(futureDict.Size() > 0);
}


void TestCheckpointingWithStatefulNodes(const DeviceDescriptor& device)
{
    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";

    size_t inputDim = 784;
    size_t numOutputClasses = 10;
    auto features = InputVariable({ inputDim }, false /*isSparse*/, DataType::Float, featureStreamName);
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, labelsStreamName);
    //auto net = BuildFFClassifierNet(features, numOutputClasses, device, 1);
    auto net = Dropout(BuildFFClassifierNet(features, numOutputClasses, device, 1), 0.5);

    auto trainer = BuildTrainer(net, labels);

    const size_t minibatchSize = 50;
    const size_t epochSize = 150;
    auto minibatchSource = TextFormatMinibatchSource(L"Train-28x28_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },  epochSize, false);
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);

    vector<double> expectedLoss;
    for (int i = 0; i < epochSize / minibatchSize; i++)
    {
        trainer->SaveCheckpoint(L"stateful_nodes.model" + std::to_wstring(i));
        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        expectedLoss.push_back(trainer->PreviousMinibatchLossAverage());
    }

    for (int i = 0; i < epochSize / minibatchSize; i++)
    {
        trainer->RestoreFromCheckpoint(L"stateful_nodes.model" + std::to_wstring(i));
        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        double loss = trainer->PreviousMinibatchLossAverage();
        FloatingPointCompare(loss, expectedLoss[i], "Post checkpoint restoration training loss does not match expectation");
    }
}


void TestCheckpointingWithStatefulNodesAndExplicitSeeds(const DeviceDescriptor& device)
{
    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";

    size_t inputDim = 784;
    size_t numOutputClasses = 10;
    auto features = InputVariable({ inputDim }, false /*isSparse*/, DataType::Float, featureStreamName);
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, labelsStreamName);

    auto net1 = BuildFFClassifierNet(features, numOutputClasses, device, 1);
    auto net2 = net1->Clone(ParameterCloningMethod::Clone, { { features , features } });
    auto net3 = net1->Clone(ParameterCloningMethod::Clone, { { features , features } });
    
    auto trainer1 = BuildTrainer(Dropout(net1, 0.5, 123), labels);
    auto trainer2 = BuildTrainer(Dropout(net2, 0.5, 123), labels);
    auto trainer3 = BuildTrainer(Dropout(net3, 0.5, 321), labels);

    const size_t minibatchSize = 50;
    const size_t maxSamples = 150;
    auto minibatchSource = TextFormatMinibatchSource(L"Train-28x28_cntk_text.txt", { { featureStreamName, inputDim },{ labelsStreamName, numOutputClasses } }, 2 * maxSamples, false);
    
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    for (int i = 0; i < maxSamples; i+=minibatchSize)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        unordered_map<Variable, MinibatchData> minibatch = { { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } };

        trainer1->TrainMinibatch(minibatch, device);
        trainer2->TrainMinibatch(minibatch, device);
        trainer3->TrainMinibatch(minibatch, device);
        auto loss1 = trainer1->PreviousMinibatchLossAverage();
        auto loss2 = trainer2->PreviousMinibatchLossAverage();
        auto loss3 = trainer3->PreviousMinibatchLossAverage();
        FloatingPointCompare(loss1, loss2, "Training loss does not match expectation");
        BOOST_TEST((abs(loss1 - loss2) <= abs(loss2 - loss3)));
    }

    trainer1->SaveCheckpoint(L"seeded_stateful_nodes.model");
    auto state = minibatchSource->GetCheckpointState();

    vector<double> expectedLoss;
    for (int i = 0; i < maxSamples; i += minibatchSize)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        unordered_map<Variable, MinibatchData> minibatch = { { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } };

        trainer1->TrainMinibatch(minibatch, device);
        expectedLoss.push_back(trainer1->PreviousMinibatchLossAverage());
    }

    trainer1->RestoreFromCheckpoint(L"seeded_stateful_nodes.model");
    minibatchSource->RestoreFromCheckpoint(state);

    for (int i = 0; i*minibatchSize < maxSamples; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        unordered_map<Variable, MinibatchData> minibatch = { { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } };

        trainer1->TrainMinibatch(minibatch, device);
        double loss = trainer1->PreviousMinibatchLossAverage();
        FloatingPointCompare(loss, expectedLoss[i], "Post checkpoint restoration training loss does not match expectation");
    }
}

void TestLoadingModelFromMemoryBuffer()
{
    ifstream modelFileStream("batch.norm.no.sample.count.v2.bin", ifstream::binary);
    modelFileStream.seekg(0, modelFileStream.end);
    size_t length = modelFileStream.tellg();
    modelFileStream.seekg(0, modelFileStream.beg);
    char* modelBuffer = new char[length];
    modelFileStream.read(modelBuffer, length);

    auto model = Function::Load(modelBuffer, length);
    if (model == nullptr) {
        ReportFailure("Failed to load a V2 model from memory buffer.");
    }
    delete[] modelBuffer;
}

void TestLoadingModelFromMemoryBufferWithException()
{
    ifstream modelFileStream("batch.norm.no.sample.count.v1.bin", ifstream::binary);
    modelFileStream.seekg(0, modelFileStream.end);
    size_t length = modelFileStream.tellg();
    modelFileStream.seekg(0, modelFileStream.beg);
    char* modelBuffer = new char[length];
    modelFileStream.read(modelBuffer, length);

    VerifyException([&length]() {
        Function::Load(nullptr, length);
    }, "Was able to load model from nullptr memory buffer.");

    VerifyException([&modelBuffer]() {
        Function::Load(modelBuffer, 0);
    }, "Was able to load model from nullptr memory buffer.");

    VerifyException([&modelBuffer, &length]() {
        Function::Load(modelBuffer, length);
    }, "Was able to load legacy model from memory buffer."); 
    delete[] modelBuffer;
}

BOOST_AUTO_TEST_SUITE(SerializationSuite)

BOOST_AUTO_TEST_CASE(LoadingModelFromMemoryBuffer)
{
    TestLoadingModelFromMemoryBuffer();
}

BOOST_AUTO_TEST_CASE(LoadingModelFromMemoryBufferWithException)
{
    TestLoadingModelFromMemoryBufferWithException();
}

BOOST_AUTO_TEST_CASE(LoadingAModelWithALoadBatchNormFunction)
{
    TestLoadingAModelWithALoadBatchNormFunction();
}

BOOST_AUTO_TEST_CASE(ExceptionsAreRaisedForNonExistentPaths)
{
    TestThatExceptionsAreRaisedForNonExistentPaths();
}

BOOST_AUTO_TEST_CASE(DictionarySerialization)
{
    TestDictionarySerialization(1);
    TestDictionarySerialization(2);
    TestDictionarySerialization(4);
    TestDictionarySerialization(8);
    TestDictionarySerialization(16);
}

BOOST_AUTO_TEST_CASE(LoadingDictionariesGeneratedFromPresentPastAndFutureProtos)
{
    TestLoadingDictionariesGeneratedFromPresentPastAndFutureProtos();
}

BOOST_AUTO_TEST_CASE(LargeValueSerialization)
{
    TestLargeValueSerialization<double>(10000000);
    TestLargeValueSerialization<float>(100000000);
}

BOOST_AUTO_TEST_CASE(LargeLernerSerializationInCpu)
{
    TestLearnerSerialization<float>(5, DeviceDescriptor::CPUDevice());
    TestLearnerSerialization<double>(10, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(FunctionsForEquality)
{
    TestFunctionsForEquality(DeviceDescriptor::CPUDevice());
    if (ShouldRunOnGpu())
    {
        TestFunctionsForEquality(DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(FunctionSerializationInCPU)
{
    TestFunctionSerialization(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ModelSerializationDuringTrainingInCPU)
{
    TestModelSerializationDuringTraining(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CheckpointingInCPU)
{
    TestCheckpointing(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(LegacyModelSavingInCPU)
{
    TestLegacyModelSaving(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CheckpointingWithStatefulNodesInCPU)
{
    TestCheckpointingWithStatefulNodes(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(LearnerSerializationInGPU)
{
    if (ShouldRunOnGpu())
    {
        TestLearnerSerialization<float>(5, DeviceDescriptor::GPUDevice(0));
        TestLearnerSerialization<double>(10, DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(LearnerSerializationBackcompat)
{
    auto device = DeviceDescriptor::CPUDevice();
    auto net = BuildLSTMClassifierNet(InputVariable({ 3 }, DataType::Float), 2, device);
    auto learner = MomentumSGDLearner(net->Parameters(), LearningRatePerSampleSchedule(0.005),
        MomentumAsTimeConstantSchedule(900), /*unitGainMomentum = */true);

    BOOST_ASSERT(learner->TotalNumberOfSamplesSeen() == 0);

    // this checkpoint contains smoothed gradients serialized as a dict, not
    // a vector (the current format).
    auto checkpoint = Dictionary::Load(L"learner.checkpoint.backcompat.bin");
    learner->RestoreFromCheckpoint(checkpoint);
    BOOST_TEST(learner->TotalNumberOfSamplesSeen() > 0);
}

BOOST_AUTO_TEST_CASE(FunctionSerializationInGPU)
{
    if (ShouldRunOnGpu())
    {
        TestFunctionSerialization(DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(ModelSerializationDuringTrainingInGPU)
{
    if (ShouldRunOnGpu())
    {
        TestModelSerializationDuringTraining(DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CheckpointingInGPU)
{
    if (ShouldRunOnGpu())
        TestCheckpointing(DeviceDescriptor::GPUDevice(0));
}


BOOST_AUTO_TEST_CASE(LegacyModelSavingInGPU)
{
    if (ShouldRunOnGpu())
        TestLegacyModelSaving(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(CheckpointingWithStatefulNodesInGPU)
{
    if (ShouldRunOnGpu())
        TestCheckpointingWithStatefulNodes(DeviceDescriptor::GPUDevice(0));
}


BOOST_AUTO_TEST_CASE(CheckpointingWithStatefulNodesAndExplicitSeedsOnCPU)
{
     TestCheckpointingWithStatefulNodesAndExplicitSeeds(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CheckpointingWithStatefulNodesAndExplicitSeedsOnGPU)
{
    if (ShouldRunOnGpu())
        TestCheckpointingWithStatefulNodesAndExplicitSeeds(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_SUITE_END()

}}
