//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
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

using namespace Microsoft::MSR::CNTK;

static const size_t maxNestingDepth = 10;
static const size_t maxNestedDictSize = 10;
static const size_t maxNestedVectorSize = 100;
static const size_t maxNDShapeSize = 100;

static const size_t maxNumAxes = 10;
static const size_t maxDimSize = 15;


static size_t keyCounter = 0;
static uniform_real_distribution<double> double_dist = uniform_real_distribution<double>();
static uniform_real_distribution<float> float_dist = uniform_real_distribution<float>();

static std::wstring tempFilePath = L"serialization.tmp";

DictionaryValue CreateDictionaryValue(DictionaryValue::Type, size_t);

DictionaryValue::Type GetType()
{
    return DictionaryValue::Type(rng() % (unsigned int) DictionaryValue::Type::NDArrayView + 1);
}

void AddKeyValuePair(Dictionary& dict, size_t depth)
{
    auto type = GetType();
    while (depth >= maxNestingDepth && 
           type == DictionaryValue::Type::Vector ||
           type == DictionaryValue::Type::Dictionary)
    {
        type = GetType();
    }
    dict[L"key" + to_wstring(keyCounter++)] = CreateDictionaryValue(type, depth);
}

Dictionary CreateDictionary(size_t size, size_t depth = 0) 
{
    Dictionary dict;
    for (auto i = 0; i < size; ++i)
    {
        AddKeyValuePair(dict, depth);
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

    if (IsGPUAvailable())
    {
        if (rng() % 2 == 0)
        {
            device = DeviceDescriptor::GPUDevice(0);
        }
    }

    return (rng() % 2 == 0) ? 
        CreateNDArrayView<float>(numAxes, device) : CreateNDArrayView<double>(numAxes, device);
}

DictionaryValue CreateDictionaryValue(DictionaryValue::Type type, size_t depth)
{
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
        return DictionaryValue(to_wstring(rng()));
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
        auto type = GetType();
        size_t size = rng() % maxNestedVectorSize + 1;
        vector<DictionaryValue> vector(size);
        for (auto i = 0; i < size; i++)
        {
            vector[i] = CreateDictionaryValue(type, depth + 1);
        }
        return DictionaryValue(vector);
    }
    case DictionaryValue::Type::Dictionary:
        return DictionaryValue(CreateDictionary(rng() % maxNestedDictSize  + 1, depth + 1));
    case DictionaryValue::Type::NDArrayView:
        return DictionaryValue(*(CreateNDArrayView()));
    default:
        NOT_IMPLEMENTED;
    }
}

void TestDictionarySerialization(size_t dictSize) 
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temporary test file 'serialization.tmp'.");

    Dictionary originalDict = CreateDictionary(dictSize);
    
    {
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << originalDict;
        stream.flush();
    }

    Dictionary deserializedDict;

    {
        fstream stream;
        OpenStream(stream, tempFilePath, true);
        stream >> deserializedDict;
    }
    
    if (originalDict != deserializedDict)
        throw std::runtime_error("TestDictionarySerialization: original and deserialized dictionaries are not identical.");
}

template <typename ElementType>
void TestLearnerSerialization(int numParameters, const DeviceDescriptor& device) 
{
    if ((_wunlink(tempFilePath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temporary test file 'serialization.tmp'.");

    NDShape shape = CreateShape(5, maxDimSize);

    vector<Parameter> parameters;
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (int i = 0; i < numParameters; i++)
    {
        Parameter parameter(NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, i, device), L"parameter_" + to_wstring(i));
        parameters.push_back(parameter);
        gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -0.5, 0.5, numParameters + i, device);
    }

    auto learner1 = SGDLearner(parameters, 0.05);
    
    learner1->Update(gradientValues, 1);

    {
        auto checkpoint = learner1->Serialize();
        fstream stream;
        OpenStream(stream, tempFilePath, false);
        stream << checkpoint;
        stream.flush();
    }

    auto learner2 = SGDLearner(parameters, 0.05);

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

     auto checkpoint1 = learner1->Serialize();
     auto checkpoint2 = learner2->Serialize();
    
    if (checkpoint1 != checkpoint2)
        throw std::runtime_error("TestLearnerSerialization: original and restored from a checkpoint learners diverge.");
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
                  static_cast<size_t>(PrimitiveOpType::Combine) == 44, 
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

FunctionPtr BuildFFClassifierNet(const Variable& inputVar, size_t numOutputClasses, const DeviceDescriptor& device, unsigned long seed = 1)
{
    Internal::SetFixedRandomSeed(seed);
    const size_t numHiddenLayers = 2;
    const size_t hiddenLayersDim = 32;
    auto nonLinearity = std::bind(Sigmoid, std::placeholders::_1, L"");
    return FullyConnectedFeedForwardClassifierNet(inputVar, numOutputClasses, hiddenLayersDim, numHiddenLayers, device, nonLinearity, L"classifierOutput");
}

FunctionPtr BuildLSTMClassifierNet(const Variable& inputVar, const size_t numOutputClasses, const DeviceDescriptor& device, unsigned long seed = 1)
{
    Internal::SetFixedRandomSeed(seed);
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    return LSTMSequenceClassiferNet(inputVar, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");
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
        throw std::runtime_error("TestFunctionSaveAndLoad: original and reloaded functions are not identical.");
    }
}

void TestFunctionsForEquality(const DeviceDescriptor& device)
{
    // TODO: add GPU version (need to reset cuda random generator each time a new function is created).
    assert(device.Type() == DeviceKind::CPU);

    auto inputVar = InputVariable({ 2 }, false, DataType::Float, L"features");

    auto f1 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 1);
    auto f2 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 1);
    if (!AreEqual(f1, f2))
    {
        throw std::runtime_error("TestFunctionsForEquality: two functions built with the same seed values are not identical.");
    }

    auto f3 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 2);
    auto f4 = BuildFFClassifierNet(inputVar, 3, device, /*seed*/ 3);
    if (AreEqual(f3, f4))
    {
        throw std::runtime_error("TestFunctionsForEquality: two functions built with different seed values are identical.");
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

Trainer BuildTrainer(const FunctionPtr& function, const Variable& labels, LearningRatesPerSample lr = 0.005, MomentumValuesPerSample m = 0.0)
{
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(function, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(function, labels, L"classificationError");
    auto learner = MomentumSGDLearner(function->Parameters(), lr, m);
    return Trainer(function, trainingLoss, prediction, { learner }); 
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

    trainer1.TrainMinibatch({ { classifierOutput1->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    auto classifierOutput2 = Function::Deserialize(model, device);

    if (AreEqual(classifierOutput1, classifierOutput2))
    {
        throw std::runtime_error("TestModelSerialization: reloaded function is still identical to the original after it was trained.");
    }

    for (int i = 0; i < 3; ++i)
    {
        Dictionary model = classifierOutput1->Serialize();

        auto classifierOutput2 = Function::Deserialize(model, device);

        if (!AreEqual(classifierOutput1, classifierOutput2))
        {
            throw std::runtime_error("TestModelSerialization: original and reloaded functions are not identical.");
        }
      
        Trainer trainer2 = BuildTrainer(classifierOutput2, labels);

        for (int j = 0; j < 3; ++j)
        {
            trainer1.TrainMinibatch({ { classifierOutput1->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
            trainer2.TrainMinibatch({ { classifierOutput2->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

            double mbLoss1 = trainer1.PreviousMinibatchLossAverage();
            double mbLoss2 = trainer2.PreviousMinibatchLossAverage();
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
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.005 }, { 2, 0.0025 }, { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);
    MomentumValuesPerSample momentumValues({ { 2, exp(-1.0 / 100) }, { 2, exp(-1.0 / 200) }, { 2, exp(-1.0 / 400) }, { 2, exp(-1.0 / 800) } }, actualMBSize);


    auto trainer1 = BuildTrainer(function1, labels, learningRateSchedule, momentumValues);
    auto trainer2 = BuildTrainer(function2, labels, learningRateSchedule, momentumValues);

    assert(AreEqual(function1, function2));

    trainer2.SaveCheckpoint(L"trainer.v2.checkpoint", false);
    trainer2.RestoreFromCheckpoint(L"trainer.v2.checkpoint", false);

    if (!AreEqual(function1, function2))
    {
        throw std::runtime_error("TestModelSerialization: reloaded function is not identical to the original.");
    }

    trainer1.TrainMinibatch({ { function1->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    if (AreEqual(function1, function2))
    {
        throw std::runtime_error("TestModelSerialization: reloaded function is still identical to the original after it was trained.");
    }

    trainer2.TrainMinibatch({ { function2->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    if (!AreEqual(function1, function2))
    {
        throw std::runtime_error("TestModelSerialization: reloaded function is not identical to the original.");
    }

    for (int i = 0; i < 3; ++i)
    {
        trainer2.SaveCheckpoint(L"trainer.v2.checkpoint", false);
        trainer2.RestoreFromCheckpoint(L"trainer.v2.checkpoint", false);

        if (!AreEqual(function1, function2))
        {
            throw std::runtime_error("TestModelSerialization: original and reloaded functions are not identical.");
        }
      
        for (int j = 0; j < 3; ++j)
        {
            trainer1.TrainMinibatch({ { function1->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
            trainer2.TrainMinibatch({ { function2->Arguments()[0], minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

            double mbLoss1 = trainer1.PreviousMinibatchLossAverage();
            double mbLoss2 = trainer2.PreviousMinibatchLossAverage();
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
    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 50;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { learner });

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    const wchar_t* modelFile = L"seq2seq.model";
    SaveAsLegacyModel(classifierOutput, modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto MB2Loss = trainer.PreviousMinibatchLossAverage();
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    classifierOutput->RestoreFromLegacyModel(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");

    classifierOutput->RestoreFromLegacyModel(modelFile);
    SaveAsLegacyModel(classifierOutput, modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);

    classifierOutput->RestoreFromLegacyModel(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");
}

void SerializationTests()
{
    fprintf(stderr, "\nSerializationTests..\n");

    TestDictionarySerialization(4);
    TestDictionarySerialization(8);
    TestDictionarySerialization(16);

    TestLearnerSerialization<float>(5, DeviceDescriptor::CPUDevice());
    TestLearnerSerialization<double>(10, DeviceDescriptor::CPUDevice());

    TestFunctionsForEquality(DeviceDescriptor::CPUDevice());
    TestFunctionSerialization(DeviceDescriptor::CPUDevice());
    TestModelSerializationDuringTraining(DeviceDescriptor::CPUDevice());
    
    TestCheckpointing(DeviceDescriptor::CPUDevice());
    TestLegacyModelSaving(DeviceDescriptor::CPUDevice());

    if (IsGPUAvailable())
    {
        TestLearnerSerialization<float>(5, DeviceDescriptor::GPUDevice(0));
        TestLearnerSerialization<double>(10, DeviceDescriptor::GPUDevice(0));
        TestFunctionSerialization(DeviceDescriptor::GPUDevice(0));
        TestModelSerializationDuringTraining(DeviceDescriptor::GPUDevice(0));
        TestCheckpointing(DeviceDescriptor::GPUDevice(0));
        TestLegacyModelSaving(DeviceDescriptor::GPUDevice(0));

    }

}
