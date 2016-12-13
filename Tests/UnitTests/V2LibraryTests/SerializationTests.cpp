#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <vector>


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
#ifndef CPUONLY
    if (rng() % 2 == 0)
    {
        device = DeviceDescriptor::GPUDevice(0);
    }
#endif

    return (rng() % 2 == 0) ? 
        CreateNDArrayView<float>(numAxes, device) : CreateNDArrayView<double>(numAxes, device);
}

DictionaryValue CreateDictionaryValue(DictionaryValue::Type type, size_t depth)
{
    switch (type)
    {
    case DictionaryValue::Type::Bool:
        return DictionaryValue(!!(rng() % 2));
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
        auto checkpoint = learner1->GetCheckpointState();
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

     auto checkpoint1 = learner1->GetCheckpointState();
     auto checkpoint2 = learner2->GetCheckpointState();
    
    if (checkpoint1 != checkpoint2)
        throw std::runtime_error("TestLearnerSerialization: original and restored from a checkpoint learners diverge.");
}

void SerializationTests()
{
    TestDictionarySerialization(4);
    TestDictionarySerialization(8);
    TestDictionarySerialization(16);

    TestLearnerSerialization<float>(5, DeviceDescriptor::CPUDevice());
    TestLearnerSerialization<double>(10, DeviceDescriptor::CPUDevice());

#ifndef CPUONLY
    TestLearnerSerialization<float>(5, DeviceDescriptor::GPUDevice(0));
    TestLearnerSerialization<double>(10, DeviceDescriptor::GPUDevice(0));;
#endif
}
