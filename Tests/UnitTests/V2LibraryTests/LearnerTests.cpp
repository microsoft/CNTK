#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <initializer_list>


using namespace CNTK;
using namespace std;

static const size_t maxMinibatchSize = 1000;

static const size_t maxNumAxes = 5;
static const size_t maxDimSize = 10;

template <typename ElementType>
void TestUpdate(LearnerPtr& learner, NDShape& shape, size_t numMinibatches, const DeviceDescriptor& device)
{
    auto seed = (unsigned long) rng();
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (auto i = 0; i < numMinibatches; i++)
    { 
        for (auto& parameter : learner->Parameters())
        {
            gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -1.0, 1.0, seed + i, device);
        }

        learner->Update(gradientValues, 1);
    }
}

template <typename ElementType>
vector<Parameter> CreateParameters(const NDShape& shape, size_t numParameters, const DeviceDescriptor& device)
{
    vector<Parameter> parameters;
    for (int i = 0; i < numParameters; i++)
    {
        parameters.push_back(
            Parameter(NDArrayView::RandomUniform<ElementType>(shape, -1.0, 1.0, i, device), 
                      L"parameter_" + to_wstring(i)));
    }
    return parameters;
}
  
template <typename ElementType>
void TestSGDLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = SGDLearner(parameters, 0.4);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestMomentumSGDLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    MomentumsPerSample momentums({ { 1, 1.0 }, { 3, 0.1 }, { 10, 0.01 } }, 2);
    auto learner = MomentumSGDLearner(parameters, vector<double>{0.3, 0.2, 0.1}, momentums);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestNesterovLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = NesterovLearner(parameters, LearningRatesPerSample({ { 1, 0.5 }, { 10, 0.25 }, { 20, 0.125 } }, 3 ), 0.2);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdaGradLearner(parameters, { vector<double>{0.5, 0.4, 0.3, 0.2, 0.1}, 2 }, true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestFSAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = FSAdaGradLearner(parameters, vector<double>{ 0.5 }, vector<double>{0.05});
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestRMSPropLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = RMSPropLearner(parameters, { { 3, 0.7 }, { 1, 0.2 } }, 0.01, 0.02, 0.03, 0.1, 0.001 );
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

void TestTrainingParametersSchedule() 
{
    LearningRatesPerSample schedule1 = 0.5;
    assert(schedule1[0] == 0.5);
    assert(schedule1[1] == 0.5);
    assert(schedule1[100] == 0.5);

    LearningRatesPerSample schedule2 = vector<double>{ 0.5 };
    assert(schedule2[0] == 0.5);
    assert(schedule2[10] == 0.5);
    assert(schedule2[100] == 0.5);

    LearningRatesPerSample schedule3 = vector<double>{ 0.5, 0.3 };
    assert(schedule3[0] == 0.5);
    assert(schedule3[1] == 0.3);
    assert(schedule3[100] == 0.3);

    LearningRatesPerSample schedule4 = { vector<double>{ 0.5 }, 10 };
    assert(schedule4[0] == 0.5);
    assert(schedule4[10] == 0.5);
    assert(schedule4[100] == 0.5);

    LearningRatesPerSample schedule5 = { vector<double>{ 0.5, 0.3, 0.2 }, 10 };
    assert(schedule5[0] == 0.5);
    assert(schedule5[9] == 0.5);
    assert(schedule5[10] == 0.3);
    assert(schedule5[19] == 0.3);
    assert(schedule5[20] == 0.2);
    assert(schedule5[100] == 0.2);

    LearningRatesPerSample schedule6 = { {1, 0.5} };
    assert(schedule6[0] == 0.5);
    assert(schedule6[10] == 0.5);
    assert(schedule6[100] == 0.5);

    LearningRatesPerSample schedule7 = { { 1, 0.5 }, { 1, 0.3 }, {1, 0.2} };
    assert(schedule7[0] == 0.5);
    assert(schedule7[1] == 0.3);
    assert(schedule7[2] == 0.2);
    assert(schedule7[100] == 0.2);

    LearningRatesPerSample schedule8( { { 1, 0.5 }, { 1, 0.3 }, { 1, 0.2 } }, 10 );
    assert(schedule8[0] == 0.5);
    assert(schedule8[9] == 0.5);
    assert(schedule8[10] == 0.3);
    assert(schedule8[19] == 0.3);
    assert(schedule8[20] == 0.2);
    assert(schedule8[100] == 0.2);

    LearningRatesPerSample schedule9 = { { 3, 0.5 }, { 2, 0.3 }, {1, 0.2} };
    assert(schedule9[0] == 0.5);
    assert(schedule9[2] == 0.5);
    assert(schedule9[3] == 0.3);
    assert(schedule9[4] == 0.3);
    assert(schedule9[5] == 0.2);
    assert(schedule9[100] == 0.2);

    LearningRatesPerSample schedule10( { { 3, 0.5 }, { 2, 0.3 }, { 1, 0.2 } }, 10 );
    assert(schedule10[0] == 0.5);
    assert(schedule10[29] == 0.5);
    assert(schedule10[30] == 0.3);
    assert(schedule10[49] == 0.3);
    assert(schedule10[50] == 0.2);
    assert(schedule10[100] == 0.2);
}


void LearnerTests()
{
    TestTrainingParametersSchedule();

    TestSGDLearner<double>(5, 3, DeviceDescriptor::CPUDevice());

#ifndef CPUONLY
    TestMomentumSGDLearner<float>(3, 11, DeviceDescriptor::GPUDevice(0));
    TestNesterovLearner<float>(1, 20, DeviceDescriptor::GPUDevice(0));
#else
    TestMomentumSGDLearner<float>(3, 11, DeviceDescriptor::CPUDevice());
    TestNesterovLearner<float>(1, 20, DeviceDescriptor::CPUDevice());
#endif
    
    TestAdaGradLearner<double>(2, 10, DeviceDescriptor::CPUDevice());
    
    TestFSAdaGradLearner<double>(10, 2, DeviceDescriptor::CPUDevice());
    TestRMSPropLearner<float>(3, 3, DeviceDescriptor::CPUDevice());
}