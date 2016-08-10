#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>

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

        learner->Update(gradientValues, rng() % maxMinibatchSize + 1);
    }
}

template <typename ElementType>
unordered_set<Parameter> CreateParameters(const NDShape& shape, size_t numParameters, const DeviceDescriptor& device)
{
    unordered_set<Parameter> parameters;
    for (int i = 0; i < numParameters; i++)
    {
        parameters.insert(
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
    map<size_t, double> momentums({ { 0, 1.0 }, { 3, 0.1 }, { 10, 0.01 } });
    auto learner = MomentumSGDLearner(parameters, {0.3, 0.2, 0.1}, momentums);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestNesterovLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner =  NesterovLearner(parameters, { { 0, 0.5 }, { 10, 0.25 }, { 20, 0.125 } }, 0.2);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdaGradLearner(parameters, {0.5, 0.4, 0.3, 0.2, 0.1}, true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestFSAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = FSAdaGradLearner(parameters, { 0.5 }, {0.05});
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestRMSPropLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = RMSPropLearner(parameters, { {0, 0.7}, {1, 0.2} }, 0.01, 0.02, 0.03, 0.1, 0.001, false);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

void LearnerTests()
{
    TestSGDLearner<double>(5, 3, DeviceDescriptor::CPUDevice());

#ifndef CPUONLY
    TestMomentumSGDLearner<float>(3, 11, DeviceDescriptor::GPUDevice(0));
    TestNesterovLearner<float>(1, 20, DeviceDescriptor::GPUDevice(0));
#else
    TestMomentumSGDLearner<float>(3, 11, DeviceDescriptor::CPUDevice());
    TestNesterovLearner<float>(1, 20, DeviceDescriptor::CPUDevice());
#endif
    
    TestAdaGradLearner<double>(2, 10, DeviceDescriptor::CPUDevice());
    
    // TODO: Both FSAdaGradLearner and TestRMSPropLearner try to resize smoothed gradients 
    // and throw
    // 'Microsoft::MSR::CNTK::CPUMatrix<double>::Resize: Cannot resize the matrix because it is a view.'
    // 
    // TestFSAdaGradLearner<double>(10, 2, DeviceDescriptor::CPUDevice());
    // TestRMSPropLearner<float>(3, 3, DeviceDescriptor::CPUDevice());
}