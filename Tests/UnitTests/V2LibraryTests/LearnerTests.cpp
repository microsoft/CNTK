//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <initializer_list>


using namespace CNTK;
using namespace std;

static const size_t maxMinibatchSize = 1000;

static const size_t maxNumAxes = 3;
static const size_t maxDimSize = 5;

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
    auto learner = SGDLearner(parameters, LearningRatePerSampleSchedule(0.4));
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestMomentumSGDLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    LearningRatePerMinibatchSchedule learnigRateSchedule = { { 3.0, 2.0, 1.0 }, numMinibatches };
    MomentumPerSampleSchedule momentumValues = { { { 1, 1.0 }, { 3, 0.1 }, { 10, 0.01 } }, 2 };
    auto learner = MomentumSGDLearner(parameters, learnigRateSchedule, momentumValues);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
    FloatingPointCompare(learner->LearningRate(), 2.0, "Learner::LearningRate does not match expectation");
}

template <typename ElementType>
void TestNesterovLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    MomentumAsTimeConstantSchedule momentumValues = { { { 1, 1 }, { 3, 5 }, { 10, 25 } }, 100 };
    auto learner = NesterovLearner(parameters, LearningRatePerMinibatchSchedule( { { 1, 0.5 }, { 10, 0.25 }, { 20, 0.125 } }, 3 ), momentumValues);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdaGradLearner(parameters, LearningRatePerMinibatchSchedule( {0.5, 0.4, 0.3, 0.2, 0.1}, 2 ), true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestFSAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdamLearner(parameters, LearningRatePerSampleSchedule({ 0.5 }), MomentumAsTimeConstantSchedule({ 10.0, 100.0, 1000.0 }));
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestRMSPropLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = RMSPropLearner(parameters, LearningRatePerMinibatchSchedule({ { 3, 0.7 }, { 1, 0.2 } }), 0.01, 0.02, 0.03, 0.1, 0.001);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

void TestTrainingParametersSchedule()
{
    VerifyException([]() {
        LearningRatePerMinibatchSchedule({ 3.0, 2.0, 1.0 }, LearningRateSchedule::EntireSweep);
    }, "Was able to create not-yet-implemented sweep-based schedule.");

    LearningRatePerSampleSchedule schedule1 = 0.5;
    assert(schedule1.Unit() == LearningRateSchedule::UnitType::Sample);
    assert(schedule1[0] == 0.5);
    assert(schedule1[1] == 0.5);
    assert(schedule1[100] == 0.5);

    LearningRatePerSampleSchedule schedule2 = { 0.5 };
    assert(schedule2.Unit() == LearningRateSchedule::UnitType::Sample);
    assert(schedule2[0] == 0.5);
    assert(schedule2[10] == 0.5);
    assert(schedule2[100] == 0.5);

    LearningRatePerSampleSchedule schedule3({ 0.5, 0.3, 0.3 });
    assert(schedule3.Unit() == LearningRateSchedule::UnitType::Sample);
    assert(schedule3[0] == 0.5);
    assert(schedule3[1] == 0.3);
    assert(schedule3[100] == 0.3);

    LearningRatePerMinibatchSchedule schedule4 = { vector<double>{ 0.5 }, 10 }; // without vector<> gcc complains that conversion here is ambiguousS
    assert(schedule4.Unit() == LearningRateSchedule::UnitType::Minibatch);
    assert(schedule4[0] == 0.5);
    assert(schedule4[10] == 0.5);
    assert(schedule4[100] == 0.5);

    LearningRatePerSampleSchedule schedule5 = { { 0.5, 0.3, 0.2 }, 10 };
    assert(schedule5.Unit() == LearningRateSchedule::UnitType::Sample);
    assert(schedule5[0] == 0.5);
    assert(schedule5[9] == 0.5);
    assert(schedule5[10] == 0.3);
    assert(schedule5[19] == 0.3);
    assert(schedule5[20] == 0.2);
    assert(schedule5[100] == 0.2);

    MomentumPerMinibatchSchedule schedule6 = { { make_pair(1, 0.5) } }; // without make_pair this is interpreted as a vector of doubles
    assert(schedule6.Unit() == MomentumSchedule::UnitType::Minibatch);
    assert(schedule6[0] == 0.5);
    assert(schedule6[10] == 0.5);
    assert(schedule6[100] == 0.5);

    LearningRatePerMinibatchSchedule schedule7 = { { { 1, 0.5 }, { 1, 0.3 }, { 1, 0.2 } } };
    assert(schedule7.Unit() == LearningRateSchedule::UnitType::Minibatch);
    assert(schedule7[0] == 0.5);
    assert(schedule7[1] == 0.3);
    assert(schedule7[2] == 0.2);
    assert(schedule7[100] == 0.2);

    MomentumPerMinibatchSchedule schedule8 = { { { 1, 0.5 }, { 1, 0.3 }, { 1, 0.2 } }, 10 };
    assert(schedule8.Unit() == MomentumSchedule::UnitType::Minibatch);
    assert(schedule8[0] == 0.5);
    assert(schedule8[9] == 0.5);
    assert(schedule8[10] == 0.3);
    assert(schedule8[19] == 0.3);
    assert(schedule8[20] == 0.2);
    assert(schedule8[100] == 0.2);

    LearningRatePerSampleSchedule schedule9 = { { { 3, 0.5 }, { 2, 0.3 }, { 1, 0.2 } } };
    assert(schedule9.Unit() == LearningRateSchedule::UnitType::Sample);
    assert(schedule9[0] == 0.5);
    assert(schedule9[2] == 0.5);
    assert(schedule9[3] == 0.3);
    assert(schedule9[4] == 0.3);
    assert(schedule9[5] == 0.2);
    assert(schedule9[100] == 0.2);

    MomentumPerMinibatchSchedule schedule10 = { { { 3, 0.5 }, { 2, 0.3 }, { 1, 0.2 } }, 10 };
    assert(schedule10.Unit() == MomentumSchedule::UnitType::Minibatch);
    assert(schedule10[0] == 0.5);
    assert(schedule10[29] == 0.5);
    assert(schedule10[30] == 0.3);
    assert(schedule10[49] == 0.3);
    assert(schedule10[50] == 0.2);
    assert(schedule10[100] == 0.2);

    MomentumAsTimeConstantSchedule schedule11 = { { 0.0, 1.0, 2.0 }, 10 };
    assert(schedule11.Unit() == MomentumAsTimeConstantSchedule::UnitType::Sample);
    assert(schedule11[0] == 0.0);
    assert(schedule11[9] == 0.0);
    assert(schedule11[10] == exp(-1.0 / 1.0));
    assert(schedule11[19] == exp(-1.0 / 1.0));
    assert(schedule11[20] == exp(-1.0 / 2.0));
    assert(schedule11[30] == exp(-1.0 / 2.0));

    MomentumAsTimeConstantSchedule schedule12 = schedule11;
    assert(schedule12.Unit() == MomentumAsTimeConstantSchedule::UnitType::Sample);
    assert(schedule12[0] == 0.0);
    assert(schedule12[9] == 0.0);
    assert(schedule12[10] == exp(-1.0 / 1.0));
    assert(schedule12[19] == exp(-1.0 / 1.0));
    assert(schedule12[20] == exp(-1.0 / 2.0));
    assert(schedule12[30] == exp(-1.0 / 2.0));

    MomentumAsTimeConstantSchedule schedule13 = 1;
    assert(schedule13.Unit() == MomentumAsTimeConstantSchedule::UnitType::Sample);
    assert(schedule13[0] == exp(-1.0 / 1.0));
    assert(schedule13[1] == exp(-1.0 / 1.0));
    assert(schedule13[100] == exp(-1.0 / 1.0));

    MomentumAsTimeConstantSchedule schedule14 = { { 1.0, 2.0, 3.0 } };
    assert(schedule14.Unit() == MomentumAsTimeConstantSchedule::UnitType::Sample);
    assert(schedule14[0] == exp(-1.0 / 1.0));
    assert(schedule14[1] == exp(-1.0 / 2.0));
    assert(schedule14[2] == exp(-1.0 / 3.0));
    assert(schedule14[100] == exp(-1.0 / 3.0));
    
    MomentumAsTimeConstantSchedule schedule15 = { { { 100, 7.0 }, { 10, 5.0 }, { 1, 3.0 } }, 100 };

    auto dict = schedule15.Serialize();

    TrainingParameterSchedule<double> schedule16 = TrainingParameterSchedule<double>::Deserialize(dict);
    assert(schedule16.Unit() == MomentumAsTimeConstantSchedule::UnitType::Sample);
    assert(schedule16[0] == exp(-1.0 / 7.0));
    assert(schedule16[9999] == exp(-1.0 / 7.0));
    assert(schedule16[10000] == exp(-1.0 / 5.0));
    assert(schedule16[10999] == exp(-1.0 / 5.0));
    assert(schedule16[11000] == exp(-1.0 / 3.0));
    assert(schedule16[99999] == exp(-1.0 / 3.0));
}


void LearnerTests()
{
    fprintf(stderr, "\nLearnerTests..\n");

    TestTrainingParametersSchedule();

    TestSGDLearner<double>(5, 3, DeviceDescriptor::CPUDevice());
    TestMomentumSGDLearner<float>(3, 4, DeviceDescriptor::CPUDevice());
    TestNesterovLearner<float>(1, 4, DeviceDescriptor::CPUDevice());
    TestAdaGradLearner<double>(2, 5, DeviceDescriptor::CPUDevice());
    TestFSAdaGradLearner<double>(10, 2, DeviceDescriptor::CPUDevice());
    TestRMSPropLearner<float>(3, 3, DeviceDescriptor::CPUDevice());

    if (IsGPUAvailable())
    {
        TestSGDLearner<double>(1, 1, DeviceDescriptor::CPUDevice());
        TestMomentumSGDLearner<float>(3, 5, DeviceDescriptor::GPUDevice(0));
        TestNesterovLearner<float>(1, 4, DeviceDescriptor::GPUDevice(0));
        TestAdaGradLearner<double>(1, 2, DeviceDescriptor::GPUDevice(0));
        TestFSAdaGradLearner<double>(2, 2, DeviceDescriptor::GPUDevice(0));
        TestRMSPropLearner<float>(3, 3, DeviceDescriptor::GPUDevice(0));
    }

}