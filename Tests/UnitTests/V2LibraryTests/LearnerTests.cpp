//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <initializer_list>

using namespace CNTK;
using namespace std;

namespace CNTK { namespace Test {

static const size_t maxMinibatchSize = 1000;

static const size_t maxNumAxes = 3;
static const size_t maxDimSize = 5;

template <typename ElementType>
void TestUpdate(LearnerPtr& learner, const NDShape& shape, size_t numMinibatches, const DeviceDescriptor& device)
{
    auto seed = (unsigned long) rng();
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (auto i = 0; i < numMinibatches; i++)
    { 
        for (auto& parameter : learner->Parameters())
        {
            gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -1.0, 1.0, seed + i, device);
        }

        learner->Update(gradientValues, 1, false);
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
    auto learner = SGDLearner(parameters, TrainingParameterPerSampleSchedule(0.4));
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestMomentumSGDLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    LearningRateSchedule learnigRateSchedule = LearningRateSchedule({ 3.0, 2.0, 1.0 }, numMinibatches);
    MomentumSchedule momentumValues = TrainingParameterPerSampleSchedule<double>( { { 1, 1.0 }, { 3, 0.1 }, { 10, 0.01 } }, 2 );
    auto learner = MomentumSGDLearner(parameters, learnigRateSchedule, momentumValues, unitGainMomentum);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
    FloatingPointCompare(learner->LearningRate(), 2.0, "Learner::LearningRate does not match expectation");
}

template <typename ElementType>
void TestNesterovLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    MomentumSchedule momentumValues = MomentumAsTimeConstantSchedule( { { 1, 1 }, { 3, 5 }, { 10, 25 } }, 100 );
    auto learner = NesterovLearner(parameters, LearningRateSchedule( { { 1, 0.5 }, { 10, 0.25 }, { 20, 0.125 } }, 3 ), momentumValues, unitGainMomentum);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdaGradLearner(parameters, LearningRateSchedule( std::vector<double>({ 0.5, 0.4, 0.3, 0.2, 0.1 }), 2), true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestFSAdaGradLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = FSAdaGradLearner(parameters, TrainingParameterPerSampleSchedule<double>({ 0.5 }), MomentumAsTimeConstantSchedule({ 10.0, 100.0, 1000.0 }), unitGainMomentum);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdamLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdamLearner(parameters, TrainingParameterPerSampleSchedule<double>({ 0.5 }), MomentumAsTimeConstantSchedule({ 10.0, 100.0, 1000.0 }), unitGainMomentum, MomentumSchedule(0.99, 1));
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdamaxLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdamLearner(parameters, TrainingParameterPerSampleSchedule<double>({ 0.5 }), MomentumAsTimeConstantSchedule({ 10.0, 100.0, 1000.0 }), unitGainMomentum, MomentumSchedule(0.99, 1), 1e-8, true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestRMSPropLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = RMSPropLearner(parameters, 
        LearningRateSchedule({ std::make_pair( 3, 0.7), std::make_pair( 1, 0.2 ) }),
        0.95, 1.2, 0.7, 10.0, 0.001);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestUniversalLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    ElementType lr = (ElementType) 0.06125;
    ParameterUpdateFunctor mysgd = [lr](Parameter p, Variable g) -> FunctionPtr 
    { 
        return Assign(p, Minus(p , ElementTimes(Constant::Scalar(lr), g))); 
    };
    auto learner = UniversalLearner(parameters, mysgd);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);

}

void TestTrainingParametersSchedule()
{
    LearningRateSchedule schedule1(0.5, 1);
    assert(schedule1.GetMinibatchSize() == 1);
    assert(schedule1[0] == 0.5);
    assert(schedule1[1] == 0.5);
    assert(schedule1[100] == 0.5);

    LearningRateSchedule schedule2(std::vector<double>({ 0.5 }), LearningRateSchedule::FullDataSweep, 1);
    assert(schedule2.GetMinibatchSize() == 1);
    assert(schedule2[0] == 0.5);
    assert(schedule2[10] == 0.5);
    assert(schedule2[100] == 0.5);

    LearningRateSchedule schedule3(std::vector<double>{ 0.5, 0.3, 0.3 }, LearningRateSchedule::FullDataSweep, 1);
    assert(schedule3.GetMinibatchSize() == 1);
    assert(schedule3[0] == 0.5);
    assert(schedule3[1] == 0.3);
    assert(schedule3[100] == 0.3);

    LearningRateSchedule schedule4(vector<double>{ 0.5 }, 10 ); // without vector<> gcc complains that conversion here is ambiguousS
    assert(schedule4.GetMinibatchSize() == LearningRateSchedule::IgnoredMinibatchSize);
    assert(schedule4[0] == 0.5);
    assert(schedule4[10] == 0.5);
    assert(schedule4[100] == 0.5);

    LearningRateSchedule schedule5{ std::vector<double>{ 0.5, 0.3, 0.2 }, 10 };
    assert(schedule5.GetMinibatchSize() == LearningRateSchedule::IgnoredMinibatchSize); //unspecified reference minibatch size is 0
    assert(schedule5[0] == 0.5);
    assert(schedule5[9] == 0.5);
    assert(schedule5[10] == 0.3);
    assert(schedule5[19] == 0.3);
    assert(schedule5[20] == 0.2);
    assert(schedule5[100] == 0.2);

    MomentumSchedule schedule6{ { make_pair(1, 0.5) } }; // without make_pair this is interpreted as a vector of doubles
    assert(schedule6.GetMinibatchSize() == MomentumSchedule::IgnoredMinibatchSize);
    assert(schedule6[0] == 0.5);
    assert(schedule6[10] == 0.5);
    assert(schedule6[100] == 0.5);

    LearningRateSchedule schedule7{ std::vector<std::pair<size_t, double>>{ { 1, 0.5 }, { 1, 0.3 }, { 1, 0.2 } } };
    assert(schedule7.GetMinibatchSize() == LearningRateSchedule::IgnoredMinibatchSize);
    assert(schedule7[0] == 0.5);
    assert(schedule7[1] == 0.3);
    assert(schedule7[2] == 0.2);
    assert(schedule7[100] == 0.2);

    MomentumSchedule schedule8{ std::vector<std::pair<size_t, double>>{ { 1, 0.5 }, { 1, 0.3 }, { 1, 0.2 } }, 10 };
    assert(schedule8.GetMinibatchSize() == MomentumSchedule::IgnoredMinibatchSize);
    assert(schedule8[0] == 0.5);
    assert(schedule8[9] == 0.5);
    assert(schedule8[10] == 0.3);
    assert(schedule8[19] == 0.3);
    assert(schedule8[20] == 0.2);
    assert(schedule8[100] == 0.2);

    LearningRateSchedule schedule9 = { std::vector<std::pair<size_t, double>>{ { 3, 0.5 }, { 2, 0.3 }, { 1, 0.2 } }, LearningRateSchedule::FullDataSweep, 1 };
    assert(schedule9.GetMinibatchSize() == 1);
    assert(schedule9[0] == 0.5);
    assert(schedule9[2] == 0.5);
    assert(schedule9[3] == 0.3);
    assert(schedule9[4] == 0.3);
    assert(schedule9[5] == 0.2);
    assert(schedule9[100] == 0.2);

    MomentumSchedule schedule10 = { std::vector<std::pair<size_t, double>>{ { 3, 0.5 }, { 2, 0.3 }, { 1, 0.2 } }, 10 };
    assert(schedule10.GetMinibatchSize() == MomentumSchedule::IgnoredMinibatchSize);
    assert(schedule10[0] == 0.5);
    assert(schedule10[29] == 0.5);
    assert(schedule10[30] == 0.3);
    assert(schedule10[49] == 0.3);
    assert(schedule10[50] == 0.2);
    assert(schedule10[100] == 0.2);

    MomentumSchedule schedule11 = MomentumAsTimeConstantSchedule(std::vector<double>{ 0.0, 1.0, 2.0 }, 10 );
    assert(schedule11.GetMinibatchSize() == 1);
    assert(schedule11[0] == 0.0);
    assert(schedule11[9] == 0.0);
    assert(schedule11[10] == exp(-1.0 / 1.0));
    assert(schedule11[19] == exp(-1.0 / 1.0));
    assert(schedule11[20] == exp(-1.0 / 2.0));
    assert(schedule11[30] == exp(-1.0 / 2.0));

    MomentumSchedule schedule12 = schedule11;
    assert(schedule12.GetMinibatchSize() == 1);
    assert(schedule12[0] == 0.0);
    assert(schedule12[9] == 0.0);
    assert(schedule12[10] == exp(-1.0 / 1.0));
    assert(schedule12[19] == exp(-1.0 / 1.0));
    assert(schedule12[20] == exp(-1.0 / 2.0));
    assert(schedule12[30] == exp(-1.0 / 2.0));

    MomentumSchedule schedule13 = MomentumAsTimeConstantSchedule(1);
    assert(schedule13.GetMinibatchSize() == 1);
    assert(schedule13[0] == exp(-1.0 / 1.0));
    assert(schedule13[1] == exp(-1.0 / 1.0));
    assert(schedule13[100] == exp(-1.0 / 1.0));

    MomentumSchedule schedule14 = MomentumAsTimeConstantSchedule(std::vector<double>{ 1.0, 2.0, 3.0 } );
    assert(schedule14.GetMinibatchSize() == 1);
    assert(schedule14[0] == exp(-1.0 / 1.0));
    assert(schedule14[1] == exp(-1.0 / 2.0));
    assert(schedule14[2] == exp(-1.0 / 3.0));
    assert(schedule14[100] == exp(-1.0 / 3.0));
    
    MomentumSchedule schedule15 = MomentumAsTimeConstantSchedule(std::vector<std::pair<size_t, double>>{ { 100, 7.0 }, { 10, 5.0 }, { 1, 3.0 } }, 100 );

    auto dict = schedule15.Serialize();

    TrainingParameterSchedule<double> schedule16 = TrainingParameterSchedule<double>::Deserialize(dict);
    assert(schedule16.GetMinibatchSize() == 1);
    assert(schedule16[0] == exp(-1.0 / 7.0));
    assert(schedule16[9999] == exp(-1.0 / 7.0));
    assert(schedule16[10000] == exp(-1.0 / 5.0));
    assert(schedule16[10999] == exp(-1.0 / 5.0));
    assert(schedule16[11000] == exp(-1.0 / 3.0));
    assert(schedule16[99999] == exp(-1.0 / 3.0));

    //test equals
    std::vector<std::pair<size_t, double>> vec1;
    vec1.push_back({ 1, 0.5 });
    vec1.push_back({ 1, 0.3 });
    LearningRateSchedule schedule17{ vec1, 100, 10 };
    LearningRateSchedule schedule18{ std::vector<std::pair<size_t, double>>{ { 1, 0.5 },{ 1, 0.3 },{ 1, 0.2 } }, 100, 10 };
    assert(schedule17 == schedule17);
    assert(!(schedule17 == schedule18));

    vec1.push_back({ 1, 0.2 });
    LearningRateSchedule schedule19{ vec1, 100, 10 };
    assert(schedule18 == schedule19);
}

void TestDefaultUnitGainGetterAndSetter()
{
    assert(DefaultUnitGainValue());

    SetDefaultUnitGainValue(false);
    assert(!DefaultUnitGainValue());

    SetDefaultUnitGainValue(true);
    assert(DefaultUnitGainValue());
}

void TestSweepBasedSchedule()
{
    DeviceDescriptor device = DeviceDescriptor::CPUDevice();
    auto schedule = LearningRateSchedule({ 1, 2, 3, 4, 5 }, LearningRateSchedule::FullDataSweep, 1);

    auto weights = Parameter({ 2 }, DataType::Float, 0, device);
    auto learner1 = SGDLearner({ weights }, schedule);
    assert(1 == learner1->LearningRate());

    
    for (auto i : {2, 3, 4, 5 })
    {
        std::vector<float> gradientValueVector(weights.Shape().TotalSize(), 0);
        auto gradientValue = MakeSharedObject<NDArrayView>(weights.Shape(), gradientValueVector);
        std::unordered_map<Parameter, NDArrayViewPtr> gradients{ { weights, gradientValue } };
        learner1->Update(gradients, 1, true);
        assert(i == learner1->LearningRate());
    }

    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTest_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });

    auto sweepSize = 603; // == wc -l SimpleDataTest_cntk_text.txt
    auto minibatchSize = 400; 
    auto featureStreamInfo = minibatchSource->StreamInfo(L"features");
    auto labelStreamInfo = minibatchSource->StreamInfo(L"labels");

    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");


    auto classifierOutput = FullyConnectedLinearLayer(input, numOutputClasses, device);
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");
    auto learner2 = SGDLearner(classifierOutput->Parameters(), schedule);
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { learner2 });

    for (auto i = 0; i <= 4000; i += minibatchSize)
    {
        auto sweepIndex1 = i / sweepSize;
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);

        if (minibatchData[featureStreamInfo].sweepEnd != minibatchData[labelStreamInfo].sweepEnd) {
            ReportFailure("TestSweepBasedSchedule failed: "
                "different streams have different end of sweep flag values.");
        }

        auto sweepIndex2 = (i + minibatchSize) / sweepSize;

        if ((sweepIndex1 != sweepIndex2) != minibatchData[labelStreamInfo].sweepEnd) {
            ReportFailure("TestSweepBasedSchedule failed: "
                "end of sweep flag value is different from expected.");
        }
       
        trainer->TrainMinibatch({ { input, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        auto expectedLR = std::min((sweepIndex2 + 1), 5);

        if (expectedLR != learner2->LearningRate()) {
            ReportFailure("TestSweepBasedSchedule failed: "
                "learning rate value is different from expected.");
        }
    }
}

struct LearnerSuiteFixture
{
    LearnerSuiteFixture()
        : unitGain{ true, false }
    {
        srand(1);
        if (ShouldRunOnCpu())
            devices.push_back(DeviceDescriptor::CPUDevice());
        if (ShouldRunOnGpu())
            devices.push_back(DeviceDescriptor::GPUDevice(0));

        numParameters = 1 + rand() % 5;
        numMinibatches = 1 + rand() % 5;
    }
    bool unitGain[2];
    vector<DeviceDescriptor> devices;
    int numParameters;
    int numMinibatches;
};

BOOST_FIXTURE_TEST_SUITE(LearnerSuite, LearnerSuiteFixture)

BOOST_AUTO_TEST_CASE(DefaultUnitGainGetterAndSetter)
{
    TestDefaultUnitGainGetterAndSetter();
}

BOOST_AUTO_TEST_CASE(SweepBasedSchedule)
{
    TestSweepBasedSchedule();
}

BOOST_AUTO_TEST_CASE(TrainingParametersSchedule)
{
    TestTrainingParametersSchedule();
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateSGDLearner)
{
    for (auto& device : devices)
    {
        TestSGDLearner<double>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateAdaGradLearner)
{
    for (auto& device : devices)
    {
        TestAdaGradLearner<double>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateRMSPropLearner)
{
    for (auto& device : devices)
    {
        TestRMSPropLearner<float>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateMomentumLearner)
{
    for (auto& device : devices)
    {
        for (auto gain : unitGain)
        {
            TestMomentumSGDLearner<float>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateNesterovLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestNesterovLearner<float>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateFSAdaGradLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestFSAdaGradLearner<double>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateAdamLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestAdamLearner<float>(numParameters, numMinibatches, gain, device);
            TestAdamLearner<double>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateUniversalLearner)
{
    for (auto& device : devices)
    {
        TestUniversalLearner<float>(numParameters, numMinibatches, device);
        TestUniversalLearner<double>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateAdamaxLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestAdamaxLearner<float>(numParameters, numMinibatches, gain, device);
            TestAdamaxLearner<double>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestResettingLearningRate)
{
    NDShape shape = { 1 };
    auto numSamples = 1; numParameters = 1, numMinibatches = 1;
    DeviceDescriptor device = DeviceDescriptor::CPUDevice();
    auto parameters = CreateParameters<float>(shape, numParameters, device);
    auto learner = SGDLearner(parameters, LearningRateSchedule(std::vector<double>{ 0.1, 1, 2, 3, 4, 5 }, numSamples, 1));
    BOOST_TEST(learner->LearningRate() == 0.1);
    for (int i = 1; i < 4; i++)
    {
        TestUpdate<float>(learner, shape, numMinibatches, device);
        BOOST_TEST(learner->LearningRate() == float(i));
    }

    learner->ResetLearningRate(LearningRateSchedule(std::vector<double>{ 9, 10, 20, 30, 40, 50 }, numSamples, 1));
    BOOST_TEST(learner->LearningRate() == 9.0);
    for (int i = 1; i < 4; i++)
    {
        TestUpdate<float>(learner, shape, numMinibatches, device);
        BOOST_TEST(learner->LearningRate() == float(i*10));
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}
