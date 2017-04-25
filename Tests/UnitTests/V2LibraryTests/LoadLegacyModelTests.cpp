//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <functional>
#include <iostream>
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

void TestLoadLegacyModelWithPrecompute(const DeviceDescriptor& device)
{
    const size_t baseFeaturesDim = 363;
    const size_t numOutputClasses = 132;

    auto modelFuncPtr = Function::Load(L"cntkSpeechFF.dnn", device);

    auto FindVariableByName = [](const std::vector<Variable>& variables, const std::wstring& name) {
        for (size_t i = 0; i < variables.size(); ++i)
            if (variables[i].Name() == name)
                return variables[i];

        BOOST_ERROR("No output found with the given name");
        return Variable();
    };

    auto arguments = modelFuncPtr->Arguments();
    auto features = FindVariableByName(arguments, L"features");
    auto labels = FindVariableByName(arguments, L"labels");

    auto outputs = modelFuncPtr->Outputs();
    FunctionPtr prediction = FindVariableByName(outputs, L"PosteriorProb");
    FunctionPtr loss = FindVariableByName(outputs, L"CrossEntropyWithSoftmax");
    FunctionPtr eval = FindVariableByName(outputs, L"EvalClassificationError");

    auto config = GetHTKMinibatchSourceConfig(baseFeaturesDim, numOutputClasses);
    config.isFrameModeEnabled = true;
    auto minibatchSource = CreateCompositeMinibatchSource(config);

    const size_t minbatchSize = 256;
    size_t numMinibatches = 10;

    auto featureStreamInfo = minibatchSource->StreamInfo(L"features");
    auto labelStreamInfo = minibatchSource->StreamInfo(L"labels");

    LearningRatePerSampleSchedule learningRatePerSample = 0.000781;
    MomentumAsTimeConstantSchedule momentumTimeConstant = 6074;
    auto learner = MomentumSGDLearner(prediction->Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true);
    auto trainer = CreateTrainer(prediction, loss, eval, { learner });

    for (size_t i = 0; i < numMinibatches; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minbatchSize, device);
        if (minibatchData.empty())
            break;

        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, 1);
    }
}

BOOST_AUTO_TEST_SUITE(LoadLegacyModelSuite)

BOOST_AUTO_TEST_CASE(LoadLegacyModelWithPrecomputeInCPU)
{
    if (ShouldRunOnCpu())
        TestLoadLegacyModelWithPrecompute(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(LoadLegacyModelWithPrecomputeInGPU)
{
    if (ShouldRunOnGpu())
        TestLoadLegacyModelWithPrecompute(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_SUITE_END()

}}
