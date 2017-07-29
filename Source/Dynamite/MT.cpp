//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "PlainTextDeseralizer.h"
#include "Layers.h"
//#include "Common.h"
//#include "TimerUtility.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;
using namespace std;

using namespace Dynamite;

const DeviceDescriptor device(DeviceDescriptor::GPUDevice(0));
//const DeviceDescriptor device(DeviceDescriptor::CPUDevice());
const size_t srcVocabSize = 2330;
const size_t tgtVocabSize = 2330;
const size_t embeddingDim = 128;
const size_t attentionDim = 128;
const size_t numEncoderLayers = 1;
const size_t encoderHiddenDim = 128;

auto BidirectionalLSTMEncoder(size_t numLayers, size_t encoderHiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    // TODO: change to LSTMs. Need to solve the problem of additional hidden state.
    //vector<BinaryModel> lstms(numLayers);
    //for (auto& lstm : lstms)
    //    lstm = Dynamite::Sequence::BiRecurrence(RNNStep(encoderHiddenDim, device), RNNStep(encoderHiddenDim, device));
    return UnaryModel({},
    {},
    [=](const Variable& x) -> Variable
    {
        return x;
    });
}

UnaryModel CreateModelFunction()
{
    auto embed = Embedding(embeddingDim, device);
    auto encoder = BidirectionalLSTMEncoder(numEncoderLayers, encoderHiddenDim, 0.8);
    //auto step = RNNStep(hiddenDim, device);
    //auto barrier = [](const Variable& x) -> Variable { return Barrier(x); };
    auto linear = Linear(tgtVocabSize, device);
    auto zero = Constant({ encoderHiddenDim }, 0.0f, device);
    return UnaryModel({},
    {
        { L"embed",   embed },
        { L"encoder", encoder },
        { L"linear",  linear }
    },
    [=](const Variable& x) -> Variable
    {
#if 1
        return x;
#else
        // 'x' is an entire sequence; last dimension is length
        let len = x.Shape().Dimensions().back();
        Variable state = zero;
        for (size_t t = 0; t < len; t++)
        {
            //if (t == 9)
            //    fprintf(stderr, "");
            auto xt = Index(x, t);
            xt = embed(xt);
            state = step(state, xt);
        }
        state = barrier(state); // for better batching
        return linear(state);
#endif
    });
}

void Train()
{
    //const size_t inputDim = 2000;
    //const size_t embeddingDim = 500;
    //const size_t hiddenDim = 250;
    //const size_t attentionDim = 20;
    //const size_t numOutputClasses = 5;
    //
    // dynamic model and criterion function
    auto model_fn = CreateModelFunction();
    auto criterion_fn = model_fn;// CreateCriterionFunction(model_fn);

    // data
    auto minibatchSourceConfig = MinibatchSourceConfig({ PlainTextDeserializer(
        {
            PlainTextStreamConfiguration(L"src", srcVocabSize, { L"d:/work/Karnak/sample-model/data/train.src" }, { L"d:/work/Karnak/sample-model/data/vocab.src", L"<s>", L"</s>", L"<unk>" }),
            PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { L"d:/work/Karnak/sample-model/data/train.tgt" }, { L"d:/work/Karnak/sample-model/data/vocab.tgt", L"<s>", L"</s>", L"<unk>" })
        })},
        /*randomize=*/true);
    minibatchSourceConfig.maxSamples = MinibatchSource::FullDataSweep;
    let minibatchSource = CreateCompositeMinibatchSource(minibatchSourceConfig);
    // BUGBUG (API): no way to specify MinibatchSource::FullDataSweep

    let parameters = model_fn.Parameters();
    auto learner = SGDLearner(parameters, LearningRatePerSampleSchedule(0.05));
}

int mt_main(int argc, char *argv[])
{
    argc; argv;
    try
    {
        Train();
        //Train(DeviceDescriptor::CPUDevice(), true);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
