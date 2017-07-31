//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
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
    vector<UnarySequenceModel> layers;
    for (size_t i = 0; i < numLayers; i++)
        layers.push_back(Dynamite::Sequence::BiRecurrence(RNNStep(encoderHiddenDim, device), Constant({ encoderHiddenDim }, 0.0f, device),
                                                          RNNStep(encoderHiddenDim, device), Constant({ encoderHiddenDim }, 0.0f, device)));
    vector<vector<Variable>> hs(2); // we need max. 2 buffers for the stack
    return UnarySequenceModel(vector<ModelParametersPtr>(layers.begin(), layers.end()),
    [=](vector<Variable>& res, const vector<Variable>& x) mutable
    {
        for (size_t i = 0; i < numLayers; i++)
        {
            const vector<Variable>& in = (i == 0) ? x : hs[i % 2];
            vector<Variable>& out = (i == numLayers - 1) ? res : hs[(i+1) % 2];
            layers[i](out, in);
        }
        hs[0].clear(); hs[1].clear();
    });
}

BinarySequenceModel CreateModelFunction()
{
    auto embed = Embedding(embeddingDim, device);
    auto encode = BidirectionalLSTMEncoder(numEncoderLayers, encoderHiddenDim, 0.8);
    //auto step = RNNStep(hiddenDim, device);
    //auto barrier = [](const Variable& x) -> Variable { return Barrier(x); };
    auto linear = Linear(tgtVocabSize, device);
    //auto zero = Constant({ encoderHiddenDim }, 0.0f, device);
    vector<Variable> e, h;
    return BinarySequenceModel({},
    {
        { L"embed",   embed },
        { L"encoder", encode },
        { L"linear",  linear }
    },
    [=](vector<Variable>& res, const vector<Variable>& x, const vector<Variable>& history) mutable
    {
        // encoder
        embed(e, x);
        encode(h, e);
        // decoder (outputting logprobs of words)
        // dummy for now
        linear(res, h);
        e.clear(); h.clear();
    });
}

BinaryFoldingModel CreateCriterionFunction(const BinarySequenceModel& model_fn)
{
    vector<Variable> features, history, labels, losses;
    // features and labels are tensors with first dimension being the length
    BinaryModel criterion = [=](const Variable& featuresAsTensor, const Variable& labelsAsTensor) mutable -> Variable
    {
        // convert sequence tensors into sequences of tensors
        // and strip the corresponding boundary markers
        //  - features: strip any?
        //  - labels: strip leading <s>
        //  - history: strip training </s>
        as_vector(features, featuresAsTensor);
        as_vector(history, labelsAsTensor);
        labels.assign(history.begin() + 1, history.end()); // make a full copy (of single-word references) without leading <s>
        history.pop_back(); // remove trailing </s>
        // apply model function
        // returns the sequence of output log probs over words
        vector<Variable> z;
        model_fn(z, features, history);
        features.clear(); history.clear(); // free some GPU memory
        // compute loss per word
        let sequenceLoss = Dynamite::Sequence::Map(BinaryModel(Dynamite::CrossEntropyWithSoftmax));
        sequenceLoss(losses, features, labels);
        let loss = Batch::sum(losses); // TODO: Batch is not the right namespace; but this does the right thing
        return loss;
    };
    // create a batch mapper (which will eventually allow suspension)
    let batchModel = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    vector<Variable> lossesPerSequence;
    return BinaryFoldingModel({}, { { L"model", model_fn } },
    [=](const /*batch*/vector<Variable>& features, const /*batch*/vector<Variable>& labels) mutable -> Variable
    {
        batchModel(lossesPerSequence, features, labels);             // batch-compute the criterion
        let collatedLosses = Splice(lossesPerSequence, Axis(0));     // collate all seq lossesPerSequence
        let mbLoss = ReduceSum(collatedLosses, Axis(0));  // aggregate over entire minibatch
        lossesPerSequence.clear();
        return mbLoss;
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
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // run something through to get the parameter matrices shaped --ugh!
    vector<Variable> d1{ Constant({ srcVocabSize }, 0.0, device) };
    vector<Variable> d2{ Constant({ tgtVocabSize }, 0.0, device) };
    vector<Variable> d3;
    model_fn(d3, d1, d2);

    // data
    auto minibatchSourceConfig = MinibatchSourceConfig({ PlainTextDeserializer(
        {
            PlainTextStreamConfiguration(L"src", srcVocabSize, { L"d:/work/Karnak/sample-model/data/train.src" }, { L"d:/work/Karnak/sample-model/data/vocab.src", L"<s>", L"</s>", L"<unk>" }),
            PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { L"d:/work/Karnak/sample-model/data/train.tgt" }, { L"d:/work/Karnak/sample-model/data/vocab.tgt", L"<s>", L"</s>", L"<unk>" })
        }) },
        /*randomize=*/true);
    minibatchSourceConfig.maxSamples = MinibatchSource::FullDataSweep;
    let minibatchSource = CreateCompositeMinibatchSource(minibatchSourceConfig);
    // BUGBUG (API): no way to specify MinibatchSource::FullDataSweep

    let parameters = model_fn.Parameters();
    auto learner = SGDLearner(parameters, LearningRatePerSampleSchedule(0.05));

    const size_t minibatchSize = 200;  // use 10 for ~3 sequences/batch
    for (size_t mbCount = 0; true; mbCount++)
    {
        // get next minibatch
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty()) // finished one data pass--TODO: really? Depends on config. We really don't care about data sweeps.
            break;
        fprintf(stderr, "#seq: %d, #words: %d\n", (int)minibatchData[minibatchSource->StreamInfo(L"src")].numberOfSequences, (int)minibatchData[minibatchSource->StreamInfo(L"src")].numberOfSamples);
        // train minibatch
        vector<vector<Variable>> args; // [variable index][batch index]
        Dynamite::FromCNTKMB(args, { minibatchData[minibatchSource->StreamInfo(L"src")].data, minibatchData[minibatchSource->StreamInfo(L"tgt")].data }, { true, true }, device);
    }
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
