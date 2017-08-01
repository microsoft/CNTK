//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "PlainTextDeseralizer.h"
#include "Layers.h"

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
const size_t srcVocabSize = 27579 + 3; // 2330;
const size_t tgtVocabSize = 21163 + 3; // 2330;
const size_t embeddingDim = 128;
const size_t attentionDim = 128;
const size_t numEncoderLayers = 1;
const size_t encoderHiddenDim = 128;
const size_t numDecoderLayers = 1;
const size_t decoderHiddenDim = 128;

UnarySequenceModel BidirectionalLSTMEncoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    vector<UnarySequenceModel> layers;
    for (size_t i = 0; i < numLayers; i++)
        layers.push_back(Dynamite::Sequence::BiRecurrence(RNNStep(hiddenDim, device), Constant({ hiddenDim }, 0.0f, device, L"fwdInitialValue"),
                                                          RNNStep(hiddenDim, device), Constant({ hiddenDim }, 0.0f, device, L"fwdInitialValue")));
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

// Bahdanau attention model
// (query, keys as tensor, data sequence as tensor) -> interpolated data vector
//  - keys used for the weights
//  - data gets interpolated
// Here they are the same.
TernaryModel AttentionModel(size_t attentionDim1)
{
    auto Q = Parameter({ attentionDim1, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"Q"); // query projection
    auto K = Parameter({ attentionDim1, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"K"); // keys projection
    auto v = Parameter({ attentionDim1 }, DataType::Float, GlorotUniformInitializer(), device, L"v"); // tanh projection
    return TernaryModel({ Q, K, v },
    [=](const Variable& query, const Variable& keys, const Variable& data) -> Variable
    {
        // compute attention weights
        let projectedQuery = Times(Q, query); // [A x 1]
        let projectedKeys  = Times(K, keys);  // [A x T]
        let tanh = Tanh(projectedQuery + projectedKeys); // [A x T]
#if 0 // this fails auto-batching
        let u = Times(v, tanh, L"vProj"); // [T] vector                         // [128] * [128 x 4 x 7] -> [4 x 7]
        let w = Dynamite::Softmax(u);                                           // [4 x 7]
        let res = Times(data, w, L"att"); // [A]                                // [128 x 4 x 7] * [4 x 7]
#else
        let u = TransposeTimes(tanh, v, L"vProj"); // [T] col vector            // [128 x 4 x 7]' * [128] = [7 x 4]         [128] * [128 x 4 x 7] -> [4 x 7]
        let w = Dynamite::Softmax(u);                                           // [7 x 4]                                  [4 x 7]
        let res = Times(data, w, L"att"); // [A]                                // [128 x 4 x 7] * [7 x 4]                  [128 x 4 x 7] * [4 x 7]
#endif
        return res;
     });
}

BinarySequenceModel AttentionDecoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    // create all the layer objects
    let initialState = Constant({ hiddenDim }, 0.0f, device, L"initialState");
    let initialContext = Constant({ 2 * hiddenDim }, 0.0f, device, L"initialContext"); // 2 * because bidirectional --TODO: can this be inferred?
    vector<BinaryModel> lstms;
    for (size_t i = 0; i < numLayers; i++)
        lstms.push_back(RNNStep(hiddenDim, device));
    let attentionModel = AttentionModel(attentionDim); // (state, encoding) -> interpolated encoding
    let barrier = Barrier();
    auto merge = Dense(hiddenDim, device); // one additional transform to merge attention into hidden state
    auto dense = Dense(tgtVocabSize, device); // dense layer without non-linearity

    vector<vector<Variable>> hs(2); // we need max. 2 buffers for the stack
    // decode from a top layer of an encoder, using history as history
    // A real decoder version would do something here, e.g. if history is empty then use its own output,
    // and maybe also take a reshuffling matrix for beam decoding.
    map<wstring, ModelParametersPtr> nestedLayers;
    for (let& lstm : lstms)
        nestedLayers[L"lstm[" + std::to_wstring(nestedLayers.size()) + L"]"] = lstm;
    nestedLayers.insert(
    {
        { L"attentionModel", attentionModel },
        { L"merge", merge },
        { L"dense", dense }
    });
    return BinarySequenceModel({}, nestedLayers,
    [=](vector<Variable>& res, const vector<Variable>& history, const vector<Variable>& hEncs) mutable
    {
        res.resize(history.size());
        // TODO: this is one layer only for now
        // convert encoder sequence into a dense tensor, so that we can do matrix products along the sequence axis
        Variable hEncsTensor = Splice(hEncs, Axis(1)); // [2*hiddenDim, inputLen]
        // decoding loop
        Variable state = initialState;
        Variable attentionContext = initialContext; // note: this is almost certainly wrong
        for (size_t t = 0; t < history.size(); t++)
        {
            // do recurrent step
            // In inference, history[t] would become res[t-1].
            // TODO: Why not learn the output of the first step, and skip the <s> and funky initial attention context?
            let input = Splice({ history[t], attentionContext }, Axis(0), L"augInput");
            state = lstms[0](state, input);
            // compute attention vector
            attentionContext = attentionModel(state, /*keys=*/hEncsTensor, /*data=*/hEncsTensor);
            // compute an enhanced hidden state with attention value merged in
            let m = Tanh(merge(Splice({ state, attentionContext }, Axis(0))));
            // compute output
            let z = dense(m);
            res[t] = z;
        }
        // ...unused for now
        hs[0].clear(); hs[1].clear();
    });
}

BinarySequenceModel CreateModelFunction()
{
    auto embed = Embedding(embeddingDim, device);
    auto encode = BidirectionalLSTMEncoder(numEncoderLayers, encoderHiddenDim, 0.8);
    auto decode = AttentionDecoder(numDecoderLayers, decoderHiddenDim, 0.8);
    vector<Variable> e, h;
    return BinarySequenceModel({},
    {
        { L"embed",   embed },
        { L"encoder", encode },
        { L"decode",  decode }
    },
    [=](vector<Variable>& res, const vector<Variable>& x, const vector<Variable>& history) mutable
    {
        // encoder
        embed(e, x);
        encode(h, e);
        // decoder (outputting logprobs of words)
        decode(res, history, h);
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
        let sequenceLoss = Dynamite::Sequence::Map(BinaryModel([](const Variable& z, const Variable& label) { return Dynamite::CrossEntropyWithSoftmax(z, label); }));
        sequenceLoss(losses, z, labels);
        let loss = Batch::sum(losses); // TODO: Batch is not the right namespace; but this does the right thing
        labels.clear(); losses.clear();
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
        let collatedLosses = Splice(lossesPerSequence, Axis(0), L"cesPerSeq");     // collate all seq lossesPerSequence
        let mbLoss = ReduceSum(collatedLosses, Axis(0), L"ceBatch");  // aggregate over entire minibatch
        lossesPerSequence.clear();
        return mbLoss;
    });
}

void Train()
{
    // dynamic model and criterion function
    auto model_fn = CreateModelFunction();
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // run something through to get the parameter matrices shaped --ugh!
    vector<Variable> d1{ Constant({ srcVocabSize }, 0.0f, device) };
    vector<Variable> d2{ Constant({ tgtVocabSize }, 0.0f, device) };
    vector<Variable> d3;
    model_fn(d3, d1, d2);

    model_fn.LogParameters();

    // data
    auto minibatchSourceConfig = MinibatchSourceConfig({ PlainTextDeserializer(
        {
            //PlainTextStreamConfiguration(L"src", srcVocabSize, { L"d:/work/Karnak/sample-model/data/train.src" }, { L"d:/work/Karnak/sample-model/data/vocab.src", L"<s>", L"</s>", L"<unk>" }),
            //PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { L"d:/work/Karnak/sample-model/data/train.tgt" }, { L"d:/work/Karnak/sample-model/data/vocab.tgt", L"<s>", L"</s>", L"<unk>" })
            PlainTextStreamConfiguration(L"src", srcVocabSize, { L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.ro.shuf" }, { L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.ro.vocab", L"<s>", L"</s>", L"<unk>" }),
            PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.en.shuf" }, { L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.en.vocab", L"<s>", L"</s>", L"<unk>" })
        }) },
        /*randomize=*/true);
    minibatchSourceConfig.maxSamples = MinibatchSource::InfinitelyRepeat;
    let minibatchSource = CreateCompositeMinibatchSource(minibatchSourceConfig);
    // BUGBUG (API): no way to specify MinibatchSource::FullDataSweep

    let parameters = model_fn.Parameters();
    //auto learner = SGDLearner(parameters, LearningRatePerSampleSchedule(0.0005));
    let epochSize = 100000; // it's a small corpus, ~50k samples
    auto learner = AdamLearner(parameters, LearningRatePerSampleSchedule({ 0.0001 * sqrt(384), 0.00005 * sqrt(384), 0.000025 * sqrt(384), 0.00001 * sqrt(384) }, epochSize), MomentumAsTimeConstantSchedule(1000), true, MomentumAsTimeConstantSchedule(10000));
    unordered_map<Parameter, NDArrayViewPtr> gradients;
    for (let& p : parameters) // TODO: test that this works outside of the loop
        gradients[p] = nullptr; // TryGetGradient(p); // TODO: get the existing gradient matrix from the parameter--or just fill it in? Would block memory free

    const size_t minibatchSize = 184;// 50;  // 384 is 32 sequences, assuming av. length ~12
    vector<vector<Variable>> args; // [variable index][batch index]  --TODO: does this really work outside the loop?
    size_t totalLabels = 0;
    for (size_t mbCount = 0; true; mbCount++)
    {
        // get next minibatch
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty()) // finished one data pass--TODO: really? Depends on config. We really don't care about data sweeps.
            break;
        let numLabels = minibatchData[minibatchSource->StreamInfo(L"tgt")].numberOfSamples;
        totalLabels += numLabels;
        fprintf(stderr, "#seq: %d, #words: %d, lr=%.8f\n",
                (int)minibatchData[minibatchSource->StreamInfo(L"src")].numberOfSequences,
                (int)minibatchData[minibatchSource->StreamInfo(L"src")].numberOfSamples,
                learner->LearningRate());
        Dynamite::FromCNTKMB(args, { minibatchData[minibatchSource->StreamInfo(L"src")].data, minibatchData[minibatchSource->StreamInfo(L"tgt")].data }, { true, true }, device);
        // train minibatch
        let mbLoss = criterion_fn(args[0], args[1]);
        let lossPerLabel = mbLoss.Value()->AsScalar<float>() / numLabels; // note: this does the GPU sync, so better do that only every N
        fprintf(stderr, "CrossEntropy loss = %.7f; PPL = %.3f; seenLabels=%d\n", lossPerLabel, exp(lossPerLabel), (int)totalLabels);
        if (std::isnan(lossPerLabel))
            throw runtime_error("Loss is NaN.");
        // backprop and model update
        mbLoss.Backward(gradients);
        learner->Update(gradients, numLabels);
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
