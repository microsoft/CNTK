//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "PlainTextDeseralizer.h"
#include "Layers.h"
#include "TimerUtility.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <io.h>
#include <boost/filesystem.hpp>

#define let const auto
#define fun const auto

using namespace CNTK;
using namespace std;

using namespace Dynamite;

// reference config [Arul]:
// \\mt-data\training_archive\mtmain_backend\rom_enu_generalnn\vCurrent\2017_06_10_00h_35m_31s\train_1\network_v3_src-3gru_tgt-1gru-4fcsc-1gru_coverage.xml
// differences:
//  - dropout not implemented yet
//  - GRU instead of LSTM
//  - ReLU not clipped/scaled
//  - no coverage/alignment model
//  - batch/length normalization
//  - no weight norm

DeviceDescriptor device(DeviceDescriptor::CPUDevice()); // dummy; will be overwritten
const size_t srcVocabSize = 27579 + 3;
const size_t tgtVocabSize = 21163 + 3;
const size_t embeddingDim = 512;
const size_t attentionDim = 512;
const size_t numEncoderLayers = 3;
const size_t encoderRecurrentDim = 512;
const size_t decoderRecurrentDim = 1024;
const size_t numDecoderResNetProjections = 4;
const size_t decoderProjectionDim = 768;
const size_t topHiddenProjectionDim = 1024;

size_t mbCount = 0; // made a global so that we can trigger debug information on it
#define DOLOG(var) (var)//((mbCount % 100 == 99) ? LOG(var) : 0)

BinarySequenceModel BidirectionalLSTMEncoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    vector<BinarySequenceModel> layers;
    for (size_t i = 0; i < numLayers; i++)
        layers.push_back(Dynamite::Sequence::BiRecurrence(GRU(hiddenDim, device), Constant({ hiddenDim }, DTYPE, 0.0, device, Named("fwdInitialValue")),
                                                          GRU(hiddenDim, device), Constant({ hiddenDim }, DTYPE, 0.0, device, Named("bwdInitialValue"))));
    vector<UnaryBroadcastingModel> bns;
    for (size_t i = 0; i < numLayers-1; i++)
        bns.push_back(Dynamite::BatchNormalization(device, Named("bnBidi")));
    vector<vector<Variable>> hs(2); // we need max. 2 buffers for the stack
    vector<Variable> hBn;
    vector<ModelParametersPtr> nested;
    nested.insert(nested.end(), layers.begin(), layers.end());
    nested.insert(nested.end(), bns.begin(), bns.end());
    return BinarySequenceModel(nested,
    [=](vector<Variable>& res, const vector<Variable>& xFwd, const vector<Variable>& xBwd) mutable
    {
        for (size_t i = 0; i < numLayers; i++)
        {
            const vector<Variable>& inFwd = (i == 0) ? xFwd : hs[i % 2];
            const vector<Variable>& inBwd = (i == 0) ? xBwd : hs[i % 2];
            vector<Variable>& out = (i == numLayers - 1) ? res : hs[(i+1) % 2];
            if (i > 0)
            {
                layers[i](hBn, inFwd, inBwd);
                bns[i - 1](out, hBn);
                hBn.clear();
            }
            else
                layers[i](out, inFwd, inBwd);
        }
        hs[0].clear(); hs[1].clear();
    });
}

// Bahdanau attention model
// (query, keys as tensor, data sequence as tensor) -> interpolated data vector
//  - keys used for the weights
//  - data gets interpolated
// Here they are the same.
TernaryModel AttentionModelBahdanau(size_t attentionDim1)
{
    auto Q = Parameter({ attentionDim1, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"Q"); // query projection
    auto v = Parameter({ attentionDim1 }, DTYPE, GlorotUniformInitializer(), device, L"v"); // tanh projection
    let normQ = LengthNormalization(device);
    return TernaryModel({ Q, /*K,*/ v }, { { L"normQ", normQ } },
    [=](const Variable& query, const Variable& projectedKeys/*keys*/, const Variable& data) -> Variable
    {
        // compute attention weights
        let projectedQuery = normQ(Times(Q, query, Named("Q"))); // [A x 1]
        let tanh = Tanh((projectedQuery + projectedKeys), Named("attTanh")); // [A x T]
#if 0
        let u = InnerProduct(tanh, v, Axis(0), Named("vProj")); // [1 x T] col vector
        let w = Dynamite::Softmax(u, Axis(1));
        let res = Reshape(InnerProduct(data, w, Axis(1), Named("att")), NDShape{ attentionDim1 }); // [A]
#else
        let u = TransposeTimes(tanh, v, Named("vProj")); // [T] col vector
        let w = Dynamite::Softmax(u);
        let res = Times(data, w, Named("att")); // [A]
#endif
        return res;
     });
}

// reference attention model
fun AttentionModelReference(size_t attentionDim1)
{
    //auto H = Parameter({ attentionDim1, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"Q"); // query projection
    auto projectQuery = Linear(attentionDim1, ProjectionOptions::weightNormalize, device);
    let normH = LengthNormalization(device); // note: can't move this inside Linear since it is applied after adding two factors
    let profiler = Function::CreateDynamicProfiler(1, L"attention");
    let zBarrier   = Barrier(20, Named("zBarrier"));
    let resBarrier = Barrier(20, Named("resBarrier"));
    vector<Variable> us, ws;
    StaticModel doToTanh(/*isBasicBlock=*/false, [=](const Variable& h, const Variable& historyProjectedKey)
    {
        let hProjected = projectQuery(h); // [A]. Batched.
        let tanh = Tanh(normH(hProjected + historyProjectedKey), Named("attTanh")); // [A]. Batched.
        return tanh;
    });
    return QuaternaryModel11NN({ }, { { L"normH", normH }, { L"projectQuery", projectQuery } },
        [=](const Variable& h,                              // [A] decoder hidden state
            const Variable& historyProjectedKey,            // [A] previous output, embedded
            const vector<Variable>& encodingProjectedKeys, // [A x T] encoder hidden state seq, projected as key >> tanh
            const vector<Variable>& encodingProjectedData           // [A x T] encoder hidden state seq, projected as data
           ) mutable -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler, false);
        // compute attention weights
        //let hProjected = projectQuery(h); // [A]. Batched.
        //let tanh = Tanh(normH(hProjected + historyProjectedKey), Named("attTanh")); // [A]. Batched.
        let tanh = doToTanh(h, historyProjectedKey);
#if 1
        // This is not well-batched yet, I don't know why. This is the solution that should realize maximum parallelization (disregarding Gather steps).
        for (let& k : encodingProjectedKeys)
            us.push_back(InnerProduct(tanh, k, Axis(0), Named("u"))); // vector<[1]>. Batched.
        // alternative: ^^ compute the inner product elementwise.
        //              Will make a broadcast copy of tanh[t] (#copies=mb size), then operate in one go instead of one per sequence
        // alternative: softmax denom: splice u, ReduceLogSum, slice up -> sequence
        //              Batched (#mb size in one go): minus, exp -> w[t]
        //              Not batched (one launch per sentence: ReduceLogSum)
        Dynamite::Sequence::Softmax(ws, us, zBarrier); // softmax over a vector
        // BUGBUG: Somehow the Exp() inside does not get batched, although they are unrolled.
        us.clear();
        // alternative: product with encodingProjectedData
        //              Batched: elementwise product -> RAM copy of (encodingProjectedData * u)[t], #items=mb size
        let res = resBarrier(Dynamite::Sequence::InnerProduct(encodingProjectedData, ws, Named("attContext"))); // inner product over a vectors
        ws.clear();
#else
        let encodingProjectedKeysTensor = Splice(encodingProjectedKeys, Axis(1), Named("encodingProjectedKeysTensor"));  // [A x T]
        let u = InnerProduct(tanh, encodingProjectedKeysTensor, Axis(0), Named("u")); // [1 x T] row vector. Batchable by stacking.
        // alternative: ^^ compute the inner product elementwise.
        //              Will make a broadcast copy of tanh[t] (#copies=mb size), then operate in one go instead of one per sequence
        // alternative: softmax denom: splice u, ReduceLogSum, slice up -> sequence
        //              Batched (#mb size in one go): minus, exp -> w[t]
        //              Not batched (one launch per sentence: ReduceLogSum)
        let w = Dynamite::Softmax(u, Axis(1), Named("w")); // [1 x T] ReduceLogSum() not parallelizable. Pad? E.g. bucket-pad?
        // alternative: product with encodingProjectedData
        //              Batched: elementwise product -> RAM copy of (encodingProjectedData * u)[t], #items=mb size
        let encodingProjectedDataTensor = Splice(encodingProjectedData, Axis(1), Named("encodingProjectedDataTensor"));  // [A x T]
        let res = Reshape(InnerProduct(encodingProjectedDataTensor, w, Axis(1), Named("attContext")), NDShape{ attentionDim1 }); // [A x T] x [1 x T] -> [A x 1] Batchable with bucket-padding.
        // alternative: reduce over encoding length. Gather the products (another copy #items=mb size)
        //              Not batched (one launch per sentence): ReduceSum
#endif
        Function::SetDynamicProfiler(prevProfiler);
        return res;
    });
}

// TODO: Break out initial step and recurrent step layers. Decoder will later pull them out frmo here.
BinarySequenceModel AttentionDecoder(double dropoutInputKeepProb)
{
    // create all the layer objects
    let encBarrier = Barrier(600, Named("encBarrier"));
    //let encoderKeysProjection = encBarrier >> Linear(attentionDim, ProjectionOptions::stabilize, device) >> BatchNormalization(device, Named("bnEncoderKeysProjection")) >> UnaryModel([](const Variable& x) { return Tanh(x, Named("tanh_bnEncoderKeysProjection")); }); // keys projection for attention
    //let encoderDataProjection = encBarrier >> Linear(attentionDim, ProjectionOptions::stabilize, device) >> BatchNormalization(device, Named("bnEncoderDataProjection")) >> UnaryModel([](const Variable& x) { return Tanh(x, Named("tanh_bnEncoderDataProjection")); }); // data projection for attention
    let encoderKeysProjection = encBarrier >> Dense(attentionDim, UnaryModel([](const Variable& x) { return Tanh(x, Named("encoderKeysProjection")); }), ProjectionOptions::batchNormalize | ProjectionOptions::bias, device); // keys projection for attention
    let encoderDataProjection = encBarrier >> Dense(attentionDim, UnaryModel([](const Variable& x) { return Tanh(x, Named("encoderDataProjection")); }), ProjectionOptions::batchNormalize | ProjectionOptions::bias, device); // data projection for attention
    let embedTarget = Barrier(600, Named("embedTargetBarrier")) >> Embedding(embeddingDim, device) /*>> BatchNormalization(device, Named("bnEmbedTarget"))*/;     // target embeddding
    let initialContext = Constant({ attentionDim }, DTYPE, 0.0, device, L"initialContext"); // 2 * because bidirectional --TODO: can this be inferred?
    let initialStateProjection = Barrier(20, Named("initialStateProjectionBarrier")) >> Dense(decoderRecurrentDim, UnaryModel([](const Variable& x) { return Tanh(x, Named("initialStateProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);
    let stepBarrier = Barrier(20, Named("stepBarrier"));
    let stepFunction = GRU(decoderRecurrentDim, device);
    //let attentionModel = AttentionModelBahdanau(attentionDim);
    let attentionModel = AttentionModelReference(attentionDim);
    let firstHiddenProjection = Barrier(600, Named("projBarrier")) >> Dense(decoderProjectionDim, UnaryModel([](const Variable& x) { return ReLU(x, Named("firstHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);
    vector<UnaryBroadcastingModel> resnets;
    for (size_t n = 0; n < numDecoderResNetProjections; n++)
        resnets.push_back(ResidualNet(decoderProjectionDim, device));
    let topHiddenProjection = Dense(topHiddenProjectionDim, UnaryModel([](const Variable& x) { return Tanh(x, Named("topHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);
    let outputProjection = Linear(tgtVocabSize, ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);  // output layer without non-linearity (no sampling yet)

    // buffers
    vector<Variable> encodingProjectedKeys, encodingProjectedData;

    // decode from a top layer of an encoder, using history as history
    map<wstring, ModelParametersPtr> nestedLayers =
    {
        { L"encoderKeysProjection",  encoderKeysProjection },
        { L"encoderDataProjection",  encoderDataProjection },
        { L"embedTarget",            embedTarget },
        { L"initialStateProjection", initialStateProjection },
        //{ L"stepBarrier",            stepBarrier },
        { L"stepFunction",           stepFunction },
        { L"attentionModel",         attentionModel },
        { L"firstHiddenProjection",  firstHiddenProjection },
        { L"topHiddenProjection",    topHiddenProjection },
        { L"outputProjection",       outputProjection },
    };
    for (let& resnet : resnets)
        nestedLayers[L"resnet[" + std::to_wstring(nestedLayers.size()) + L"]"] = resnet;
    let profiler = Function::CreateDynamicProfiler(1, L"decode");

#if 1 // this fails because some dimensions are not known yet
    StaticModel doToOutput(/*isBasicBlock=*/false, [=](const Variable& state, const Variable& attentionContext)
    {
        // first one brings it into the right dimension
        auto state1 = firstHiddenProjection(state);
        // then a bunch of ResNet layers
        for (auto& resnet : resnets)
            state1 = resnet(state1);
        // one more transform, bringing back the attention context
        let topHidden = topHiddenProjection(Splice({ state1, attentionContext }, Axis(0)));
        // ^^ batchable; currently one per target word in MB (e.g. 600); could be one per batch (2 launches)
        // TODO: dropout layer here
        dropoutInputKeepProb;
        // compute output
        let z = outputProjection(topHidden);
        return z;
    });
#endif

    return BinarySequenceModel({ }, nestedLayers,
    [=](vector<Variable>& res, const vector<Variable>& history, const vector<Variable>& hEncs) mutable
    {
        res.resize(history.size());
        // decoding loop
        Variable state = Slice(hEncs.front(), Axis(0), encoderRecurrentDim, 2 * encoderRecurrentDim); // initial state for the recurrence is the final encoder state of the backward recurrence
        state = initialStateProjection(state);      // match the dimensions
        Variable attentionContext = initialContext; // note: this is almost certainly wrong
        // common subexpression of attention.
        // We pack the result into a dense matrix; but only after the matrix product, to allow for it to be batched.
        encoderKeysProjection(encodingProjectedKeys, hEncs);
        encoderDataProjection(encodingProjectedData, hEncs);
        // ^^ these are 2 x #sequences splices per MB, (40 for 20 seq in MB)
        for (size_t t = 0; t < history.size(); t++)
        {
            // do recurrent step (in inference, history[t] would become res[t-1])
            let historyProjectedKey = embedTarget(history[t]);
            // ^^ fully batched, 1 launch
            let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph
            let input = Splice({ historyProjectedKey, attentionContext }, Axis(0), Named("augInput"));
            state = stepFunction(state, stepBarrier(input));
            // compute attention vector
            //attentionContext = attentionModel(state, /*keys=*/projectedKeys, /*data=*/hEncsTensor);
            attentionContext = attentionModel(state, historyProjectedKey, encodingProjectedKeys, encodingProjectedData);
            // ^^ two operations that are per-sequence
            Function::SetDynamicProfiler(prevProfiler);
            // stack of non-recurrent projections
            // vv fully batchable
#if 1
            let z = doToOutput(state, attentionContext);
#else
            auto state1 = firstHiddenProjection(state); // first one brings it into the right dimension
            for (auto& resnet : resnets)                // then a bunch of ResNet layers
                state1 = resnet(state1);
            // one more transform, bringing back the attention context
            let topHidden = topHiddenProjection(Splice({ state1, attentionContext }, Axis(0)));
            // ^^ batchable; currently one per target word in MB (e.g. 600); could be one per batch (2 launches)
            // TODO: dropout layer here
            dropoutInputKeepProb;
            // compute output
            let z = outputProjection(topHidden);
#endif
            res[t] = z;
        }
        encodingProjectedKeys.clear();
        encodingProjectedData.clear();
    });
}

BinarySequenceModel CreateModelFunction()
{
    auto embedFwd = Embedding(embeddingDim, device) /*>> BatchNormalization(device, Named("bnEmbedFwd"))*/;
    auto embedBwd = Embedding(embeddingDim, device) /*>> BatchNormalization(device, Named("bnEmbedBwd"))*/;
    auto encode = BidirectionalLSTMEncoder(numEncoderLayers, encoderRecurrentDim, 0.8);
    auto decode = AttentionDecoder(0.8);
    vector<Variable> eFwd,eBwd, h;
    return BinarySequenceModel({},
    {
        { L"embedSourceFwd", embedFwd },
        { L"embedSourceBwd", embedBwd },
        { L"encode",         encode   },
        { L"decode",         decode   }
    },
    [=](vector<Variable>& res, const vector<Variable>& x, const vector<Variable>& history) mutable
    {
        // embedding
        embedFwd(eFwd, x);
        embedBwd(eBwd, x);
        // encoder
        encode(h, eFwd, eBwd);
        eFwd.clear(); eBwd.clear();
        // decoder (outputting logprobs of words)
        decode(res, history, h);
        h.clear();
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
    let profiler = Function::CreateDynamicProfiler(1, L"all");
    // create a batch mapper (which will eventually allow suspension)
    let batchModel = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    vector<Variable> lossesPerSequence;
    return BinaryFoldingModel({}, { { L"model", model_fn } },
    [=](const /*batch*/vector<Variable>& features, const /*batch*/vector<Variable>& labels) mutable -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph
        batchModel(lossesPerSequence, features, labels);             // batch-compute the criterion
        let collatedLosses = Splice(lossesPerSequence, Axis(0), Named("seqLosses"));     // collate all seq lossesPerSequence
        // ^^ this is one launch per MB
        let mbLoss = /*Reshape*/(ReduceSum(collatedLosses, /*Axis(0)*/Axis_DropLastAxis, Named("batchLoss"))/*, NDShape{}*/);  // aggregate over entire minibatch
        lossesPerSequence.clear();
        Function::SetDynamicProfiler(prevProfiler);
        return mbLoss;
    });
}

void Train(wstring outputDirectory)
{
    let communicator = MPICommunicator();
#if 1 // while we are running with MPI, we always start from start
    let numGpus = DeviceDescriptor::AllDevices().size() -1;
    let ourRank = communicator->CurrentWorker().m_globalRank;
    if (numGpus > 0)
        device = DeviceDescriptor::GPUDevice((unsigned int)(ourRank % numGpus));
    else
        device = DeviceDescriptor::CPUDevice();
#else
    device = DeviceDescriptor::UseDefaultDevice();
#endif
    // open log file
    // Log path = "$workingDirectory/$experimentId.log.$ourRank" where $ourRank is missing for rank 0
    let logPath = outputDirectory + L"/train.log" + (ourRank == 0 ? L"" : (L"." + to_wstring(ourRank)));
    boost::filesystem::create_directories(boost::filesystem::path(logPath).parent_path());
    FILE* outStream =
        /*if*/ (communicator->CurrentWorker().IsMain()) ?
            _wpopen((L"tee " + logPath).c_str(), L"wt")
        /*else*/:
            _wfopen(logPath.c_str(), L"wt");
    if (!outStream)
        InvalidArgument("error %d opening log file '%S'", errno, logPath.c_str());
    fprintf(stderr, "redirecting stderr to %S\n", logPath.c_str());
    if (_dup2(_fileno(outStream), _fileno(stderr)))
        InvalidArgument("error %d redirecting stderr to '%S'", errno, logPath.c_str());
    fprintf(stderr, "starting training as worker[%d]\n", (int)ourRank), fflush(stderr); // write something to test

    // dynamic model and criterion function
    auto model_fn = CreateModelFunction();
    auto criterion_fn = CreateCriterionFunction(model_fn);

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
    minibatchSourceConfig.isMultithreaded = false;
    minibatchSourceConfig.enableMinibatchPrefetch = false;
    // BUGBUG: ^^ I see two possibly related bugs
    //  - when running on CPU, this fails reliably with what looks like a race condition
    //  - even with GPU, training unreliably fails after precisely N data passes minus one data pass. That minus one may indicate a problem in prefetch?
    // -> Trying without, to see if the problems go away.
    let minibatchSource = CreateCompositeMinibatchSource(minibatchSourceConfig);
    // BUGBUG (API): no way to specify MinibatchSource::FullDataSweep in a single expression

    // run something through to get the parameter matrices shaped --ugh!
    vector<Variable> d1{ Constant({ srcVocabSize }, DTYPE, 0.0, device) };
    vector<Variable> d2{ Constant({ tgtVocabSize }, DTYPE, 0.0, device) };
    vector<Variable> d3;
    model_fn(d3, d1, d2);

    model_fn.LogParameters();

    let parameters = model_fn.Parameters();
    size_t numParameters = 0;
    for (let& p : parameters)
        numParameters += p.Shape().TotalSize();
    fprintf(stderr, "Total number of learnable parameters: %u\n", (unsigned int)numParameters), fflush(stderr);
    let epochSize = 10000000; // 10M is a bit more than half an epoch of ROM-ENG (~16M words)
    let minibatchSize = 4096      * communicator->Workers().size() /6; // for debugging: switch to smaller MB when running without MPI
    AdditionalLearningOptions learnerOptions;
    learnerOptions.gradientClippingThresholdPerSample = 0.2;
#if 0
    auto baseLearner = SGDLearner(parameters, LearningRatePerSampleSchedule(0.0005), learnerOptions);
#else
    // AdaGrad correction-correction:
    //  - LR is specified for av gradient
    //  - numer should be /minibatchSize
    //  - denom should be /sqrt(minibatchSize)
    let f = 1 / sqrt(minibatchSize)/*AdaGrad correction-correction*/;
    let lr0 = 0.0003662109375 * f;
    auto baseLearner = AdamLearner(parameters, LearningRatePerSampleSchedule({ lr0, lr0/2, lr0/4, lr0/8 }, epochSize),
        MomentumAsTimeConstantSchedule(40000), true, MomentumAsTimeConstantSchedule(400000), /*eps=*/1e-8, /*adamax=*/false,
        learnerOptions);
#endif
    let& learner = CreateDataParallelDistributedLearner(communicator, baseLearner, /*distributeAfterSamples =*/ 0, /*useAsyncBufferedParameterUpdate =*/ false);
    unordered_map<Parameter, NDArrayViewPtr> gradients;
    for (let& p : parameters)
        gradients[p] = nullptr;

    vector<vector<Variable>> args;
    size_t totalLabels = 0;
    class SmoothedVar
    {
        double smoothedNumer = 0; double smoothedDenom = 0;
        const double smoothedDecay = 0.99;
    public:
        double Update(double avgVal, size_t count)
        {
            // TODO: implement the correct smoothing
            smoothedNumer = smoothedDecay * smoothedNumer + (1 - smoothedDecay) * avgVal * count;
            smoothedDenom = smoothedDecay * smoothedDenom + (1 - smoothedDecay) * count;
            return Value();
        }
        double Value() const
        {
            return smoothedNumer / smoothedDenom;
        }
    } smoothedLoss;
    Microsoft::MSR::CNTK::Timer timer;
    class // helper for timing GPU-side operations
    {
        Microsoft::MSR::CNTK::Timer m_timer;
        double syncGpu() { return CNTK::NDArrayView::Sync(device); }
    public:
        void Restart(bool syncGPU = false)
        {
            if (syncGPU)
                syncGpu();
            m_timer.Restart();
        }
        double Elapsed(bool syncGPU = false)
        {
            if (syncGPU)
                return syncGpu();
            return m_timer.ElapsedSeconds();
        }
        //void Log(const char* what, size_t numItems)
        //{
        //    if (enabled)
        //    {
        //        syncGpu();
        //        let elapsed = Elapsed();
        //        fprintf(stderr, "%s: %d items, items/sec=%.2f, ms/item=%.2f\n", what, (int)numItems, numItems / elapsed, 1000.0/*ms*/ * elapsed / numItems);
        //    }
        //}
    } partTimer;
    //wstring modelPath = L"d:/me/tmp_dynamite_model.cmf";
    //wstring modelPath = L"d:/me/tmp_dynamite_model_wn.cmf";
    //wstring modelPath = L"d:/me/tmp_dynamite_model_attseq.cmf";
    wstring modelPath = outputDirectory + L"/model.dmf"; // DMF=Dynamite model file
    size_t saveEvery = 100;
    size_t startMbCount = 0;
    let ModelPath = [&](size_t currentMbCount) // helper to form the model filename
    {
        char currentMbCountBuf[20];
        sprintf(currentMbCountBuf, "%06d", (int)currentMbCount); // append the minibatch index with a fixed width for sorted directory listings
        return modelPath + L"." + wstring(currentMbCountBuf, currentMbCountBuf + strlen(currentMbCountBuf)); // (simplistic string->wstring converter)
    };
    if (startMbCount > 0)
    {
        // restarting after crash. Note: not checkpointing the reader yet.
        let path = ModelPath(startMbCount);
        fprintf(stderr, "restarting from: %S\n", path.c_str()), fflush(stderr);
        // TODO: The code below is copy-paste from Trainer.cpp, since it is not public. Move this down through the API.
        // TODO: Can we just wrap everything in an actual Trainer instance, and use that?
        //model_fn.RestoreParameters(path);
        std::wstring trainerStateCheckpointFilePath = path + L".ckp";

        // Restore the model's parameters
        auto compositeFunction = model_fn.ParametersCombined();
        compositeFunction->Restore(path);

        // restore remaining state
        Dictionary checkpoint = Dictionary::Load(trainerStateCheckpointFilePath);

        const std::wstring internalWorkerStateKey = L"internal_worker_state"; // these are from Serialization.h
        const std::wstring externalWorkerStateKey = L"external_worker_state";

        // TODO: reuse from Trainer.cpp:
        const std::wstring versionPropertyName = L"Version";
        const std::wstring learnersPropertyName = L"Learners";
        const std::wstring externalStatePropertyName = L"ExternalState";
        const std::wstring distributedStatePropertyName = L"DistributedState";

        auto learnerState = checkpoint[learnersPropertyName].Value<std::vector<DictionaryValue>>().front().Value<Dictionary>();
        auto externalState = checkpoint[externalStatePropertyName].Value<Dictionary>();

        learner->RestoreFromCheckpoint(learnerState);

        if (communicator->Workers().size() > 1 || true)
        {
            // this ensures that nobody will start writing to the model/checkpoint files, until
            // everybody is done reading them.
            DistributedCommunicatorPtr checkpointCommunicator = MPICommunicator();
            checkpointCommunicator->Barrier();

            auto mainWorkerId = std::to_wstring(0);
            auto localWorkerId = std::to_wstring(checkpointCommunicator->CurrentWorker().m_globalRank);

            Dictionary distributedState = checkpoint[distributedStatePropertyName].Value<Dictionary>();

            if (!checkpointCommunicator->CurrentWorker().IsMain() && distributedState.Contains(localWorkerId))
            {
                // the checkpoint contains internal state for this worker.
                Dictionary localState = distributedState[localWorkerId].Value<Dictionary>();
                externalState = localState[externalWorkerStateKey].Value<Dictionary>();
            }
        }

        minibatchSource->RestoreFromCheckpoint(externalState[/*s_trainingMinibatchSource=*/L"TrainingMinibatchSource"].Value<Dictionary>());
    }
    fflush(stderr);
    // MINIBATCH LOOP
    for (mbCount = startMbCount; true; mbCount++)
    {
        // checkpoint
        if (mbCount % saveEvery == 0 &&
            (/*startMbCount == 0 ||*/ mbCount > startMbCount)) // don't overwrite the starting model
        {
            let path = ModelPath(mbCount);
            fprintf(stderr, "%ssaving: %S\n", communicator->CurrentWorker().IsMain() ? "" : "not ", path.c_str()), fflush(stderr); // indicate time of saving, but only main worker actually saves
            //model_fn.SaveParameters(path);

            // reader state
            Dictionary externalState(/*s_trainingMinibatchSource=*/L"TrainingMinibatchSource", minibatchSource->GetCheckpointState());

            // learner state
            std::vector<DictionaryValue> learnersState{ learner->CreateCheckpoint() };

            auto compositeFunction = model_fn.ParametersCombined();

            Dictionary aggregatedState;
            DistributedCommunicatorPtr checkpointCommunicator;
            if (communicator->Workers().size() > 1    ||true)
                checkpointCommunicator = MPICommunicator();
            if (checkpointCommunicator)
            {
                const std::wstring internalWorkerStateKey = L"internal_worker_state"; // these are from Serialization.h
                const std::wstring externalWorkerStateKey = L"external_worker_state";
                Dictionary localState(internalWorkerStateKey, Dictionary(), externalWorkerStateKey, externalState);

                // Collect distrbuted external localState.
                checkpointCommunicator = MPICommunicator();
                checkpointCommunicator->Barrier();

                std::vector<DictionaryPtr> remoteState;
                checkpointCommunicator->Gather(localState, remoteState, checkpointCommunicator->Workers());

                for (const auto& w : checkpointCommunicator->Workers())
                    aggregatedState[std::to_wstring(w.m_globalRank)] = *remoteState[w.m_globalRank];
            }

            if (!checkpointCommunicator || checkpointCommunicator->CurrentWorker().IsMain())
            {
                // TODO: reuse from Trainer.cpp:
                const std::wstring versionPropertyName = L"Version";
                const std::wstring learnersPropertyName = L"Learners";
                const std::wstring externalStatePropertyName = L"ExternalState";
                const std::wstring distributedStatePropertyName = L"DistributedState";
                static const size_t trainerCheckpointVersion = 1;

                Dictionary state(
                    versionPropertyName         , trainerCheckpointVersion,
                    learnersPropertyName        , learnersState,
                    externalStatePropertyName   , externalState,
                    distributedStatePropertyName, aggregatedState);

                // TODO: rename must check return code. To fix this, just move this down to the API and use the functions in Trainer.cpp.
                std::wstring tempModelFile = path + L".tmp";
                std::wstring trainerStateCheckpointFilePath = path + L".ckp";
                std::wstring tempCheckpointFile = trainerStateCheckpointFilePath + L".tmp";
                compositeFunction->Save(tempModelFile);
                state.Save(tempCheckpointFile);
                _wunlink(path.c_str());
                _wunlink(trainerStateCheckpointFilePath.c_str());
                _wrename(tempModelFile.c_str(), path.c_str());
                _wrename(tempCheckpointFile.c_str(), trainerStateCheckpointFilePath.c_str());
            }

            if (checkpointCommunicator)
                // all workers need to sync up after saving model to avoid read-after-write hazard
                // i.e. one worker is in the middle of write while another tries to read
                checkpointCommunicator->Barrier();
                // Note: checkpointCommunicator is destructed at end of this block

            // test model saving
            //for (auto& param : parameters) // destroy parameters as to prove that we reloaded them correctly.
            //    param.Value()->SetValue(0.0);
            //model_fn.RestoreParameters(path);
        }
        timer.Restart();
        // get next minibatch
        partTimer.Restart();
        auto minibatchData = minibatchSource->GetNextMinibatch(/*minibatchSizeInSequences=*/ (size_t)0, (size_t)minibatchSize, communicator->Workers().size(), communicator->CurrentWorker().m_globalRank, device);
        if (minibatchData.empty()) // finished one data pass--TODO: really? Depends on config. We really don't care about data sweeps.
            break;
        Dynamite::FromCNTKMB(args, { minibatchData[minibatchSource->StreamInfo(L"src")].data, minibatchData[minibatchSource->StreamInfo(L"tgt")].data }, { true, true }, DTYPE, device);
        let timeGetNextMinibatch = partTimer.Elapsed();
        //partTimer.Log("FromCNTKMB", minibatchData[minibatchSource->StreamInfo(L"tgt")].numberOfSamples);
        //args[0].resize(1);
        //args[1].resize(1);
#if 0   // for debugging: reduce #sequences to 3, and reduce their lengths
        args[0].resize(3);
        args[1].resize(3);
        let TrimLength = [](Variable& seq, size_t len) // chop off all frames after 'len', assuming the last axis is the length
        {
            seq = Slice(seq, Axis((int)seq.Shape().Rank()-1), 0, (int)len);
        };
        // input
        TrimLength(args[0][0], 2);
        TrimLength(args[0][1], 4);
        TrimLength(args[0][2], 3);
        // target
        TrimLength(args[1][0], 3);
        TrimLength(args[1][1], 2);
        TrimLength(args[1][2], 2);
#endif
        let numSeq = args[0].size();
        size_t numLabels = 0, numSamples = 0, maxSamples = 0, maxLabels = 0;
        for (let& seq : args[0])
        {
            let len = seq.Shape().Dimensions().back();
            numSamples += len;
            maxSamples = max(maxSamples, len);
        }
        for (let& seq : args[1])
        {
            let len = seq.Shape().Dimensions().back();
            numLabels += len;
            maxLabels = max(maxLabels, len);
        }
        //partTimer.Log("GetNextMinibatch", numLabels);
        fprintf(stderr, "%d: #seq: %d, #words: %d -> %d, max len %d -> %d, lr=%.8f * %.8f\n", (int)mbCount,
                (int)numSeq, (int)numSamples, (int)numLabels, (int)maxSamples, (int)maxLabels,
                lr0, learner->LearningRate() / lr0);
        // train minibatch
        partTimer.Restart();
        auto mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        //mbLoss = criterion_fn(args[0], args[1]);
        let timeBuildGraph = partTimer.Elapsed();
        //partTimer.Log("criterion_fn", numLabels);
        // backprop and model update
        partTimer.Restart();
        mbLoss.Value()->AsScalar<float>();
        let timeForward = partTimer.Elapsed();
        //fprintf(stderr, "%.5f\n", mbLoss.Value()->AsScalar<float>()), fflush(stderr);
        //partTimer.Log("ForwardProp", numLabels);
        partTimer.Restart();
        mbLoss.Backward(gradients);
        let timeBackward = partTimer.Elapsed();
        //partTimer.Log("BackProp", numLabels);
        // note: we must use numScoredLabels here
        let numScoredLabels = numLabels - numSeq; // the <s> is not scored; that's one per sequence. Do not count for averages.
        fprintf(stderr, "{%.2f, %d-%d}\n", mbLoss.Value()->AsScalar<float>(), (int)numLabels, (int)numSeq);
        MinibatchInfo info{ /*atEndOfData=*/false, /*sweepEnd=*/false, /*numberOfSamples=*/numScoredLabels, mbLoss.Value(), mbLoss.Value() };
        info.trainingLossValue->AsScalar<float>();
        partTimer.Restart();
        learner->Update(gradients, info);
        let timePerUpdate = partTimer.Elapsed();
        //partTimer.Log("Update", numLabels);
        let lossPerLabel = info.trainingLossValue->AsScalar<float>() / info.numberOfSamples; // note: this does the GPU sync, so better do that only every N
        totalLabels += info.numberOfSamples;
        // I once saw a strange (impossible) -1e23 or so CE loss, no idea where that comes from. Skip it in smoothed loss. Does not seem to hurt the convergence.
        if (lossPerLabel < 0)
        {
            fprintf(stderr, "%d STRANGE CrossEntropy loss = %.7f, not counting in accumulative loss, seenLabels=%d, words/sec=%.1f\n",
                (int)mbCount, lossPerLabel, (int)totalLabels,
                info.numberOfSamples / timer.ElapsedSeconds());
            continue;
        }
        let smoothedLossVal = smoothedLoss.Update(lossPerLabel, info.numberOfSamples);
        let elapsed = timer.ElapsedSeconds(); // [sec]
        if (communicator->CurrentWorker().IsMain())
            fprintf(stderr, "%d: >> loss = %.7f; PPL = %.3f << smLoss = %.7f, smPPL = %.2f, seenLabels=%d, %.1f w/s, %.1f ms/w, m=%.0f, g=%.0f, f=%.0f, b=%.0f, u=%.0f ms\n",
                            (int)mbCount, lossPerLabel, exp(lossPerLabel), smoothedLossVal, exp(smoothedLossVal), (int)totalLabels,
                            info.numberOfSamples / elapsed, 1000.0/*ms*/ * elapsed / info.numberOfSamples,
                            1000.0 * timeGetNextMinibatch, 1000.0 * timeBuildGraph, 1000.0 * timeForward, 1000.0 * timeBackward, 1000.0 * timePerUpdate);
        // log
        // Do this last, which forces a GPU sync and may avoid that "cannot resize" problem
        if (mbCount < 400 || mbCount % 5 == 0)
            fflush(stderr);
        if (std::isnan(lossPerLabel))
            throw runtime_error("Loss is NaN.");
        //if (mbCount == 10)
        //    exit(0);
    }
}


int mt_main(int argc, char *argv[])
{
    Internal::PrintBuiltInfo();
    try
    {
        // minimalistic argument parser, only to get a pathname
        if (argc != 3)
            throw invalid_argument("required command line: --id IDSTRING, where IDSTRING is used to form the log and model path for now");
        let* pExpId = argv[2];
        wstring experimentId(pExpId, pExpId + strlen(pExpId)); // (cheap conversion to wchar_t)
        wstring workingDirectory = L"d:/mt/experiments";       // output dir = "$workingDirectory/$experimentId/"
        Train(workingDirectory + L"/" + experimentId);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
        fflush(stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
