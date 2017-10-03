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

fun BidirectionalLSTMEncoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    vector<BinaryModel> layers;
    for (size_t i = 0; i < numLayers; i++)
        layers.push_back(Dynamite::Sequence::BiRecurrence(GRU(hiddenDim, device), Constant({ hiddenDim }, DTYPE, 0.0, device, Named("fwdInitialValue")),
                                                          GRU(hiddenDim, device), Constant({ hiddenDim }, DTYPE, 0.0, device, Named("bwdInitialValue"))));
    vector<UnaryBroadcastingModel> bns;
    for (size_t i = 0; i < numLayers-1; i++)
        bns.push_back(Dynamite::BatchNormalization(device, 1, Named("bnBidi")));
    vector<ModelParametersPtr> nested;
    nested.insert(nested.end(), layers.begin(), layers.end());
    nested.insert(nested.end(), bns.begin(), bns.end());
    // BUGBUG: If I change to Dynamite::Model, the model trains differently or causes compilation errors.
    return /*Dynamite::Model*/BinaryModel({}, NameNumberedParameters(nested),
    [=](const Variable& xFwd, const Variable& xBwd) -> Variable
    {
        // the first layer has different inputs for forward and backward
        auto h = layers[0](xFwd, xBwd);
        for (size_t i = 1; i < numLayers; i++)
        {
            // do another layer
            h = layers[i](h, h);
            // after each additional layer, so batch norm
            // BUGBUG: Why not the first? Seems like a bug.
            h = bns[i - 1](h);
        }
        return h;
    });
}

// Bahdanau attention model
// (query, keys as tensor, data sequence as tensor) -> interpolated data vector
//  - keys used for the weights
//  - data gets interpolated
// Here they are the same.
fun AttentionModelBahdanau(size_t attentionDim1)
{
    auto Q = Parameter({ attentionDim1, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"Q"); // query projection
    auto v = Parameter({ attentionDim1 }, DTYPE, GlorotUniformInitializer(), device, L"v"); // tanh projection
    let normQ = LengthNormalization(device);
    return Dynamite::Model({ Q, /*K,*/ v }, { { L"normQ", normQ } },
    [=](const Variable& query, const Variable& projectedKeys/*keys*/, const Variable& data) -> Variable
    {
        // compute attention weights
        CountAPICalls(3);
        let projectedQuery = normQ(Times(Q, query, Named("Q"))); // [A x 1]
        let tanh = Tanh((projectedQuery + projectedKeys), Named("attTanh")); // [A x T]
#if 0
        CountAPICalls(1);
        let u = InnerProduct(tanh, v, Axis(0), Named("vProj")); // [1 x T] col vector
        let w = Dynamite::Softmax(u, Axis(1));
        CountAPICalls(2);
        let res = Reshape(InnerProduct(data, w, Axis(1), Named("att")), NDShape{ attentionDim1 }); // [A]
#else
        CountAPICalls(1);
        let u = TransposeTimes(tanh, v, Named("vProj")); // [T] col vector
        let w = Dynamite::Softmax(u);
        CountAPICalls(1);
        let res = Times(data, w, Named("att")); // [A]
#endif
        return res;
     });
}

// simple helper for temporary conversions; inefficient
vector<Variable> AsVector(const Variable& seq)
{
    vector<Variable> res;
    as_vector(res, seq);
    return res;
}

// reference attention model
fun AttentionModelReference(size_t attentionDim1)
{
    let projectQuery = Linear(attentionDim1, ProjectionOptions::weightNormalize, device);
    let normH = LengthNormalization(device); // note: can't move this inside Linear since it is applied after adding two factors
    let profiler = Function::CreateDynamicProfiler(1, L"attention");
    let zBarrier   = Barrier(20, Named("zBarrier"));
    let resBarrier = Barrier(20, Named("resBarrier"));
    StaticModel doToTanh(/*isBasicBlock=*/false, [=](const Variable& h, const Variable& historyProjectedKey)
    {
        let hProjected = projectQuery(h); // [A]. Batched.
        CountAPICalls(2);
        let tanh = Tanh(normH(hProjected + historyProjectedKey), Named("attTanh")); // [A]. Batched.
        return tanh;
    });
    return Dynamite::Model({ }, { { L"normH", normH }, { L"projectQuery", projectQuery } },
        [=](const Variable& h,                        // [A] decoder hidden state
            const Variable& historyProjectedKey,      // [A] previous output, embedded
            const Variable& encodingProjectedKeysSeq, // [A x T] encoder hidden state seq, projected as key >> tanh
            const Variable& encodingProjectedDataSeq  // [A x T] encoder hidden state seq, projected as data
           ) -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler, false);
        // compute attention weights
        let tanh = doToTanh(h, historyProjectedKey); // [A]
        CountAPICalls(1);
        let uSeq = InnerProduct(tanh, encodingProjectedKeysSeq, Axis(0), Named("u")); // [1 x T]
        let wSeq = Dynamite::Softmax(uSeq, Axis(1), Named("attSoftmax"), zBarrier);   // [1 x T]
        CountAPICalls(1);
        let res = resBarrier(InnerProduct(encodingProjectedDataSeq, wSeq, Axis_DropLastAxis, Named("attContext"))); // [.] inner product over a vectors
        Function::SetDynamicProfiler(prevProfiler);
        return res;
    });
}

// TODO: Break out initial step and recurrent step layers. Decoder will later pull them out frmo here.
fun AttentionDecoder(double dropoutInputKeepProb)
{
    // create all the layer objects
    let encBarrier = Barrier(600, Named("encBarrier"));
    let encoderKeysProjection = encBarrier >> Dense(attentionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("encoderKeysProjection")); }), ProjectionOptions::batchNormalize | ProjectionOptions::bias, device); // keys projection for attention
    let encoderDataProjection = encBarrier >> Dense(attentionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("encoderDataProjection")); }), ProjectionOptions::batchNormalize | ProjectionOptions::bias, device); // data projection for attention
    let embedTarget = Barrier(600, Named("embedTargetBarrier")) >> Embedding(embeddingDim, device, Named("embedTarget"));     // target embeddding
    let initialContext = Constant({ attentionDim }, DTYPE, 0.0, device, L"initialContext"); // 2 * because bidirectional --TODO: can this be inferred?
    let initialStateProjection = Barrier(20, Named("initialStateProjectionBarrier")) >> Dense(decoderRecurrentDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("initialStateProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);
    let stepBarrier = Barrier(20, Named("stepBarrier"));
    let stepFunction = GRU(decoderRecurrentDim, device);
    auto attentionModel = AttentionModelReference(attentionDim);
    let firstHiddenProjection = Barrier(600, Named("projBarrier")) >> Dense(decoderProjectionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return ReLU(x, Named("firstHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);
    vector<UnaryBroadcastingModel> resnets;
    for (size_t n = 0; n < numDecoderResNetProjections; n++)
        resnets.push_back(ResidualNet(decoderProjectionDim, device));
    let topHiddenProjection = Dense(topHiddenProjectionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("topHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);
    let outputProjection = Linear(tgtVocabSize, ProjectionOptions::weightNormalize | ProjectionOptions::bias, device);  // output layer without non-linearity (no sampling yet)

    // decode from a top layer of an encoder, using history as history
    map<wstring, ModelParametersPtr> nestedLayers =
    {
        { L"encoderKeysProjection",  encoderKeysProjection },
        { L"encoderDataProjection",  encoderDataProjection },
        { L"embedTarget",            embedTarget },
        { L"initialStateProjection", initialStateProjection },
        { L"stepFunction",           stepFunction },
        { L"attentionModel",         attentionModel },
        { L"firstHiddenProjection",  firstHiddenProjection },
        { L"topHiddenProjection",    topHiddenProjection },
        { L"outputProjection",       outputProjection },
    };
    for (let& resnet : resnets)
        nestedLayers[L"resnet[" + std::to_wstring(nestedLayers.size()) + L"]"] = resnet;
    let profiler = Function::CreateDynamicProfiler(1, L"decode");

    let outProjProfiler = Function::CreateDynamicProfiler(1, L"outProj");
    StaticModel doToOutput(/*isBasicBlock=*/false, [=](const Variable& state, const Variable& attentionContext)
    {
        // first one brings it into the right dimension
        let prevProfiler = Function::SetDynamicProfiler(outProjProfiler, false);
        auto state1 = firstHiddenProjection(state);
        // then a bunch of ResNet layers
        for (auto& resnet : resnets)
            state1 = resnet(state1);
        // one more transform, bringing back the attention context
        CountAPICalls(1);
        let topHidden = topHiddenProjection(Splice({ state1, attentionContext }, Axis(0)));
        // ^^ batchable; currently one per target word in MB (e.g. 600); could be one per batch (2 launches)
        // TODO: dropout layer here
        dropoutInputKeepProb;
        // compute output
        let z = outputProjection(topHidden);
        Function::SetDynamicProfiler(prevProfiler);
        return z;
    });

    return Dynamite::Model/*BinaryModel*/({ }, nestedLayers,
    [=](const Variable& history, const Variable& hEncs) -> Variable
    {
        // decoding loop
        CountAPICalls(2);
        Variable state = Slice(hEncs[0], Axis(0), encoderRecurrentDim, 2 * encoderRecurrentDim); // initial state for the recurrence is the final encoder state of the backward recurrence
        state = initialStateProjection(state);      // match the dimensions
        Variable attentionContext = initialContext; // note: this is almost certainly wrong
        // common subexpression of attention.
        // We pack the result into a dense matrix; but only after the matrix product, to allow for it to be batched.
        let encodingProjectedKeys = encoderKeysProjection(hEncs); // this projects the entire sequence
        let encodingProjectedData = encoderDataProjection(hEncs);
        vector<Variable> states(history.size());           // we put the time steps here
        vector<Variable> attentionContexts(states.size()); // and the attentionContexts
        let historyEmbedded = embedTarget(history);
        for (size_t t = 0; t < history.size(); t++)
        {
            // do recurrent step (in inference, history[t] would become states[t-1])
            CountAPICalls(1);
            let historyProjectedKey = historyEmbedded[t];
            let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph
            CountAPICalls(1);
            let input = stepBarrier(Splice({ historyProjectedKey, attentionContext }, Axis(0), Named("augInput")));
            state = stepFunction(state, input);
            // compute attention vector
            //attentionContext = attentionModel(state, /*keys=*/projectedKeys, /*data=*/hEncsTensor);
            attentionContext = attentionModel(state, historyProjectedKey, encodingProjectedKeys, encodingProjectedData);
            Function::SetDynamicProfiler(prevProfiler);
            states[t] = state;
            attentionContexts[t] = attentionContext;
            // TODO: This has now become just a recurrence with a tuple state. Let's have a function for that.
        }
        // stack of output transforms
        let z = doToOutput(Splice(move(states)/*state*/, Axis::EndStaticAxis()), Splice(move(attentionContexts), Axis::EndStaticAxis()));
        return z;
    });
}

fun CreateModelFunction()
{
    let embedFwd = Embedding(embeddingDim, device, Named("embedFwd"));
    let embedBwd = Embedding(embeddingDim, device, Named("embedBwd"));
    let encode = BidirectionalLSTMEncoder(numEncoderLayers, encoderRecurrentDim, 0.8);
    auto decode = AttentionDecoder(0.8);
    return BinaryModel({},
    {
        { L"embedSourceFwd", embedFwd },
        { L"embedSourceBwd", embedBwd },
        { L"encode",         encode   },
        { L"decode",         decode   }
    },
    [=](const Variable& sourceSeq, const Variable& historySeq) -> Variable
    {
        // embedding
        let eFwd = embedFwd(sourceSeq);
        let eBwd = embedBwd(sourceSeq);
        // encoder
        let hSeq = encode(eFwd, eBwd);
        // decoder (outputting log probs of words)
        let zSeq = decode(historySeq, hSeq);
        return zSeq;
    });
}

fun CreateCriterionFunction(const BinaryModel& model_fn)
{
    vector<Variable> /*features, historyVector,*/ labelsVector, zVector, losses; // TODO: remove this; leave it to Splice(&&)
    // features and labels are tensors with first dimension being the length
    BinaryModel criterion = [=](const Variable& source, const Variable& target) mutable -> Variable
    {
        // convert sequence tensors into sequences of tensors
        // and strip the corresponding boundary markers
        //  - features: strip any?
        //  - labels: strip leading <s>
        //  - history: strip training </s>
        CountAPICalls(2);
        let labels  = Slice(target, Axis(-1), 1, (int)target.size()    ); // labels  = targets without leading <s>
        let history = Slice(target, Axis(-1), 0, (int)target.size() - 1); // history = targets without trailing </s>
        // apply model function
        // returns the sequence of output log probs over words
        let z = model_fn(source, history);
        // compute loss per word
#if 1
        let sequenceLoss = Dynamite::Sequence::Map(BinaryModel([](const Variable& zVector, const Variable& label) { return Dynamite::CrossEntropyWithSoftmax(zVector, label); }));
        as_vector(zVector, z);
        as_vector(labelsVector, labels);
        sequenceLoss(losses, zVector, labelsVector);
        zVector.clear(); labelsVector.clear();
        let loss = Batch::sum(losses); // TODO: Batch is not the right namespace; but this does the right thing
        losses.clear();
#else
#if 1
        //let lossSeq = Dynamite::Sequence::map(zSeq, labelsSeq, [](const Variable& z, const Variable& y) { return Dynamite::CrossEntropyWithSoftmax(z, y); }, losses);
        as_vector(zVector, z);
        as_vector(labelsVector, labelsSeq);
        let sequenceLoss = Dynamite::Sequence::Map(BinaryModel([](const Variable& z, const Variable& y) { return Dynamite::CrossEntropyWithSoftmax(z, y); }));
        sequenceLoss(losses, zVector, labelsVector);
        let lossSeq = Splice(losses, Axis(0));
        zVector.clear(); labelsVector.clear();
        losses.clear();
#else
        let lossSeq = Dynamite::CrossEntropyWithSoftmax(zSeq, labelsSeq, Axis(0));
#endif
        CountAPICalls(1);
        let loss = ReduceSum(lossSeq, Axis_DropLastAxis);
#endif
        return loss;
    };
    let profiler = Function::CreateDynamicProfiler(1, L"all");
    // create a batch mapper (which will eventually allow suspension)
    let batchModel = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    return BinaryFoldingModel({}, { { L"model", model_fn } },
    [=](const /*batch*/vector<Variable>& features, const /*batch*/vector<Variable>& labels) -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph
        vector<Variable> lossesPerSequence;
        batchModel(lossesPerSequence, features, labels);             // batch-compute the criterion
        CountAPICalls(1);
        let collatedLosses = Splice(move(lossesPerSequence), Axis(0), Named("seqLosses"));     // collate all seq lossesPerSequence
        // ^^ this is one launch per MB
        CountAPICalls(1);
        let mbLoss = ReduceSum(collatedLosses, Axis_DropLastAxis, Named("batchLoss"));  // aggregate over entire minibatch
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
    //vector<Variable> d1{ Constant({ srcVocabSize }, DTYPE, 0.0, device) };
    //vector<Variable> d2{ Constant({ tgtVocabSize }, DTYPE, 0.0, device) };
    //vector<Variable> d3;
    //model_fn(d3, d1, d2);
    model_fn(Constant({ srcVocabSize, 1 }, DTYPE, 0.0, device), Constant({ tgtVocabSize, 1 }, DTYPE, 0.0, device));

    model_fn.LogParameters();

    let parameters = model_fn.Parameters();
    size_t numParameters = 0;
    for (let& p : parameters)
        numParameters += p.Shape().TotalSize();
    fprintf(stderr, "Total number of learnable parameters is %u in %d parameter tensors.\n", (unsigned int)numParameters, (int)parameters.size()), fflush(stderr);
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
#if 0   // hack (until we fix the reader): pack sequences of similar length
        //  - get 10 x the minibatch
        //  - pick a length, e.g. of the first  --use target length here
        //  - pick the 10% that are closest
        //  - sort them by length --use source length to aid the attention model
        auto minibatchData = minibatchSource->GetNextMinibatch(/*minibatchSizeInSequences=*/ (size_t)0, 5 * (size_t)minibatchSize, communicator->Workers().size(), communicator->CurrentWorker().m_globalRank, device);
        Dynamite::FromCNTKMB(args, { minibatchData[minibatchSource->StreamInfo(L"src")].data, minibatchData[minibatchSource->StreamInfo(L"tgt")].data }, { true, true }, DTYPE, device);
        auto& sourceSents = args[0];
        auto& targetSents = args[1];
        let desiredLength = targetSents[0].size(); // length of first entry
        vector<size_t> indices; // (sentIndex, someValue) array to sort
        for (let& _ : targetSents)
            indices.push_back(indices.size()), _;
        // keep the N closest in target length
        sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return abs((int)targetSents[i].size() - (int)desiredLength) < abs((int)targetSents[j].size() - (int)desiredLength); });
        size_t numTargetWordsInBatch = 0;
        let desiredNumWords = minibatchSize / communicator->Workers().size();
        for (size_t i = 0; i < indices.size(); i++)
        {
            let len = targetSents[indices[i]].size();
            if (i > 0 && numTargetWordsInBatch + len > desiredNumWords)
            {
                indices.resize(i);
                break;
            }
            numTargetWordsInBatch += len;
        }
        // now sort by source length, longest first
        sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return (int)sourceSents[i].size() > (int)sourceSents[j].size(); }); // longest first
        // and update the args
        let updateSents = [&](vector<Variable>& sents)
        {
            vector<Variable> newSents;
            for (auto index : indices)
                newSents.push_back(sents[index]);
            sents = move(newSents);
        };
        updateSents(sourceSents);
        updateSents(targetSents);
#else
        auto minibatchData = minibatchSource->GetNextMinibatch(/*minibatchSizeInSequences=*/ (size_t)0, (size_t)minibatchSize, communicator->Workers().size(), communicator->CurrentWorker().m_globalRank, device);
        if (minibatchData.empty()) // finished one data pass--TODO: really? Depends on config. We really don't care about data sweeps.
            break;
        Dynamite::FromCNTKMB(args, { minibatchData[minibatchSource->StreamInfo(L"src")].data, minibatchData[minibatchSource->StreamInfo(L"tgt")].data }, { true, true }, DTYPE, device);
#endif
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
        let numAPICalls0 = CountAPICalls(0);
        //criterion_fn(args[0], args[1]); // call it once before, to flush that thing that we otherwise also measure, whatever that is
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
        let numAPICalls = CountAPICalls(0) - numAPICalls0;
        numAPICalls;
        //fprintf(stderr, "#API calls = %d\n", (int)numAPICalls), fflush(stderr);
        //exit(1);
        //partTimer.Log("criterion_fn", numLabels);
        // backprop and model update
        partTimer.Restart();
        mbLoss.Value()->AsScalar<float>();
        let timeForward = partTimer.Elapsed();
        //fprintf(stderr, "%.5f\n", mbLoss.Value()->AsScalar<float>()), fflush(stderr);
        //partTimer.Log("ForwardProp", numLabels);
        // note: we must use numScoredLabels here
        let numScoredLabels = numLabels - numSeq; // the <s> is not scored; that's one per sequence. Do not count for averages.
        fprintf(stderr, "{%.2f, %d-%d}\n", mbLoss.Value()->AsScalar<float>(), (int)numLabels, (int)numSeq), fflush(stderr);
        partTimer.Restart();
        mbLoss.Backward(gradients);
        let timeBackward = partTimer.Elapsed();
        //partTimer.Log("BackProp", numLabels);
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
