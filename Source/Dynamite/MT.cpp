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

// --- high-level configuration ---
// all configurable items that vary across systems
size_t srcVocabSize, tgtVocabSize;
wstring srcTxtFile, srcVocabFile, tgtTxtFile, tgtVocabFile;
//int srcPositionMin = -100;  // positions -100..100
//int srcPositionMax = +100;
//size_t srcPosition0 = 105; // 201 position embeddings, with 0 at this word index (they live in the same index space and embedding matrices as the word identities)
//int tgtPositionMin = 0;    // positions 0..100
//int tgtPositionMax = 100;
//size_t tgtPosition0 = 5;

size_t embeddingDim = 512;
size_t attentionDim = 512;
size_t numEncoderLayers = 3;
size_t encoderRecurrentDim = 512;
size_t decoderRecurrentDim = 1024;
size_t numDecoderResNetProjections = 4;
size_t decoderProjectionDim = 768;
size_t topHiddenProjectionDim = 1024;
size_t subMinibatches = 1;// 0;

static void SetConfigurationVariablesFor(string systemId) // set variables; overwrite defaults
{
    if (systemId == "chs_enu")
    {
        srcVocabSize = 78440;
        tgtVocabSize = 79439;
        srcTxtFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.CHS.txt"; srcVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.source.vocab";
        tgtTxtFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.ENU.txt"; tgtVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.target_input.vocab";
    }
    else if (systemId == "chs_enu_small")
    {
        srcVocabSize = 78440;
        tgtVocabSize = 79439;
        srcTxtFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.small.CHS.txt"; srcVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.source.vocab";
        tgtTxtFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.small.ENU.txt"; tgtVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.target_input.vocab";
    }
    else if (systemId == "rom_enu")
    {
        srcVocabSize = 27579 + 3;
        tgtVocabSize = 21163 + 3;
        srcTxtFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.ro.shuf"; srcVocabFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.ro.vocab";
        tgtTxtFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.en.shuf"; tgtVocabFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.en.vocab";
    }
    else if (systemId == "karnak_sample")
    {
        // what was the vocab size again here?
        srcTxtFile = L"d:/work/Karnak/sample-model/data/train.src"; srcVocabFile = L"d:/work/Karnak/sample-model/data/vocab.src";
        tgtTxtFile = L"d:/work/Karnak/sample-model/data/train.tgt"; tgtVocabFile = L"d:/work/Karnak/sample-model/data/vocab.tgt";
    }
    else
        InvalidArgument("Invalid system id '%S'", systemId.c_str());
}

size_t mbCount = 0; // made a global so that we can trigger debug information on it
#define DOLOG(var) (var)//((mbCount % 100 == 99) ? LOG(var) : 0)

fun BidirectionalLSTMEncoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    vector<BinaryModel> layers;
    for (size_t i = 0; i < numLayers; i++)
        layers.push_back(Dynamite::Sequence::BiRecurrence(GRU(hiddenDim), Constant({ hiddenDim }, CurrentDataType(), 0.0, CurrentDevice(), Named("fwdInitialValue")),
                                                          GRU(hiddenDim), Constant({ hiddenDim }, CurrentDataType(), 0.0, CurrentDevice(), Named("bwdInitialValue"))));
    vector<UnaryBroadcastingModel> bns;
    for (size_t i = 0; i < numLayers-1; i++)
        bns.push_back(Dynamite::BatchNormalization(1, Named("bnBidi")));
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

// reference attention model
// Returns the attention probabilities. Caller must do the weighted average.
fun AttentionModelReference(size_t attentionDim1)
{
    let projectQuery = Linear(attentionDim1, ProjectionOptions::weightNormalize);
    let normH = LengthNormalization(); // note: can't move this inside Linear since it is applied after adding two factors
    let profiler = Function::CreateDynamicProfiler(1, L"attention");
    let zBarrier = Barrier(20, Named("zBarrier"));
    let doToTanh = StaticModel(/*isBasicBlock=*/false, [=](const Variable& h, const Variable& historyProjectedKey)
    {
        let hProjected = projectQuery(h); // [A]. Batched.
        let tanh = Tanh(normH(hProjected + historyProjectedKey), Named("attTanh")); CountAPICalls(2); // [A]. Batched.
        return tanh;
    });
    return /*Dynamite::Model*/TernaryModel({ }, { { L"normH", normH }, { L"projectQuery", projectQuery } },
        [=](const Variable& h,                       // [A] decoder hidden state
            const Variable& historyProjectedKey,     // [A] previous output, embedded
            const Variable& encodingProjectedKeysSeq // [A x T] encoder hidden state seq, projected as key >> tanh
           ) -> Variable
        {
            let prevProfiler = Function::SetDynamicProfiler(profiler, false);
            let tanh = doToTanh(h, historyProjectedKey); // [A]
            let uSeq = InnerProduct(tanh, encodingProjectedKeysSeq, Axis(0), Named("u")); CountAPICalls(1); // [1 x T]
            let wSeq = Dynamite::Softmax(uSeq, Axis(1), Named("attSoftmax"), zBarrier);                     // [1 x T]
            Function::SetDynamicProfiler(prevProfiler);
            return wSeq;
        });
}

// TODO: Break out initial step and recurrent step layers. Decoder will later pull them out frmo here.
fun AttentionDecoder(double dropoutInputKeepProb)
{
    // create all the layer objects
    let encBarrier = Barrier(600, Named("encBarrier"));
    let encoderKeysProjection = encBarrier >> Dense(attentionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("encoderKeysProjection")); }), ProjectionOptions::batchNormalize | ProjectionOptions::bias); // keys projection for attention
    let encoderDataProjection = encBarrier >> Dense(attentionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("encoderDataProjection")); }), ProjectionOptions::batchNormalize | ProjectionOptions::bias); // data projection for attention
    let embedTarget = Barrier(600, Named("embedTargetBarrier")) >> Embedding(embeddingDim, Named("embedTarget"));     // target embeddding
    let initialContext = Constant({ attentionDim }, CurrentDataType(), 0.0, CurrentDevice(), L"initialContext"); // 2 * because bidirectional --TODO: can this be inferred?
    let initialStateProjection = Barrier(20, Named("initialStateProjectionBarrier")) >> Dense(decoderRecurrentDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("initialStateProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias);
    let stepBarrier = Barrier(20, Named("stepBarrier"));
    let stepFunction = GRU(decoderRecurrentDim);
    auto attentionModel = AttentionModelReference(attentionDim);
    let attBarrier = Barrier(20, Named("attBarrier"));
    let firstHiddenProjection = Barrier(600, Named("projBarrier")) >> Dense(decoderProjectionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return ReLU(x, Named("firstHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias);
    vector<UnaryBroadcastingModel> resnets;
    for (size_t n = 0; n < numDecoderResNetProjections; n++)
        resnets.push_back(ResidualNet(decoderProjectionDim));
    let topHiddenProjection = Dense(topHiddenProjectionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("topHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias);
    let outputProjection = Linear(tgtVocabSize, ProjectionOptions::weightNormalize | ProjectionOptions::bias);  // output layer without non-linearity (no sampling yet)

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
    // BUGBUG: Setting this to true fails with an off batch axis.
    let doToOutput = StaticModel(/*isBasicBlock=*/false, [=](const Variable& state, const Variable& attentionContext)
    {
        // first one brings it into the right dimension
        let prevProfiler = Function::SetDynamicProfiler(outProjProfiler, false);
        auto state1 = firstHiddenProjection(state);
        // then a bunch of ResNet layers
        for (auto& resnet : resnets)
            state1 = resnet(state1);
        // one more transform, bringing back the attention context
        let topHidden = topHiddenProjection(Splice({ state1, attentionContext }, Axis(0))); CountAPICalls(1);
        // TODO: dropout layer here
        dropoutInputKeepProb;
        // compute output
        let z = outputProjection(topHidden);
        Function::SetDynamicProfiler(prevProfiler);
        return z;
    }, Named("doToOutput"));

    return /*Dynamite::Model*/BinaryModel({ }, nestedLayers,
    [=](const Variable& history, const Variable& hEncoderSeq) -> Variable
    {
        // decoding loop
        CountAPICalls(2);
        Variable state = Slice(hEncoderSeq[0], Axis(0), (int)encoderRecurrentDim, 2 * (int)encoderRecurrentDim); // initial state for the recurrence is the final encoder state of the backward recurrence
        state = initialStateProjection(state);      // match the dimensions
        Variable attentionContext = initialContext; // note: this is almost certainly wrong
        // common subexpression of attention.
        // We pack the result into a dense matrix; but only after the matrix product, to allow for it to be batched.
        let encodingProjectedKeysSeq = encoderKeysProjection(hEncoderSeq); // this projects the entire sequence
        let encodingProjectedDataSeq = encoderDataProjection(hEncoderSeq);
        vector<Variable> states(history.size());           // we put the time steps here
        vector<Variable> attentionContexts(states.size()); // and the attentionContexts
        let historyEmbedded = embedTarget(history);
        for (size_t t = 0; t < history.size(); t++)
        {
            // do recurrent step (in inference, history[t] would become states[t-1])
            let historyProjectedKey = historyEmbedded[t]; CountAPICalls(1);
            let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph
            let input = stepBarrier(Splice({ historyProjectedKey, attentionContext }, Axis(0), Named("augInput"))); CountAPICalls(1);
            state = stepFunction(state, input);

            // compute attention vector
            let attentionWeightSeq = attentionModel(state, historyProjectedKey, encodingProjectedKeysSeq);
            attentionContext = attBarrier(InnerProduct(encodingProjectedDataSeq, attentionWeightSeq, Axis_DropLastAxis, Named("attContext"))); CountAPICalls(1); // [.] inner product over a vectors
            Function::SetDynamicProfiler(prevProfiler);

            // save the results
            // TODO: This has now become just a recurrence with a tuple state. Let's have a function for that. Or an iterator!
            states[t] = state;
            attentionContexts[t] = attentionContext;
        }
        let stateSeq = Splice(move(states)/*state*/, Axis::EndStaticAxis());
        let attenContextSeq = Splice(move(attentionContexts), Axis::EndStaticAxis());
        // stack of output transforms
        let z = doToOutput(stateSeq, attenContextSeq);
        return z;
    });
}

fun CreateModelFunction()
{
    let embedFwd = Embedding(embeddingDim, Named("embedFwd"));
    let embedBwd = Embedding(embeddingDim, Named("embedBwd"));
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
    BinaryModel criterion = [=](const Variable& sourceSeq, const Variable& targetSeq) mutable -> Variable
    {
        // convert sequence tensors into sequences of tensors
        // and strip the corresponding boundary markers
        //  - labels: strip leading <s>
        //  - history: strip training </s>
        let labelsSeq  = Slice(targetSeq, Axis(-1), 1, (int)targetSeq.size()    ); CountAPICalls(); // labels  = targets without leading <s>
        let historySeq = Slice(targetSeq, Axis(-1), 0, (int)targetSeq.size() - 1); CountAPICalls(); // history = targets without trailing </s>
        // apply model function
        // returns the sequence of output log probs over words
        let zSeq = model_fn(sourceSeq, historySeq);
        // compute loss per word
        let lossSeq = Dynamite::CrossEntropyWithSoftmax(zSeq, labelsSeq);
        return lossSeq;
    };
    let profiler = Function::CreateDynamicProfiler(1, L"all");
    // create a batch mapper (which will eventually allow suspension)
    let batchModel = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    return BinaryFoldingModel({}, { { L"model", model_fn } },
    [=](const /*batch*/vector<Variable>& features, const /*batch*/vector<Variable>& labels) -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph
        vector<Variable> lossSequences;
        batchModel(lossSequences, features, labels);             // batch-compute the criterion
        let collatedLosses = Splice(move(lossSequences), Axis(-1), Named("seqLosses")); CountAPICalls(1);    // concatenate all seq lossSequences
        let mbLoss = ReduceSum(collatedLosses, Axis_DropLastAxis, Named("batchLoss")); CountAPICalls(1); // aggregate over entire minibatch
        Function::SetDynamicProfiler(prevProfiler);
        return mbLoss;
    });
}

void Train(string systemId, wstring outputDirectory)
{
    SetConfigurationVariablesFor(systemId);
    let communicator = MPICommunicator();
#if 1 // while we are running with MPI, we always start from start
    let numGpus = DeviceDescriptor::AllDevices().size() -1;
    let ourRank = communicator->CurrentWorker().m_globalRank;
    if (numGpus > 0)
        SetCurrentDevice(DeviceDescriptor::GPUDevice((unsigned int)(ourRank % numGpus)));
    else
        SetCurrentDevice(DeviceDescriptor::CPUDevice());
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
        PlainTextStreamConfiguration(L"src", srcVocabSize, { srcTxtFile }, { srcVocabFile, L"<s>", L"</s>", L"<unk>" }),
        PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { tgtTxtFile }, { tgtVocabFile, L"<s>", L"</s>", L"<unk>" })
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
    model_fn(Constant({ srcVocabSize, 1 }, CurrentDataType(), 0.0, CurrentDevice()), Constant({ tgtVocabSize, 1 }, CurrentDataType(), 0.0, CurrentDevice()));

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
    auto baseLearner = AdamLearner(parameters, TrainingParameterPerSampleSchedule(vector<double>{ lr0, lr0/2, lr0/4, lr0/8 }, epochSize),
        MomentumAsTimeConstantSchedule(40000), true, MomentumAsTimeConstantSchedule(400000), /*eps=*/1e-8, /*adamax=*/false,
        learnerOptions);
#endif
    let& learner = CreateDataParallelDistributedLearner(communicator, baseLearner, /*distributeAfterSamples =*/ 0, /*useAsyncBufferedParameterUpdate =*/ false);
    unordered_map<Parameter, NDArrayViewPtr> gradients;
    for (let& p : parameters)
        gradients[p] = nullptr;

    vector<vector<vector<Variable>>> args; // [subMinibatchIndex, streamIndex, sequenceIndex]
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
        double syncGpu() { return CNTK::NDArrayView::Sync(CurrentDevice()); }
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
    size_t saveEvery = 2000;
    size_t startMbCount = 0;// 12000;
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
    mbCount = startMbCount;
    for (;;)
    {
        // checkpoint
        // BUGBUG: For now, 'saveEvery' must be a multiple of subMinibatches, otherwise it won't save
        if (mbCount % saveEvery == 0 &&
            (/*startMbCount == 0 ||*/ mbCount > startMbCount)) // don't overwrite the starting model
        {
            // TODO: Move this to a separate function.
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
        Dynamite::GetSubBatches(args, { L"src", L"tgt" }, subMinibatches, /*shuffleSeed=*/mbCount, minibatchSource, minibatchSize, communicator, CurrentDataType(), CurrentDevice());
        let timeGetNextMinibatch = partTimer.Elapsed();
        //partTimer.Log("FromCNTKMB", minibatchData[minibatchSource->StreamInfo(L"tgt")].numberOfSamples);

        // SUB-MINIBATCH LOOP
        // We get 10 x the minibatch, sort it by source length, and then process it in consecutive chunks of 1/10, which form the actual minibatch for training
        for (let& subBatchArgs : args)
        {
            timer.Restart();
#if 0       // for debugging: reduce #sequences to 3, and reduce their lengths
            subBatchArgs[0].resize(3);
            subBatchArgs[1].resize(3);
            let TrimLength = [](Variable& seq, size_t len) // chop off all frames after 'len', assuming the last axis is the length
            {
                seq = Slice(seq, Axis((int)seq.Shape().Rank()-1), 0, (int)len);
            };
            // input
            TrimLength(subBatchArgs[0][0], 2);
            TrimLength(subBatchArgs[0][1], 4);
            TrimLength(subBatchArgs[0][2], 3);
            // target
            TrimLength(subBatchArgs[1][0], 3);
            TrimLength(subBatchArgs[1][1], 2);
            TrimLength(subBatchArgs[1][2], 2);
#endif
            let numSeq = subBatchArgs[0].size();
            size_t numLabels = 0, numSamples = 0, maxSamples = 0, maxLabels = 0;
            for (let& seq : subBatchArgs[0])
            {
                let len = seq.Shape().Dimensions().back();
                numSamples += len;
                maxSamples = max(maxSamples, len);
            }
            for (let& seq : subBatchArgs[1])
            {
                let len = seq.Shape().Dimensions().back();
                numLabels += len;
                maxLabels = max(maxLabels, len);
            }
            //partTimer.Log("GetNextMinibatch", numLabels);
            fprintf(stderr, "%d: #seq: %d, #words: %d -> %d, max len %d -> %d, lr=%.8f * %.8f\n", (int)mbCount,
                    (int)numSeq, (int)numSamples, (int)numLabels, (int)maxSamples, (int)maxLabels,
                    lr0, learner->LearningRate() / lr0);
#if 0       // log the sequences
            for (size_t n = 0; n < numSeq; n++)
            {
                subBatchArgs[0][n].Value()->LogToFile(L"Source_" + to_wstring(n), stderr, SIZE_MAX);
                subBatchArgs[1][n].Value()->LogToFile(L"Target_" + to_wstring(n), stderr, SIZE_MAX);
            }
#endif
            // train minibatch
            let numAPICalls0 = CountAPICalls(0);
            //criterion_fn(subBatchArgs[0], subBatchArgs[1]); // call it once before, to flush that thing that we otherwise also measure, whatever that is
            partTimer.Restart();
            auto mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
            //mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
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
            //fprintf(stderr, "%.7f\n", mbLoss.Value()->AsScalar<float>()), fflush(stderr);
            //exit(1);
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
            mbCount++;
        }
    }
}

// minimalist command-line eargument parser, requires all arguments in order
class GetCommandLineArguments
{
    size_t argc; char** argv;
    string Front() const { if (argc == 0) LogicError("GetCommandLineArguments: out of command-line arguments??"); return *argv; }
    string Pop() { let val = Front(); argc--, argv++; return val; }
    string PopArg(const string& argTag) // check that next arg is argTag, Pop it, then Pop and return the subsequent arg
    {
        if (argc < 2 || Pop() != argTag)
            InvalidArgument("command-line argument '%s' missing or out of order", argTag.c_str());
        return Pop();
    }
    // recursive templates. Each one pops the next argument, where the C++ type selects the conversion
    template <typename ...ArgTypes>
    void Get(const string& argTag, wstring& argVal, ArgTypes&& ...remainingArgs)
    {
        let val = PopArg(argTag);
        argVal = wstring(val.begin(), val.end()); // note: this only presently works for ASCII arguments, easy to fix if ever needed
        Get(std::forward<ArgTypes>(remainingArgs)...); // recurse
    }
    template <typename ...ArgTypes>
    void Get(const string& argTag, string& argVal, ArgTypes&& ...remainingArgs)
    {
        argVal = PopArg(argTag);
        Get(std::forward<ArgTypes>(remainingArgs)...); // recurse
    }
    // if needed, we can add Get() for other types, and with defaults (triples)
    void Get() // end of recursion
    {
        if (argc > 0)
            InvalidArgument("unexpected extraneous command-line argument '%s'", Pop());
    }
public:
    // call this way: GetCommandLineArguments(argc, argv, "--tag1", variable1, "--tag2", variable2...)
    template <typename ...ArgTypes>
    GetCommandLineArguments(int argc, char *argv[], ArgTypes&& ...remainingArgs) :
        argc(argc), argv(argv)
    {
        Pop(); // pop program name itself
        Get(std::forward<ArgTypes>(remainingArgs)...);
    }
};

int mt_main(int argc, char *argv[])
{
    Internal::PrintBuiltInfo();
    try
    {
        string systemId;
        wstring experimentId;
        try
        {
            GetCommandLineArguments(argc, argv, "--system", systemId, "--id", experimentId);
        }
        catch (const exception& e)
        {
            fprintf(stderr, "%s\n", e.what()), fflush(stderr);
            throw invalid_argument("required command line: --system SYSTEMID --id IDSTRING\n SYSTEMID = chs_enu, rom_enu, etc\n IDSTRING is used to form the log and model path for now");
        }
        //let* pExpId = argv[2];
        //wstring experimentId(pExpId, pExpId + strlen(pExpId)); // (cheap conversion to wchar_t)
        wstring workingDirectory = L"d:/mt/experiments";       // output dir = "$workingDirectory/$experimentId/"
        Train(systemId, workingDirectory + L"/" + experimentId + L"_" + wstring(systemId.begin(), systemId.end()));
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
        fflush(stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
