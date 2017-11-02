//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "PlainTextDeseralizer.h"
#include "Models.h"
#include "Layers.h"
#include "TimerUtility.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <io.h>
#include <boost/filesystem.hpp>
#include <sstream>

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
wstring srcTrainFile, srcDevFile, srcTestFile, srcVocabFile, tgtTrainFile, tgtDevFile, tgtTestFile, tgtVocabFile;
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
size_t subMinibatches = 10;
string learnerType = "adam";
double learningRate = 0.0003662109375;
bool use1BitSgd = false;
size_t saveEvery = 2000;
double  pruningThreshold = 10.0;
size_t maxBeam = 5;

static void SetConfigurationVariablesFor(string systemId) // set variables; overwrite defaults
{
    if (systemId == "chs_enu")
    {
        srcVocabSize = 78440;
        tgtVocabSize = 79439;
        srcTrainFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.CHS.txt";
        tgtTrainFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.ENU.txt";
        srcDevFile   = L"f:/local/data/2017_10_05_21h_46m_39s/val.CHS.txt";
        tgtDevFile   = L"f:/local/data/2017_10_05_21h_46m_39s/val.ENU.txt";
        srcTestFile  = L"f:/local/data/2017_10_05_21h_46m_39s/test.CHS.txt";
        tgtTestFile  = L"f:/local/data/2017_10_05_21h_46m_39s/test.ENU.txt";
        srcVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.source.vocab";
        tgtVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.target_input.vocab";
        subMinibatches = 10;
        learningRate *= 10;
    }
    else if (systemId == "chs_enu_small")
    {
        srcVocabSize = 78440;
        tgtVocabSize = 79439;
        srcTrainFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.CHS.txt";
        tgtTrainFile = L"f:/local/data/2017_10_05_21h_46m_39s/train.ENU.txt";
        srcDevFile   = L"f:/local/data/2017_10_05_21h_46m_39s/val.CHS.txt";
        tgtDevFile   = L"f:/local/data/2017_10_05_21h_46m_39s/val.ENU.txt";
        srcTestFile  = L"f:/local/data/2017_10_05_21h_46m_39s/test.CHS.txt";
        tgtTestFile  = L"f:/local/data/2017_10_05_21h_46m_39s/test.ENU.txt";
        srcVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.source.vocab";
        tgtVocabFile = L"f:/local/data/2017_10_05_21h_46m_39s/CHS.ENU.generalnn.target_input.vocab";
        // this config uses a much smaller system configuration than the default system
        embeddingDim = 128;
        attentionDim = 128;
        numEncoderLayers = 1;
        encoderRecurrentDim = 128;
        decoderRecurrentDim = 128;
        numDecoderResNetProjections = 0;
        decoderProjectionDim = 128;
        topHiddenProjectionDim = 128;
    }
    else if (systemId == "rom_enu")
    {
        srcVocabSize = 27579 + 3;
        tgtVocabSize = 21163 + 3;
        srcTrainFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.ro.shuf"; srcVocabFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.ro.vocab";
        tgtTrainFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.en.shuf"; tgtVocabFile = L"f:/hanyh-ws2/shared/forFrank/ROM-ENU-WMT/Data/corpus.bpe.en.vocab";
        // TODO: dev and test paths
    }
    else if (systemId == "karnak_sample")
    {
        // what was the vocab size again here?
        srcTrainFile = L"d:/work/Karnak/sample-model/data/train.src"; srcVocabFile = L"d:/work/Karnak/sample-model/data/vocab.src";
        tgtTrainFile = L"d:/work/Karnak/sample-model/data/train.tgt"; tgtVocabFile = L"d:/work/Karnak/sample-model/data/vocab.tgt";
        // TODO: dev and test paths
    }
    else
        InvalidArgument("Invalid system id '%S'", systemId.c_str());
}

size_t mbCount = 0; // made a global so that we can trigger debug information on it
#define DOLOG(var)                  ((mbCount % 50 == 1 && Dynamite::Batch::CurrentMapIndex() < 2) ? LOG(var)                                : 0)
#define DOPRINT(prefix, var, vocab) ((mbCount % 50 == 1 && Dynamite::Batch::CurrentMapIndex() < 2) ? PrintSequence((prefix), (var), (vocab)) : 0)
static void PrintSequence(const char* prefix, const Variable& seq, const wstring& vocabFile);

fun BidirectionalLSTMEncoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
{
    dropoutInputKeepProb;
    vector<BinaryModel> layers;
    for (size_t i = 0; i < numLayers; i++)
        layers.push_back(Dynamite::Sequence::BiRecurrence(GRU(hiddenDim), Constant({ hiddenDim }, CurrentDataType(), 0.0, CurrentDevice(), Named("fwdInitialValue")),
                                                          GRU(hiddenDim), Constant({ hiddenDim }, CurrentDataType(), 0.0, CurrentDevice(), Named("bwdInitialValue"))));
    vector<UnaryModel> bns;
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
            // But BN after this does not help, so probably it should be between those layers, not after.
            h = bns[i - 1](h);
        }
        DOLOG(h);
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
    // tanh(W1 h + W2 prevWord)
    let doToTanh = StaticModel(/*isBasicBlock=*/false,
        [=](const Variable& h, const Variable& historyEmbedded)
        {
            let hProjected = projectQuery(h); // [A]. Batched.
            let tanh = Tanh(normH(hProjected + historyEmbedded), Named("attTanh")); CountAPICalls(2); // [A]. Batched.
            return tanh;
        }, Named("doToTanh"));
    return /*Dynamite::Model*/TernaryModel({ }, { { L"normH", normH }, { L"projectQuery", projectQuery } },
        [=](const Variable& h,                       // [A] decoder hidden state
            const Variable& historyEmbedded,         // [A] previous output, embedded
            const Variable& encodingProjectedKeysSeq // [A x T] encoder hidden state seq, projected as key >> tanh
           ) -> Variable
        {
            let prevProfiler = Function::SetDynamicProfiler(profiler, false);
            let tanh = doToTanh(h, historyEmbedded); // [A]
            let uSeq = InnerProduct(tanh, encodingProjectedKeysSeq, Axis(0), Named("u")); CountAPICalls(1); // [1 x T]
            let wSeq = Dynamite::Softmax(uSeq, Axis(1), Named("attSoftmax"), zBarrier);                     // [1 x T]
            Function::SetDynamicProfiler(prevProfiler);
            DOLOG(wSeq);
            return wSeq;
        });
}

// helper to extract a struct member
template<typename CollectionType, typename CollectionMemberType>
auto TransformSelectMember(const CollectionType& fromCollection, CollectionMemberType pMember)
{
    typedef typename CollectionType::value_type ValueType;
    // BUGBUG: Figure out the return type. Getting random compilation errors.
    return vector<Variable/*ValueType*/>(Transform(fromCollection, [&](const ValueType& item) { return item.*pMember; }));
}

// the state we carry forward across decoding steps
struct DecoderState
{
    // actual state
    Variable state, attentionContext;
    // stuff that gets computed only once
    Variable encodingProjectedKeysSeq;
    Variable encodingProjectedDataSeq;
};

// this is the main beam decoder, for a single sentence
template<typename InitFunctionType, typename StepFunctionType, typename OutputFunctionType>
Variable BeamDecode(const Variable& hEncoderSeq, const InitFunctionType& initFunction, const StepFunctionType& stepFunction, const OutputFunctionType& outputFunction)
{
    let sentStartId = 1;
    //let sentEndId = 2; // TODO: get those from the actual dictionary
    let sentStartIdVar = Constant({ }, CurrentDataType(), sentStartId, CurrentDevice(), L"sentStart");
    auto axism1 = Axis(-1); // API BUGBUG: must change to const&, next time we change the lib header anyway
    Variable sentStart = OneHotOp(sentStartIdVar, tgtVocabSize, /*outputSparse=*/true, axism1/*Axis(-1)*/);
    sentStart.Value();
    list<DecoderState> state = { initFunction(hEncoderSeq) }; // initial state
    state; sentStart; 
    outputFunction; stepFunction;
#if 0
    // expansion
    for (;;)
    {
    }





    let historyEmbeddedSeq = embedTarget(historySeq);
    // TODO: Forward iterator
    for (size_t t = 0; t < historyEmbeddedSeq.size(); t++)
    {
        // do recurrent step (in inference, historySeq[t] would become states[t-1])
        let historyEmbedded = historyEmbeddedSeq[t]; CountAPICalls(1);

        // do one decoding step
        decoderState = decoderStepFunction(decoderState, historyEmbedded);

        // save the results
        // TODO: This has now become just a recurrence with a tuple state. Let's have a function for that. Or an iterator!
        decoderStates[t] = decoderState;
    }
    let stateSeq = Splice(TransformSelectMember(decoderStates, &DecoderState::state), Axis::EndStaticAxis());
    let attentionContextSeq = Splice(TransformSelectMember(decoderStates, &DecoderState::attentionContext), Axis::EndStaticAxis());
    // stack of output transforms
    let z = doToOutput(stateSeq, attentionContextSeq);
    return z;
#endif
    return Variable();
}

// TODO: Break out initial step and recurrent step layers. Decoder will later pull them out from here.
fun AttentionDecoder(double dropoutInputKeepProb)
{
    // create all the layer objects
    let encBarrier = Barrier(600, Named("encBarrier"));
    let encoderKeysProjection = encBarrier // keys projection for attention
                             >> Linear(attentionDim, ProjectionOptions_batchNormalize | ProjectionOptions::bias)
                             >> Activation(Tanh)
                             >> Label(Named("encoderKeysProjection"));
    let encoderDataProjection = encBarrier // data projection for attention
                             >> Dense(attentionDim, ProjectionOptions_batchNormalize | ProjectionOptions::bias)
                             >> Activation(Tanh)
                             >> Label(Named("encoderDataProjection"));
    let embedTarget = Barrier(600, Named("embedTargetBarrier"))     // target embeddding
                   >> Embedding(embeddingDim)
                   >> Label(Named("embedTarget"));
    let initialContext = Constant({ attentionDim }, CurrentDataType(), 0.0, CurrentDevice(), L"initialContext"); // 2 * because bidirectional --TODO: can this be inferred?
    let initialStateProjection = Barrier(20, Named("initialStateProjectionBarrier"))
                              >> Dense(decoderRecurrentDim, ProjectionOptions::weightNormalize | ProjectionOptions::bias)
                              >> Activation(Tanh)
                              >> Label(Named("initialStateProjection"));
    let stepBarrier = Barrier(20, Named("stepBarrier"));
    let stepFunction = GRU(decoderRecurrentDim);
    auto attentionModel = AttentionModelReference(attentionDim);
    let attBarrier = Barrier(20, Named("attBarrier"));
    let firstHiddenProjection = Barrier(600, Named("projBarrier"))
                             >> Dense(decoderProjectionDim, ProjectionOptions::weightNormalize | ProjectionOptions::bias)
                             >> Activation(Tanh)
                             //>> Activation(ReLU)
                             >> Label(Named("firstHiddenProjection"));
    vector<UnaryModel> resnets;
    for (size_t n = 0; n < numDecoderResNetProjections; n++)
        resnets.push_back(ResidualNet(decoderProjectionDim));
    // BUGBUG: The following call leads to a different weight layout, and prevents the model from being loaded after this change.
    //let topHiddenProjection = Dense(topHiddenProjectionDim, ProjectionOptions::weightNormalize | ProjectionOptions::bias)
    //                       >> Activation(Tanh)
    //                       >> Label(Named("topHiddenProjection"));
    let topHiddenProjection = Dense(topHiddenProjectionDim, UnaryModel([](const Variable& x) { CountAPICalls(); return Tanh(x, Named("topHiddenProjection")); }), ProjectionOptions::weightNormalize | ProjectionOptions::bias);
    let outputProjection = Linear(tgtVocabSize, ProjectionOptions::weightNormalize | ProjectionOptions::bias);  // output layer without non-linearity (no sampling yet)

    let profiler = Function::CreateDynamicProfiler(1, L"decode");

    let outProjProfiler = Function::CreateDynamicProfiler(1, L"outProj");

    // initialization function
    let decoderInitFunction =
        [=](const Variable& hEncoderSeq) -> DecoderState
        {
            CountAPICalls(2);
            Variable state = Slice(hEncoderSeq[0], Axis(0), (int)encoderRecurrentDim, 2 * (int)encoderRecurrentDim); // initial state for the recurrence is the final encoder state of the backward recurrence
            state = initialStateProjection(state);      // match the dimensions
            Variable attentionContext = initialContext; // note: this is almost certainly wrong
            // common subexpression of attention.
            // We pack the result into a dense matrix; but only after the matrix product, to allow for it to be batched.
            let encodingProjectedKeysSeq = encoderKeysProjection(hEncoderSeq); // this projects the entire sequence
            let encodingProjectedDataSeq = encoderDataProjection(hEncoderSeq);
            return{ state, attentionContext, encodingProjectedKeysSeq, encodingProjectedDataSeq };
        };

    // step function
    let decoderStepFunction =
        [=](const DecoderState& decoderState, const Variable& historyEmbedded) -> DecoderState
        {
            // get the state
            auto state = decoderState.state;
            auto attentionContext = decoderState.attentionContext;
            let& encodingProjectedKeysSeq = decoderState.encodingProjectedKeysSeq;
            let& encodingProjectedDataSeq = decoderState.encodingProjectedDataSeq;

            let prevProfiler = Function::SetDynamicProfiler(profiler, false); // use true to display this section of batched graph

            // perform one recurrent step
            let input = stepBarrier(Splice({ historyEmbedded, attentionContext }, Axis(0), Named("augInput"))); CountAPICalls(1);
            state = stepFunction(state, input);

            // compute attention-context vector
            let attentionWeightSeq = attentionModel(state, historyEmbedded, encodingProjectedKeysSeq);
            attentionContext = attBarrier(InnerProduct(encodingProjectedDataSeq, attentionWeightSeq, Axis_DropLastAxis, Named("attContext"))); CountAPICalls(1); // [.] inner product over a vectors

            Function::SetDynamicProfiler(prevProfiler);
            return{ state, attentionContext, encodingProjectedKeysSeq, encodingProjectedDataSeq };
        };

    // non-recurrent further output processing
    // BUGBUG: Setting this to true fails with an off batch axis.
    let doToOutput = StaticModel(/*isBasicBlock=*/false,
        [=](const Variable& state, const Variable& attentionContext)
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

    // perform the entire thing
    return /*Dynamite::Model*/BinaryModel({ }, nestedLayers,
        [=](const Variable& historySeq, const Variable& hEncoderSeq) -> Variable
        {
            // if no history then instead use beam decoder
            if (historySeq == Variable())
                return BeamDecode(hEncoderSeq, decoderInitFunction, decoderStepFunction, doToOutput);
            // decoding loop
            DecoderState decoderState = decoderInitFunction(hEncoderSeq);
            vector<DecoderState> decoderStates(historySeq.size()); // inner state and attentionContext remembered here
            let historyEmbeddedSeq = embedTarget(historySeq);
            // TODO: Forward iterator
            for (size_t t = 0; t < historyEmbeddedSeq.size(); t++)
            {
                // do recurrent step (in inference, historySeq[t] would become states[t-1])
                let historyEmbedded = historyEmbeddedSeq[t]; CountAPICalls(1);

                // do one decoding step
                decoderState = decoderStepFunction(decoderState, historyEmbedded);

                // save the results
                // TODO: This has now become just a recurrence with a tuple state. Let's have a function for that. Or an iterator!
                decoderStates[t] = decoderState;
            }
            let stateSeq            = Splice(TransformSelectMember(decoderStates, &DecoderState::state           ), Axis::EndStaticAxis());
            let attentionContextSeq = Splice(TransformSelectMember(decoderStates, &DecoderState::attentionContext), Axis::EndStaticAxis());
            // stack of output transforms
            let z = doToOutput(stateSeq, attentionContextSeq);
            return z;
        });
}

fun CreateModelFunction()
{
    let embedFwd = Embedding(embeddingDim, Named("embedFwd"));
    //let embedBwd = Embedding(embeddingDim, Named("embedBwd"));
    let encode = BidirectionalLSTMEncoder(numEncoderLayers, encoderRecurrentDim, 0.8);
    auto decode = AttentionDecoder(0.8);

    return BinaryModel({},
        {
            { L"embedSourceFwd", embedFwd   },
            //{ L"embedSourceBwd", embedBwd   },
            { L"encode",         encode     },
            { L"decode",         decode     },
        },
        [=](const Variable& sourceSeq, const Variable& historySeq) -> Variable
        {
            DOLOG(sourceSeq);
            DOLOG(historySeq);
            // embedding
            //let& W = embedFwd.Nested(L"embed")[L"W"];
            //DOLOG(W);
            let eFwd = embedFwd(sourceSeq);
            let eBwd = eFwd;// embedBwd(sourceSeq);
            // encoder
            let hSeq = encode(eFwd, eBwd);
            // decoder (outputting log probs of words)
            let zSeq = decode(historySeq, hSeq);
            DOLOG(zSeq);
            return zSeq;
        });
}

fun CreateCriterionFunction(const BinaryModel& model_fn)
{
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
        DOPRINT("src", sourceSeq, srcVocabFile);
        DOPRINT("tgt", labelsSeq, tgtVocabFile);
        DOPRINT("hyp", zSeq, tgtVocabFile);
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

// join
template<typename StringCollection>
static auto Join(const StringCollection& tokens)
{
    StringCollection::value_type res;
    for (let token : tokens)
    {
        if (!res.empty())
            res.push_back(' ');
        res.append(token);
    }
    return res;
}

// helper to get a dictionary from file. First call will read it in.
static const vector<string>& GetAndCacheDictionary(const wstring& vocabFile)
{
    // get the dictionary. If first time then read it in; otherwise cache in static variable.
    static map<wstring, vector<string>> dicts;
    auto iter = dicts.find(vocabFile);
    if (iter != dicts.end())
        return iter->second;
    // not in cache: create and read from file
    auto& dict = dicts[vocabFile];
    ifstream f(vocabFile);
    copy(istream_iterator<string>(f), istream_iterator<string>(), back_inserter(dict));
    return dict;
}

// helper to convert a one-hot tensor that represents a word sequence to a readable string
static string OneHotToWordSequence(const NDArrayViewPtr& seq, const wstring& vocabFile)
{
    // convert tensor to index sequence
    auto outputShape = seq->Shape();
    outputShape[0] = 1;
    vector<size_t> indices;
    let out = MakeSharedObject<NDArrayView>(seq->GetDataType(), /*StorageFormat::Dense,*/ outputShape, seq->Device());
    NDArrayView::NumericOperation({ seq }, 1, L"Copy", out, 0, L"Argmax")->CopyDataTo(indices);
    // transform to word sequence
    let& dict = GetAndCacheDictionary(vocabFile);
#if 1 // somehow the #else branch crashes in release
    string res;
    for (let index : indices)
    {
        string token = dict[index];
        if (!res.empty())
            res.push_back(' ');
        res.append(token);
    }
    return res;
#else
    let words = Transform(indices, [&](size_t wordIndex) -> string { return dict[wordIndex]; });
    // join the strings
    return Join(words);
#endif
}

static inline void PrintSequence(const char* prefix, const Variable& seq, const wstring& vocabFile)
{
    fprintf(stderr, "\n%s=%s\n", prefix, OneHotToWordSequence(seq.Value(), vocabFile).c_str()), fflush(stderr);
}

MinibatchSourcePtr CreateMinibatchSource(const wstring& srcFile, const wstring& tgtFile, bool infinitelyRepeat)
{
    auto minibatchSourceConfig = MinibatchSourceConfig({ PlainTextDeserializer(
        {
            PlainTextStreamConfiguration(L"src", srcVocabSize, { srcFile }, { srcVocabFile, L"<s>", L"</s>", L"<unk>" }),
            PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { tgtFile }, { tgtVocabFile, L"<s>", L"</s>", L"<unk>" })
        }) },
        /*randomize=*/true);
    minibatchSourceConfig.maxSamples = infinitelyRepeat ? MinibatchSource::InfinitelyRepeat : MinibatchSource::FullDataSweep;
    minibatchSourceConfig.isMultithreaded = false;
    minibatchSourceConfig.enableMinibatchPrefetch = false; // TODO: reenable the multi-threading and see if (1) it works and (2) makes things faster
    // BUGBUG: ^^ I see two possibly related bugs
    //  - when running on CPU, this fails reliably with what looks like a race condition
    //  - even with GPU, training unreliably fails after precisely N data passes minus one data pass. That minus one may indicate a problem in prefetch?
    // -> Trying without, to see if the problems go away.
    return CreateCompositeMinibatchSource(minibatchSourceConfig);
    // BUGBUG (API): no way to specify MinibatchSource::FullDataSweep in a single expression
}

static wstring IntermediateModelPath(const wstring& modelPath, size_t currentMbCount) // helper to form the model filename
{
    char currentMbCountBuf[20];
    sprintf(currentMbCountBuf, "%06d", (int)currentMbCount); // append the minibatch index with a fixed width for sorted directory listings
    return modelPath + L"." + wstring(currentMbCountBuf, currentMbCountBuf + strlen(currentMbCountBuf)); // (simplistic string->wstring converter)
};

static void Train(const DistributedCommunicatorPtr& communicator, const wstring& modelPath, size_t startMbCount)
{
    // dynamic model and criterion function
    auto model_fn = CreateModelFunction();
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // data
    let minibatchSource = CreateMinibatchSource(srcTrainFile, tgtTrainFile, /*infinitelyRepeat=*/true);

    // run something through to get the parameter matrices shaped --ugh!
    model_fn(Constant({ srcVocabSize, (size_t)1 }, CurrentDataType(), 0.0, CurrentDevice()), Constant({ tgtVocabSize, (size_t)1 }, CurrentDataType(), 0.0, CurrentDevice()));

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
    LearnerPtr baseLearner;
    double lr0;
    if (learnerType == "sgd")
    {
        let f = 1.0;
        lr0 = learningRate * f;
        baseLearner = SGDLearner(parameters, TrainingParameterPerSampleSchedule(vector<double>{ lr0, lr0 / 2, lr0 / 4, lr0 / 8 }, epochSize), learnerOptions);
    }
    else if (learnerType == "adam")
    {
        // AdaGrad correction-correction:
        //  - LR is specified for av gradient
        //  - numer should be /minibatchSize
        //  - denom should be /sqrt(minibatchSize)
        let f = 1 / sqrt(minibatchSize)/*AdaGrad correction-correction*/;
        // ...TODO: Haven't I already added that to the base code??
        lr0 = learningRate * f;
        baseLearner = AdamLearner(parameters, TrainingParameterPerSampleSchedule(vector<double>{ lr0, lr0/2, lr0/4, lr0/8 }, epochSize),
                                  MomentumAsTimeConstantSchedule(40000), true, MomentumAsTimeConstantSchedule(400000), /*eps=*/1e-8, /*adamax=*/false,
                                  learnerOptions);
    }
    else InvalidArgument("Invalid --learner %s", learnerType.c_str());
    // TODO: move this out
    let CreateDistributedLearner = [](const LearnerPtr& baseLearner, const DistributedCommunicatorPtr& communicator)
    {
        if (dynamic_cast<QuantizedDistributedCommunicator*>(communicator.get()))
            return CreateQuantizedDataParallelDistributedLearner(static_pointer_cast<QuantizedDistributedCommunicator>(communicator), baseLearner, /*distributeAfterSamples =*/ 0, /*useAsyncBufferedParameterUpdate =*/ false);
        else
            return CreateDataParallelDistributedLearner(communicator, baseLearner, /*distributeAfterSamples =*/ 0, /*useAsyncBufferedParameterUpdate =*/ false);
    };
    let& learner = CreateDistributedLearner(baseLearner, communicator);
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
    // TODO: move this to a different place, e.g. the helper header
    //let SaveCheckpoint = [](const wstring& path, const FunctionPtr& compositeFunction,
    //    size_t numWorkers, const MinibatchSourcePtr& minibatchSource, const DistributedLearnerPtr& learner)
    if (startMbCount > 0)
    {
        let path = IntermediateModelPath(modelPath, startMbCount);
        fprintf(stderr, "restarting from: %S... ", path.c_str()), fflush(stderr);
        Dynamite::RestoreFromCheckpoint(path, model_fn.ParametersCombined(), communicator->Workers().size(), minibatchSource, learner);
        fprintf(stderr, "done\n"), fflush(stderr);
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
            let modelPathN = IntermediateModelPath(modelPath, mbCount);
            fprintf(stderr, "ssaving: %S... ", modelPathN.c_str()), fflush(stderr); // indicate time of saving, but only main worker actually saves
            Dynamite::SaveCheckpoint(modelPathN, model_fn.ParametersCombined(), communicator->Workers().size(), minibatchSource, learner);
            fprintf(stderr, "done%s\n", communicator->CurrentWorker().IsMain() ? "" : " by main worker"), fflush(stderr);
            // test model saving
            //for (auto& param : parameters) // destroy parameters as to prove that we reloaded them correctly.
            //    param.Value()->SetValue(0.0);
            //model_fn.RestoreParameters(modelPathN);
        }
        timer.Restart();

        // get next minibatch
        partTimer.Restart();
#if 1
        // dynamically adjust the MB size lower at the start to ramp up
        let fullMbSizeAt = 1000000;
        let lowMbSize = minibatchSize / 16;
        let clamp = [](size_t v, size_t lo, size_t hi) { if (v < lo) return lo; else if (v > hi) return hi; else return v; };
        let actualMinibatchSize = clamp(lowMbSize + (minibatchSize - lowMbSize) * totalLabels / fullMbSizeAt, lowMbSize, minibatchSize);
#else
        let actualMinibatchSize = minibatchSize;
#endif
        Dynamite::GetSubBatches(args, { L"src", L"tgt" }, subMinibatches, /*shuffleSeed=*/mbCount, minibatchSource, actualMinibatchSize,
                                communicator->Workers().size(), communicator->CurrentWorker().m_globalRank, CurrentDataType(), CurrentDevice());
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
                let len = seq.size();
                numSamples += len;
                maxSamples = max(maxSamples, len);
            }
            for (let& seq : subBatchArgs[1])
            {
                let len = seq.size();
                numLabels += len;
                maxLabels = max(maxLabels, len);
            }
            //partTimer.Log("GetNextMinibatch", numLabels);
            fprintf(stderr, "%5d: #seq: %d, #words: %d -> %d, max len %d -> %d, lr=%.8f * %.8f, mbSize=%d\n", (int)mbCount,
                    (int)numSeq, (int)numSamples, (int)numLabels, (int)maxSamples, (int)maxLabels,
                    lr0, learner->LearningRate() / lr0, (int)actualMinibatchSize);
#if 0       // log the sequences
            for (size_t n = 0; n < numSeq; n++)
            {
                subBatchArgs[0][n].Value()->LogToFile(L"Source_" + to_wstring(n), stderr, SIZE_MAX);
                subBatchArgs[1][n].Value()->LogToFile(L"Target_" + to_wstring(n), stderr, SIZE_MAX);
            }
#endif
#if 0       // log the first sequence
            fprintf(stderr, "src=%s\n", OneHotToWordSequence(subBatchArgs[0][0].Value(), srcVocabFile).c_str());
            fprintf(stderr, "tgt=%s\n", OneHotToWordSequence(subBatchArgs[1][0].Value(), tgtVocabFile).c_str());
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
            let numScoredLabels = numLabels - numSeq; // the <s> is not scored; that's one per sequence. Do not count for averages.
#if 1       // use 0 to measure graph building only
            partTimer.Restart();
            fprintf(stderr, "%5d:  ", (int)mbCount); // prefix for the log
            mbLoss.Value()->AsScalar<float>(); // trigger computation
            let timeForward = partTimer.Elapsed();
            //fprintf(stderr, "%.7f\n", mbLoss.Value()->AsScalar<float>()), fflush(stderr);
            //exit(1);
            //partTimer.Log("ForwardProp", numLabels);
            // note: we must use numScoredLabels here
            //fprintf(stderr, "{%.2f, %d-%d}\n", mbLoss.Value()->AsScalar<float>(), (int)numLabels, (int)numSeq), fflush(stderr);
            fprintf(stderr, "%5d:  ", (int)mbCount); // prefix for the log
            //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
            partTimer.Restart();
            mbLoss.Backward(gradients);
            //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
            let timeBackward = partTimer.Elapsed();
            //partTimer.Log("BackProp", numLabels);
            MinibatchInfo info{ /*atEndOfData=*/false, /*sweepEnd=*/false, /*numberOfSamples=*/numScoredLabels, mbLoss.Value(), mbLoss.Value() };
            info.trainingLossValue->AsScalar<float>();
            //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
            partTimer.Restart();
            learner->Update(gradients, info);
            //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
            let timePerUpdate = partTimer.Elapsed();
            // log the parameters
            if (mbCount % 50 == 1) for (let& p : parameters)
            {
                p.Value()->LogToFile(p.Name(), stderr, 10);
                if (gradients[p]->GetStorageFormat() != StorageFormat::SparseBlockCol)
                    gradients[p]->LogToFile(L"grad " + p.Name(), stderr, 10);
            }
            //partTimer.Log("Update", numLabels);
            let lossPerLabel = info.trainingLossValue->AsScalar<float>() / info.numberOfSamples; // note: this does the GPU sync, so better do that only every N
            totalLabels += info.numberOfSamples;
            let smoothedLossVal = smoothedLoss.Update(lossPerLabel, info.numberOfSamples);
            let elapsed = timer.ElapsedSeconds(); // [sec]
            partTimer.Restart();
            mbLoss = Variable(); // this destructs the entire graph
            let timeDeleteGraph = partTimer.Elapsed();
            if (communicator->CurrentWorker().IsMain())
                fprintf(stderr, "%5d:  loss, PPL = [smoothed] %4.2f, ### %8.2f ### [this] %9.7f, %6.3f, seenLabels=%d, %.1f w/s, %.1f ms/w, m=%.0f, g=%.0f, f=%.0f, b=%.0f, u=%.0f, d=%.0f ms\n",
                                (int)mbCount, smoothedLossVal, exp(smoothedLossVal), lossPerLabel, exp(lossPerLabel), (int)totalLabels,
                                info.numberOfSamples / elapsed, 1000.0/*ms*/ * elapsed / info.numberOfSamples,
                                1000.0 * timeGetNextMinibatch, 1000.0 * timeBuildGraph, 1000.0 * timeForward, 1000.0 * timeBackward, 1000.0 * timePerUpdate, 1000.0 * timeDeleteGraph);
            // log
            // Do this last, which forces a GPU sync and may avoid that "cannot resize" problem
            if (mbCount < 400 || mbCount % 5 == 0)
                fflush(stderr);
            if (std::isnan(lossPerLabel))
                throw runtime_error("Loss is NaN.");
#else
            let elapsed = timer.ElapsedSeconds(); // [sec]
            double lossPerLabel = 0, smoothedLossVal = 0;
            fprintf(stderr, "%d: >> loss = %.7f; PPL = %.3f << smLoss = %.7f, smPPL = %.2f, seenLabels=%d, %.1f w/s, %.1f ms/w, m=%.0f, g=%.0f, d=%.0f ms\n",
                            (int)mbCount, lossPerLabel, exp(lossPerLabel), smoothedLossVal, exp(smoothedLossVal), (int)totalLabels,
                            numScoredLabels / elapsed, 1000.0/*ms*/ * elapsed / numScoredLabels,
                            1000.0 * timeGetNextMinibatch, 1000.0 * timeBuildGraph, 1000.0 * timeDeleteGraph);
            totalLabels;
#endif
            //if (mbCount == 10)
            //    exit(0);
            mbCount++;
        }
    }
}

static void Evaluate(const wstring& modelPath, size_t modelMbCount, const wstring& srcEvalFile, const wstring& tgtEvalFile)
{
    // dynamic model and criterion function
    // This must recreate the same model as in training, so that we can restore its model parameters.
    auto model_fn = CreateModelFunction();
#if 1 // BUGBUG: currently needed so that parameter names and shapes match, otherwise the model cannot be loaded
    model_fn.LogParameters();
    // run something through to get the parameter matrices shaped --ugh!
    model_fn(Constant({ srcVocabSize, (size_t)1 }, CurrentDataType(), 0.0, CurrentDevice()), Constant({ tgtVocabSize, (size_t)1 }, CurrentDataType(), 0.0, CurrentDevice()));
#endif
    let path = IntermediateModelPath(modelPath, modelMbCount);
    fprintf(stderr, "loading model: %S... ", path.c_str()), fflush(stderr);
    model_fn.RestoreParameters(path);
    fprintf(stderr, "done\n"), fflush(stderr);

    // data
    let minibatchSource = CreateMinibatchSource(srcEvalFile, tgtEvalFile, /*infinitelyRepeat=*/false);

    let minibatchSize = 700; // this fits

    size_t totalLabels = 0; // total scored labels (excluding the <s>)
    double totalLoss = 0;   // corresponding total aggregate loss

    // MINIBATCH LOOP
    auto criterion_fn = CreateCriterionFunction(model_fn); // ...for now
    vector<vector<vector<Variable>>> args; // [subMinibatchIndex, streamIndex, sequenceIndex]
    for (mbCount = 0; ; mbCount++)
    {
        // get next minibatch
        bool gotData = Dynamite::GetSubBatches(args, { L"src", L"tgt" }, /*subMinibatches=*/1, /*shuffleSeed=*/0, minibatchSource, minibatchSize, /*numWorkers=*/1, /*currentWorker=*/0, CurrentDataType(), CurrentDevice());
        if (!gotData)
            break;
        auto& subBatchArgs = args.front(); // there is only one

        let numSeq = subBatchArgs[0].size();
        size_t numLabels = 0, numSamples = 0, maxSamples = 0, maxLabels = 0;
        for (let& seq : subBatchArgs[0])
        {
            let len = seq.size();
            numSamples += len;
            maxSamples = max(maxSamples, len);
        }
        for (let& seq : subBatchArgs[1])
        {
            let len = seq.size();
            numLabels += len;
            maxLabels = max(maxLabels, len);
        }
#if 0   // beam decoding
        for (size_t seqId = 0; seqId < numSeq; seqId++)
        {
            let& srcSeq = subBatchArgs[0][seqId];
            let& tgtSeq = subBatchArgs[1][seqId];
            PrintSequence("src", srcSeq, srcVocabFile);
            PrintSequence("tgt", tgtSeq, tgtVocabFile);
            let outSeq = model_fn(srcSeq, Variable());
            PrintSequence("out", outSeq, tgtVocabFile);
        }
#endif
        //partTimer.Log("GetNextMinibatch", numLabels);
        fprintf(stderr, "%5d: #seq: %d, #words: %d -> %d, max len %d -> %d\n", (int)mbCount,
                (int)numSeq, (int)numSamples, (int)numLabels, (int)maxSamples, (int)maxLabels);
        // decode all sequences
        auto mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]).Value()->AsScalar<double>();
        let numScoredLabels = numLabels - numSeq; // the <s> is not scored; that's one per sequence. Do not count for averages.
        let lossPerLabel = mbLoss / numScoredLabels; // note: this does the GPU sync, so better do that only every N
        totalLabels += numScoredLabels;
        totalLoss += mbLoss;
        fprintf(stderr, "%5d:  loss, PPL = [aggregate] %5.2f, ### %8.2f ### [this] %10.7f, %9.3f, seenLabels=%d\n",
                        (int)mbCount, totalLoss/ totalLabels, exp(totalLoss / totalLabels), lossPerLabel, exp(lossPerLabel), (int)totalLabels);
        fflush(stderr);
    }
    fprintf(stderr, "\n%5d:  loss, PPL = [total] %5.2f, ### %8.2f ###, seenLabels=%d\n",
        (int)mbCount, totalLoss / totalLabels, exp(totalLoss / totalLabels), (int)totalLabels);
}

// minimalist command-line eargument parser, requires all arguments in order
class GetCommandLineArguments
{
    size_t argc; char** argv;
    string Front() const { if (argc == 0) LogicError("GetCommandLineArguments: out of command-line arguments??"); return *argv; }
    string Pop() { let val = Front(); argc--, argv++; return val; }
    template<typename T>
    void PopArg(const char* argTag, T& argVal) // check that next arg is argTag, Pop it, then Pop and return the subsequent arg
    {
        if (argc == 1)
            InvalidArgument("last command-line arg '--%s' is missing a value", argTag);
        bool optional = argTag[0] == '?';
        if (optional)
            argTag++;
        if (argc > 0 && Front() == "--"s + argTag)
        {
            let argStr = (Pop(), Pop());
            SetArg(argStr, argVal);
            fprintf(stderr, "%-30s = %s\n", argTag, argStr.c_str());
        }
        else
        {
            if (!optional)
                InvalidArgument("command-line argument '%s' missing or out of order", argTag);
            fprintf(stderr, "%-30s = %S\n", argTag, ToWString(argVal).c_str());
        }
    }
    // back conversion for printing --thanks to Billy O'Neal for the SFINAE hacking
    struct unique_c1xx_workaround_tag_give_this_a_fun_name;
    template<typename T, typename = void> struct has_towstring : false_type {};
    template<typename T> struct has_towstring<T, void_t<unique_c1xx_workaround_tag_give_this_a_fun_name, decltype(declval<T>().ToWString())>> : true_type {};
    template<typename T> static wstring ToWStringImpl(const T& val, true_type) { return val.ToWString(); }
    template<typename T> static wstring ToWStringImpl(const T& val, false_type) { return to_wstring(val); }
    template<typename T> static wstring ToWString(const T& val) { return ToWStringImpl(val, has_towstring<T>{}); }
    static wstring ToWString(const wstring& val) { return val; }
    static wstring ToWString(const string& val) { return wstring(val.begin(), val.end()); } // ASCII only
    // conversion to target type via overloads
    // if needed, we can add Get() for other types, and with defaults (triples)
    template<typename T> // template, for types that accept a string; such as... std::stringm but more so special hacks like SystemSentinel
    static void SetArg(const string& argStr, T& argVal) { argVal = argStr; }
    static void SetArg(const string& argStr, wstring& argVal) { argVal.assign(argStr.begin(), argStr.end()); } // note: this only presently works for ASCII arguments, easy to fix if ever needed
    static void SetArg(const string& argStr, int& argVal) { SetArgViaStream(argStr, argVal); }
    static void SetArg(const string& argStr, size_t& argVal) { SetArgViaStream(argStr, argVal); }
    static void SetArg(const string& argStr, float& argVal) { SetArgViaStream(argStr, argVal); }
    static void SetArg(const string& argStr, double& argVal) { SetArgViaStream(argStr, argVal); }
    template<typename T> // template, for types that accept a string; such as... std::stringm but more so special hacks like SystemSentinel
    static void SetArgViaStream(const string& argStr, T& argVal)
    {
        istringstream iss(argStr);
        iss >> argVal;
        if (iss.bad()) InvalidArgument("command-line argument value '%s' cannot be parsed to %s", argStr.c_str(), typeid(argVal).name());
    }
private:
    // recursive template. Each invocation pops the next argument, where the C++ type selects the conversion in PopArg()
    template <typename T, typename ...ArgTypes>
    void Get(const char* argTag, T& argVal, ArgTypes&& ...remainingArgs)
    {
        PopArg(argTag, argVal);
        Get(std::forward<ArgTypes>(remainingArgs)...); // recurse
    }
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
        wstring command;
        wstring workingDirectory = L"d:/mt/experiments"; // output dir = "$workingDirectory/$experimentId/"
        wstring modelPath;
        size_t fromMbCount = 0;
        size_t firstGpu = 0;
        size_t numBits = 4;
        struct SystemSentinel : public string
        {
            wstring ToWString() const { return wstring(begin(), end()); }
            void operator=(const string& systemId)
            {
                string::operator=(systemId);
                SetConfigurationVariablesFor(systemId);
            }
        } systemId;
        wstring experimentId;
        try
        {
            GetCommandLineArguments(argc, argv,
                "command", command,
                "system", systemId,
                "id", experimentId,
                // optional overrides of global stuff
                "?workingDirectory", workingDirectory,
                "?modelPath", modelPath,
                "?firstGpu", firstGpu,
                "?numBits", numBits,
                // these are optional to override the system settings
                "?learner", learnerType,
                "?learningRate", learningRate,
                "?fromMb", fromMbCount);
        }
        catch (const exception& e)
        {
            fprintf(stderr, "%s\n", e.what()), fflush(stderr);
            throw invalid_argument("required command line: --command train|test --system SYSTEMID --id IDSTRING\n SYSTEMID = chs_enu, rom_enu, etc\n IDSTRING is used to form the log and model path for now");
        }

        //let* pExpId = argv[2];
        //wstring experimentId(pExpId, pExpId + strlen(pExpId)); // (cheap conversion to wchar_t)
        auto outputDirectory = workingDirectory + L"/" + experimentId + L"_" + wstring(systemId.begin(), systemId.end());
        if (modelPath.empty())
            modelPath = outputDirectory + L"/model.dmf"; // DMF=Dynamite model file

        // set up parallel communicator
        use1BitSgd = numBits != 0;
        fprintf(stderr, "Using %d-bit quantization (0=off)\n", (int)numBits);
        let communicator =
            /*if*/use1BitSgd ?
                QuantizedMPICommunicator(/*zeroThresholdFor1Bit=*/true, /*useQuantizationForSelfStripe=*/true, /*numQuantizationBits=*/numBits)
            /*else*/ :
                MPICommunicator();
#if 1 // while we are running with MPI, we always start from start
        let numGpus = DeviceDescriptor::AllDevices().size() - 1;
        let ourRank = communicator->CurrentWorker().m_globalRank;
        if (numGpus > 0)
        {
            let ourGpu = (ourRank + firstGpu) % numGpus;
            fprintf(stderr, "Worker %d using GPU %d\n", (int)ourRank, (int)ourGpu);
            SetCurrentDevice(DeviceDescriptor::GPUDevice((unsigned int)ourGpu));
        }
        else
            SetCurrentDevice(DeviceDescriptor::CPUDevice());
#endif

        // open log file. The path depends on the worker rank.
        // Log path = "$workingDirectory/$experimentId.log.$ourRank" where $ourRank is missing for rank 0
        let logPath = outputDirectory + L"/" + command + L".log" + (ourRank == 0 ? L"" : (L"." + to_wstring(ourRank)));
        boost::filesystem::create_directories(boost::filesystem::path(logPath).parent_path());
        FILE* outStream =
            /*if*/ (communicator->CurrentWorker().IsMain()) ?
            _wpopen((L"tee " + logPath).c_str(), L"wt")
            /*else*/ :
            _wfopen(logPath.c_str(), L"wt");
        if (!outStream)
            InvalidArgument("error %d opening log file '%S'", errno, logPath.c_str());
        fprintf(stderr, "redirecting stderr to %S\n", logPath.c_str());
        if (_dup2(_fileno(outStream), _fileno(stderr)))
            InvalidArgument("error %d redirecting stderr to '%S'", errno, logPath.c_str());
        fprintf(stderr, "starting %S as worker[%d]\n", command.c_str(), (int)ourRank), fflush(stderr); // write something to test

        // perform the command
        if (command == L"train")
            Train(communicator, modelPath, fromMbCount);
        else if (command == L"validate")
            Evaluate(modelPath, fromMbCount, srcDevFile, tgtDevFile);
        else if (command == L"test")
            Evaluate(modelPath, fromMbCount, srcTestFile, tgtTestFile);
        else
            InvalidArgument("Unknonw --command %S", command.c_str());
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
        fflush(stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
