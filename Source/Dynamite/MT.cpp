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

#include "marian.h"
#include "transformer.h"
using namespace marian;

#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include <sstream>
#ifdef _MSC_VER
#include <Windows.h> // for process killing
#endif

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

static size_t CeilDiv(size_t numer, size_t denom) { return (numer + denom - 1) / denom; };

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
size_t bucketingFactor = 10; // group 10 minibatches together, sort, re-split; for more homogenous batch sizes
string learnerType = "adam";
double learningRate = 0.0003662109375;
bool use1BitSgd = false;
size_t saveEvery = 2000;
size_t maxBeam = 5;
double beamWidth = 2.0; // logProb beam width
int/*bool*/ runProfiling = false;
size_t minibatchSize = 4096;
size_t maxBatchSizePerWorker = 2000;// CeilDiv(4096, 6); // this much fits into RAM
bool insertBOS = true;

static void SetConfigurationVariablesFor(string systemId) // set variables; overwrite defaults
{
    if (systemId == "en_de_bpe") // Marian setup
    {
        // cat vocab.ende.yml | sed 's/: [0-9]*$//' | tr -d ^" > vocab.ende.txt && "C:\Program Files\Git\usr\bin\echo.exe" >> ..\data\vocab.ende.txt
        srcVocabSize = 36000;
        tgtVocabSize = 36000;
        srcTrainFile = L"f:/data2/fseide/marian-examples/transformer/data/corpus.bpe.en";
        tgtTrainFile = L"f:/data2/fseide/marian-examples/transformer/data/corpus.bpe.de";
        srcDevFile   = L"f:/data2/fseide/marian-examples/transformer/data/valid.bpe.en";
        tgtDevFile   = L"f:/data2/fseide/marian-examples/transformer/data/valid.bpe.de";
        srcTestFile  = L"f:/data2/fseide/marian-examples/transformer/data/test2016.bpe.en";
        tgtTestFile  = L"f:/data2/fseide/marian-examples/transformer/data/test2016.bpe.de";
        srcVocabFile = L"f:/data2/fseide/marian-examples/transformer/data/vocab.ende.txt";
        tgtVocabFile = L"f:/data2/fseide/marian-examples/transformer/data/vocab.ende.txt";
        bucketingFactor = 1;//0;
        insertBOS = false; // Marian model does not expect <s>
    }
    else if (systemId == "chs_enu")
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
        bucketingFactor = 10;
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
    else if (systemId == "chs_enu_wmt17")
    {
        srcVocabSize = 68266;
        tgtVocabSize = 55144;
        srcTrainFile = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/train.src";
        tgtTrainFile = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/train.tgt";
        srcDevFile   = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/valid.src";
        tgtDevFile   = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/valid.tgt";
        srcTestFile  = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/test.src";
        tgtTestFile  = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/test.tgt";
        srcVocabFile = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/vocab.src";
        tgtVocabFile = L"f:/mt-data/mtdrop/mtdrop/babel/Official/WMT17/ProcessedData/vocab.tgt";
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
        InvalidArgument("Invalid system id '%s'", systemId.c_str());
}

size_t mbCount = 0; // made a global so that we can trigger debug information on it
#define DOLOG(var)                  ((mbCount % 50 == 1 && Dynamite::Batch::CurrentMapIndex() < 2 && !runProfiling) ? (LOG(var),0)                            : 0)
#define DOPRINT(prefix, var, vocab) ((mbCount % 50 == 1 && Dynamite::Batch::CurrentMapIndex() < 2 && !runProfiling) ? PrintSequence((prefix), (var), (vocab)) : string())
static string PrintSequence(const char* prefix, const Variable& seq, const wstring& vocabFile);

BinaryModel/*auto*/ BidirectionalLSTMEncoder(size_t numLayers, size_t hiddenDim, double dropoutInputKeepProb)
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
            // before each additional layer, so batch norm
            h = bns[i - 1](h);
            // do another layer
            h = layers[i](h, h);
            //// after each additional layer, so batch norm
            //// BUGBUG: Why not the first? Seems like a bug.
            //// But BN after this does not help, so probably it should be between those layers, not after.
            ////h = bns[i - 1](h);
        }
        //DOLOG(h);
        return h;
    });
}

// reference attention model
// Returns the attention probabilities. Caller must do the weighted average.
TernaryModel/*auto*/ AttentionModelReference(size_t attentionDim1)
{
    let projectQuery = Linear(attentionDim1, ProjectionOptions::weightNormalize);
    //let normH = BatchNormalization(1, Named("bnAtt")); // note: can't move this inside Linear since it is applied after adding two factors
    let normH = LengthNormalization(); // note: can't move this inside Linear since it is applied after adding two factors
    //let normH = Identity;// LengthNormalization(); // note: can't move this inside Linear since it is applied after adding two factors
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
            //DOLOG(wSeq);
            return wSeq;
        });
}

// helper to extract a struct member
template<typename CollectionType, typename CollectionMemberType>
vector<Variable/*ValueType*/>/*auto*/ TransformSelectMember(const CollectionType& fromCollection, CollectionMemberType pMember)
{
    typedef typename CollectionType::value_type ValueType;
    // BUGBUG: Figure out the return type. Getting random compilation errors.
    return vector<Variable/*ValueType*/>(Transform(fromCollection, [&](const ValueType& item) { return item.*pMember; }));
}

// the state we carry forward across decoding steps
struct DecoderRecurrentState
{
    // actual state
    Variable state, attentionContext;
    // stuff that gets computed only once
    Variable encodingProjectedKeysSeq;
    Variable encodingProjectedDataSeq;
};

struct BeamDecoderToken
{
    const BeamDecoderToken* backPointer;  // where we came from -- pointer to best previous token
    DecoderRecurrentState recurrentState; // new internal state -- after calling recurrence
    size_t wordIndex;                     // resulting output word, as a scalar index value
    Variable pathLogP;                    // accumulated partial log probability (scalar)
};

// helper to create a CSC-sparse NDArrayView in the current device that represents the passed index sequence
// TODO: Suboptimal. We want a view here, and then construct a Constant from that.
static Variable OneHotConstant(const vector<double>& ids)
{
    let indexView = MakeSharedObject<NDArrayView>(ids.data(), CurrentDataType(), NDShape{ ids.size() }, CurrentDevice(), /*readOnly=*/true);
    return OneHotOp(Constant(indexView), tgtVocabSize, /*outputSparse=*/true, Axis(0));
}

static const vector<string>& GetAndCacheDictionary(const wstring& vocabFile);

// this is the main beam decoder, for a single sentence
template<typename InitFunctionType, typename EmbedFunctionType, typename StepFunctionType, typename OutputFunctionType>
Variable BeamDecode(const Variable& hEncoderSeq, const InitFunctionType& decoderInitFunction, const EmbedFunctionType& decoderEmbedOutputFunction, const StepFunctionType& decoderStepFunction, const OutputFunctionType& decoderOutputFunction)
{
    let& dict = GetAndCacheDictionary(tgtVocabFile); // for debugging
    size_t sentStartId = 1;
    size_t sentEndId = 2; // TODO: get those from the actual dictionary
    let initialPathLogP = Constant({}, CurrentDataType(), 0, CurrentDevice(), L"initialPathLogP");
    BeamDecoderToken initialToken = { /*backPointer=*/nullptr, decoderInitFunction(hEncoderSeq), sentStartId, initialPathLogP };
    // expansion
    list<vector<BeamDecoderToken>> allTokens{ { initialToken } }; // search space over time steps
    vector<BeamDecoderToken> newTokens; // buffer for new tokens
    vector<Variable> pathLogPVectors;
    vector<DecoderRecurrentState> newRecurrentStates;
    vector<const BeamDecoderToken*> backPointers;
    vector<vector<float>> probCopyBuffers(maxBeam);
    vector<pair<size_t, size_t>> sortBuffer;
    size_t totalTokens = 1;
    for (;;)
    {
        let& tokens = allTokens.back();
        // end?
        // It is theoretically correct to stop once the best-scoring hypothesis is </s>.
        // We are trying to find the highest-scoring *sentence* probability, which is
        // computed via Bayes rule. Hence, any subsequent path expansions cannot have a higher
        // sentence probability, since whatever gets multiplied on later is < 1.
        if (tokens.front().wordIndex == sentEndId)
            break;
        // expand all present tokens
        //  - recurrent step
        //  - determine path probabilities for all words
        backPointers      .clear();
        newRecurrentStates.clear();
        pathLogPVectors   .clear();
        fprintf(stderr, "--- step %d from %d tokens ---\n", (int)allTokens.size(), (int)tokens.size());
        for (let& token : tokens)
        {
            if (token.wordIndex == sentEndId) // if </s> reached then stop. This hypothesis is worse than others that have not ended.
                continue;
            //LOG(token.recurrentState.state);
            //LOG(token.recurrentState.attentionContext);
            // embed it for the next step  --note the isVolatile flag, to make sure BatchNorm runs in eval mode
            let wordIndexVar = Constant({}, CurrentDataType(), /*isVolatile=*/true, (double)token.wordIndex, CurrentDevice(), L"wordIndexVar");
            let word = OneHotOp(wordIndexVar, tgtVocabSize, /*outputSparse=*/true, Axis(0));
            let wordEmbedded = decoderEmbedOutputFunction(word);
            //LOG(wordEmbedded);
            // update the recurrent model
            let newRecurrentState = decoderStepFunction(token.recurrentState, wordEmbedded);
            // stack of output transforms
            let z = decoderOutputFunction(newRecurrentState.state, newRecurrentState.attentionContext);
            // conditional probability log P(word|state)
            let logPVector = LogSoftmax(z);
            //logPVector.Value()->LogToFile(L"logPs", stderr, 1000000);
            //LOG(logPVector);
            // expand it
            let pathLogPVector = logPVector + token.pathLogP;
            backPointers      .push_back(&token           ); // where we came from
            newRecurrentStates.push_back(newRecurrentState); // the new internal state (which does not depend on the choice)
            pathLogPVectors   .push_back(pathLogPVector   ); // the path probability vectors
        }
        let numTokens = backPointers.size();
        // now we have gotten all we wanted from the token
        // determine new token set
        // That is, all tokens that in the top N and within the probability beam.
        // we go into CPU land now, since there is no sorting function...
        for (size_t i = 0; i < numTokens; i++)
            pathLogPVectors[i].Value()->CopyDataTo(probCopyBuffers[i]);
        sortBuffer.clear();
        for (size_t i = 0; i < numTokens; i++)
            for (size_t k = 0; k < tgtVocabSize; k++)
                sortBuffer.push_back({ i, k });
        sort(sortBuffer.begin(), sortBuffer.end(),
            [&](const pair<size_t, size_t>& ik1, const pair<size_t, size_t>& ik2)
            {
                return probCopyBuffers[ik1.first][ik1.second] > probCopyBuffers[ik2.first][ik2.second];
            });
        size_t i, k;
        tie(i,k) = sortBuffer.front();
        let maxPathLogP = probCopyBuffers[i][k];
        // create the new tokens
        newTokens.clear();
        for (let& ik : sortBuffer)
        {
            if (newTokens.size() == maxBeam) // top-N pruning
                break;
            tie(i, k) = ik;
            let logPathProb = probCopyBuffers[i][k];
            if (logPathProb < maxPathLogP - beamWidth) // probability-beam pruning
                break;
            // token should survive
            //LOG(k);
            //LOG(word);
            string hist;
            for (let* bp = backPointers[i]; bp; bp = bp->backPointer)
                hist = dict[bp->wordIndex] + " " + hist;
            fprintf(stderr, "word[%d] %.3f: %d %s | %s\n", (int)i, logPathProb, (int)k, dict[k].c_str(), hist.c_str());
            // new token
            let newPathLogP = pathLogPVectors[i][k];
            //LOG(newPathLogP);
            newTokens.emplace_back/*BeamDecoderToken newToken = */(BeamDecoderToken{ backPointers[i], newRecurrentStates[i], k, newPathLogP });
            totalTokens++;
        }
        // expand the remembered tokens
        allTokens.push_back(newTokens); // makes a copy. and we reuse newTokens[]
    }

    // traceback
    let numWords = allTokens.size();
    vector<double> resultWords(numWords);
    let* pTok = &allTokens.back().front();
    auto iter = resultWords.end();
    while (pTok)
    {
        if (iter == resultWords.begin())
            LogicError("BeamDecode: length screwed up??");
        iter--;
        // get word in this token
        *iter = (double)pTok->wordIndex;
        // trace back one step
        pTok = pTok->backPointer;
    }
    if (iter != resultWords.begin())
        LogicError("BeamDecode: length screwed up??");
    let hyp = OneHotConstant(resultWords);
    //let hyp = Splice(resultWords, Axis::EndStaticAxis());
    //LOG(hyp);
    let totalLogP = allTokens.back().front().pathLogP.Value()->AsScalar<double>();
    let logPPerWord = totalLogP / (numWords - 1);
    let ppl = exp(-logPPerWord);
    fprintf(stderr, "logP = %.3f * %d ; PPL=%.2f ; N = %.2f * %d\n", logPPerWord, (int)(numWords - 1), ppl, totalTokens / (double)numWords, (int)numWords), fflush(stderr);
    return hyp;
}

// TODO: Break out initial step and recurrent step layers. Decoder will later pull them out from here.
BinaryModel/*auto*/ AttentionDecoder(double dropoutInputKeepProb)
{
    // create all the layer objects
    let encBarrier = Barrier(600, Named("encBarrier"));
    let encoderKeysProjection = encBarrier // keys projection for attention
        // batch norm after Linear causes huge std dev values
                             >> Linear(attentionDim, ProjectionOptions::weightNormalize |/*ProjectionOptions_batchNormalize |*/ ProjectionOptions::bias)
                             >> Activation(Tanh)
                             >> Label(Named("encoderKeysProjection"));
    let encoderDataProjection = encBarrier // data projection for attention
        // stabilizer causes fluctuations
                             >> Dense(attentionDim, ProjectionOptions::weightNormalize |/*ProjectionOptions_batchNormalize |*/ ProjectionOptions::bias)
                             >> Activation(Tanh)
                             >> Label(Named("encoderDataProjection"));
    let embedTarget = Barrier(600, Named("embedTargetBarrier"))     // target embeddding
                   >> Embedding(embeddingDim)
#if 0
                   >> Dynamite::Sequence::Recurrence(GRU(encoderRecurrentDim), Constant({ encoderRecurrentDim }, CurrentDataType(), 0.0, CurrentDevice(), Named("historyInitialValue")))
                   >> Dense(attentionDim, ProjectionOptions::weightNormalize | ProjectionOptions_batchNormalize |/*ProjectionOptions_batchNormalize |*/ ProjectionOptions::bias)
                   >> Activation(Tanh)
#endif
                   >> Label(Named("embedTarget"));
    let initialContext = Constant({ attentionDim }, CurrentDataType(), 0.0, CurrentDevice(), L"initialContext"); // 2 * because bidirectional --TODO: can this be inferred?
    let initialStateProjection = Barrier(20, Named("initialStateProjectionBarrier"))
                              >> Dense(decoderRecurrentDim, ProjectionOptions::weightNormalize | /*ProjectionOptions_batchNormalize |*/ ProjectionOptions::bias)
                              >> Activation(Tanh)
                              >> Label(Named("initialStateProjection"));
    let stepBarrier = Barrier(20, Named("stepBarrier"));
    let stepFunction = GRU(decoderRecurrentDim);
    auto attentionModel = AttentionModelReference(attentionDim);
    let attBarrier = Barrier(20, Named("attBarrier"));
#if 0
    let firstHiddenProjection = Barrier(600, Named("projBarrier"));
#else
    let firstHiddenProjection = Barrier(600, Named("projBarrier"))
                             >> Dense(decoderProjectionDim, ProjectionOptions::weightNormalize | /*ProjectionOptions_batchNormalize |*/ ProjectionOptions::bias)
                             >> Activation(ReLU)
                             >> Label(Named("firstHiddenProjection"));
#endif
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
        [=](const Variable& hEncoderSeq) -> DecoderRecurrentState
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
        [=](const DecoderRecurrentState& decoderState, const Variable& historyEmbedded) -> DecoderRecurrentState
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
    let decoderOutputFunction = StaticModel(/*isBasicBlock=*/false,
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
        }, Named("decoderOutputFunction"));

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
    size_t n = 0;
    for (let& resnet : resnets)
        nestedLayers[L"resnet[" + std::to_wstring(n++) + L"]"] = resnet;

    // perform the entire thing
    return /*Dynamite::Model*/BinaryModel({ }, nestedLayers,
        [=](const Variable& historySeq, const Variable& hEncoderSeq) -> Variable
        {
            // if no history then instead use beam decoder
            if (historySeq == Variable())
                return BeamDecode(hEncoderSeq, decoderInitFunction, embedTarget, decoderStepFunction, decoderOutputFunction);
            // decoding loop
            DecoderRecurrentState decoderState = decoderInitFunction(hEncoderSeq);
            vector<DecoderRecurrentState> decoderStates(historySeq.size()); // inner state and attentionContext remembered here
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
            let stateSeq            = Splice(TransformSelectMember(decoderStates, &DecoderRecurrentState::state           ), Axis::EndStaticAxis());
            let attentionContextSeq = Splice(TransformSelectMember(decoderStates, &DecoderRecurrentState::attentionContext), Axis::EndStaticAxis());
            // stack of output transforms
            let z = decoderOutputFunction(stateSeq, attentionContextSeq);
            return z;
        });
}

BinaryModel/*auto*/ CreateModelFunction()
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
            //DOLOG(sourceSeq);
            //if (historySeq != Variable())
            //    DOLOG(historySeq);
            // embedding
            let eFwd = embedFwd(sourceSeq);
            let eBwd = eFwd;// embedBwd(sourceSeq);
            // encoder
            let hSeq = encode(eFwd, eBwd);
            // decoder (outputting log probs of words)
            let zSeq = decode(historySeq, hSeq);
            //DOLOG(zSeq);
            return zSeq;
        });
}

BinaryFoldingModel/*auto*/ CreateCriterionFunction(const BinaryModel& model_fn)
{
    // features and labels are tensors with first dimension being the length
    BinaryModel criterion = [=](const Variable& sourceSeq, const Variable& targetSeq) -> Variable
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
        //DOPRINT("src", sourceSeq, srcVocabFile);
        //DOPRINT("tgt", labelsSeq, tgtVocabFile);
        //DOPRINT("hyp", zSeq, tgtVocabFile);
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
        let collatedLosses = Splice(move(lossSequences), Axis(-1), Named("collatedLosses")); CountAPICalls(1);    // concatenate all seq lossSequences
        //DOLOG(collatedLosses);
        //GetValueAsTensor(collatedLosses)->LogToFile(L"collatedLosses", stderr, 10000);
        let mbLoss = ReduceSum(collatedLosses, Axis_DropLastAxis, Named("batchLoss")); CountAPICalls(1); // aggregate over entire minibatch
        Function::SetDynamicProfiler(prevProfiler);
        return mbLoss;
    });
}

// join
template<typename StringCollection>
static typename StringCollection::value_type/*auto*/ Join(const StringCollection& tokens)
{
    typename StringCollection::value_type res;
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
    auto f = _wifstream(vocabFile);
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

static string PrintSequence(const char* prefix, const Variable& seq, const wstring& vocabFile)
{
    let wordSequence = OneHotToWordSequence(seq.Value(), vocabFile);
    fprintf(stderr, "\n%s=%s\n", prefix, wordSequence.c_str()), fflush(stderr);
    return wordSequence;
}

MinibatchSourcePtr CreateMinibatchSource(const wstring& srcFile, const wstring& tgtFile, bool isTraining)
{
    auto minibatchSourceConfig = MinibatchSourceConfig({ PlainTextDeserializer(
        {
            PlainTextStreamConfiguration(L"src", srcVocabSize, { srcFile }, { srcVocabFile, insertBOS ? L"<s>" : L"", L"</s>", L"<unk>" }),
            PlainTextStreamConfiguration(L"tgt", tgtVocabSize, { tgtFile }, { tgtVocabFile, insertBOS ? L"<s>" : L"", L"</s>", L"<unk>" })
        }) },
        /*randomize=*/isTraining);
    minibatchSourceConfig.maxSamples = isTraining ? MinibatchSource::InfinitelyRepeat : MinibatchSource::FullDataSweep;
    minibatchSourceConfig.isMultithreaded = false;
    minibatchSourceConfig.enableMinibatchPrefetch = false; // TODO: reenable the multi-threading and see if (1) it works and (2) makes things faster
    // BUGBUG: ^^ I see two possibly related bugs --TODO: This has been fixed already. Once everything works, switch back to normal.
    //  - when running on CPU, this fails reliably with what looks like a race condition.
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

// helper to smooth a variable over time through a low-pass filter
class SmoothedCriterion
{
    NDArrayViewPtr smoothedNumer; // loss value is accumulated on the GPU
    double smoothedDenom = 0;
    const double smoothingTimeConstant = 409600; // e.g. smooth over 100 minibatches
public:
    // 'mbLoss' = sum over 'count' loss values
    void Update(NDArrayViewPtr mbLoss, size_t count)
    {
        // first time we allocate the accumulator to match device etc. of the loss value
        if (!smoothedNumer)
            smoothedNumer = make_shared<NDArrayView>(0, mbLoss->GetDataType(), mbLoss->Shape(), mbLoss->Device(), /*readOnly=*/false);
        // update smoothed numerator and denominator
        // this smoothing will converge to 1.0 for the denominator
        // (actually a little higher since we add 'count' items instead of 1)
        let decay      =     exp(-(double)count / smoothingTimeConstant);
        let complement = 1 - exp(-1             / smoothingTimeConstant);
        let alpha = complement;
        let beta = decay;
        smoothedDenom   = alpha * count  + beta * smoothedDenom;
        //smoothedNumer = alpha * mbLoss + beta * smoothedNumer;
        NDArrayView::NumericOperation({ mbLoss }, alpha, L"Copy", smoothedNumer, beta);
        // TODO: when we have to rebuild GPUTensor, then change return val for invalid ops from 0 to NaN
    }
    double RunningAverage() const
    {
        return smoothedNumer->AsScalar<double>() / smoothedDenom;
    }
};

namespace marian { namespace models {
    // direct copy from Marian
    Ptr<EncoderBase> EncoderFactory::construct() {
        //if (options_->get<std::string>("type") == "s2s")
        //    return New<EncoderS2S>(options_);
        //if (options_->get<std::string>("type") == "char-s2s")
        //    return New<CharS2SEncoder>(options_);
        if (options_->get<std::string>("type") == "transformer")
            return New<EncoderTransformer>(options_);

        ABORT("Unknown encoder type");
    }

    Ptr<DecoderBase> DecoderFactory::construct() {
        //if (options_->get<std::string>("type") == "s2s")
        //    return New<DecoderS2S>(options_);
        if (options_->get<std::string>("type") == "transformer")
            return New<DecoderTransformer>(options_);
        //if (options_->get<std::string>("type") == "hard-att")
        //    return New<DecoderHardAtt>(options_);
        //if (options_->get<std::string>("type") == "hard-soft-att")
        //    return New<DecoderHardAtt>(options_);

        ABORT("Unknown decoder type");
    }

    Ptr<EncoderDecoder> EncoderDecoderFactory::construct() {
        Ptr<EncoderDecoder> encdec;

        //if (options_->get<std::string>("type") == "amun")
        //    encdec = New<Amun>(options_);
        //if (options_->get<std::string>("type") == "nematus")
        //    encdec = New<Nematus>(options_);
        //
        if (!encdec)
            encdec = New<EncoderDecoder>(options_);

        for (auto& ef : encoders_)
            encdec->push_back(ef(options_).construct());

        for (auto& df : decoders_)
            encdec->push_back(df(options_).construct());

        return encdec;
    }
} }

static void Train(const DistributedCommunicatorPtr& communicator, const wstring& modelPath, size_t startMbCount)
{
    let numWorkers = communicator->Workers().size();
    let workerId = communicator->CurrentWorker().m_globalRank;

#if 0   // old Dynamite model
    // dynamic model and criterion function
    auto model_fn = CreateModelFunction();
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // run something through to get the parameter matrices shaped --ugh!
    //model_fn(Constant({ srcVocabSize, (size_t)1 }, CurrentDataType(), 0.0, CurrentDevice()), Constant({ tgtVocabSize, (size_t)1 }, CurrentDataType(), 0.0, CurrentDevice()));

#else // Marian model
    auto moptions = Dictionary
    (
        // These are all options given in the log. Not all are used inside the model.
        L"after-batches",                 0,    
        L"after-epochs",                  0,    
        L"allow-unk",                     false,    
        L"batch-flexible-lr",             false,    
        L"batch-normal-words",            1920,    
        L"beam-size",                     6,    
        L"best-deep",                     false,    
        L"clip-norm",                     5,    
        L"cost-type",                     L"ce-sum",
        //L"cost-type",                     L"ce-mean",
        L"dec-cell",                      L"gru",    
        L"dec-cell-base-depth",           2,    
        L"dec-cell-high-depth",           1,    
        L"dec-depth",                     6,    
        L"devices",                       Options::VectorOf({ 0, 1, 2, 3 }),    
        L"dim-emb",                       512,    
        L"dim-rnn",                       1024,    
        L"dim-vocabs",                    Options::VectorOf({ 36000, 36000 }),    // changed from 0,0
        L"disp-freq",                     500,    
        L"dropout-rnn",                   0.0f,    
        L"dropout-src",                   0.0f,    
        L"dropout-trg",                   0.0f,    
        L"early-stopping",                10,    
        L"embedding-fix-src",             false,    
        L"embedding-fix-trg",             false,    
        L"embedding-normalization",       false,    
        L"enc-cell",                      L"gru",    
        L"enc-cell-depth",                1,    
        L"enc-depth",                     6,    
        L"enc-type",                      L"bidirectional",    
        L"exponential-smoothing",         0.0001,    
        L"gradient-dropping",             0,    
        L"guided-alignment-cost",         L"ce",    
        L"guided-alignment-weight",       1,    
        L"ignore-model-config",           false,    
        L"keep-best",                     false,    
        L"label-smoothing",               0.1f, // disable for easier debugging
        L"layer-normalization",           false,    
        //L"learn-rate",                    0.0003,     // not used in Dynamite
        L"log",                           L"model/train.log",    
        L"log-level",                     L"info",    
        L"lr-decay",                      0,    
        L"lr-decay-freq",                 50000,    
        L"lr-decay-inv-sqrt",             16000,    
        L"lr-decay-repeat-warmup",        false,    
        L"lr-decay-reset-optimizer",      false,    
        L"lr-decay-start",                Options::VectorOf({ 10, 1 }),    
        L"lr-decay-strategy",             L"epoch+stalled",    
        L"lr-report",                     true,    
        L"lr-warmup",                     16000,    
        L"lr-warmup-at-reload",           false,    
        L"lr-warmup-cycle",               false,    
        L"lr-warmup-start-rate",          0,    
        L"max-length",                    100,    
        L"max-length-crop",               false,    
        L"maxi-batch",                    1000,    
        L"maxi-batch-sort",               L"trg",    
        L"mini-batch",                    64,    
        L"mini-batch-fit",                true,    
        L"mini-batch-words",              0,    
        L"model",                         L"model/model.npz",    
        L"n-best",                        false,    
        L"no-reload",                     false,    
        L"no-shuffle",                    true,    
        L"normalize",                     1,    
        L"optimizer",                     L"adam",    
        L"optimizer-delay",               1,    
        //L"optimizer-params",              Options::VectorOf({ 0.9, 0.98, 1e-09 }),     // not used in Dynamite
        L"overwrite",                     false,    
        L"quiet",                         false,    
        L"quiet-translation",             true,    
        L"relative-paths",                false,    
        L"save-freq",                     5000,    
        L"seed",                          1111,    
        L"skip",                          false,    
        L"sync-sgd",                      true,    
        L"tempdir",                       L"/tmp",    
        L"tied-embeddings",               false,    
        L"tied-embeddings-all",           true,    
        L"tied-embeddings-src",           false,    
        L"train-sets",                    Options::VectorOf({ L"data/corpus.bpe.en", L"data/corpus.bpe.de" }),    
        L"transformer-dim-ffn",           2048,    
        L"transformer-dropout",           0.1f,    
        L"transformer-dropout-attention", 0.0f,    
        L"transformer-heads",             8,    
        L"transformer-postprocess",       L"an",    
        L"transformer-postprocess-emb",   L"",    
        //L"transformer-postprocess",       L"dan",    
        //L"transformer-postprocess-emb",   L"d",    
        L"transformer-preprocess",        L"",    
        L"type",                          L"transformer",    
        L"valid-freq",                    5000,    
        L"valid-log",                     L"model/valid.log",    
        L"valid-max-length",              1000,    
        L"valid-metrics",                 Options::VectorOf({ L"cross-entropy", L"perplexity", L"translation" }),    
        L"valid-mini-batch",              64,    
        L"valid-script-path",             L"./scripts/validate.sh",    
        L"valid-sets",                    Options::VectorOf({ L"data/valid.bpe.en", L"data/valid.bpe.de" }),    
        L"valid-translation-output",      L"data/valid.bpe.en.output",    
        L"vocabs",                        Options::VectorOf({ L"model/vocab.ende.yml", L"model/vocab.ende.yml" }),    
        L"workspace",                     10000    
    );
    auto mmodel = models::encoder_decoder()(New<Options>(moptions))
        .push_back(models::encoder()("type", "transformer"))
        .push_back(models::decoder()("type", "transformer"))
        .construct();
    // run through once to create all params, so that we can pull them out
    auto graph = New<ExpressionGraph>();
    auto fakeBatch = data::CorpusBatch::fakeBatch(std::vector<size_t>{ /*srcLen=*/3, /*tgtLen*/4 }, /*batchSize=*/1);
    mmodel->build(graph, fakeBatch);
    auto mparamsVector = graph->GetAllParameters();
    auto mparams = shared_ptr<Dynamite::ModelParameters>(new Dynamite::ModelParameters(mparamsVector, {}));
#if 1 // for comparison to Marian, read all initial values from Marian
    vector<float> buf;
    for (auto& p : mparamsVector)
    {
        p.Value()->LogToFile(p.Name() + L" (CNTK)");
        // load it from the Marian init file
        wstring path = L"/data2/fseide/marian-examples/transformer/initvals/" + p.Name();
        let numElem = p.Shape().TotalSize();
        fprintf(stderr, "Loading %d init vals: %S\n", (int)numElem, path.c_str()), fflush(stderr);
        FILE* f = _wfopen(path.c_str(), L"rw");
        if (f)
        {
            buf.resize(numElem);
            fread(buf.data(), sizeof(*buf.data()), buf.size(), f) == 0;
            fclose(f);
            fprintf(stderr, "first val: %.f\n", buf.front());
            auto temp = CNTK::NDArrayView(CNTK::DataType::Float, p.Shape(),
                                      (void*)buf.data(), buf.size() * sizeof(*buf.data()),
                                      CNTK::DeviceDescriptor::CPUDevice(), /*readOnly=*/true)
                                      .DeepClone(p.Value()->Device(), /*readOnly=*/false);
            p.SetValue(temp);
            p.Value()->LogToFile(p.Name() + L" (Marian)");
        }
        else
            fprintf(stderr, "######### Failed to open\n"), fflush(stderr);
    }
#endif
    // TODO: figure out why make_shared does not work here ^^
    //mparams->LogParameters();
    auto model_fn = BinaryFoldingModel(mparamsVector,
            [=](const /*batch*/vector<Variable>& sources, const /*batch*/vector<Variable>& targets) -> Variable
    {
        // convert source batch to Marian CorpusBatch
        // TODO: pass this in
        auto batch = New<marian::data::CorpusBatch>(vector<Ptr<marian::data::SubBatch>>
        {
            New<marian::data::SubBatch>(sources), // wrap references to the CNTK variables in SubBatch instances
            New<marian::data::SubBatch>(targets)
        });

        // encoder
        let state = mmodel->startState(graph, batch);

        // decoder input
        // This sets state->targetEmbeddings_ and state->targetMask_.
        mmodel->getDecoders().front()->groundTruth(state, graph, batch);

        // decoder
        // This sets state->probs_.
        let nextState = mmodel->step(graph, state);

        return nextState->getProbs();
    });
    auto criterion_fn = BinaryFoldingModel({}, { { L"model", model_fn } },
        [=](const /*batch*/vector<Variable>& sources, const /*batch*/vector<Variable>& targets) -> Variable
    {
        using namespace keywords;

        // convert source batch to Marian CorpusBatch
        let batch = New<marian::data::CorpusBatch>(vector<Ptr<marian::data::SubBatch>>
        {
            New<marian::data::SubBatch>(sources), // wrap references to the CNTK variables in SubBatch instances
            New<marian::data::SubBatch>(targets)
        });

        // invoke the model function
        // TODO: pass in the Corpus batch instead of (source, targets), so that we compute that only once
        let probs = model_fn(sources, targets);

        //probs.Value(); // currently fails in DetermineBatchAxis for Transpose operation
        //probs.Value()->LogToFile(L"probs");

        // apply the criterion
        let& trgSubBatch = batch->back();
        let dimBatch = trgSubBatch->batchSize();
        let dimWords = trgSubBatch->batchWidth();
        let  trgMask = graph->constant({ dimWords, dimBatch, 1 }, init = inits::from_vector(trgSubBatch->mask()));
        let& trgData = trgSubBatch->oneHot();

        std::string costType = "ce-sum";// mmodel->opt<string>("cost-type");
        bool inference = false; // TODO
        float ls = inference ? 0.f : mmodel->opt<float>("label-smoothing");

        auto cost = Cost(probs, trgData, trgMask, costType, ls);
        //fprintf(stderr, "====> cost/target = %.8f\n", cost.Value()->AsScalar<float>() / trgSubBatch->batchWords()), fflush(stderr);

        if (mmodel->getOptions()->has("guided-alignment") && !inference) {
            auto alignments = mmodel->getDecoders()[0]->getAlignments();
            ABORT_IF(alignments.empty(), "Model does not seem to support alignments");

            auto att = concatenate(alignments, axis = 3);
            cost = cost + guidedAlignmentCost(graph, batch, mmodel->getOptions(), att);
        }
        cost = Reshape(cost, { 1 }); // Learner.Update() expects a 1-dim vector for some reason
        return cost;
    });
#endif
    // run something through to get the parameter matrices shaped --ugh!
#if 0
    criterion_fn(
    {
        Constant({ srcVocabSize, (size_t)2 }, CurrentDataType(), 0.0, CurrentDevice()),
        Constant({ srcVocabSize, (size_t)3 }, CurrentDataType(), 0.0, CurrentDevice())
    },
    {
        Constant({ tgtVocabSize, (size_t)3 }, CurrentDataType(), 0.0, CurrentDevice()),
        Constant({ tgtVocabSize, (size_t)2 }, CurrentDataType(), 0.0, CurrentDevice())
    });
#endif

    // data
    if (runProfiling     /*||true*/) // if profiling then use small files so we don't measure the load time
        srcTrainFile = srcDevFile, tgtTrainFile = tgtDevFile;
    let minibatchSource = CreateMinibatchSource(srcTrainFile, tgtTrainFile, /*isTraining=*/false);                   //true);

    model_fn.LogParameters();

    let parameters = model_fn.Parameters();
    size_t numParameters = 0;
    for (let& p : parameters)
        numParameters += p.Shape().TotalSize();
    fprintf(stderr, "Total number of learnable parameters is %u in %d parameter tensors.\n", (unsigned int)numParameters, (int)parameters.size()), fflush(stderr);
    let epochSize = 8192 * 10000; // Frantic epoch

    let isDebugging = numWorkers == 1;
    // determine how large a batch size we can stomach in a single go
    // Data parallelism splits up the minibatch over workers.
    // If the resulting size is still too large for GPU RAM to do it in a single forward/backward,
    // then we further reduce the local worker's minibatch size by an integer factor, and use local gradient
    // aggregation before model update/data exchange.
    // (We ignore MB size ramp-up here, i.e. we use unnecessarily small MBs for a while.)
    let workerMinibatchSize = minibatchSize / numWorkers; // worker processes this many samples between model updates
    let numPartialBatchesPerWorker = CeilDiv(workerMinibatchSize, maxBatchSizePerWorker); // need to break local worker's minibatch if >1
    let partialMinibatchSize = minibatchSize / numPartialBatchesPerWorker; // partial MB size across all workers
    fprintf(stderr, "Minibatch size = %d, per worker = %d, per fw/bw = %d across all for %d fw/bws, for each of %d workers\n",
            (int)minibatchSize, (int)workerMinibatchSize, (int)partialMinibatchSize, (int)numPartialBatchesPerWorker, (int)numWorkers), fflush(stderr);
    AdditionalLearningOptions learnerOptions;
    LearnerPtr baseLearner;
    double lr0;
    if (learnerType == "sgd")
    {
        let f = 1.0;
        lr0 = learningRate * f;
        //learnerOptions.gradientClippingThresholdPerSample = 0.001 / 0.002 / 4096; // mimics Frantic but only before LR decay
        baseLearner = SGDLearner(parameters, TrainingParameterPerSampleSchedule(vector<double>{ lr0, lr0 / 2, lr0 / 4, lr0 / 8 }, epochSize), learnerOptions);
    }
    else if (learnerType == "adam")
    {
        // AdaGrad correction-correction:
        //  - LR is specified for av gradient
        //  - numer should be /minibatchSize
        //  - denom should be /sqrt(minibatchSize)
        let f = 1.0;// / sqrt(4096/*minibatchSize*/)/*AdaGrad correction-correction*/;
        // ...TODO: Haven't I already added that to the base code?? Or is this only for compat with Jacob's parameters?
        lr0 = learningRate * f;
        //learnerOptions.gradientClippingThresholdPerSample = 0.001 / 0.002 / 4096;
        baseLearner = AdamLearner(parameters, TrainingParameterPerSampleSchedule(vector<double>{ lr0, lr0/2, lr0/4, lr0/8 }, epochSize),
                                  MomentumAsTimeConstantSchedule(40000), true, MomentumAsTimeConstantSchedule(400000), /*eps=*/1e-9, /*adamax=*/false,
                                  learnerOptions);
    }
    else InvalidArgument("Invalid --learner %s", learnerType.c_str());
    let globalNormClipping = 0.0; // set to 0 to disable. For ce-sum, this does not make much sense.
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
    unordered_map<Parameter, NDArrayViewPtr> parameterNorms; // for in-place weight norm

    vector<vector<vector<Variable>>> args; // [subMinibatchIndex, streamIndex, sequenceIndex]
    size_t totalLabels = 0;
    Microsoft::MSR::CNTK::Timer updateTimer; // timer between Update() calls end-to-end
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
        Dynamite::RestoreFromCheckpoint(path, model_fn.ParametersCombined(), numWorkers, minibatchSource, learner);
        fprintf(stderr, "done\n"), fflush(stderr);
    }
    fflush(stderr);
    // data structure used for data exchange between workers
    MinibatchInfo info
    {
        /*atEndOfData=*/false, /*sweepEnd=*/false, /*numberOfSamples=*/0,
        make_shared<NDArrayView>(0, CurrentDataType(), NDShape{ 1 }, CurrentDevice(), /*readOnly=*/false),
        make_shared<NDArrayView>(0, CurrentDataType(), NDShape{ 1 }, CurrentDevice(), /*readOnly=*/false)
        // TODO: ^^ the above should be NDShape{}
    };

    // MINIBATCH LOOP
    // BUGBUG: The minibatch loop actually runs over partial worker batches.
    //         In current code, the meaning of startMbCount is broken. mbCount should refer to full minibatches.
    SmoothedCriterion smoothedLoss;
    updateTimer.Restart();
    size_t lastUpdateLogTotalLabels = totalLabels; // sample count for updateTimer
    for (mbCount = startMbCount; ; mbCount++)
    {
        let logThisMb = mbCount <= 20 || mbCount % 10 == 0; // (use this to cut down on logging)
        // checkpoint
        if (mbCount % saveEvery == 0 &&
            (/*startMbCount == 0 ||*/ mbCount > startMbCount)) // don't overwrite the starting model
        {
            let modelPathN = IntermediateModelPath(modelPath, mbCount);
            fprintf(stderr, "ssaving: %S... ", modelPathN.c_str()), fflush(stderr); // indicate time of saving, but only main worker actually saves
            Dynamite::SaveCheckpoint(modelPathN, model_fn.ParametersCombined(), numWorkers, minibatchSource, learner);
            fprintf(stderr, "done%s\n", communicator->CurrentWorker().IsMain() ? "" : " by main worker"), fflush(stderr);
            // test model saving
            //for (auto& param : parameters) // destroy parameters as to prove that we reloaded them correctly.
            //    param.Value()->SetValue(0.0);
            //model_fn.RestoreParameters(modelPathN);
        }

        // get next minibatch
        partTimer.Restart();
#if 0
        // dynamically adjust the MB size lower at the start to ramp up
        let fullMbSizeAt = 4000000;
        let lowMbSize = (isDebugging || runProfiling || startMbCount > 0) ? partialMinibatchSize : 750;//partialMinibatchSize / (numWorkers > 6 ? 8 : 16);
        let clamp = [](size_t v, size_t lo, size_t hi) { if (v < lo) return lo; else if (v > hi) return hi; else return v; };
        let partialMinibatchSizeConsideringRampUp = clamp(lowMbSize + (partialMinibatchSize - lowMbSize) * totalLabels / fullMbSizeAt, lowMbSize, partialMinibatchSize);
#else
        let partialMinibatchSizeConsideringRampUp = partialMinibatchSize;
#endif
        if (mbCount % bucketingFactor == 0)
            Dynamite::GetSubBatches(args, { L"src", L"tgt" }, bucketingFactor, /*shuffleSeed=*/mbCount, minibatchSource, partialMinibatchSizeConsideringRampUp,
                                    numWorkers, workerId,
                                    /*inferenceOnly=*/false, CurrentDataType(), CurrentDevice());
        let timeGetNextMinibatch = partTimer.Elapsed();
        //partTimer.Log("FromCNTKMB", minibatchData[minibatchSource->StreamInfo(L"tgt")].numberOfSamples);

        // We get 10 x the minibatch, sort it by source length, and then process it in consecutive chunks of 1/10, which form the actual minibatch for training
        let numAPICalls00 = CountAPICalls(0);
        let& subBatchArgs = args[mbCount % bucketingFactor];
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
#if 1
        let numPartialWorkerScoredLabels = numLabels; // Marian: data has no <s>, so don't subtract anything. TODO: Get this info from reader or setup.
#else
        let numPartialWorkerScoredLabels = numLabels - numSeq; // the <s> is not scored; that's one per sequence. Do not count for averages.
#endif
        if (logThisMb)
            fprintf(stderr, "%5d: #seq: %d, #words: %d -> %d, max len %d -> %d, lr=%.8f * %.8f, partial worker mbSize=%d\n",
                    (int)mbCount,
                    (int)numSeq, (int)numSamples, (int)numLabels, (int)maxSamples, (int)maxLabels,
                    lr0, learner->LearningRate() / lr0, (int)numPartialWorkerScoredLabels);
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
        // BUGBUG: In extreme cases, we can have 0 sentences. HANDLE THAT! Then we can use more GPUs.
        auto partialWorkerLossVar = criterion_fn(subBatchArgs[0], subBatchArgs[1]);
        let timeBuildGraph = partTimer.Elapsed();
        let numAPICalls = CountAPICalls(0) - numAPICalls0;
        numAPICalls;
        //fprintf(stderr, "#API calls = %d\n", (int)numAPICalls), fflush(stderr);
        //exit(1);
        //partTimer.Log("criterion_fn", numLabels);
        // backprop and model update

        // special code branch for profiling the forward operation
        // This code is presently not up-to-date.
        if (runProfiling)
        {
            double lossPerLabel = partialWorkerLossVar.Value()->AsScalar<float>() / numPartialWorkerScoredLabels, smoothedLossVal = 0;
            partTimer.Restart();
            partialWorkerLossVar = Variable(); // this destructs the entire graph
            let timeDeleteGraph = partTimer.Elapsed();
            let elapsed = updateTimer.ElapsedSeconds(); // [sec]
            fprintf(stderr, "%d: >> loss = %.7f; PPL = %.3f << smLoss = %.7f, smPPL = %.2f, seenLabels=%d, %.1f w/s, %.1f ms/w, m=%.0f, g=%.0f, d=%.0f ms\n",
                (int)mbCount, lossPerLabel, exp(lossPerLabel), smoothedLossVal, exp(smoothedLossVal), (int)totalLabels,
                numPartialWorkerScoredLabels / elapsed, 1000.0/*ms*/ * elapsed / numPartialWorkerScoredLabels,
                1000.0 * timeGetNextMinibatch, 1000.0 * timeBuildGraph, 1000.0 * timeDeleteGraph);
            totalLabels;
            updateTimer.Restart(); // BUGBUG: check this w.r.t. partial
            lastUpdateLogTotalLabels = totalLabels;
            continue;
        }

        //if (logThisMb)
        //    fprintf(stderr, "%5d:   ", (int)mbCount); // prefix for the log
        partTimer.Restart();
        let partialWorkerLoss = partialWorkerLossVar.Value(); // trigger computation. Note: This is GPU submission only, not waiting for GPU completion.
        let timeForward = partTimer.Elapsed();
        //fprintf(stderr, "%.7f\n", partialWorkerLoss->AsScalar<float>()), fflush(stderr);
        //exit(1);
        partTimer.Restart();
#if 0
        partialWorkerLoss->AsScalar<float>(); // wait for completion of computation  --don't do this unless we want time measurement
#endif
        let timeForwardGpu = partTimer.Elapsed(); // note: enable the #if above to see the remaining GPU time
        //partTimer.Log("ForwardProp", numLabels);
        // note: we must use numPartialWorkerScoredLabels here
        //fprintf(stderr, "{%.2f, %d-%d}\n", partialWorkerLoss->AsScalar<float>(), (int)numLabels, (int)numSeq), fflush(stderr);
        //if (logThisMb)
        //    fprintf(stderr, "%5d:   ", (int)mbCount); // prefix for the log
        //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
        partTimer.Restart();

#if 0
        let LogGradByName = [&](const wstring& name)
        {
            for (let& kv : gradients)
                if (kv.second && kv.first.Name() == name)
                    return kv.second->LogToFile(kv.first.Name());
        };
        let logName = L"encode.[2].stepBwd.projectInput.W"s;
        LogGradByName(logName);
#endif
        // implement local gradient aggregation if the desired MB size does not fit into local GPU RAM
        let isFirstPartialBatch = mbCount % numPartialBatchesPerWorker == 0;
        let isFinalPartialBatch = mbCount % numPartialBatchesPerWorker == (numPartialBatchesPerWorker - 1);
        double beta = isFirstPartialBatch ? 0 : 1; // partial batches except for first are aggregated into the existing gradient memory, without model update
        partialWorkerLossVar.Backward(gradients, beta);
        //LogGradByName(logName);
        //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
        let timeBackward = partTimer.Elapsed();
        //partTimer.Log("BackProp", numLabels);
        if (isFirstPartialBatch)
        {
            info.numberOfSamples = numPartialWorkerScoredLabels;
            info.trainingLossValue ->CopyFrom(*partialWorkerLoss);
            info.evalCriterionValue->CopyFrom(*partialWorkerLoss);
        }
        else // statistics is aggregated over multiple partial batches
        {
            info.numberOfSamples += numPartialWorkerScoredLabels;
            info.trainingLossValue  += partialWorkerLoss; // note: these are GPU objects
            info.evalCriterionValue += partialWorkerLoss;
        }
        //info.trainingLossValue->AsScalar<float>();
        //CNTK::NDArrayView::Sync(DeviceDescriptor::CPUDevice()); // (currently a special sentinel to flush the GPU...)
#if 0       // log the gradients
        //for (let& p : parameters) if (p.Name() == L"project1.W")
        //{
        //    p.Value()->LogToFile(p.Name(), stderr, 800);
        //}
        //if (mbCount > 1)
        //    exit(0);
        if (mbCount % 50 == 0)
        {
            if (mbCount > 0) for (let& p : parameters)
            {
                if (gradients[p]->GetStorageFormat() != StorageFormat::SparseBlockCol)
                    gradients[p]->LogToFile(L"grad " + p.Name(), stderr, 10);
            }
            // log the parameters
            for (let& p : parameters)
                p.Value()->LogToFile(p.Name(), stderr, 10);
        }
#endif

        // model update
        partTimer.Restart();
        if (isFinalPartialBatch) // if partial batches then skip Update() for all but the last partial batch
        {
            // Marian global-gradient-norm clipping
            if (globalNormClipping != 0) // Marian clips the global gradient vector (all gradient values concatenated)
            {
                let normSqrAccVal = make_shared<NDArrayView>(0, CurrentDataType(), NDShape{}, CurrentDevice());
                for (let& p : parameters)
                    NDArrayView::NumericOperation({ gradients[p] }, /*alpha=*/1, L"Sqr", normSqrAccVal, /*beta=*/1);
                let totalL2Norm = sqrt(normSqrAccVal->AsScalar<float>());
                if (totalL2Norm > globalNormClipping)
                {
                    // clip it
                    fprintf(stderr, "Warning: global gradient L2Norm = %.8f, rescaling to norm %.2f\n", totalL2Norm, globalNormClipping), fflush(stderr);
                    for (let& p : parameters)
                        NDArrayView::NumericOperation({ gradients[p] }, /*alpha=*/globalNormClipping / totalL2Norm, L"Copy", gradients[p], /*beta=*/0);
                }
            }
            learner->Update(gradients, info);
            totalLabels += info.numberOfSamples; // also remember #target labels trained into this model
        }
        // keep track of loss
        let mbLoss          = isFinalPartialBatch ? info.trainingLossValue : partialWorkerLoss;
        let numScoredLabels = isFinalPartialBatch ? info.numberOfSamples   : numPartialWorkerScoredLabels;
        if (isFinalPartialBatch) // TODO: only needed on the main thread which logs
            smoothedLoss.Update(mbLoss, numScoredLabels); // note: this happens on the GPU; no GPU sync here
        // TODO: if Update() is distributed, then the resulting NDArrayView should be allowed to be on the CPU (in case of no NCCL)
        let timePerUpdate = partTimer.Elapsed();

#if 0       // weight normalization (hack for now, since I cannot configure it)
        let EndsWith = [](const wstring& s, const wstring& what)
        {
            return s.size() >= what.size() && s.substr(s.size() - what.size()) == what;
        };
        // This filters by name. ".W" is from Dense/Linear, and we don't use weight norm for embed and projectState.
        for (let& p : parameters) if (EndsWith(p.Name(), L".W") && !EndsWith(p.Name(), L"embed.W") && !EndsWith(p.Name(), L"projectState.W"))
        {
            if (parameterNorms.find(p) == parameterNorms.end())
                parameterNorms[p] = make_shared<NDArrayView>(0, CurrentDataType(), NDShape{ p.Shape()[0] }, CurrentDevice());
            static NDArrayViewPtr minusHalf;
            if (!minusHalf)
                minusHalf = make_shared<NDArrayView>(-0.5, CurrentDataType(), NDShape{ }, CurrentDevice());
            //minusHalf->LogToFile(L"minusHalf");
            let norm = parameterNorms[p];
            let W = p.Value();
            NDArrayView::NumericOperation({ W }, 1.0, L"Sqr", norm, 0, L"Sum");
            //norm->LogToFile(L"sqrSum");
            let eps = DEFAULT_EPSILON;
            NDArrayView::NumericOperation({ }, eps*eps, L"ConstOne", norm, 1.0); // sqr += eps^2
            //norm->LogToFile(L"sqrSum+eps^2");
            NDArrayView::NumericOperation({ norm, minusHalf }, 1.0, L"Pow", norm);
            //norm->LogToFile(L"1/sqrt(sqrSum+eps^2)");
            //norm->LogToFile(L"sqrt(sqrSum)+eps");
            //W->LogToFile(p.Name() + L" (before)");
            NDArrayView::NumericOperation({ W, norm }, 1.0, L"ElementwiseProduct", W); // W /= norm
            //W->LogToFile(p.Name() + L" (after)");
            //fprintf(stderr, "in-place weight norm done for %S\n", p.Name().c_str());
        }
#endif

        // clean up
        partTimer.Restart();
        partialWorkerLossVar = Variable(); // this destructs the entire graph
        let timeDeleteGraph = partTimer.Elapsed();

        // log progress
        // Note: Without logging, there is no GPU-CPU transfer.
        if (logThisMb && communicator->CurrentWorker().IsMain())
        {
            fprintf(stderr, "%5d:   loss, PPL = ", (int)mbCount);
            if (isFinalPartialBatch)
            {
                let smoothedLossVal = smoothedLoss.RunningAverage();
                fprintf(stderr, "[smoothed] %4.2f, ### %8.2f ### [this] ", smoothedLossVal, exp(smoothedLossVal));
            }
            else
                fprintf(stderr, "[partial] ");
            let lossPerLabel = mbLoss->AsScalar<double>() / numScoredLabels;
            fprintf(stderr, "%9.7f * %d, %6.3f, seenLabels=%d, ", lossPerLabel, (int)numScoredLabels, exp(lossPerLabel), (int)totalLabels);
            if (isFinalPartialBatch)
            {
                let elapsed = updateTimer.ElapsedSeconds(); // elapsed time between updates
                updateTimer.Restart();                      // restart timer right away so that we get a true end-to-end measurement including everything
                let numTimedLabels = totalLabels - lastUpdateLogTotalLabels;
                lastUpdateLogTotalLabels = totalLabels;
                fprintf(stderr, "%.1f w/s, %.1f ms/w, ", numTimedLabels / elapsed, 1000.0/*ms*/ * elapsed / numTimedLabels);
            }
            fprintf(stderr, "m=%.0f, g=%.0f, f=%.0f+%.0f, b=%.0f, u=%.0f, d=%.0f ms\n",
                    1000.0 * timeGetNextMinibatch, 1000.0 * timeBuildGraph, 1000.0 * timeForward, 1000.0 * timeForwardGpu, 1000.0 * timeBackward, 1000.0 * timePerUpdate, 1000.0 * timeDeleteGraph);
            if (mbCount < 400 || mbCount % 5 == 0) // flush log
                fflush(stderr);
            if (std::isnan(lossPerLabel))
                throw runtime_error("Loss is NaN.");
        }
        //if (mbCount == 11)
        //{
        //    let numAPICalls = CountAPICalls(0) - numAPICalls00;
        //    fprintf(stderr, "#API calls in last minibatch = %.1f * %d\n", numAPICalls / (float)bucketingFactor, (int)bucketingFactor), fflush(stderr);
        //    //if (runProfiling)
        //        return;
        //}
    }
}

static void Evaluate(const wstring& modelPath, size_t modelMbCount,
                     const wstring& srcEvalFile, const wstring& tgtEvalFile,
                     const wstring& outputHypFile)
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
    fprintf(stderr, "Evaluate: loading model: %S... ", path.c_str()), fflush(stderr);
    model_fn.RestoreParameters(path);
    fprintf(stderr, "done\n"), fflush(stderr);

    // data
    let minibatchSource = CreateMinibatchSource(srcEvalFile, tgtEvalFile, /*isTraining=*/false);

    // output file
    fprintf(stderr, "Evaluate: writing output to %S$$\n", outputHypFile.c_str()), fflush(stderr);
    unique_ptr<FILE, void(*)(FILE*)> fOut(_wfopen((outputHypFile + L"$$").c_str(), L"w"), [](FILE* f) { fclose(f); });
    if (!fOut)
        InvalidArgument("Evaluate: Failed to create output file %S", outputHypFile.c_str());

    size_t totalLabels = 0; // total scored labels (excluding the <s>)
    double totalLoss = 0;   // corresponding total aggregate loss

    // MINIBATCH LOOP
    auto criterion_fn = CreateCriterionFunction(model_fn); // ...for now
    vector<vector<vector<Variable>>> args; // [subMinibatchIndex, streamIndex, sequenceIndex]
    for (mbCount = 0; ; mbCount++)
    {
        // get next minibatch
        bool gotData = Dynamite::GetSubBatches(args, { L"src", L"tgt" }, /*bucketingFactor=*/1, /*shuffleSeed=*/0, minibatchSource, /*minibatchSize=*/1,
                                               /*numWorkers=*/1, /*currentWorker=*/0,
                                               /*inferenceOnly=*/true, CurrentDataType(), CurrentDevice());
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
        // beam decoding
        for (size_t seqId = 0; seqId < numSeq; seqId++)
        {
            let& srcSeq = subBatchArgs[0][seqId];
            let& tgtSeq = subBatchArgs[1][seqId];
            let outSeq = model_fn(srcSeq, Variable());
            let src = PrintSequence("src", srcSeq, srcVocabFile);
            let tgt = PrintSequence("tgt", tgtSeq, tgtVocabFile);
            let out = PrintSequence("out", outSeq, tgtVocabFile);
            fflush(stderr);
            fprintf(fOut.get(), "%s\n", out.c_str());
            fflush(fOut.get());
            if (ferror(fOut.get()))
                RuntimeError("Evaluate: Failed to write to output file: %s", strerror(errno));
        }
        //partTimer.Log("GetNextMinibatch", numLabels);
        fprintf(stderr, "%5d: #seq: %d, #words: %d -> %d, max len %d -> %d\n", (int)mbCount,
                (int)numSeq, (int)numSamples, (int)numLabels, (int)maxSamples, (int)maxLabels);
        // decode all sequences
        auto mbLoss = criterion_fn(subBatchArgs[0], subBatchArgs[1]).Value()->AsScalar<double>();
        let numPartialWorkerScoredLabels = numLabels - numSeq; // the <s> is not scored; that's one per sequence. Do not count for averages.
        let lossPerLabel = mbLoss / numPartialWorkerScoredLabels; // note: this does the GPU sync, so better do that only every N
        totalLabels += numPartialWorkerScoredLabels;
        totalLoss += mbLoss;
        fprintf(stderr, "%5d:  loss, PPL = [aggregate] %5.2f, ### %8.2f ### [this] %10.7f, %9.3f, seenLabels=%d\n",
                        (int)mbCount, totalLoss/ totalLabels, exp(totalLoss / totalLabels), lossPerLabel, exp(lossPerLabel), (int)totalLabels);
        fflush(stderr);
    }
    fprintf(stderr, "\n%5d:  loss, PPL = [total] %5.2f, ### %8.2f ###, seenLabels=%d\n",
        (int)mbCount, totalLoss / totalLabels, exp(totalLoss / totalLabels), (int)totalLabels);
    // finalize output file
    fOut.reset(); // close the file
    _wunlink(outputHypFile.c_str());
    if (_wrename((outputHypFile + L"$$").c_str(), outputHypFile.c_str()) != 0)
        RuntimeError("Evaluate: Failed to rename output file to final name: %s", strerror(errno));
    fprintf(stderr, "Evaluate: Output created at %S\n", outputHypFile.c_str()), fflush(stderr);
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
        if (argc > 0 && Front() == string("--") + argTag)
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
#ifndef _MSC_VER // missing in C++11
    template<typename... Ts> struct make_void { typedef void type; };
    template<typename... Ts> using void_t = typename make_void<Ts...>::type;
#endif
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
            InvalidArgument("unexpected extraneous command-line argument '%s'", Pop().c_str());
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
    wstring logPath;
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
                "?runProfiling", runProfiling,
                "system", systemId,
                "id", experimentId,
                // optional overrides of global stuff
                "?workingDirectory", workingDirectory,
                "?modelPath", modelPath,
                "?minibatchSize", minibatchSize,
                "?maxBatchSizePerWorker", maxBatchSizePerWorker,
                "?firstGpu", firstGpu,
                "?numBits", numBits,
                // these are optional to override the system settings
                "?learner", learnerType,
                "?learningRate", learningRate,
                // decoding parameters
                "?maxBeam", maxBeam,
                "?beamWidth", beamWidth,
                "?fromMb", fromMbCount);
        }
        catch (const exception& e)
        {
            fprintf(stderr, "%s\n", e.what()), fflush(stderr);
            throw invalid_argument("required command line: --command train|test --system SYSTEMID --id IDSTRING\n SYSTEMID = chs_enu, rom_enu, etc\n IDSTRING is used to form the log and model path for now");
        }
        //SetCurrentDataType(DataType::Double);
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
        for (size_t retry = 0; ; retry++)
        {
            logPath = outputDirectory + L"/" + command +
                      (retry == 0 ? L"" : (L"." + to_wstring(retry))) +
                      L".log" +
                      (ourRank == 0 ? L"" : (L"." + to_wstring(ourRank)));
#if 1
            FILE* f = _wfopen(logPath.c_str(), L"r");
            if (!f)
                break;
            fclose(f);
#else
            // for a bizarre unknown reason, this crashes on Linux
            if (!boost::filesystem::exists(logPath))
                break;
#endif
            fprintf(stderr, "%S already exists, bumping up the retry count\n", logPath.c_str());
        }
        boost::filesystem::create_directories(boost::filesystem::path(logPath).parent_path());
        FILE* outStream =
            /*if*/ (communicator->CurrentWorker().IsMain()) ?
            _wpopen((L"tee " + logPath).c_str(), L"w")
            /*else*/ :
            _wfopen(logPath.c_str(), L"wt");
        if (!outStream)
            InvalidArgument("error %d opening log file '%S'", errno, logPath.c_str());
        fprintf(stderr, "redirecting stderr to %S\n", logPath.c_str());
        if (_dup2(_fileno(outStream), _fileno(stderr)) == -1)
            InvalidArgument("error %d redirecting stderr to '%S'", errno, logPath.c_str());
        fprintf(stderr, "command line:");
        for (let* p : Span<char**>(argv, argv + argc))
            fprintf(stderr, " %s", p);
        fprintf(stderr, "\nstarting %S as worker[%d]\n", command.c_str(), (int)ourRank), fflush(stderr); // write something to test

        // output file (for evaluation commands)
        let outPath = outputDirectory + L"/" + command +
            L"_fromMb_" + to_wstring(fromMbCount) +
            L"_beamWidth_" + to_wstring(beamWidth) +
            L"_maxBeam_" + to_wstring(maxBeam) +
            L".hyp";

        // perform the command
        if (command == L"train")
            Train(communicator, modelPath, fromMbCount);
        else if (command == L"validate")
            Evaluate(modelPath, fromMbCount, srcDevFile, tgtDevFile, outPath);
        else if (command == L"test")
            Evaluate(modelPath, fromMbCount, srcTestFile, tgtTestFile, outPath);
        else
            InvalidArgument("Unknonw --command %S", command.c_str());
        fprintf(stderr, "redirected stderr to %S\n", logPath.c_str());
    }
    catch (exception& e)
    {
        fprintf(stderr, "redirected stderr to %S\n", logPath.c_str());
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
        fflush(stderr);
#ifdef _MSC_VER
        TerminateProcess(OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, TRUE, GetCurrentProcessId()), 0);
#else
        std::terminate(); // work around the crash when unloading DLLs with CUDA
#endif
        // BUGBUG: Does not do what I thought it would do.
        //return EXIT_FAILURE;
    }
#ifdef _MSC_VER
    if (runProfiling)
        TerminateProcess(OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, TRUE, GetCurrentProcessId()), 0);
#endif
    return EXIT_SUCCESS;
}
