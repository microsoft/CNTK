//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// This implements SequenceClassification.py as an example in CNTK Dynamite.
// Both CNTK Static and CNTK Dynamite are run in parallel to show that both produce the same result.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "PlainTextDeseralizer.h"
#include "Layers.h"
#include "Common.h"
#include "TimerUtility.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;
using namespace std;

using namespace Dynamite;

// baseline model for CNTK Static
UnaryModel CreateModelFunction(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    return StaticSequential({
        Embedding(embeddingDim, device),
        StaticSequence::Fold(RNNStep(hiddenDim, device)),
        Linear(numOutputClasses, device)
    });
}

BinaryModel CreateCriterionFunction(UnaryModel model)
{
    return [=](const Variable& features, const Variable& labels)
    {
        let z = model(features);

        //let loss   = CNTK::CrossEntropyWithSoftmax(z, labels);
        //let loss = Minus(ReduceLogSum(z, Axis::AllStaticAxes()), TransposeTimes(labels, z));
        let loss = Minus(ReduceLogSum(z, Axis(0)), TransposeTimes(labels, z));
        //let metric = CNTK::ClassificationError    (z, labels);
        return loss;
        //return make_pair(loss, metric);
    };
}

// CNTK Dynamite model
UnaryModel CreateModelFunctionUnrolled(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    auto embed   = Embedding(embeddingDim, device);
    auto step    = RNNStep(hiddenDim, device);
    auto zero = Constant({ hiddenDim }, 0.0f, device);
    auto fold = Dynamite::Sequence::Fold(step, zero);
    auto linear  = Linear(numOutputClasses, device);
    vector<Variable> xvec;
    vector<Variable> evec;
    return UnaryModel({},
    {
        { L"embed",  embed  },
        { L"fold",   fold   },
        { L"linear", linear }
    },
    [=](const Variable& x) mutable -> Variable
    {
        // 'x' is an entire sequence; last dimension is length
        as_vector(xvec, x);
        embed(evec, xvec);
        auto h = fold(evec);
        evec.clear(); xvec.clear(); // release the memory
        return linear(h);
    });
}

function<Variable(const vector<Variable>&, const vector<Variable>&)> CreateCriterionFunctionUnrolled(UnaryModel model)
{
    BinaryModel criterion = [=](const Variable& feature, const Variable& label) -> Variable
    {
        let z = model(feature);
        return Dynamite::CrossEntropyWithSoftmax(z, label);
    };
    // create a batch mapper (which will eventually allow suspension)
    let batchModel = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    vector<Variable> losses;
    return [=](const vector<Variable>& features, const vector<Variable>& labels) mutable
    {
        batchModel(losses, features, labels);             // batch-compute the criterion
        let collatedLosses = Splice(losses, Axis(0));     // collate all seq losses
        let mbLoss = ReduceSum(collatedLosses, Axis(0));  // aggregate over entire minibatch
        losses.clear();
        return mbLoss;
    };
}

#if 0
Variable softmax(const Variable& z)
{
    //let Z = ReduceLogSum(z, Axis::AllStaticAxes());
    let Z = ReduceLogSum(z, Axis(0));
    let P = Exp(z - Z);
    return P;
}

void Flush(const Variable& x)
{
    x.Value();
}
void Flush(const FunctionPtr& f)
{
    f->Output().Value();
}

function<Variable(const vector<Variable>&, const Variable&)> AttentionModel(size_t attentionDim, const DeviceDescriptor& device)
{
    auto Wenc = Parameter({ attentionDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto Wdec = Parameter({ attentionDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto v    = Parameter({ attentionDim }, DataType::Float, GlorotUniformInitializer(), device);
    return [=](const vector<Variable>& hEncs, const Variable& hDec)
    {
        // BUGBUG: suboptimal, redoing attention projection for inputs over again; need CSE
        Variable hEncsTensor = Splice(hEncs, Axis(1)); // [hiddenDim, inputLen]
        let hEncsProj = Times(Wenc, hEncsTensor, /*outputRank=*/1);
        let hDecProj  = Times(Wdec, hDec);
        let u = Tanh(hEncsProj + hDecProj); // // [hiddenDim, inputLen]
        let u1 = Times(v, u, /*outputRank=*/0); // [inputLen]   --BUGBUG: fails, but no need
        let w = softmax(u1);  // [inputLen] these are the weights
        let hEncsAv = Times(hEncsTensor, w);
        return hEncsAv;
    };
}

// create a s2s translator
//     auto d_model_fn1 = CreateModelFunctionS2SAtt(inputDim, embeddingDim, 2 * hiddenDim, attentionDim, device); // (Splice cannot concat, so hidden and embedding must be the same)
BinarySequenceModel CreateModelFunctionS2SAtt(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, size_t attentionDim, const DeviceDescriptor& device)
{
    numOutputClasses; hiddenDim;
    let embed = Embedding(embeddingDim, device);
    let fwdEnc = RNNStep(hiddenDim, device);
    let bwdEnc = RNNStep(hiddenDim, device);
    let zero = Constant({ hiddenDim }, 0.0f, device);
    //let encoder = BiRecurrence(fwdEnc, bwdEnc, zero);
    let encoder = Recurrence(fwdEnc, zero);
    let outEmbed = Embedding(embeddingDim, device);
    let bos = Constant({ numOutputClasses }, 0.0f, device); // one-hot representation of BOS symbol --TODO currently using zero for simplicity
    let fwdDec = RNNStep(hiddenDim, device);
    let attentionModel = AttentionModel(attentionDim, device);
    let outProj = Linear(numOutputClasses, device);
    let decode = [=](const vector<Variable>& encoded, const Variable& recurrenceState, const Variable& prevWord)
    {
        // compute the attention state
        let attentionAugmentedState = attentionModel(encoded, recurrenceState);
        // combine attention abnd previous state
        let prevWordEmbedded = outEmbed(prevWord);
        //Flush(prevWordEmbedded);
        //Flush(attentionAugmentedState);
        let input1 = Splice({ prevWordEmbedded, attentionAugmentedState }, Axis(1));
        let input = Reshape(input1, { prevWordEmbedded.Shape().Dimensions()[0] * 2});
        // Splice is not implemented yet along existing axis, so splice into new and flatten
        //Flush(input);
        return fwdDec(recurrenceState, input);
    };
    return [=](const vector<Variable>& input, const vector<Variable>& label) -> vector<Variable>
    {
        // embed the input sequence
        let seq = embed(input);
        // bidirectional recurrence
        let encoded = encoder(seq); // vector<Variable>
        // decode, this version emits unnormalized log probs and uses labels as history
        let outLen = label.size();
        auto losses = vector<Variable>(outLen);
        //auto state = encoded.back(); // RNN initial state --note: bidir makes little sense here
        Variable state = zero; // RNN initial state
        for (size_t t = 0; t < outLen; t++)
        {
            let& prevOut = t == 0 ? bos : label[t - 1];
            state = decode(encoded, state, prevOut);
            let z = outProj(state);
            //let loss = Minus(ReduceLogSum(z, Axis::AllStaticAxes()), Times(label[t], z, /*outputRank=*/0));
            //let loss = Minus(ReduceLogSum(z, Axis::AllStaticAxes()), Times(label[t], z, /*outputRank=*/0));
            let loss = Minus(ReduceLogSum(z, Axis(0)), Times(label[t], z, /*outputRank=*/0));
            Flush(loss);
            losses[t] = loss;
        }
        return losses;
    };
}
#endif

// helper for logging a Variable's value
void LogVal(const Variable& x)
{
    let& val = *x.Value();
    let& shape = val.Shape();
    let* data = val.DataBuffer<float>();
    let total = shape.TotalSize();
    fprintf(stderr, "%S:", shape.AsString().c_str());
    for (size_t i = 0; i < total && i < 5; i++)
        fprintf(stderr, " %.6f", data[i]);
    fprintf(stderr, "\n");
}

void TrainSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels)
{
    const size_t inputDim         = 2000; // TODO: it's only 1000??
    const size_t embeddingDim     = 500;
    const size_t hiddenDim        = 250;
    const size_t attentionDim     = 20;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    // static model and criterion function
    auto model_fn = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim, device);
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // dynamic model and criterion function
    auto d_model_fn = CreateModelFunctionUnrolled(numOutputClasses, embeddingDim, hiddenDim, device);
    auto d_criterion_fn = CreateCriterionFunctionUnrolled(d_model_fn);

    // data
    const wstring featuresName = L"features";
    const wstring labelsName   = L"labels";

#if 1 // test bed for new PlainTextReader
    auto minibatchSourceConfig = MinibatchSourceConfig({ PlainTextDeserializer(
        {
            PlainTextStreamConfiguration(featuresName, inputDim,         { L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.x.txt" }, { L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.x.vocab", L"", L"", L"" }),
            PlainTextStreamConfiguration(labelsName,   numOutputClasses, { L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.y.txt" }, { L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.y.vocab", L"", L"", L"" })
        }) },
        /*randomize=*/true);
    minibatchSourceConfig.maxSamples = MinibatchSource::FullDataSweep;
    let minibatchSource = CreateCompositeMinibatchSource(minibatchSourceConfig);
    // BUGBUG (API): no way to specify MinibatchSource::FullDataSweep
#else
    auto minibatchSource = TextFormatMinibatchSource(trainingCTFPath,
    {
        { featuresName, inputDim,         true,  L"x" },
        { labelsName,   numOutputClasses, false, L"y" }
    }, MinibatchSource::FullDataSweep);
#endif

    auto featureStreamInfo = minibatchSource->StreamInfo(featuresName);
    auto labelStreamInfo   = minibatchSource->StreamInfo(labelsName);

    // build the graph
    useSparseLabels;
    auto features = InputVariable({ inputDim },         true/*false*/ /*isSparse*/, DataType::Float, featuresName);
    auto labels   = InputVariable({ numOutputClasses }, false/*useSparseLabels*/,   DataType::Float, labelsName, { Axis::DefaultBatchAxis() });

    auto criterion = criterion_fn(features, labels); // this sets the shapes and initializes all parameters
    auto loss   = criterion;
    //auto metric = criterion.second;

    // tie model parameters
    d_model_fn.Nested(L"embed" )[L"E"].SetValue(model_fn.Nested(L"[0]")[L"E"]                .Value());
#if 1
    d_model_fn.Nested(L"fold").Nested(L"step")[L"W"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"W"].Value());
    d_model_fn.Nested(L"fold").Nested(L"step")[L"R"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"R"].Value());
    d_model_fn.Nested(L"fold").Nested(L"step")[L"b"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"b"].Value());
#else
    d_model_fn.Nested(L"step"  )[L"W"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"W"].Value());
    d_model_fn.Nested(L"step"  )[L"R"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"R"].Value());
    d_model_fn.Nested(L"step"  )[L"b"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"b"].Value());
#endif
    d_model_fn.Nested(L"linear")[L"W"].SetValue(model_fn.Nested(L"[2]")[L"W"]                .Value());
    d_model_fn.Nested(L"linear")[L"b"].SetValue(model_fn.Nested(L"[2]")[L"b"]                .Value());

    // train
    let d_parameters = d_model_fn.Parameters();
    auto d_learner = SGDLearner(d_parameters, LearningRatePerSampleSchedule(0.05));

    auto learner = SGDLearner(FunctionPtr(loss)->Parameters(), LearningRatePerSampleSchedule(0.05));
    auto trainer = CreateTrainer(nullptr, loss, loss/*metric*/, { learner });

    // force synchronized GPU operation so we can profile more meaningfully
    // better do this externally: set CUDA_LAUNCH_BLOCKING=1
    //DeviceDescriptor::EnableSynchronousGPUKernelExecution();
    fprintf(stderr, "CUDA_LAUNCH_BLOCKING=%s\n", getenv("CUDA_LAUNCH_BLOCKING"));

    // force-ininitialize the GPU system
    // This is a hack for profiling only.
    // Without this, an expensive CUDA call will show up as part of the GatherBatch kernel
    // and distort the measurement.
    auto t1 = make_shared<NDArrayView>(DataType::Float, NDShape({  1,42 }), device);
    auto t2 = make_shared<NDArrayView>(DataType::Float, NDShape({ 13,42 }), device);
    auto t3 = make_shared<NDArrayView>(DataType::Float, NDShape({ 13,42 }), device);
    t3->NumericOperation({ t1, t2 }, 1.0, 26/*PrimitiveOpType::Plus*/);

    const size_t minibatchSize = 200;  // use 10 for ~3 sequences/batch
    for (size_t repeats = 0; true; repeats++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;
        fprintf(stderr, "#seq: %d, #words: %d\n", (int)minibatchData[featureStreamInfo].numberOfSequences, (int)minibatchData[featureStreamInfo].numberOfSamples);

        //LogVal(d_model_fn.Nested(L"linear")[L"b"]); LogVal(model_fn.Nested(L"[2]")[L"b"]);
        //LogVal(d_model_fn.Nested(L"linear")[L"W"]); LogVal(model_fn.Nested(L"[2]")[L"W"]);
        //LogVal(d_model_fn.Nested(L"step")[L"W"]);   LogVal(model_fn.Nested(L"[1]").Nested(L"step")[L"W"]);
        //LogVal(d_model_fn.Nested(L"embed")[L"E"]);  LogVal(model_fn.Nested(L"[0]")[L"E"]);

#if 1
        // Dynamite
        vector<vector<Variable>> args;
        Variable mbLoss;
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "FromCNTKMB:     %.6f sec\n");
            FromCNTKMB(args, { minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, { true, false }, device);
        }
        //vector<vector<vector<Variable>>> vargs(args.size());
        //for (size_t i = 0; i < args.size(); i++)
        //{
        //    let& batch = args[i]; // vector of variable-len tensors
        //    auto& vbatch = vargs[i];
        //    vbatch.resize(batch.size());
        //    for (size_t j = i; j < batch.size(); j++)
        //        vbatch[j] = std::move(ToVector(batch[j]));
        //}
        //mbLoss = d_criterion_fn(args[0], args[1]); mbLoss.Value()->AsScalar<float>();
        //let s2sLoss = Batch::sum(Batch::Map(d_model_fn1)(vargs[0], vargs[0])); // for now auto-encoder
        //s2sLoss.Value();
        {
            // compute not directly comparable due to (1) no batching and (2) sparse, which may be expensive w.r.t. slicing, or not
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "d_criterion_fn: %.6f sec\n");
            mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
        }
        //fprintf(stderr, "uid of first parameter: %S\n", mbLoss.Uid().c_str());
        //fprintf(stderr, "uid of loss: %S\n", d_parameters[0].Uid().c_str());
        unordered_map<Parameter, NDArrayViewPtr> gradients;
        for (let& p : d_parameters)
            gradients[p] = nullptr; // TryGetGradient(p); // TODO: get the existing gradient matrix from the parameter
        double loss1;
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "/// ### CNTK Dynamite:  %.6f sec\n");
#if 1       // model update with Dynamite
            mbLoss.Backward(gradients);
            d_learner->Update(gradients, minibatchData[labelStreamInfo].numberOfSamples);
#endif
            loss1 = mbLoss.Value()->AsScalar<float>(); // note: this does the GPU sync
        }
        fprintf(stderr, "Dynamite:    CrossEntropy loss = %.7f\n", loss1 / minibatchData[labelStreamInfo].numberOfSamples);
#endif
#if 1   // static CNTK
        double crit;// = trainer->PreviousMinibatchLossAverage();
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "\\\\\\ ### CNTK Static:    %.6f sec\n");
            trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            crit = trainer->PreviousMinibatchLossAverage(); // note: this does the GPU sync
        }
        PrintTrainingProgress(trainer, repeats, /*outputFrequencyInMinibatches=*/ 1);
#endif
    }
}

extern int mt_main(int argc, char *argv[]);

int main(int argc, char *argv[])
{
#if 1
    return mt_main(argc, argv);
#else
    argc; argv;
    try
    {
        TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
        //TrainSequenceClassifier(DeviceDescriptor::CPUDevice(), true);
        // BUGBUG: CPU currently outputs loss=0??
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
    }
#endif
}
