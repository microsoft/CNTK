//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// This implements SequenceClassification.py as an example in CNTK Dynamite.
// Both CNTK Static and CNTK Dynamite are run in parallel to show that both produce the same result.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
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
    return Sequential({
        Embedding(embeddingDim, device),
        Dynamite::Sequence::Fold(RNNStep(hiddenDim, device)),
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
    auto barrier = [](const Variable& x) -> Variable { return Barrier(x); };
    auto linear  = Linear(numOutputClasses, device);
    auto zero    = Constant({ hiddenDim }, 0.0f, device);
    return UnaryModel({},
    {
        { L"embed",  embed  },
        { L"step",   step   },
        { L"linear", linear }
    },
    [=](const Variable& x) -> Variable
    {
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
    });
}

function<Variable(const vector<Variable>&, const vector<Variable>&)> CreateCriterionFunctionUnrolled(UnaryModel model)
{
    BinaryModel criterion = [=](const Variable& feature, const Variable& label) -> Variable
    {
        let z = model(feature);
        //let loss = CNTK::CrossEntropyWithSoftmax(z, label);
        auto s1 = label.Shape();
        auto z1 = z.Shape();
        //let loss = Minus(ReduceLogSum(z, Axis::AllStaticAxes()), TransposeTimes(label, z, /*outputRank=*/0));
        //let loss = Minus(ReduceLogSum(z, Axis::AllStaticAxes()), Times(label, z, /*outputRank=*/0));
        // TODO: reduce ops must be able to drop the axis
        // TODO: dynamite should rewrite Times() that is really a dot product
        let loss = Reshape(Minus(ReduceLogSum(z, Axis(0)), ReduceSum(ElementTimes(label, z), Axis(0))), NDShape());
        return loss;
    };
    // create a batch mapper (which will allow suspension)
    let batchModel = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    return [=](const vector<Variable>& features, const vector<Variable>& labels)
    {
        let losses = batchModel(features, labels);
        let collatedLosses = Splice(losses, Axis(0));     // collate all seq losses
        let mbLoss = ReduceSum(collatedLosses, Axis(0));  // aggregate over entire minibatch
        return mbLoss;
    };
}

// helper to convert a tensor to a vector of slices
vector<Variable> ToVector(const Variable& x)
{
    vector<Variable> res;
    let len = x.Shape().Dimensions().back();
    res.reserve(len);
    for (size_t t = 0; t < len; t++)
        res.emplace_back(Index(x, t));
    return res;
}

UnarySequenceModel Recurrence(const BinaryModel& step, const Variable& initialState, bool goBackwards = false)
{
    return [=](const vector<Variable>& seq)
    {
        let len = seq.size();
        vector<Variable> res(len);
        for (size_t t = 0; t < len; t++)
        {
            if (!goBackwards)
            {
                let& prev = t == 0 ? initialState : res[t - 1];
                res[t] = step(prev, seq[t]);
            }
            else
            {
                let& prev = t == 0 ? initialState : res[len - 1 - (t - 1)];
                res[len - 1 - t] = step(prev, seq[len - 1 - t]);
            }
        }
        return res;
    };
}

UnarySequenceModel BiRecurrence(const BinaryModel& stepFwd, const BinaryModel& stepBwd, const Variable& initialState)
{
    let fwd = Recurrence(stepFwd, initialState);
    let bwd = Recurrence(stepBwd, initialState, true);
    let splice = Batch::Map(BinaryModel([](Variable a, Variable b) { return Splice({ a, b }, Axis(0)); }));
    return [=](const vector<Variable>& seq)
    {
        // does not work since Gather canm currently not concatenate
        let rFwd = fwd(seq);
        let rBwd = bwd(seq);
        return splice(rFwd, rBwd);
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
    const size_t inputDim         = 2000;
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

    auto minibatchSource = TextFormatMinibatchSource(trainingCTFPath,
    {
        { featuresName, inputDim,         true,  L"x" },
        { labelsName,   numOutputClasses, false, L"y" }
    }, MinibatchSource::FullDataSweep);

    auto featureStreamInfo = minibatchSource->StreamInfo(featuresName);
    auto labelStreamInfo   = minibatchSource->StreamInfo(labelsName);

    // build the graph
    useSparseLabels;
    auto features = InputVariable({ inputDim },         false/*true*/ /*isSparse*/, DataType::Float, featuresName);
    auto labels   = InputVariable({ numOutputClasses }, false/*useSparseLabels*/,   DataType::Float, labelsName, { Axis::DefaultBatchAxis() });

    auto criterion = criterion_fn(features, labels); // this sets the shapes and initializes all parameters
    auto loss   = criterion;
    //auto metric = criterion.second;

    // tie model parameters
    d_model_fn.Nested(L"embed" )[L"E"].SetValue(model_fn.Nested(L"[0]")[L"E"]                .Value());
    d_model_fn.Nested(L"step"  )[L"W"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"W"].Value());
    d_model_fn.Nested(L"step"  )[L"R"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"R"].Value());
    d_model_fn.Nested(L"step"  )[L"b"].SetValue(model_fn.Nested(L"[1]").Nested(L"step")[L"b"].Value());
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

    // force-ininitialize the GPU system
    // This is a hack for profiling only.
    // Without this, an expensive CUDA call will show up as part of the GatherBatch kernel
    // and distort the measurement.
    auto t1 = make_shared<NDArrayView>(DataType::Float, NDShape({  1,42 }), device);
    auto t2 = make_shared<NDArrayView>(DataType::Float, NDShape({ 13,42 }), device);
    auto t3 = make_shared<NDArrayView>(DataType::Float, NDShape({ 13,42 }), device);
    t3->NumericOperation({ t1, t2 }, 1.0, 19/*PrimitiveOpType::Plus*/);

    const size_t minibatchSize = 200;  // use 6 for ~2 sequences/batch
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
            //args = FromCNTKMB({ minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, FunctionPtr(loss)->Arguments(), device);
            args = FromCNTKMB({ minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data },
                              //{ InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featuresName),
                              //  InputVariable({ numOutputClasses }, useSparseLabels, DataType::Float, labelsName,{ Axis::DefaultBatchAxis() }) }, device);
                              { features, labels }, device);
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
        fprintf(stderr, "Dynamite:    CrossEntropy loss = %.7f\n", loss1 / minibatchData[featureStreamInfo].numberOfSequences);
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

int main(int argc, char *argv[])
{
    argc; argv;
    try
    {
        TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
        //TrainSequenceClassifier(DeviceDescriptor::CPUDevice(), true);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
    }
}
