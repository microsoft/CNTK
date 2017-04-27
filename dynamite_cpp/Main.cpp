//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include "TimerUtility.h"
#include "Layers.h"

#include <iostream>
#include <cstdio>

#define let const auto

using namespace CNTK;

using namespace std;

namespace Dynamite {

template<class Base>
class ModelT : public Base
{
public:
    ModelT(const Base& f) : Base(f){}
    // need to think a bit how to store nested NnaryModels
};
typedef ModelT<function<Variable(Variable)>> UnaryModel;
typedef ModelT<function<Variable(Variable,Variable)>> BinaryModel;
typedef ModelT<function<vector<Variable>(const vector<Variable>&)>> UnarySequenceModel;
typedef ModelT<function<vector<Variable>(const vector<Variable>&, const vector<Variable>&)>> BinarySequenceModel;

struct Batch
{
    // UNTESTED
    // This function would trigger the complex behavior.
    static vector<Variable> map(const UnaryModel& f, const vector<Variable>& batch)
    {
        vector<Variable> res;
        res.reserve(batch.size());
        for (const auto& x : batch)
            res.push_back(f(x));
        return res;
    }

    // UNTESTED
    static function<const vector<Variable>&(const vector<Variable>&)> Map(UnaryModel f)
    {
        return [=](const vector<Variable>& batch)
        {
#if 0
            return map(f, batch);
#else
            vector<Variable> res;
            res.reserve(batch.size());
            for (const auto& x : batch)
                res.push_back(f(x));
            return res;
#endif
        };
    }
    static function<vector<Variable>(const vector<Variable>&, const vector<Variable>&)> Map(BinaryModel f)
    {
        return [=](const vector<Variable>& xBatch, const vector<Variable>& yBatch)
        {
            vector<Variable> res;
            res.reserve(xBatch.size());
            assert(yBatch.size() == xBatch.size());
            for (size_t i = 0; i < xBatch.size(); i++)
                res.emplace_back(f(xBatch[i], yBatch[i]));
            return res;
        };
    }
    static function<vector<vector<Variable>>(const vector<vector<Variable>>&, const vector<vector<Variable>>&)> Map(BinarySequenceModel f)
    {
        return [=](const vector<vector<Variable>>& xBatch, const vector<vector<Variable>>& yBatch)
        {
            vector<vector<Variable>> res;
            res.reserve(xBatch.size());
            assert(yBatch.size() == xBatch.size());
            for (size_t i = 0; i < xBatch.size(); i++)
                res.emplace_back(f(xBatch[i], yBatch[i]));
            return res;
        };
    }
};

// UNTESTED
struct UnaryBroadcastingModel : public UnaryModel
{
    typedef UnaryModel Base;
    UnaryBroadcastingModel(const UnaryModel& f) : UnaryModel(f) { }
    Variable operator() (Variable x) const
    {
        return Base::operator()(x);
    }
    vector<Variable> operator() (const vector<Variable>& x) const
    {
        return Batch::map(*this, x);
    }
};

UnaryBroadcastingModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
{
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    return UnaryModel([=](Variable x)
    {
        return Times(E, x);
    });
}

BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto R = Parameter({ outputDim, outputDim                  }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);
    return [=](Variable prevOutput, Variable input)
    {
        return ReLU(Times(W, input) + Times(R, prevOutput) + b);
    };
}

UnaryModel Linear(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);
    return [=](Variable x) { return Times(W, x) + b; };
}

UnaryModel Sequential(const vector<UnaryModel>& fns)
{
    return [=](Variable x)
    {
        auto arg = Combine({ x });
        for (const auto& f : fns)
            arg = f(arg);
        return arg;
    };
}

struct Sequence
{
    const static function<Variable(Variable)> Last;

    static UnaryModel Recurrence(const BinaryModel& stepFunction)
    {
        return [=](Variable x)
        {
            auto dh = PlaceholderVariable();
            auto rec = stepFunction(PastValue(dh), x);
            FunctionPtr(rec)->ReplacePlaceholders({ { dh, rec } });
            return rec;
        };
    }

    static UnaryModel Fold(const BinaryModel& stepFunction)
    {
        auto recurrence = Recurrence(stepFunction);
        return [=](Variable x)
        {
            return Sequence::Last(recurrence(x));
        };
    }

    static function<vector<Variable>(const vector<Variable>&)> Map(UnaryModel f)
    {
        return Batch::Map(f);
    }

    static function<vector<Variable>(const vector<Variable>&)> Embedding(size_t embeddingDim, const DeviceDescriptor& device)
    {
        return Map(Dynamite::Embedding(embeddingDim, device));
    }
};
const /*static*/ function<Variable(Variable)> Sequence::Last = [](Variable x) -> Variable { return CNTK::Sequence::Last(x); };

// slice the last dimension (index with index i; then drop the axis)
Variable Index(Variable x, size_t i)
{
    auto dims = x.Shape().Dimensions();
    x = Slice(x, { Axis((int)x.Shape().Rank() - 1) }, { (int)i }, { (int)i + 1 });
    dims = x.Shape().Dimensions();
    return x;
}

// slice the last dimension (index with index i; then drop the axis)
NDArrayViewPtr Index(NDArrayViewPtr data, size_t i)
{
    auto dims = data->Shape().Dimensions();
    auto startOffset = vector<size_t>(dims.size(), 0); // TODO: get a simpler interface without dynamic vector allocation
    auto extent = dims;
    if (startOffset.back() != i || extent.back() != 1)
    {
        startOffset.back() = i;
        extent.back()      = 1;
        data = data->SliceView(startOffset, extent, true); // slice it
        dims = data->Shape().Dimensions();
    }
    let newShape = NDShape(vector<size_t>(dims.begin(), dims.end() - 1));
    data = data->AsShape(newShape); // and drop the final dimension
    return data;
}

vector<vector<Variable>> FromCNTKMB(const vector<ValuePtr>& inputs, const vector<Variable>& variables, const DeviceDescriptor& device) // variables needed for axis info only
// returns vector[numArgs] OF vector[numBatchItems] OF Constant[seqLen,sampleShape]
{
    let numArgs = inputs.size();
    vector<vector<Variable>> res(numArgs);
    size_t numSeq = 0;
    for (size_t i = 0; i < numArgs; i++)
    {
        // prepare argument i
        let& input    = inputs[i];
        let& variable = variables[i];

        auto sequences = input->UnpackVariableValue(variable, device); // vector[numBatchItems] of NDArrayViews
        if (numSeq == 0)
            numSeq = sequences.size();
        else if (numSeq != sequences.size())
            CNTK::LogicError("inconsistent MB size");
        auto hasAxis = variable.DynamicAxes().size() > 1;

        auto& arg = res[i];
        arg.resize(numSeq);   // resulting argument
        for (size_t s = 0; s < numSeq; s++)
        {
            auto data = sequences[s]; // NDArrayView
            // convert sparse if needed
/*
                global cached_eyes
                dim = shape[1] # (BUGBUG: won't work for >1D sparse objects)
                if dim not in cached_eyes:
                    eye_np = np.array(np.eye(dim), np.float32)
                    cached_eyes[dim] = cntk.NDArrayView.from_dense(eye_np)
                eye = cached_eyes[dim]
                data = data @ eye
                assert shape == data.shape
*/
            // ... needs NDArrayView, and not needed for just building the graph
            // return in correct shape
            if (!hasAxis)
            {
                assert(data->Shape().Dimensions().back() == 1);
                data = Index(data, 0); // slice off sample axis (the last in C++)
            }
            arg[s] = Constant(data);
        }
    }
    return res;
}

}; // namespace

using namespace Dynamite;

UnaryModel CreateModelFunction(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    return Sequential({
        Embedding(embeddingDim, device),
        Dynamite::Sequence::Fold(RNNStep(hiddenDim, device)),
        Linear(numOutputClasses, device)
    });
}

// SequenceClassification.py
UnaryModel CreateModelFunctionUnrolled(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    auto embed  = Embedding(embeddingDim, device);
    auto step   = RNNStep(hiddenDim, device);
    auto linear = Linear(numOutputClasses, device);
    auto zero   = Constant({ hiddenDim }, 0.0f, device);
    return [=](Variable x) -> Variable
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
        return linear(state);
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
        let rFwd = fwd(seq);
        let rBwd = bwd(seq);
        return splice(rFwd, rBwd);
    };
}

function<Variable(const vector<Variable>&, Variable)> AttentionModel(size_t attentionDim, const DeviceDescriptor& device)
{
    auto Wenc = Parameter({ attentionDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto Wdec = Parameter({ attentionDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto v    = Parameter({ attentionDim }, DataType::Float, GlorotUniformInitializer(), device);
    return [=](const vector<Variable>& hEncs, Variable hDec)
    {
        // BUGBUG: suboptimal, redoing attention projection for inputs over again; need CSE
        Variable hEncsTensor = Splice(hEncs, Axis(1)); // [hiddenDim, inputLen]
        let hEncsProj = Times(Wenc, hEncsTensor, /*outputRank=*/1);
        let hDecProj  = Times(Wdec, hDec);
        let u = Tanh(hEncsProj + hDecProj); // // [hiddenDim, inputLen]
        let u1 = Times(v, u, /*outputRank=*/0); // [inputLen]
        let w = Softmax(u1);  // [inputLen] these are the weights
        let hEncsAv = Times(hEncsTensor, w);
        return hEncsAv;
    };
}

// create a s2s translator
BinarySequenceModel CreateModelFunctionS2SAtt(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, size_t attentionDim, const DeviceDescriptor& device)
{
    numOutputClasses; hiddenDim;
    let embed = Embedding(embeddingDim, device);
    let fwdEnc = RNNStep(hiddenDim, device);
    let bwdEnc = RNNStep(hiddenDim, device);
    let zero = Constant({ hiddenDim }, 0.0f, device);
    let encoder = BiRecurrence(fwdEnc, bwdEnc, zero);
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
        let input = Splice({ prevWordEmbedded, attentionAugmentedState }, Axis(0));
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
        auto out = vector<Variable>(outLen);
        //auto state = encoded.back(); // RNN initial state --note: bidir makes little sense here
        Variable state = zero; // RNN initial state
        for (size_t t = 0; t < outLen; t++)
        {
            let& prevOut = t == 0 ? bos : label[t - 1];
            state = decode(encoded, state, prevOut);
            out[t] = outProj(state);
        }
        return encoded;
    };
}

function<pair<Variable,Variable>(Variable, Variable)> CreateCriterionFunction(UnaryModel model)
{
    return [=](Variable features, Variable labels)
    {
        let z = model(features);

        let loss   = CNTK::CrossEntropyWithSoftmax(z, labels);
        let metric = CNTK::ClassificationError    (z, labels);

        return make_pair(loss, metric);
    };
}

function<Variable(const vector<Variable>&, const vector<Variable>&)> CreateCriterionFunctionUnrolled(UnaryModel model)
{
    BinaryModel criterion = [=](Variable feature, Variable label) -> Variable
    {
        let z = model(feature);
        let loss = CNTK::CrossEntropyWithSoftmax(z, label);
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

void TrainSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels)
{
    const size_t inputDim         = 2000;
    const size_t embeddingDim     = 50;
    const size_t hiddenDim        = 25;
    const size_t attentionDim     = 20;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    // static model and criterion function
    //auto model_fn     = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim, device);
    //auto criterion_fn = CreateCriterionFunction(model_fn);

    // dybamic model and criterion function
    auto d_model_fn     = CreateModelFunctionUnrolled(numOutputClasses, embeddingDim, hiddenDim, device);
    auto d_model_fn1    = CreateModelFunctionS2SAtt(inputDim, embeddingDim, hiddenDim, attentionDim, device);
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
    auto features = InputVariable({ inputDim },         true /*isSparse*/, DataType::Float, featuresName);
    auto labels   = InputVariable({ numOutputClasses }, useSparseLabels,   DataType::Float, labelsName, { Axis::DefaultBatchAxis() });

    //auto criterion = criterion_fn(features, labels);
    //auto loss   = criterion.first;
    //auto metric = criterion.second;
    //
    //// train
    //auto learner = SGDLearner(FunctionPtr(loss)->Parameters(), LearningRatePerSampleSchedule(0.05));
    //auto trainer = CreateTrainer(nullptr, loss, metric, { learner });

    const size_t minibatchSize = 200;
    for (size_t repeats = 0; true; repeats++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

#if 1
        // Dynamite
        fprintf(stderr, "#seq: %d, #words: %d\n", (int)minibatchData[featureStreamInfo].numberOfSequences, (int)minibatchData[featureStreamInfo].numberOfSamples);
        vector<vector<Variable>> args;
        Variable mbLoss;
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "FromCNTKMB:     %.6f sec\n");
            //args = FromCNTKMB({ minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, FunctionPtr(loss)->Arguments(), device);
            args = FromCNTKMB({ minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, { InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featuresName), InputVariable({ numOutputClasses }, useSparseLabels, DataType::Float, labelsName,{ Axis::DefaultBatchAxis() }) }, device);
        }
        vector<vector<vector<Variable>>> vargs(args.size());
        for (size_t i = 0; i < args.size(); i++)
        {
            let& batch = args[i]; // vector of variable-len tensors
            auto& vbatch = vargs[i];
            vbatch.resize(batch.size());
            for (size_t j = i; j < batch.size(); j++)
                vbatch[j] = std::move(ToVector(batch[j]));
        }
        let s2sOut = Batch::Map(d_model_fn1)(vargs[0], vargs[0]); // for now auto-encoder
        for (size_t xxx = 0; xxx < 10; xxx++)
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "d_criterion_fn: %.6f sec\n");
            mbLoss = d_criterion_fn(args[0], args[1]);
            mbLoss = d_criterion_fn(args[0], args[1]);
            mbLoss = d_criterion_fn(args[0], args[1]);
            mbLoss = d_criterion_fn(args[0], args[1]);
            mbLoss = d_criterion_fn(args[0], args[1]);
        }
#endif

#if 0
        // static CNTK
        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, /*outputFrequencyInMinibatches=*/ 1);
#endif
    }
}

int main(int argc, char *argv[])
{
    argc; argv;
    try
    {
        //TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
        TrainSequenceClassifier(DeviceDescriptor::CPUDevice(), true);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
    }
}
