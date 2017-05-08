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
#include "GetValue.h" // meat is here

#include <iostream>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;

using namespace std;

namespace Dynamite {

struct ModelParameters
{
    map<wstring, Parameter> m_parameters;
    map<wstring, shared_ptr<ModelParameters>> m_parentParameters;
    ModelParameters(const vector<Parameter>& parameters, const map<wstring, shared_ptr<ModelParameters>>& parentParameters)
        : m_parentParameters(parentParameters)
    {
        for (const auto& p : parameters)
            if (p.Name().empty())
                LogicError("parameters must be named");
            else
                m_parameters.insert(make_pair(p.Name(), p));
    }
    /*const*/ Parameter& operator[](const wstring& name) const
    {
        auto iter = m_parameters.find(name);
        if (iter == m_parameters.end())
            LogicError("no such parameter: %ls", name.c_str());
        //return iter->second;
        return const_cast<Parameter&>(iter->second);
    }
    const ModelParameters& Nested(const wstring& name) const
    {
        auto iter = m_parentParameters.find(name);
        if (iter == m_parentParameters.end())
            LogicError("no such captured model: %ls", name.c_str());
        return *iter->second;
    }
};
typedef shared_ptr<ModelParameters> ModelParametersPtr;

template<class Base>
class TModel : public Base, public ModelParametersPtr
{
public:
    TModel(const Base& f) : Base(f){}
    // need to think a bit how to store nested NnaryModels
    TModel(const vector<Parameter>& parameters, const Base& f)
        : Base(f), ModelParametersPtr(make_shared<ModelParameters>(parameters, map<wstring, shared_ptr<ModelParameters>>()))
    {
    }
    TModel(const vector<Parameter>& parameters, const map<wstring, shared_ptr<ModelParameters>>& nested, const Base& f)
        : Base(f), ModelParametersPtr(make_shared<ModelParameters>(parameters, nested))
    {
    }
    const Parameter& operator[](const wstring& name) { return Parameters()[name]; }
    const ModelParameters& Nested(const wstring& name) { return Parameters().Nested(name); }
    const ModelParameters& Parameters() const { return **this; }
};
typedef TModel<function<Variable(const Variable&)>> UnaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&)>> BinaryModel;
typedef TModel<function<vector<Variable>(const vector<Variable>&)>> UnarySequenceModel;
typedef TModel<function<vector<Variable>(const vector<Variable>&, const vector<Variable>&)>> BinarySequenceModel;

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

    static Variable sum(const vector<Variable>& batch)
    {
        let& shape = batch.front().Shape();
        let axis = (int)shape.Rank(); // add a new axis
        return Reshape(ReduceSum(Splice(batch, Axis(axis)), Axis(axis)), shape);
    }

    static Variable sum(const vector<vector<Variable>>& batch)
    {
        vector<Variable> allSummands;
        for (const auto& batchItem : batch)
            for (const auto& seqItem : batchItem)
                allSummands.push_back(seqItem);
        return sum(allSummands);
    }
};

// UNTESTED
struct UnaryBroadcastingModel : public UnaryModel
{
    typedef UnaryModel Base;
    UnaryBroadcastingModel(const UnaryModel& f) : UnaryModel(f) { }
    Variable operator() (const Variable& x) const
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
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"E");
    return UnaryModel({ E }, [=](const Variable& x)
    {
        return Times(E, x);
    });
}

BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"W");
    auto R = Parameter({ outputDim, outputDim                  }, DataType::Float, GlorotUniformInitializer(), device, L"R");
    auto b = Parameter({ outputDim }, 0.0f, device, L"b");
    return BinaryModel({ W, R, b }, [=](const Variable& prevOutput, const Variable& input)
    {
        return ReLU(Times(W, input) + b + Times(R, prevOutput));
    });
}

UnaryBroadcastingModel Linear(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"W");
    auto b = Parameter({ outputDim }, 0.0f, device, L"b");
    return UnaryModel({ W, b }, [=](const Variable& x) { return Times(W, x) + b; });
}

UnaryModel Sequential(const vector<UnaryModel>& fns)
{
    map<wstring, shared_ptr<ModelParameters>> captured;
    for (size_t i = 0l; i < fns.size(); i++)
    {
        auto name = L"[" + std::to_wstring(i) + L"]";
        captured[name] = fns[i];
    }
    return UnaryModel({}, captured, [=](const Variable& x)
    {
        auto arg = Combine({ x });
        for (const auto& f : fns)
            arg = f(arg);
        return arg;
    });
}

struct Sequence
{
    const static function<Variable(Variable)> Last;

    static UnaryModel Recurrence(const BinaryModel& stepFunction)
    {
        return [=](const Variable& x)
        {
            auto dh = PlaceholderVariable();
            auto rec = stepFunction(PastValue(dh), x);
            FunctionPtr(rec)->ReplacePlaceholders({ { dh, rec } });
            return rec;
        };
    }

    static UnaryModel Fold(const BinaryModel& stepFunction)
    {
        map<wstring, shared_ptr<ModelParameters>> captured;
        captured[L"step"] = stepFunction;
        auto recurrence = Recurrence(stepFunction);
        return UnaryModel({}, captured, [=](const Variable& x)
        {
            return Sequence::Last(recurrence(x));
        });
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
Variable Index(const Variable& input, size_t i)
{
    auto dims = input.Shape().Dimensions();
    Variable x = Slice(input, { Axis((int)input.Shape().Rank() - 1) }, { (int)i }, { (int)i + 1 });
    dims = x.Shape().Dimensions();
    dims.pop_back(); // drop last axis
    x = Reshape(x, dims);
    return x;
}

// slice the last dimension if an NDArrayView (index with index i; then drop the axis)
// This is used for MB conversion.
NDArrayViewPtr Index(NDArrayViewPtr data, size_t i)
{
    auto dims = data->Shape().Dimensions();
    auto startOffset = vector<size_t>(dims.size(), 0);
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

// helper for converting data to dense
template<typename ElementType>
NDArrayViewPtr MakeEye(size_t n, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
{
    vector<ElementType> buffer(n*n, 0);
    for (size_t i = 0; i < n; i++)
        buffer[i*n + i] = 1;
    auto eye = make_shared<NDArrayView>(dataType, NDShape{ n, n }, buffer.data(), buffer.size() * sizeof(ElementType), DeviceDescriptor::CPUDevice(), /*readOnly=*/false);
    eye = eye->DeepClone(device);
    return eye;
}
NDArrayViewPtr Eye(size_t n, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
{
    static map<pair<size_t, CNTK::DataType>, NDArrayViewPtr> cached;
    let key = make_pair(n, dataType);
    auto iter = cached.find(key);
    if (iter != cached.end())
        return iter->second;
    // need to create it
    NDArrayViewPtr eye;  device;
    switch (dataType)
    {
    case DataType::Float:  eye = MakeEye<float> (n, dataType, device);  break;
    case DataType::Double: eye = MakeEye<double>(n, dataType, device); break;
    default: throw logic_error("Eye: Unsupported ype.");
    }
    cached[key] = eye;
    return eye;
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
            // return in correct shape
            if (!hasAxis)
            {
                assert(data->Shape().Dimensions().back() == 1);
                data = Index(data, 0); // slice off sample axis (the last in C++)
            }
#if 1
            // convert sparse, since currently we cannot GatherBatch() sparse data
            if (data->IsSparse())
            {
                // multiply with  an identity matrix
                auto eye = Eye(data->Shape()[0], data->GetDataType(), data->Device());
                data = NDArrayView::MatrixProduct(false, eye, false, data, false, 1.0, 1);
            }
#endif
            arg[s] = Constant(data);
            data->GetDataType();
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

//function<pair<Variable,Variable>(Variable, Variable)> CreateCriterionFunction(UnaryModel model)
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

// SequenceClassification.py
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
    const size_t attentionDim     = 200;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    // static model and criterion function
    auto model_fn = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim, device);
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // dybamic model and criterion function
    auto d_model_fn = CreateModelFunctionUnrolled(numOutputClasses, embeddingDim, hiddenDim, device);
    auto d_model_fn1 = CreateModelFunctionS2SAtt(inputDim, embeddingDim, 2 * hiddenDim, attentionDim, device); // (Splice cannot concat, so hidden and embedding must be the same)
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

    auto criterion = criterion_fn(features, labels); // this sets the shapes
    auto loss   = criterion;
    //auto metric = criterion.second;

    // tie model parameters
    d_model_fn.Nested(L"embed" )[L"E"].TieValueWith(model_fn.Nested(L"[0]")[L"E"]                );
    d_model_fn.Nested(L"step"  )[L"W"].TieValueWith(model_fn.Nested(L"[1]").Nested(L"step")[L"W"]);
    d_model_fn.Nested(L"step"  )[L"R"].TieValueWith(model_fn.Nested(L"[1]").Nested(L"step")[L"R"]);
    d_model_fn.Nested(L"step"  )[L"b"].TieValueWith(model_fn.Nested(L"[1]").Nested(L"step")[L"b"]);
    d_model_fn.Nested(L"linear")[L"W"].TieValueWith(model_fn.Nested(L"[2]")[L"W"]                );
    d_model_fn.Nested(L"linear")[L"b"].TieValueWith(model_fn.Nested(L"[2]")[L"b"]                );

    // train
    auto learner = SGDLearner(FunctionPtr(loss)->Parameters(), LearningRatePerSampleSchedule(0.05));
    auto trainer = CreateTrainer(nullptr, loss, loss/*metric*/, { learner });

    const size_t minibatchSize = 200;
    for (size_t repeats = 0; true; repeats++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;
        fprintf(stderr, "#seq: %d, #words: %d\n", (int)minibatchData[featureStreamInfo].numberOfSequences, (int)minibatchData[featureStreamInfo].numberOfSamples);

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
        for (size_t xxx = 0; xxx < 1; xxx++)
        {
            // compute not directly comparable due to (1) no batching and (2) sparse, which may be expensive w.r.t. slicing, or not
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "d_criterion_fn: %.6f sec\n");
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //GetValue(mbLoss)->AsScalar<float>();
            mbLoss = d_criterion_fn(args[0], args[1]);// mbLoss.Value();//->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
            //mbLoss.Value()->AsScalar<float>();
        }
        double loss1;
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "dynamite eval:  %.6f sec\n");
            loss1 = GetValue(mbLoss)->AsScalar<float>();
        }
        fprintf(stderr, "Dynamite:    CrossEntropy loss = %.7f\n", loss1 / minibatchData[featureStreamInfo].numberOfSequences);
#endif
#if 1
        // static CNTK
        //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
        double crit;// = trainer->PreviousMinibatchLossAverage();
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "CNTK static:    %.6f sec\n");
            trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            //trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            crit = trainer->PreviousMinibatchLossAverage();
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
