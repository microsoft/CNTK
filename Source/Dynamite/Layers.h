//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// experimental/prototypical layers lib in C++

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"

#include <functional>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;
using namespace std;

#define Barrier Alias

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
    // recursively traverse and collect all Parameters
    const void AppendParametersTo(vector<Parameter>& res) const
    {
        for (let& kv : m_parameters)
            res.push_back(kv.second);
        for (let& kv : m_parentParameters)
            kv.second->AppendParametersTo(res);
    }
};
typedef shared_ptr<ModelParameters> ModelParametersPtr;

template<class Base>
class TModel : public Base, public ModelParametersPtr
{
    const ModelParameters& ParameterSet() const { return **this; }
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
    const Parameter& operator[](const wstring& name) { return ParameterSet()[name]; }
    const ModelParameters& Nested(const wstring& name) { return ParameterSet().Nested(name); }
    vector<Parameter> Parameters() const
    {
        vector<Parameter> res;
        ParameterSet().AppendParametersTo(res);
        return res;
    }
};
typedef TModel<function<Variable(const Variable&)>> UnaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&)>> BinaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&, const Variable&)>> TernaryModel;
typedef TModel<function<void(vector<Variable>&, const vector<Variable>&)>> UnarySequenceModel;
typedef TModel<function<void(vector<Variable>&, const vector<Variable>&, const vector<Variable>&)>> BinarySequenceModel;
typedef TModel<function<Variable(const vector<Variable>&)>> UnaryFoldingModel;

struct Batch
{
    // TODO: this is code dup with Sequence; but it is weird that the batches are SequenceModels. Fix this.
    static UnarySequenceModel Map(UnaryModel f)
    {
        return UnarySequenceModel({}, { { L"f", f } },
            [=](vector<Variable>& res, const vector<Variable>& batch)
        {
#if 0
            return map(f, batch);
#else
            res.clear();
            for (const auto& x : batch)
                res.push_back(f(x));
            return res;
#endif
        });
    }

    // for binary functions
    static BinarySequenceModel Map(BinaryModel f)
    {
        return BinarySequenceModel({}, { { L"f", f } },
            [=](vector<Variable>& res, const vector<Variable>& x, const vector<Variable>& y)
        {
            assert(y.size() == x.size());
            res.resize(x.size());
            for (size_t i = 0; i < x.size(); i++)
                res[i] = f(x[i], y[i]);
        });
    }

    // TODO: get rid of this
    // This function would trigger the complex behavior.
    static vector<Variable> map(const UnaryModel& f, const vector<Variable>& batch)
    {
        vector<Variable> res;
        res.reserve(batch.size());
        for (const auto& x : batch)
            res.push_back(f(x));
        return res;
    }

    // batch map
    static function<vector<vector<Variable>>(const vector<vector<Variable>>&, const vector<vector<Variable>>&)> Map(BinarySequenceModel f)
    {
        return [=](const vector<vector<Variable>>& xBatch, const vector<vector<Variable>>& yBatch)
        {
            vector<vector<Variable>> res;
            res.resize(xBatch.size());
            assert(yBatch.size() == xBatch.size());
            for (size_t i = 0; i < xBatch.size(); i++)
                f(res[i], xBatch[i], yBatch[i]);
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
    void operator() (vector<Variable>& res, const vector<Variable>& x) const
    {
        res = Batch::map(*this, x);
    }
    // TODO: get rid if this variant:
    vector<Variable> operator() (const vector<Variable>& x) const
    {
        return Batch::map(*this, x);
    }
};

static UnaryBroadcastingModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
{
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"E");
    return UnaryModel({ E }, [=](const Variable& x)
    {
        return Times(E, x);
    });
}

static BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"W");
    auto R = Parameter({ outputDim, outputDim                  }, DataType::Float, GlorotUniformInitializer(), device, L"R");
    auto b = Parameter({ outputDim }, 0.0f, device, L"b");
    return BinaryModel({ W, R, b }, [=](const Variable& prevOutput, const Variable& input)
    {
        return ReLU(Times(W, input) + b + Times(R, prevOutput));
    });
}

static TernaryModel LSTMStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"W");
    auto R = Parameter({ outputDim, outputDim }, DataType::Float, GlorotUniformInitializer(), device, L"R");
    auto b = Parameter({ outputDim }, 0.0f, device, L"b");
    return TernaryModel({ W, R, b }, [=](const Variable& prevH, const Variable& prevC, const Variable& input)
    {
        // TODO: complete this
        prevC;
        return ReLU(Times(W, input) + b + Times(R, prevH));
    });
}

static UnaryBroadcastingModel Linear(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device, L"W");
    auto b = Parameter({ outputDim }, 0.0f, device, L"b");
    return UnaryModel({ W, b }, [=](const Variable& x) { return Times(W, x) + b; });
}

struct Sequence
{
    static UnarySequenceModel Map(UnaryModel f)
    {
        return UnarySequenceModel({}, { { L"f", f } },
            [=](vector<Variable>& res, const vector<Variable>& batch)
        {
#if 0
            return map(f, batch);
#else
            res.clear();
            for (const auto& x : batch)
                res.push_back(f(x));
            return res;
#endif
        });
    }

    // for binary functions
    static BinarySequenceModel Map(BinaryModel f)
    {
        return BinarySequenceModel({}, { { L"f", f } },
            [=](vector<Variable>& res, const vector<Variable>& x, const vector<Variable>& y)
        {
            assert(y.size() == x.size());
            res.resize(x.size());
            for (size_t i = 0; i < x.size(); i++)
                res[i] = f(x[i], y[i]);
        });
    }

    static UnarySequenceModel Recurrence(const BinaryModel& step, const Variable& initialState, bool goBackwards = false)
    {
        auto barrier = [](const Variable& x) -> Variable { return Barrier(x); };
        // if initialState is a learnable parameter, then we must keep it
        vector<Parameter> rememberedInitialState;
        if (initialState.IsParameter())
            rememberedInitialState.push_back((Parameter)initialState);
        return UnarySequenceModel(rememberedInitialState, { { L"step", step } },
        [=](vector<Variable>& res, const vector<Variable>& seq)
        {
            let len = seq.size();
            res.resize(len);
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
            if (!goBackwards)
                res.back() = barrier(res.back());
            else
                res.front() = barrier(res.front());
        });
    }

    static UnarySequenceModel BiRecurrence(const BinaryModel& stepFwd, const BinaryModel& stepBwd, const Variable& initialStateFwd, const Variable& initialStateBwd)
    {
        let fwd = Recurrence(stepFwd, initialStateFwd);
        let bwd = Recurrence(stepBwd, initialStateBwd, true);
        let splice = Sequence::Map(BinaryModel([](const Variable& a, const Variable& b) { return Splice({ a, b }, Axis(0)); }));
        vector<Variable> rFwd, rBwd;
        return UnarySequenceModel({}, { { L"stepFwd", stepFwd },{ L"stepBwd", stepBwd } },
        [=](vector<Variable>& res, const vector<Variable>& seq) mutable
        {
            fwd(rFwd, seq);
            bwd(rBwd, seq);
            splice(res, rFwd, rBwd);
            rFwd.clear(); rBwd.clear(); // don't hold references
        });
    }

    static UnaryFoldingModel Fold(const BinaryModel& step, const Variable& initialState)
    {
        auto barrier = [](const Variable& x) -> Variable { return Barrier(x); };
        return UnaryFoldingModel({}, { { L"step", step }  },
        [=](const vector<Variable>& x) -> Variable
        {
            Variable state = initialState;
            for (let& xt : x)
                state = step(state, xt);
            return barrier(state);
        });
    }

    // TODO: This is somewhat broken presently.
    static UnarySequenceModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
    {
        let embed = Dynamite::Embedding(embeddingDim, device);
        return UnarySequenceModel(embed.Parameters(), {},
        [=](vector<Variable>& res, const vector<Variable>& x)
        {
            return Map(embed);
        });
    }
};

// slice the last dimension (index with index i; then drop the axis)
static Variable Index(const Variable& input, size_t i)
{
    auto dims = input.Shape().Dimensions();
    Variable x = Slice(input, { Axis((int)input.Shape().Rank() - 1) }, { (int)i }, { (int)i + 1 });
    dims = x.Shape().Dimensions();
    dims.pop_back(); // drop last axis
    x = Reshape(x, dims);
    return x;
}

static inline void as_vector(vector<Variable>& res, const Variable& x)
{
    // 'x' is an entire sequence; last dimension is length
    let len = x.Shape().Dimensions().back();
    res.resize(len);
    for (size_t t = 0; t < len; t++)
        res[t] = Index(x, t);
}

// TODO: move this out, and don't use Dynamite Model structure

static UnaryModel StaticSequential(const vector<UnaryModel>& fns)
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

struct StaticSequence // for CNTK Static
{
    //const static function<Variable(Variable)> Last;
    //static Variable Last(Variable x) { return CNTK::Sequence::Last(x); };

    static UnaryModel Recurrence(const BinaryModel& step)
    {
        return [=](const Variable& x)
        {
            auto dh = PlaceholderVariable();
            auto rec = step(PastValue(dh), x);
            FunctionPtr(rec)->ReplacePlaceholders({ { dh, rec } });
            return rec;
        };
    }

    static UnaryModel Fold(const BinaryModel& step)
    {
        map<wstring, shared_ptr<ModelParameters>> captured;
        captured[L"step"] = step;
        auto recurrence = Recurrence(step);
        return UnaryModel({}, captured, [=](const Variable& x)
        {
            return CNTK::Sequence::Last(recurrence(x));
        });
    }
};

}; // namespace
