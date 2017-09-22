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

#define Axis_DropLastAxis (Axis(-1)) // TODO: make this a CNTK construct, Axis::DropLastAxis(), with a special sentinel; or an official flag to ReduceXXX()

//#define DISABLE_NORMALIZATIONS // #define this to disable all normalizations such as Batch norm, LengthNormalization, and Droppo scaling

#define let const auto
//#define Named(n) (L##n)
#define Named(n) (std::wstring())

using namespace CNTK;
using namespace std;

#define DTYPE DataType::Float
//#define DTYPE DataType::Double

#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced function was removed

namespace Dynamite {

// debugging helper
static inline NDArrayViewPtr GetValueAsTensor(const Variable& var) { return var.Value(); }
static inline NDArrayViewPtr GetValueAsTensor(const FunctionPtr & fun) { return fun->Output().Value(); }
static inline NDArrayViewPtr GetValueAsTensor(const vector<Variable>& vec) { return (Splice(vec, Axis((int)vec[0].Shape().Rank())))->Output().Value(); }
#define LOG(var) (GetValueAsTensor(var)->LogToFile(L#var, stderr, 10)) // helper to log a value

static inline FunctionPtr operator*(const Variable& leftOperand, const Variable& rightOperand)
{
    return ElementTimes(leftOperand, rightOperand);
}

static inline FunctionPtr operator/(const Variable& leftOperand, const Variable& rightOperand)
{
    return ElementDivide(leftOperand, rightOperand);
}

// structure to hold model parameters of a Dynamite layer
// The actual Model instance doubles up as a shared_ptr to this.
struct ModelParameters
{
    map<wstring, Parameter> m_parameters;
    typedef shared_ptr<ModelParameters> ModelParametersPtr;
    map<wstring, ModelParametersPtr> m_nestedParameters;
    ModelParameters(const vector<Parameter>& parameters, const map<wstring, ModelParametersPtr>& parentParameters)
    {
        // remove nested parameters that are empty (which happens for plain lambdas without parameters)
        for (let& kv : parentParameters)
            if (kv.second)
                m_nestedParameters.insert(kv);
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
        auto iter = m_nestedParameters.find(name);
        if (iter == m_nestedParameters.end())
            LogicError("no such captured model: %ls", name.c_str());
        return *iter->second;
    }
public:
    // recursively traverse and collect all Parameters
    void CollectParameters(vector<Parameter>& res, unordered_set<Variable>& visited) const
    {
        for (let& kv : m_parameters)
            if (visited.insert(kv.second).second)
                res.push_back(kv.second);
        for (let& kv : m_nestedParameters)
            kv.second->CollectParameters(res, visited);
    }
    void LogParameters(const wstring& prefix = L"") const
    {
        for (let& kv : m_nestedParameters) // log nested functions
            kv.second->LogParameters(kv.first + L".");
        for (let& kv : m_parameters) // log parameters defined right here
        {
            let name = prefix + kv.first;
            fprintf(stderr, "%S : %S\n", name.c_str(), kv.second.AsString().c_str());
            // for debugging, implant the full name. This way, the full name will show up in AutoBatch log output.
            const_cast<Parameter&>(kv.second).DebugUpdateName(name);
        }
    }
};
typedef ModelParameters::ModelParametersPtr ModelParametersPtr;

template<class Base>
class TModel : public Base, public ModelParametersPtr
{
public:
    TModel(const Base& f) : Base(f){}
    // constructor with parameters (their names are the Name() properties)
    TModel(const vector<Parameter>& parameters, const Base& f)
        : Base(f), ModelParametersPtr(make_shared<ModelParameters>(parameters, map<wstring, ModelParametersPtr>()))
    {
    }
    // constructor with nested items that have names
    TModel(const vector<Parameter>& parameters, const map<wstring, ModelParametersPtr>& nested, const Base& f)
        : Base(f), ModelParametersPtr(make_shared<ModelParameters>(parameters, nested))
    {
    }
    // constructor with nested items that are indexed
private:
    // create a named map where names are [%d]
    map<wstring, ModelParametersPtr> NameNumberedParameters(const vector<ModelParametersPtr>& nested)
    {
        map<wstring, ModelParametersPtr> res;
        for (let& p : nested)
            res[L"[" + std::to_wstring(res.size()) + L"]"] = p;
        return res;
    }
public:
    TModel(const vector<ModelParametersPtr>& nested, const Base& f)
        : Base(f), ModelParametersPtr(make_shared<ModelParameters>(vector<Parameter>(), NameNumberedParameters(nested)))
    {
    }
    // TODO: would be neat to support a vector of strings for tested paths, or even . separated paths
    const Parameter& operator[](const wstring& name) { return (*get())[name]; } // TODO: This may not have a test currently.
    const ModelParameters& Nested(const wstring& name) { return get()->Nested(name); }
    vector<Parameter> Parameters() const
    {
        vector<Parameter> res;
        unordered_set<Variable> visited;
        get()->CollectParameters(res, visited);
        return res;
    }
    void LogParameters() const { get()->LogParameters(); }
    // saving and loading--we go through a proxy Combine() function so that we can use the standard CNTK functions
    void SaveParameters   (const std::wstring& filepath) { ParametersCombined()->Save   (filepath); }
    void RestoreParameters(const std::wstring& filepath) { ParametersCombined()->Restore(filepath); }
    // we use this for checkpointing  --TODO: encapsulate this better
    FunctionPtr ParametersCombined() const
    {
        auto parameters = Parameters();
        return Combine(vector<Variable>(parameters.begin(), parameters.end())); // need to cast from Parameter to Variable
    }
};
typedef TModel<function<Variable(const Variable&)>> UnaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&)>> BinaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&, const Variable&)>> TernaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&, const Variable&, const Variable&)>> QuaternaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&, const vector<Variable>&, const vector<Variable>&)>> QuaternaryModel11NN;
typedef TModel<function<void(vector<Variable>&, const vector<Variable>&)>> UnarySequenceModel;
typedef TModel<function<void(vector<Variable>&, const vector<Variable>&, const vector<Variable>&)>> BinarySequenceModel;
typedef TModel<function<Variable(const vector<Variable>&)>> UnaryFoldingModel;
typedef TModel<function<Variable(const vector<Variable>&, const vector<Variable>&)>> BinaryFoldingModel;

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
        return /*Reshape*/(ReduceSum(Splice(batch, Axis(axis)), /*Axis(axis)*/Axis_DropLastAxis)/*, shape, Named("sum")*/);
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
    //vector<Variable> operator() (const vector<Variable>& x) const
    //{
    //    return Batch::map(*this, x);
    //}
};

// function composition
// TODO: Do we need other overloads as well? SequenceModel, and going back and forth?
static inline UnaryBroadcastingModel operator>> (const UnaryBroadcastingModel& before, const UnaryBroadcastingModel& after)
{
    return UnaryModel({}, { { L"f", before },{ L"g", after } }, [=](const Variable& x) -> Variable
    {
        return after(before(x));
    });
}

// identity function object; makes it easy to disable stuff
static UnaryModel Identity = [](const Variable& x) { return x; };

enum ProjectionOptions
{
    none            = 0x00,
    bias            = 0x01,
#ifndef DISABLE_NORMALIZATIONS
    stabilize       = 0x02,
    batchNormalize  = 0x04,
    lengthNormalize = 0x08,
    weightNormalize = 0x10
#else
    stabilize       = 0,//x02,
    batchNormalize  = 0,//x04,
    lengthNormalize = 0,//x08,
    weightNormalize = 0//x10
#endif
};
static ProjectionOptions operator|(ProjectionOptions a, ProjectionOptions b) { return (ProjectionOptions)(((size_t)a) | ((size_t)b)); }
static UnaryBroadcastingModel Linear(size_t outputDim, ProjectionOptions opts, const DeviceDescriptor& device);
// TODO: sort these functions vv after Linear()
static UnaryBroadcastingModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
{
    // BUGBUG: We would not want a bias here, right? (but BN always comes with one)
    auto embed = Linear(embeddingDim, ProjectionOptions::batchNormalize | ProjectionOptions::bias, device);
    return UnaryModel({ }, { { L"embed", embed } }, [=](const Variable& x)
    {
        return embed(x);
    });
}

#if 1
// create a Barrier function
static UnaryBroadcastingModel Barrier(size_t depthHint, const wstring& name = wstring())
{
    // TODO: we can save just a little by wrapping this into a static function. We'd save the attribute Dictionary (which can be shared).
    return UnaryModel([=](const Variable& x) -> Variable
    {
        return BatchSync(x, depthHint, name);
    });
}
#else
// create a Barrier function
static UnaryBroadcastingModel Barrier(const wstring& name = wstring())
{
    static size_t id = 0; // unique id
    auto thisId = ++id;   // note: don't use 'id' in lambda; it will access the static variable directly
    return UnaryModel([=](const Variable& x) -> Variable
    {
        return BatchSync(x, thisId, name);
    });
}
#endif

// helper to create a unary static lambda by running a lambda over a Placeholder
class StaticModel
{
    size_t m_arity; // actual number of function arguments. Note: m_argsMap contains additional leaves, so its size() is not sufficient.
    mutable vector<pair<Variable, Variable>> m_argsMap; // [*].first contains the Variable in the composite; [*].second are overwritten upon each call
    // ^^ This is not nice. Better: Split into two arrays, one that's constant, and one that is filled in. That second one can use a template function for Invoke().
    FunctionPtr m_composite; // Note: multiple calls to Invoke() assume that the composite does not change; so encapsulate it here.
    // TODO: Move this down into the V2 API; and make Invoke() a private function of this class. Invoke() is a brittle API since it requires control of the composite.
    bool m_isBasicBlock;
    void BeginConstruct(size_t arity, bool isBasicBlock)
    {
        m_arity = arity;
        m_isBasicBlock = isBasicBlock;
        // allocate m_argsMap and populate the Placeholder section (later we will add Parameters)
        m_argsMap.resize(m_arity);
        for (auto& arg : m_argsMap)
            arg.first = PlaceholderVariable();
    }
    void EndConstruct(FunctionPtr&& composite, const std::wstring& name)
    {
        // note: the graph is built by calling the lambda on Placeholders
        // We must pass in the Placeholders and remember them, since the composite itself will not remember their ordering.
        if (!name.empty())
            composite = Alias(composite, name);
        m_composite = move(composite);
        // complete the m_argsMap pairs by including all learnable Parameters in it as well
        // This is needed so that the auto-batcher can see all Parameters that are inside, without having to traverse it.
        for (let& p : m_composite->Parameters())
            m_argsMap.push_back({ p,p }); // presently also must pass all Parameters
    }
    void CheckArity(size_t arity) const
    {
        if (m_arity != arity)
            LogicError("StaticModel: It was attempted to invoke a %d-nary function with %d arguments.", (int)m_arity, (int)arity);
    }
public:
    StaticModel(bool isBasicBlock, const function<Variable(const Variable&)>& f, std::wstring name = std::wstring())
    {
        BeginConstruct(1, isBasicBlock), EndConstruct(move(f(m_argsMap[0].first)), name);
    }
    StaticModel(bool isBasicBlock, const function<Variable(const Variable&, const Variable&)>& f, std::wstring name = std::wstring())
    {
        BeginConstruct(2, isBasicBlock), EndConstruct(move(f(m_argsMap[0].first, m_argsMap[1].first)), name);
    }
    StaticModel(bool isBasicBlock, const function<Variable(const Variable&, const Variable&, const Variable&)>& f, std::wstring name = std::wstring())
    {
        BeginConstruct(3, isBasicBlock), EndConstruct(move(f(m_argsMap[0].first, m_argsMap[1].first, m_argsMap[2].first)), name);
    }
    // To invoke it, we place the arguments into the m_argsMap array next to the corresponding Placeholder.
    // We leave the Parameters in the m_argsMap array untouched (they are at the end).
    // After the call, we destruct the argument as to not accidentally keep a reference to the argument around.
    // BUGBUG: ^^ this caused some expired shared_ptr...? Try again and track it down. Is it not keeping some reference in the called function? A composite?
    Variable operator()(const Variable& x1) const
    {
        CheckArity(1);
        m_argsMap.front().second = x1;
        let res = Invoke(m_composite, m_argsMap, m_isBasicBlock);
        //m_argsMap.front().second = Variable();
        return res;
    }
    Variable operator()(const Variable& x1, const Variable& x2) const
    {
        CheckArity(2);
        m_argsMap[0].second = x1;
        m_argsMap[1].second = x2;
        let res = Invoke(m_composite, m_argsMap, m_isBasicBlock);
        //m_argsMap[0].second = Variable();
        //m_argsMap[1].second = Variable();
        return res;
    }
    Variable operator()(const Variable& x1, const Variable& x2, const Variable& x3) const
    {
        CheckArity(3);
        m_argsMap[0].second = x1;
        m_argsMap[1].second = x2;
        m_argsMap[2].second = x3;
        let res = Invoke(m_composite, m_argsMap, m_isBasicBlock);
        //m_argsMap[0].second = Variable();
        //m_argsMap[1].second = Variable();
        //m_argsMap[2].second = Variable();
        return res;
    }
};

// layer normalization without bias term (which makes not much sense since we have a bias outside anyway in many cases)
static UnaryBroadcastingModel LengthNormalization(const DeviceDescriptor& device, const Axis& axis = Axis(0))
{
#ifdef DISABLE_NORMALIZATIONS
    axis; device;
    return UnaryModel(vector<Parameter>{ }, [=](const Variable& x)
    {
        return x;
    });
#else
    auto scale = Parameter({ }, DTYPE, 1.0, device, L"scale");
    let eps = Constant::Scalar(DTYPE, 1e-16, device);
    let minusHalf = Constant::Scalar(DTYPE, -0.5, device);
    let profiler = Function::CreateDynamicProfiler(1, L"lnorm");
#if 1

    // for efficiency, we set this up as a static graph
    let asBasicBlock = false; // true does not work fully yet
    StaticModel doLengthNorm(asBasicBlock, [=](const Variable& x) -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler);
        let mean = ReduceMean(x, axis); // it would be faster to say mean(x*x)-mu*mu, except that we need to consider rounding errors
        let x0 = x - mean;
        //let invLen = Pow(ReduceSum(x0 * x0, axis) + eps, minusHalf); // TODO: change to InnerProduct
        let invLen = Pow(InnerProduct(x0, x0, axis) + eps, minusHalf);
        let res = x0 * (invLen * scale);
        Function::SetDynamicProfiler(prevProfiler);
        return res;
    }, Named("lengthNorm"));

    // Note: Arguments() is slow. Don't call this inside graph generation.
    return UnaryModel(vector<Parameter>{ scale }, [=](const Variable& x) mutable
    {
#if 1
        return doLengthNorm(x);
#else
        argBuf.front().second = x; // (avoid the repeated malloc)
        let res = Invoke(lengthNormGraph, argBuf, /*isBasicBlock=*/false); // FOR NOW, can't handle basic block yet
        return res;
#endif
    });
#else
    return UnaryModel(vector<Parameter>{ scale }, [=](const Variable& x)
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler);
        let mean = ReduceMean(x, axis); // it would be faster to say mean(x*x)-mu*mu, except that we need to consider rounding errors
        let x0 = x - mean;
        //LOG(x0);
        // BUGBUG: Sqrt() seems hosed!!
        let invLen = Pow(InnerProduct(x0, x0, axis) + eps, minusHalf); // TODO: change to InnerProduct (but we don't have the dims upfront)
        //let invLen = Pow(ReduceMean(x0 * x0, axis) + eps, minusHalf); // TODO: change to InnerProduct (but we don't have the dims upfront)
        //LOG(len);
        // Note: ^^ this parallelizes, while this vv does not
        //let len = Sqrt(TransposeTimes(x, x));
        //let res = x * (invLen /*+ eps*/) * scale;
        //LOG(scale);
        //LOG(res);
        let res = x0 * (invLen * scale);
        Function::SetDynamicProfiler(prevProfiler);
        return res;
    });
#endif
#endif
}

static BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W");
    auto R = Parameter({ outputDim, outputDim                  }, DTYPE, GlorotUniformInitializer(), device, L"R");
    auto b = Parameter({ outputDim }, DTYPE, 0.0, device, L"b");
    return BinaryModel({ W, R, b }, [=](const Variable& prevOutput, const Variable& input)
    {
        return /*Sigmoid*/ReLU(Times(W, input) + b + Times(R, prevOutput), Named("RNNStep.h"));
    });
}

#if 0
static BinaryModel GRU(size_t outputDim, const DeviceDescriptor& device)
{
    let activation = [](const Variable& x) { return Tanh(x); };
    auto W  = Parameter({ outputDim * 3, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W");
    auto R  = Parameter({ outputDim * 2, outputDim }, DTYPE, GlorotUniformInitializer(), device, L"R");
    auto R1 = Parameter({ outputDim    , outputDim }, DTYPE, GlorotUniformInitializer(), device, L"R1");
    auto b  = Parameter({ outputDim * 3 }, DTYPE, 0.0f, device, L"b");
    let normW = LengthNormalization(device);
    let normR = LengthNormalization(device);
    let normR1 = LengthNormalization(device);
    let stackAxis = Axis(0);
    let stackedDim = (int)outputDim;
    let one = Constant::Scalar(DTYPE, 1.0, device); // for "1 -"...
    // e.g. https://en.wikipedia.org/wiki/Gated_recurrent_unit
    return BinaryModel({ W, R, R1, b },
    {
        { L"normW",  normW  },
        { L"normR",  normR  },
        { L"normR1", normR1 }
    },
    [=](const Variable& dh, const Variable& x)
    {
        let& dhs = dh;
        // projected contribution from input(s), hidden, and bias
        let projx3 = b + normW(Times(W, x));
        let projh2 = normR(Times(R, dh));
        let zt_proj = Slice(projx3, stackAxis, 0 * stackedDim, 1 * stackedDim) + Slice(projh2, stackAxis, 0 * stackedDim, 1 * stackedDim);
        let rt_proj = Slice(projx3, stackAxis, 1 * stackedDim, 2 * stackedDim) + Slice(projh2, stackAxis, 1 * stackedDim, 2 * stackedDim);
        let ct_proj = Slice(projx3, stackAxis, 2 * stackedDim, 3 * stackedDim);

        let zt = Sigmoid(zt_proj)->Output();        // update gate z(t)

        let rt = Sigmoid(rt_proj);                  // reset gate r(t)

        let rs = dhs * rt;                          // "cell" c
        let ct = activation(ct_proj + normR1(Times(R1, rs)));

        let ht = (one - zt) * ct + zt * dhs; // hidden state ht / output

        //# for comparison: CUDNN_GRU
        //# i(t) = sigmoid(W_i x(t) + R_i h(t - 1) + b_Wi + b_Ru)
        //# r(t) = sigmoid(W_r x(t) + R_r h(t - 1) + b_Wr + b_Rr)   --same up to here
        //# h'(t) =   tanh(W_h x(t) + r(t) .* (R_h h(t-1)) + b_Wh + b_Rh)   --r applied after projection? Would make life easier!
        //# h(t) = (1 - i(t).*h'(t)) + i(t) .* h(t-1)                     --TODO: need to confirm bracketing with NVIDIA

        return ht;
    });
}
#else
static BinaryModel GRU(size_t outputDim, const DeviceDescriptor& device)
{
    // matrices are stacked in order (i, r, h)
    auto R  = Parameter({ outputDim * 3, outputDim                  }, DTYPE, GlorotUniformInitializer(), device, L"R");
    auto projectInput = Linear(outputDim * 3, ProjectionOptions::lengthNormalize | ProjectionOptions::weightNormalize, device);
    //auto projectState = Linear(outputDim * 3, ProjectionOptions::none, device);
    auto b  = Parameter({ outputDim * 3 }, DTYPE, 0.0f, device, L"b");
    let normR = LengthNormalization(device);
    let stackAxis = Axis(0);
    let stackedDim = (int)outputDim;
    let profiler = Function::CreateDynamicProfiler(1, L"GRU");
    let irBarrier = Barrier(2, Named("irBarrier"));
    // e.g. https://en.wikipedia.org/wiki/Gated_recurrent_unit
    let gru3 = [=](const Variable& dh, const Variable& projdh3, const Variable& projx3) // note: the input has already been projected. That op is batched more widely.
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler, false);
        // projected contribution from input(s), hidden, and bias
        // BUGBUG: Why can we not project R in here again? It's only one composite instance, there can be no batching.
        let i_proj  = Slice(projx3, stackAxis, 0 * stackedDim, 1 * stackedDim, Named("ix_proj")) + Slice(projdh3, stackAxis, 0 * stackedDim, 1 * stackedDim, Named("ih_proj"));
        let r_proj  = Slice(projx3, stackAxis, 1 * stackedDim, 2 * stackedDim, Named("rx_proj")) + Slice(projdh3, stackAxis, 1 * stackedDim, 2 * stackedDim, Named("rh_proj"));
        let cx_proj = Slice(projx3, stackAxis, 2 * stackedDim, 3 * stackedDim, Named("cx_proj"));
        let ch_proj =                                                                              Slice(projdh3, stackAxis, 2 * stackedDim, 3 * stackedDim, Named("ch_proj"));

        let i = Sigmoid(irBarrier(i_proj), Named("i"));  // update gate z(t)  --if 1 then take new input; if 0 then retain state
        let r = Sigmoid(irBarrier(r_proj), Named("r"));  // reset gate r(t)   --new input + projected old state mixed in

        let c_proj = cx_proj + r * ch_proj;
        let c = Tanh(c_proj, Named("c"));                // "cell"

        let h = dh + i * (c - dh);                       // state
        //    = i * c  +  (1 - i) * dh;

        //# for comparison: CUDNN_GRU
        //# i(t) = sigmoid(W_i x(t) + R_i h(t - 1) + b_Wi + b_Ru)
        //# r(t) = sigmoid(W_r x(t) + R_r h(t - 1) + b_Wr + b_Rr)   --same up to here
        //# h'(t) =   tanh(W_h x(t) + r(t) .* (R_h h(t-1)) + b_Wh + b_Rh)   --r applied after projection? Would make life easier!
        //# h(t) = (1 - i(t).*h'(t)) + i(t) .* h(t-1)                     --TODO: need to confirm bracketing with NVIDIA

        Function::SetDynamicProfiler(prevProfiler);
        return h;
    };
    vector<pair<Variable, Variable>> gruArgs = { { PlaceholderVariable(), Variable() }, { PlaceholderVariable(), Variable() },{ PlaceholderVariable(), Variable() } };
    let gru3Composite = Alias(gru3(gruArgs[0].first, gruArgs[1].first, gruArgs[2].first), L"gru");
    for (let& p : gru3Composite->Parameters())
        gruArgs.push_back({ p,p }); // presently also must pass all Parameters
    let gruAsBasicBlock = false;
    StaticModel doGRU(gruAsBasicBlock, [=](const Variable& dh, const Variable& x)->Variable
    {
        let projx3 = b + projectInput(x); // TODO: fold 'b' into the Linear layer
        let projdh3 = normR(Times(R, dh));
        return gru3(dh, projdh3, projx3);
    }, Named("gru"));
    return BinaryModel({ R, b },
    {
        { L"projectInput",  projectInput },
        //{ L"projectState",  projectState },
        { L"normR",  normR  },
    },
    [=](const Variable& dh, const Variable& x) mutable
    {
#if 1
        // TODO: Somehow this increases #nodes from ~300k to ~450k --what's going on?
        return doGRU(dh, x);
#else
        let projx3 = b + projectInput(x); // TODO: fold 'b' into the Linear layer
        let projdh3 = normR(Times(R, dh));
#if 0   // using the composite
        gruArgs[0].second = dh;
        gruArgs[1].second = projdh3;
        gruArgs[2].second = projx3;
        return Invoke(gru3Composite, gruArgs, /*isBasicBlock=*/false); // basic block not working, as it does not know not to batch multiple matrix products
#else   // using imperative code
        return gru3(dh, projdh3, projx3);
#endif
#endif
    });
}
#endif

static TernaryModel LSTM(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W");
    auto R = Parameter({ outputDim, outputDim }, DTYPE, GlorotUniformInitializer(), device, L"R");
    auto b = Parameter({ outputDim }, DTYPE, 0.0f, device, L"b");
    return TernaryModel({ W, R, b }, [=](const Variable& prevH, const Variable& prevC, const Variable& input)
    {
        // TODO: complete this
        prevC;
        return ReLU(Times(W, input) + b + Times(R, prevH));
    });
}

static UnaryBroadcastingModel BatchNormalization(const DeviceDescriptor& device, const wstring& name = wstring());

static UnaryBroadcastingModel Dense(size_t outputDim, const UnaryModel& activation, ProjectionOptions opts, const DeviceDescriptor& device)
{
    let hasBatchNorm  = (opts & (ProjectionOptions::batchNormalize )) != 0;
    let hasLengthNorm = (opts & (ProjectionOptions::lengthNormalize)) != 0;
    let hasWeightNorm = (opts & (ProjectionOptions::weightNormalize)) != 0;
    let hasBias       = (opts & (ProjectionOptions::bias           )) != 0;
#ifdef DISABLE_NORMALIZATIONS
    let hasScale = false;
#else
    let hasScale      = (opts & (ProjectionOptions::stabilize      )) != 0; // Droppo stabilizer
#endif
    if (hasBatchNorm && !hasBias)
        InvalidArgument("Dense: ProjectionOptions::batchNormalize requires ProjectionOptions::bias to be specified as well");
    if (hasScale && (hasBatchNorm || hasLengthNorm))
        InvalidArgument("Dense: ProjectionOptions::stabilize is not meaningful (will cancel out) with batch or layer normalization");
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W");
    auto b = Parameter({ outputDim }, DTYPE, 0.0f, device, L"b");
    auto scale = Parameter({}, DTYPE, 1.0, device, L"Wscale");
    auto weightNormRescale = Parameter({ outputDim }, DTYPE, 1.0, device, L"Wscale");
    let weightNormMinusHalf = Constant::Scalar(DTYPE, -0.5, device);
    let batchNorm = hasBatchNorm ? BatchNormalization(device, Named("DenseBN")) : Identity;
    let lengthNorm = hasLengthNorm ? LengthNormalization(device) : Identity;
    vector<Parameter> parameters{ W };
    if (hasBias && !hasBatchNorm) // batchNorm supplies its own bias
        parameters.push_back(b);
    if (hasScale)
        parameters.push_back(scale);
    if (hasWeightNorm)
        parameters.push_back(weightNormRescale);
    map<wstring, ModelParametersPtr> nested{ { L"activation", activation } };
    if (hasBatchNorm)
        nested[L"batchNorm"] = batchNorm;
    if (hasLengthNorm)
        nested[L"lengthNorm"] = lengthNorm;
    let asBasicBlock = false;
    StaticModel doDense(asBasicBlock, [=](const Variable& x)->Variable
    {
        auto y = x;
        if (hasScale) // (note: could speed this up by moving this before or after, wherever the dimension is lower)
            y = y * scale;
        y = Times(W, y);
        if (hasWeightNorm)
        {
            // pretend W had rows of length 1, by dividing by the row length after the fact
            // Note that this is generated over again, but will be computed only once since it is ready upfront.
            // BUGBUG: Does not work with sparse input, as that implies a sparse gradient, for which we cannot compute the elementwise ops.
            let rowNorm = /*Reshape*/(InnerProduct(W, W, /*Axis(1)*/Axis_DropLastAxis)/*, NDShape{ outputDim }*/);
            // BUGBUG: ^^ this reduction is wrong if W has more than one input axes, e.g. for image
            // TODO: need a ReduceToShape operation? Where instead of an axis, the target shape is specified?
            let invLen = Pow(rowNorm, weightNormMinusHalf);
            let scale1 = invLen * weightNormRescale; // invLen normalizes the weight; weightNormRescale scales it back
            y = scale1 * y;
        }
        if (hasLengthNorm) // note: has no bias
            y = lengthNorm(y);
        if (hasBatchNorm)
            y = batchNorm(y); // note: batchNorm has its own bias
        else if (hasBias)
            y = y + b;
        return activation(y);
    }, Named("dense"));
    return UnaryModel(parameters, nested, [=](const Variable& x)
    {
        return doDense(x);
    });
}

// by default we have a bias and weight norm
//static UnaryBroadcastingModel Dense(size_t outputDim, const UnaryModel& activation, const DeviceDescriptor& device)
//{
//    return Dense(outputDim, activation, ProjectionOptions::bias, device);
//    //return Dense(outputDim, activation, ProjectionOptions::stabilize | ProjectionOptions::bias, device);
//    //return Dense(outputDim, activation, ProjectionOptions::bias | ProjectionOptions::weightNormalize, device);
//}

static UnaryBroadcastingModel Linear(size_t outputDim, ProjectionOptions opts, const DeviceDescriptor& device)
{
    return Dense(outputDim, Identity, opts, device);
}

// by default we have a bias
//static UnaryBroadcastingModel Linear(size_t outputDim, const DeviceDescriptor& device)
//{
//    return Linear(outputDim, ProjectionOptions::stabilize | ProjectionOptions::bias, device);
//}

// create a Barrier function
static UnaryBroadcastingModel BatchNormalization(const DeviceDescriptor& device, const wstring& name /*= wstring()*/)
{
#ifdef DISABLE_NORMALIZATIONS
    device; name;
    return Identity;
#else
    static size_t id = 0; // unique id
    auto thisId = ++id;   // note: don't use 'id' in lambda; it will access the static variable directly
    auto scale = Parameter({ NDShape::InferredDimension }, DTYPE, 1.0, device, L"scale");
    auto bias  = Parameter({ NDShape::InferredDimension }, DTYPE, 0.0, device, L"bias");
    auto runningMean   = Parameter({ NDShape::InferredDimension }, DTYPE, 0.0, device, L"runningMean");
    auto runningInvStd = Parameter({ NDShape::InferredDimension }, DTYPE, 1.0, device, L"runningInvStd");
    auto runningCount  = Parameter({                            }, DTYPE, 0.0, device, L"runningCount");
    // TODO: figure out this Parameter mess for BN
    return UnaryModel({ scale, bias, runningMean, runningInvStd, runningCount }, [=](const Variable& x) -> Variable
    {
        return CNTK::BatchNormalization(x, thisId, scale, bias, runningMean, runningInvStd, runningCount, /*spatial=*/false, 0, 0, 0.0001, name);
    });
#endif
}

// ResNet layer
// Two Dense(ReLU) with skip connection and batch normalization after the matrix product.
static UnaryBroadcastingModel ResidualNet(size_t outputDim, const DeviceDescriptor& device)
{
    let project1 = Linear(outputDim, ProjectionOptions::batchNormalize | ProjectionOptions::bias, device);
    let project2 = Linear(outputDim, ProjectionOptions::batchNormalize | ProjectionOptions::bias, device);
    StaticModel doResidualNet(/*isBasicBlock=*/false, [=](const Variable& x)
    {
        let h = ReLU(project1(x)    , Named("hRes"));
        let r = ReLU(project2(h) + x, Named("rRes"));
        return r;
    });
    return UnaryModel({ },
    {
        { L"project1", project1 },
        { L"project2", project2 },
    },
    [=](const Variable& x)
    {
        return doResidualNet(x);
    });
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
        let barrier = Barrier(600, Named("Recurrence"));
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
            for (size_t t = 0; t < len; t++)
                res[t] = barrier(res[t]);
        });
    }

    // this layer takes two inputs, one forward one backward, to mimic Frantic's config
    static BinarySequenceModel BiRecurrence(const BinaryModel& stepFwd, const Variable& initialStateFwd, 
                                            const BinaryModel& stepBwd, const Variable& initialStateBwd)
    {
        let fwd = Recurrence(stepFwd, initialStateFwd);
        let bwd = Recurrence(stepBwd, initialStateBwd, true);
        //let barrier = Barrier(600, Named("BiRecurrence"));
        let splice = Sequence::Map(BinaryModel([=](const Variable& a, const Variable& b) { return Splice({ /*barrier*/(a), b }, Axis(0), Named("bidi")); }));
        vector<Variable> rFwd, rBwd;
        return BinarySequenceModel({}, { { L"stepFwd", stepFwd },{ L"stepBwd", stepBwd } },
        [=](vector<Variable>& res, const vector<Variable>& inFwd, const vector<Variable>& inBwd) mutable
        {
            fwd(rFwd, inFwd);
            bwd(rBwd, inBwd);
            splice(res, rFwd, rBwd);
            // ^^ batchable
            //    Would bring it down from #source words in MB to 2 launches (rFwd, rBwd) (and associated Gather)
            //    For 3 layers a 600 -> from 1800 launches to 6
            //    But we have 3300 splices, where is the rest?
            rFwd.clear(); rBwd.clear(); // don't hold references
        });
    }

    static UnaryFoldingModel Fold(const BinaryModel& step, const Variable& initialState)
    {
        let barrier = Barrier(600, Named("Fold"));
        return UnaryFoldingModel({}, { { L"step", step }  },
        [=](const vector<Variable>& x) -> Variable
        {
            Variable state = initialState;
            for (let& xt : x)
                state = step(state, xt);
            state = barrier(state);
            return state;
        });
    }

    // Softmax over a vector producing a vector
    static void Softmax(vector<Variable>& res, const vector<Variable>& z, const UnaryModel& barrier = Identity)
    {
        let& shape = z[0].Shape();
        let axis = Axis((int)shape.Rank());
        auto Z = /*Reshape*/(ReduceLogSum(Splice(z, axis), /*axis*/Axis_DropLastAxis)/*, shape*/); // -> [1]
        Z = barrier(Z);
        res.resize(z.size());
        for (size_t t = 0; t < z.size(); t++)
            res[t] = Exp(Minus(z[t], Z, Named("vecSoftmaxMinus")));
    }

    // InnerProduct over a pair of vectors (dot product over the vector dimension)
    static Variable InnerProduct(const vector<Variable>& xs, const vector<Variable>& ys, const std::wstring& name = std::wstring())
    {
        let xRank = xs[0].Shape().Rank();
        let yRank = ys[0].Shape().Rank();
        let axis = Axis((int)max(xRank, yRank));
        // PERF BUGBUG: malloc. Avoidable?
        vector<Variable> temps(xs.size());
        for (size_t t = 0; t < temps.size(); t++)
            temps[t] = xs[t] * ys[t]; // Batched
        let res = /*Reshape*/(ReduceSum(Splice(temps, axis), /*axis*/Axis_DropLastAxis, name)/*, temps[0].Shape(), name*/);
        // TODO: This should be a primitive.
        return res;
    }
};

// built-in Softmax requires temp memory, so we use an explicit expression instead
static Variable LogSoftmax(const Variable& z, const Axis& axis = Axis::AllStaticAxes(), const std::wstring& name = std::wstring())
{
    //LOG(z);
    //LOG(ReduceLogSum(z, axis, L"smLogDenom"));
    return z - ReduceLogSum(z, axis, name);
}

// built-in Softmax requires temp memory, so we use an explicit expression instead
static Variable Softmax(const Variable& z, const Axis& axis = Axis::AllStaticAxes(), const std::wstring& name = std::wstring())
{
    //LOG(LogSoftmax(z, axis));
    return Exp(LogSoftmax(z, axis, name), name);
}

// built-in Softplus is a BlockFunction, so need to replace it here
static Variable Softplus(const Variable& z, const std::wstring& name)
{
    // TODO: This will create a Constant object every single time--better create it once. Or pre-define constant 0 and 1.
    return LogAddExp(z, Constant::Scalar(z.GetDataType(), 0.0), name);
}

// we need a special definition since the built-in one creates a BlockFunction, which costs too much each time
// BUGBUG: AllStaticAxes (=> keepDimensions=false) leads to incorrect auto-batching. Some screwup of batching axis.
//static Variable CrossEntropyWithSoftmax(const Variable& z, const Variable& label, const Axis& axis = Axis::AllStaticAxes())
static Variable CrossEntropyWithSoftmax(const Variable& z, const Variable& label, const Axis& axis = Axis(0))
{
    Variable ceLogNumer;
#if 1
    ceLogNumer = InnerProduct(label, z, axis, Named("ceLogNumer"));
#else
    if (label.IsSparse() && label.Shape().Rank() == 1)
        ceLogNumer = Times(label, z, /*outputRank=*/0, Named("ceLogNumer"));
    else
        ceLogNumer = ReduceSum(ElementTimes(label, z, Named("ceLabel")), axis, Named("ceLogNumer"));
#endif
    return Minus(ReduceLogSum(z, axis, Named("ceLogDenom")), ceLogNumer, Named("ce"));
}

static inline void as_vector(vector<Variable>& res, const Variable& x)
{
    // 'x' is an entire sequence; last dimension is length
    let len = x.size();
    res.resize(len);
    for (size_t t = 0; t < len; t++)
        res[t] = x[t];
}

// TODO: the following are helpers for Static CNTK from C++. Move them out, and don't use Dynamite data types.

static UnaryModel StaticSequential(const vector<UnaryModel>& fns)
{
    map<wstring, ModelParametersPtr> captured;
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
        map<wstring, ModelParametersPtr> captured;
        captured[L"step"] = step;
        auto recurrence = Recurrence(step);
        return UnaryModel({}, captured, [=](const Variable& x)
        {
            return CNTK::Sequence::Last(recurrence(x));
        });
    }
};

}; // namespace

#pragma warning(pop)
