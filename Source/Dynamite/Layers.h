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

//#define DISABLE_NORMALIZATIONS // #define this to disable all normalizations such as Batch norm, LengthNormalization, and Droppo scaling. Weight norm is kept enabled, since it is cheap.

#define let const auto
//#define Named(n) (L##n)
#define Named(n) (std::wstring())

using namespace CNTK;
using namespace std;

#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced function was removed

// helper to count API calls
// Call with 0 to get the current count.
__declspec(selectany) size_t g_numAPICallsSoFar = 0;
static inline size_t CountAPICalls(size_t n = 1)
{
    g_numAPICallsSoFar += n;
    return g_numAPICallsSoFar;
}

namespace Dynamite {

// globally set options
// Use these to set the DataType and Device to be used for any call onwards.
static auto& CurrentOptions()
{
    static struct
    {
        DataType         dataType = DataType::Float;
        DeviceDescriptor device   = DeviceDescriptor::UseDefaultDevice();
    } s_currentOptions;
    return s_currentOptions;
};

static inline DeviceDescriptor CurrentDevice()                                  { return CurrentOptions().device; }
static inline void             SetCurrentDevice(const DeviceDescriptor& device) { CurrentOptions().device = device; }
static inline DataType         CurrentDataType()                                { return CurrentOptions().dataType; }
static inline void             SetCurrentDataType(DataType dataType)            { CurrentOptions().dataType = dataType; }

// debugging helper
static inline NDArrayViewPtr GetValueAsTensor(const Variable& var) { return var.Value(); }
static inline NDArrayViewPtr GetValueAsTensor(const FunctionPtr & fun) { return fun->Output().Value(); }
static inline NDArrayViewPtr GetValueAsTensor(const vector<Variable>& vec) { return (Splice(vec, Axis((int)vec[0].Shape().Rank())))->Output().Value(); }
#define LOG(var) (GetValueAsTensor(var)->LogToFile(L#var, stderr, 10)) // helper to log a value

static inline FunctionPtr operator*(const Variable& leftOperand, const Variable& rightOperand)
{
    CountAPICalls();
    return ElementTimes(leftOperand, rightOperand);
}

static inline FunctionPtr operator/(const Variable& leftOperand, const Variable& rightOperand)
{
    CountAPICalls();
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

// create a named map where names are [%d]
static inline map<wstring, ModelParametersPtr> NameNumberedParameters(const vector<ModelParametersPtr>& nested)
{
    map<wstring, ModelParametersPtr> res;
    for (let& p : nested)
        res[L"[" + std::to_wstring(res.size()) + L"]"] = p;
    return res;
}

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
    // This is the most general one.
    TModel(const vector<Parameter>& parameters, const map<wstring, ModelParametersPtr>& nested, const Base& f)
        : Base(f), ModelParametersPtr(make_shared<ModelParameters>(parameters, nested))
    {
    }
    // constructor with nested items that are indexed
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

template<typename Lambda>
static inline TModel<Lambda> Model(const vector<Parameter>& parameters, const map<wstring, ModelParametersPtr>& nested, const Lambda& f)
{
    return TModel<Lambda>(parameters, nested, f);
}

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
        CountAPICalls(2);
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

#if 1
// create a Barrier function
static UnaryBroadcastingModel Barrier(size_t depthHint, const wstring& name = wstring())
{
    // TODO: we can save just a little by wrapping this into a static function. We'd save the attribute Dictionary (which can be shared).
    return UnaryModel([=](const Variable& x) -> Variable
    {
        CountAPICalls();
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

struct Sequence
{
    // map a tensor along its last axis via a given lambda
    template<typename Lambda>
    static Variable map(const Variable& x, const Lambda& f, vector<Variable>& buffer)
    {
        let len = x.size();
        buffer.resize(len);
        for (size_t t = 0; t < len; t++)
            buffer[t] = f(x[t]);
        let res = Splice(buffer, Axis::EndStaticAxis());
        buffer.clear();
        return res;
    }

    // map two tensors along its last axis via a given lambda
    template<typename Lambda>
    static Variable map(const Variable& x, const Variable& y, const Lambda& f, vector<Variable>& buffer)
    {
        let len = x.size();
        if (y.size() != len)
            InvalidArgument("map: x and y have different lengths %d vs. %d", (int)len, (int)y.size());
        buffer.resize(len);
        for (size_t t = 0; t < len; t++)
            buffer[t] = f(x[t], y[t]);
        let res = Splice(buffer, Axis::EndStaticAxis());
        buffer.clear();
        return res;
    }

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

    // The last tensor dimension is the sequence axis.
    static UnaryModel Recurrence(const BinaryModel& step, const Variable& initialState, bool goBackwards = false)
    {
        let barrier = Barrier(600, Named("Recurrence"));
        // if initialState is a learnable parameter, then we must keep it
        vector<Parameter> rememberedInitialState;
        if (initialState.IsParameter())
            rememberedInitialState.push_back((Parameter)initialState);
        return UnaryModel(rememberedInitialState, { { L"step", step } },
        [=](const Variable& x) -> Variable
        {
            let len = x.size();
            vector<Variable> res(len);
            auto state = initialState;
            for (size_t n = 0; n < len; n++)
            {
                let t = goBackwards ? len - 1 - n : n;
                // recurrent step
                state = step(state, x[t]);
                // remember result for output
                res[t] = state;
            }
            CountAPICalls(1);
            let h = Splice(move(res), Axis::EndStaticAxis());
            // The barrier will force the Splice() to happen batch-side.
            return barrier(h);
        });
    }

    // this layer takes two inputs, one forward one backward, to mimic Frantic's config
    // The last tensor dimension is the sequence axis.
    static BinaryModel BiRecurrence(const BinaryModel& stepFwd, const Variable& initialStateFwd, 
                                    const BinaryModel& stepBwd, const Variable& initialStateBwd)
    {
        let fwd = Recurrence(stepFwd, initialStateFwd);
        let bwd = Recurrence(stepBwd, initialStateBwd, true);
        return BinaryModel({}, { { L"stepFwd", stepFwd },{ L"stepBwd", stepBwd } },
        [=](const Variable& inFwd, const Variable& inBwd) -> Variable
        {
            let rFwd = fwd(inFwd);
            let rBwd = bwd(inBwd);
            return Splice({ rFwd, rBwd }, Axis(0), Named("bidi"));
        });
    }

#if 1   // TODO: update to accept a single Variable
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
#endif

    // Softmax over a vector producing a vector
    static void Softmax(vector<Variable>& res, const vector<Variable>& z, const UnaryModel& barrier = Identity)
    {
        let& shape = z[0].Shape();
        let axis = Axis((int)shape.Rank());
        CountAPICalls(2);
        auto Z = ReduceLogSum(Splice(z, axis), /*axis*/Axis_DropLastAxis); // -> [1]
        Z = barrier(Z);
        res.resize(z.size());
        for (size_t t = 0; t < z.size(); t++)
            res[t] = Exp(Minus(z[t], Z, Named("vecSoftmaxMinus")));
        CountAPICalls(2 * z.size());
    }

    // InnerProduct over a pair of vectors (dot product over the vector dimension)
    static Variable InnerProduct(const vector<Variable>& xs, const vector<Variable>& ys, const std::wstring& name = std::wstring())
    {
        let xRank = xs[0].Shape().Rank();
        let yRank = ys[0].Shape().Rank();
        let axis = Axis((int)max(xRank, yRank));
        // PERF BUGBUG: malloc. Avoidable?
        vector<Variable> temps(xs.size());
        CountAPICalls(temps.size());
        for (size_t t = 0; t < temps.size(); t++)
            temps[t] = xs[t] * ys[t]; // Batched
        CountAPICalls(2);
        let res = /*Reshape*/(ReduceSum(Splice(temps, axis), /*axis*/Axis_DropLastAxis, name)/*, temps[0].Shape(), name*/);
        // TODO: This should be a primitive.
        return res;
    }
};

enum ProjectionOptions
{
    none            = 0x00,
    bias            = 0x01,
#ifndef DISABLE_NORMALIZATIONS
    stabilize       = 0x02,
    batchNormalize  = 0x04,
    lengthNormalize = 0x08,
#else
    stabilize       = 0,//x02,
    batchNormalize  = 0,//x04,
    lengthNormalize = 0,//x08,
#endif
    weightNormalize = 0x10
};
static ProjectionOptions operator|(ProjectionOptions a, ProjectionOptions b) { return (ProjectionOptions)(((size_t)a) | ((size_t)b)); }
static UnaryBroadcastingModel Linear(size_t outputDim, ProjectionOptions opts, const wstring& name = wstring());
// TODO: sort these functions vv after Linear()
static UnaryBroadcastingModel Embedding(size_t embeddingDim, const wstring& name = wstring())
{
    // BUGBUG: We would not want a bias here, right? (but BN always comes with one)
    auto embed = Linear(embeddingDim, ProjectionOptions::batchNormalize | ProjectionOptions::bias, name);
    return UnaryModel({ }, { { L"embed", embed } }, [=](const Variable& x)
    {
        return embed(x);
    });
}

// helper to create a unary static lambda by running a lambda over a Placeholder
class StaticModel
{
    shared_ptr<CNTK::Invocable> m_invocable; // this is the only member, so that we can copy this with shared state
    static const size_t batchAxis = 1; // TODO: make this a parameter
public:
    template<typename Lambda>
    StaticModel(bool isBasicBlock, const Lambda& f, std::wstring name = std::wstring()) :
        m_invocable(make_shared<CNTK::Invocable>(isBasicBlock, batchAxis, f, name))
    { }

    template <typename ...ArgTypes>
    Variable operator()(ArgTypes&& ...args) const
    {
        CountAPICalls();
        return m_invocable->operator()(std::forward<ArgTypes>(args)...);
    }
};

// layer normalization without bias term. Normalize each sample to zero mean and length 1, then scale it back up, element-wise.
// This is meant to be invoked via Dense(), where users can select that a bias term should be used as well.
static UnaryBroadcastingModel LengthNormalization(const Axis& axis = Axis(0))
{
#ifdef DISABLE_NORMALIZATIONS
    axis;
    return UnaryModel(vector<Parameter>{ }, [=](const Variable& x)
    {
        return x;
    });
#else
    auto scale    = Parameter({ },   CurrentDataType(), 1.0,   CurrentDevice(), L"scale");
    let eps       = Constant::Scalar(CurrentDataType(), 1e-16, CurrentDevice());
    let minusHalf = Constant::Scalar(CurrentDataType(), -0.5,  CurrentDevice());
    let profiler = Function::CreateDynamicProfiler(1, L"lnorm");

    // for efficiency, we set this up as a set of static graphs
    // subtract a sample's mean
    let doMeanNorm = StaticModel(/*isBasicBlock=*/true, [=](const Variable& x) -> Variable
    {
        CountAPICalls(2);
        let mean = ReduceMean(x, axis);
        return x - mean;
    }, Named("doMeanNorm"));
    // determine the length (L2 norm) of each sample
    let doGetInverseOfL2Norm = StaticModel(/*isBasicBlock=*/true, [=](const Variable& x0) -> Variable
    {
        CountAPICalls(3);
        let invLen = Pow(InnerProduct(x0, x0, axis) + eps, minusHalf);
        return invLen;
    }, Named("doGetInverseOfL2Norm"));
    // perform the length-normalization operation
    let doLengthNorm = StaticModel(/*isBasicBlock=*/false, [=](const Variable& x) -> Variable
    {
        let prevProfiler = Function::SetDynamicProfiler(profiler);
        let x0 = doMeanNorm(x);
        let invLen = doGetInverseOfL2Norm(x0);
        let res = x0 * (invLen * scale); CountAPICalls(2); // note: (invLen*scale), a scalar product, can be batched across multiple invocations
        Function::SetDynamicProfiler(prevProfiler);
        return res;
    }, Named("lengthNorm"));

    // this is the actual function
    return UnaryModel(vector<Parameter>{ scale }, [=](const Variable& x)
    {
        return doLengthNorm(x);
    });
#endif
}

static BinaryModel RNNStep(size_t outputDim)
{
    auto W = Parameter({ (NDShapeDimension)outputDim, NDShape::InferredDimension }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"W");
    auto R = Parameter({ outputDim,        outputDim                             }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"R");
    auto b = Parameter({ outputDim        },                                        CurrentDataType(), 0.0,                        CurrentDevice(), L"b");
    return BinaryModel({ W, R, b }, [=](const Variable& prevOutput, const Variable& input)
    {
        CountAPICalls(5);
        return /*Sigmoid*/ReLU(Times(W, input) + b + Times(R, prevOutput), Named("RNNStep.h"));
    });
}

#if 0
static BinaryModel GRU(size_t outputDim)
{
    let activation = [](const Variable& x) { return Tanh(x); };
    auto W  = Parameter({ outputDim * 3, NDShape::InferredDimension }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"W");
    auto R  = Parameter({ outputDim * 2, outputDim }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"R");
    auto R1 = Parameter({ outputDim    , outputDim }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"R1");
    auto b  = Parameter({ outputDim * 3 }, CurrentDataType(), 0.0f, CurrentDevice(), L"b");
    let normW = LengthNormalization();
    let normR = LengthNormalization();
    let normR1 = LengthNormalization();
    let stackAxis = Axis(0);
    let stackedDim = (int)outputDim;
    let one = Constant::Scalar(CurrentDataType(), 1.0, CurrentDevice()); // for "1 -"...
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
static BinaryModel GRU(size_t outputDim)
{
    // matrices are stacked in order (i, r, h)
    auto projectInput = Linear(outputDim * 3, ProjectionOptions::lengthNormalize | ProjectionOptions::weightNormalize | ProjectionOptions::bias, Named("projectInput"));
    //auto projectState = Linear(outputDim * 3, ProjectionOptions::none, CurrentDevice());
    auto R  = Parameter({ outputDim * 3, outputDim }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"R");
    //auto b  = Parameter({ outputDim * 3            }, CurrentDataType(), 0.0f, CurrentDevice(), L"b");
    let normR = LengthNormalization();
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
        CountAPICalls(8);
        let i_proj  = Slice(projx3, stackAxis, 0 * stackedDim, 1 * stackedDim, Named("ix_proj")) + Slice(projdh3, stackAxis, 0 * stackedDim, 1 * stackedDim, Named("ih_proj"));
        let r_proj  = Slice(projx3, stackAxis, 1 * stackedDim, 2 * stackedDim, Named("rx_proj")) + Slice(projdh3, stackAxis, 1 * stackedDim, 2 * stackedDim, Named("rh_proj"));
        let cx_proj = Slice(projx3, stackAxis, 2 * stackedDim, 3 * stackedDim, Named("cx_proj"));
        let ch_proj =                                                                              Slice(projdh3, stackAxis, 2 * stackedDim, 3 * stackedDim, Named("ch_proj"));

        CountAPICalls(2);
        let i = Sigmoid(irBarrier(i_proj), Named("i"));  // update gate z(t)  --if 1 then take new input; if 0 then retain state
        let r = Sigmoid(irBarrier(r_proj), Named("r"));  // reset gate r(t)   --new input + projected old state mixed in

        CountAPICalls(3);
        let c_proj = cx_proj + r * ch_proj;
        let c = Tanh(c_proj, Named("c"));                // "cell"

        CountAPICalls(3);
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
    let gru3Composite = StaticModel(/*isBasicBlock=*/true, [=](const Variable& dh, const Variable& projdh3, const Variable& projx3)
    {
        return gru3(dh, projdh3, projx3);
    }, L"gru3Composite");
    let doGRU = StaticModel(/*isBasicBlock=*/false, [=](const Variable& dh, const Variable& x) -> Variable
    {
        let projx3 = projectInput(x); // note: this has a bias
        let projdh3 = normR(Times(R, dh)); CountAPICalls(1);
        return gru3Composite(dh, projdh3, projx3);
    }, Named("gru"));
    return BinaryModel({ R },
    {
        { L"projectInput",  projectInput },
        //{ L"projectState",  projectState },
        { L"normR",  normR  },
    },
    // TODO: can we pass doGRU here directly, instead of creating a new lambda? Needs some form of type cast of StaticModel to this lambda.
    [=](const Variable& dh, const Variable& x) //mutable
    {
        return doGRU(dh, x);
    });
}
#endif

static TernaryModel LSTM(size_t outputDim)
{
    auto W = Parameter({ (NDShapeDimension)outputDim, NDShape::InferredDimension }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"W");
    auto R = Parameter({ outputDim, outputDim }, CurrentDataType(), GlorotUniformInitializer(), CurrentDevice(), L"R");
    auto b = Parameter({ outputDim }, CurrentDataType(), 0.0f, CurrentDevice(), L"b");
    return TernaryModel({ W, R, b }, [=](const Variable& prevH, const Variable& prevC, const Variable& input)
    {
        // TODO: complete this
        prevC;
        CountAPICalls(5);
        return ReLU(Times(W, input) + b + Times(R, prevH));
    });
}

static UnaryBroadcastingModel BatchNormalization(size_t axis, const wstring& name = wstring());

static UnaryBroadcastingModel Dense(size_t outputDim, const UnaryModel& activation, ProjectionOptions opts, const wstring& name = wstring())
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
    auto W                  = Parameter({ (NDShapeDimension)outputDim, NDShape::InferredDimension }, CurrentDataType(),  GlorotUniformInitializer(), CurrentDevice(), L"W");
    auto b                  = Parameter({                   outputDim                             }, CurrentDataType(),  0.0f,                       CurrentDevice(), L"b");
    auto scale              = Parameter({                                                         }, CurrentDataType(),  1.0,                        CurrentDevice(), L"scale");
    auto weightNormRescale  = Parameter({                   outputDim                             }, CurrentDataType(),  1.0,                        CurrentDevice(), L"weightNormRescale");
    let weightNormMinusHalf = Constant::Scalar(                                                      CurrentDataType(), -0.5,                        CurrentDevice());
    let batchNorm = hasBatchNorm ? BatchNormalization(/*axis=*/1, Named("DenseBN")) : Identity;
    let lengthNorm = hasLengthNorm ? LengthNormalization() : Identity;
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
    let normWeight = StaticModel(/*isBasicBlock=*/true , [=]() -> Variable
    {
        if (!hasWeightNorm)
            return W; // TODO: this is a dummy so that we don't reference the weightNormRescale parameter
        // pretend W had rows of length 1, by dividing by the row length after the fact
        // Note that this is generated over again, but will be computed only once since it is ready upfront.
        // BUGBUG: Does not work with sparse input, as that implies a sparse gradient, for which we cannot compute the elementwise ops.
        CountAPICalls(4);
        let rowNorm = InnerProduct(W, W, /*Axis(1)*/Axis_DropLastAxis);
        // BUGBUG: ^^ this reduction is wrong if W has more than one input axes, e.g. for image
        // TODO: need a ReduceToShape operation? Where instead of an axis, the target shape is specified?
        let invLen = Pow(rowNorm, weightNormMinusHalf);
        //if (hasBatchNorm && !hasLengthNorm) // batchNorm does element-wise rescaling, so no need to do it here as well
        //    return invLen;
        let scale1 = invLen * weightNormRescale; // invLen normalizes the weight; weightNormRescale scales it back
        return scale1;
        //y = scale1 * y;
    }, Named("dense.normWeight"));
    let doDense = StaticModel(/*isBasicBlock=*/false, [=](const Variable& x) -> Variable
    {
        auto y = x;
        CountAPICalls(1);
        y = Times(W, y);
        CountAPICalls(hasScale);
        if (hasScale) // (note: could speed this up by moving this before or after, wherever the dimension is lower)
            y = y * scale;
        if (hasWeightNorm)
            y = normWeight() * y;
        if (hasLengthNorm) // note: has no bias
            y = lengthNorm(y);
        CountAPICalls(hasBias && !hasBatchNorm);
        if (hasBatchNorm)
            y = batchNorm(y); // note: batchNorm has its own bias
        else if (hasBias)
            y = y + b;
        return activation(y);
    }, name);
    return UnaryModel(parameters, nested, [=](const Variable& x)
    {
        return doDense(x);
    });
}

static UnaryBroadcastingModel Linear(size_t outputDim, ProjectionOptions opts, const wstring& name /*= wstring()*/)
{
    return Dense(outputDim, Identity, opts, name);
}

// create a BatchNormalization layer
// TODO: the API must take an axis parameter to declare where the axis is.
static UnaryBroadcastingModel BatchNormalization(const size_t axis, const wstring& name /*= wstring()*/)
{
#ifdef DISABLE_NORMALIZATIONS
    name; axis;
    return Identity;
#else
    static size_t id = 0; // unique id
    auto thisId = ++id;   // note: don't use 'id' in lambda; it will access the static variable directly
    auto scale = Parameter({ NDShape::InferredDimension }, CurrentDataType(), 1.0, CurrentDevice(), L"scale");
    auto bias  = Parameter({ NDShape::InferredDimension }, CurrentDataType(), 0.0, CurrentDevice(), L"bias");
    auto runningMean   = Parameter({ NDShape::InferredDimension }, CurrentDataType(), 0.0, CurrentDevice(), L"runningMean");
    auto runningInvStd = Parameter({ NDShape::InferredDimension }, CurrentDataType(), 1.0, CurrentDevice(), L"runningInvStd");
    auto runningCount  = Parameter({                            }, CurrentDataType(), 0.0, CurrentDevice(), L"runningCount");
    axis;
    //vector<Variable> buffer;
    // TODO: figure out this Parameter mess for BN
    return UnaryModel({ scale, bias, runningMean, runningInvStd, runningCount }, [=](const Variable& x) -> Variable
    {
        let batchNorm = [&](const Variable& x) // apply to one sample
        {
            CountAPICalls(1);
            return CNTK::BatchNormalization(x, thisId, scale, bias, runningMean, runningInvStd, runningCount, /*spatial=*/false, 0, 0, 0.0001, name);
        };
        // BUGBUG: This cannot work once we reenable static graphs.
        //if (x.Shape().Rank() == axis) // single item
            return batchNorm(x);
        //else // a batch of items
        //    return Dynamite::Sequence::map(x, batchNorm, buffer);
    });
#endif
}

// ResNet layer
// Two Dense(ReLU) with skip connection and batch normalization after the matrix product.
static UnaryBroadcastingModel ResidualNet(size_t outputDim)
{
    // TODO: why not combine with weightNormalize?
    let project1 = Linear(outputDim, ProjectionOptions::batchNormalize | ProjectionOptions::bias, Named("project1"));
    let project2 = Linear(outputDim, ProjectionOptions::batchNormalize | ProjectionOptions::bias, Named("project2"));
    let doResidualNet = StaticModel(/*isBasicBlock=*/false, [=](const Variable& x)
    {
        CountAPICalls(3);
        let h = ReLU(project1(x)    , Named("hRes"));
        let r = ReLU(project2(h) + x, Named("rRes"));
        return r;
    }, Named("doResidualNet"));
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

// built-in Softmax requires temp memory, so we use an explicit expression instead
static Variable LogSoftmax(const Variable& z, const Axis& axis = Axis::AllStaticAxes(), const std::wstring& name = std::wstring(), const UnaryModel& barrier = Identity)
{
    //LOG(z);
    //LOG(ReduceLogSum(z, axis, L"smLogDenom"));
    CountAPICalls(2);
    let Z = barrier(ReduceLogSum(z, axis, name));
    return z - Z;
}

// built-in Softmax requires temp memory, so we use an explicit expression instead
static Variable Softmax(const Variable& z, const Axis& axis = Axis::AllStaticAxes(), const std::wstring& name = std::wstring(), const UnaryModel& barrier = Identity)
{
    //LOG(LogSoftmax(z, axis));
    CountAPICalls(1);
    return Exp(LogSoftmax(z, axis, name, barrier), name);
}

// built-in Softplus is a BlockFunction, so need to replace it here
static Variable Softplus(const Variable& z, const std::wstring& name)
{
    // TODO: This will create a Constant object every single time--better create it once. Or pre-define constant 0 and 1.
    CountAPICalls(2);
    return LogAddExp(z, Constant::Scalar(z.GetDataType(), 0.0), name);
}

// we need a special definition since the built-in one creates a BlockFunction, which costs too much each time
// BUGBUG: AllStaticAxes (=> keepDimensions=false) leads to incorrect auto-batching. Some screwup of batching axis.
//static Variable CrossEntropyWithSoftmax(const Variable& z, const Variable& label, const Axis& axis = Axis::AllStaticAxes())
static Variable CrossEntropyWithSoftmax(const Variable& z, const Variable& label, const Axis& axis = Axis(0))
{
    Variable ceLogNumer;
#if 1
    CountAPICalls(1);
    ceLogNumer = InnerProduct(label, z, axis, Named("ceLogNumer"));
#else
    if (label.IsSparse() && label.Shape().Rank() == 1)
        ceLogNumer = Times(label, z, /*outputRank=*/0, Named("ceLogNumer"));
    else
        ceLogNumer = ReduceSum(ElementTimes(label, z, Named("ceLabel")), axis, Named("ceLogNumer"));
#endif
    CountAPICalls(2);
    return Minus(ReduceLogSum(z, axis, Named("ceLogDenom")), ceLogNumer, Named("ce"));
}

static inline void as_vector(vector<Variable>& res, const Variable& x)
{
    // 'x' is an entire sequence; last dimension is length
    let len = x.size();
    res.resize(len);
    CountAPICalls(len); // x[t] is a Slice()
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
