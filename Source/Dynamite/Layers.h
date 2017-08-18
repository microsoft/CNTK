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

//#define DISABLE_NORMALIZATIONS // #define this to disable LengthNormalization and Droppo scaling

#define let const auto

using namespace CNTK;
using namespace std;

#define DTYPE DataType::Float
//#define DTYPE DataType::Double

#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced function was removed --TODO: use push/pop

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
    FunctionPtr ParametersCombined() const
    {
        auto parameters = Parameters();
        return Combine(vector<Variable>(parameters.begin(), parameters.end())); // need to cast from Parameter to Variable
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
};
typedef TModel<function<Variable(const Variable&)>> UnaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&)>> BinaryModel;
typedef TModel<function<Variable(const Variable&, const Variable&, const Variable&)>> TernaryModel;
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
        return Reshape(ReduceSum(Splice(batch, Axis(axis)), Axis(axis)), shape, L"sum");
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
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"E");
    return UnaryModel({ E }, [=](const Variable& x)
    {
        return Times(E, x);
    });
}

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
    return UnaryModel(vector<Parameter>{ scale }, [=](const Variable& x)
    {
        let mean = ReduceMean(x, axis); // it would be faster to say mean(x*x)-mu*mu, except that we need to consider rounding errors
        let x0 = x - mean;
        //LOG(x0);
        // BUGBUG: Sqrt() seems hosed!!
        let invLen = Pow(ReduceMean(x0 * x0, axis) + eps, minusHalf);
        //LOG(len);
        // Note: ^^ this parallelizes, while this vv does not
        //let len = Sqrt(TransposeTimes(x, x));
        //let res = x * (invLen /*+ eps*/) * scale;
        //LOG(scale);
        //LOG(res);
        let res = x0 * invLen * scale;
        return res;
    });
#endif
}

static BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W");
    auto R = Parameter({ outputDim, outputDim                  }, DTYPE, GlorotUniformInitializer(), device, L"R");
    auto b = Parameter({ outputDim }, DTYPE, 0.0, device, L"b");
    return BinaryModel({ W, R, b }, [=](const Variable& prevOutput, const Variable& input)
    {
        return /*Sigmoid*/ReLU(Times(W, input) + b + Times(R, prevOutput), L"RNNStep.h");
    });
}

#if 1
// TODO: change outputDim to an NDShape
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

// Frantic's GRU kernel:
int input_width = attention_size + 3 * recurrent_size;
int source_width = 3 * recurrent_size;
int aur_recurrent_width = attention_size + 2 * recurrent_size;
float u_x = aurc_input[i*input_width + attention_size + j];
float u_s = urc_source[i*source_width + j];
float u_r = aur_recurrent[i*aur_recurrent_width + attention_size + j];
float u_b = u_bias[j];
float update = LogisticFunction(u_x + u_s + u_r + u_b);

float c_x = aurc_input[i*input_width + attention_size + 2 * recurrent_size + j];
float c_s = urc_source[i*source_width + 2 * recurrent_size + j];
float c_r = c_recurrent[i*recurrent_size + j];
float c_b = c_bias[j];
float cand_value = TanhFunction(c_x + c_s + c_r + c_b);

float prev_value = prev_recurrent[i*recurrent_size + j];
output[idx] = update*cand_value + (1.0f - update)*prev_value;

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

static UnaryBroadcastingModel Dense(size_t outputDim, const UnaryModel& activation, bool bias, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W");
#ifdef DISABLE_NORMALIZATIONS
    if (bias)
    {
        auto b = Parameter({ outputDim }, DTYPE, 0.0f, device, L"b");
        return UnaryModel({ W,  b }, [=](const Variable& x) { return Times(W, x) + b; });
    }
    else
        return UnaryModel({ W, }, [=](const Variable& x) { return Times(W, x); });
#else
    auto scale = Parameter({ }, DTYPE, 1.0, device, L"Wscale");
    if (bias)
    {
        auto b = Parameter({ outputDim }, DTYPE, 0.0f, device, L"b");
        return UnaryModel({ W, scale, b }, { { L"activation", activation}  }, [=](const Variable& x) { return Times(W, x * scale) + b; });
    }
    else
        return UnaryModel({ W, scale    }, { { L"activation", activation } }, [=](const Variable& x) { return Times(W, x * scale); });
#endif
}

// by default we have a bias
static UnaryBroadcastingModel Dense(size_t outputDim, const UnaryModel& activation, const DeviceDescriptor& device)
{
    return Dense(outputDim, activation, true, device);
}

// create an identity function; makes it easy to disable stuff
static UnaryModel Identity = [](const Variable& x) { return x; };

static UnaryBroadcastingModel Linear(size_t outputDim, bool bias, const DeviceDescriptor& device)
{
    return Dense(outputDim, Identity, bias, device);
}

// by default we have a bias
static UnaryBroadcastingModel Linear(size_t outputDim, const DeviceDescriptor& device)
{
    return Linear(outputDim, true, device);
}

// ResNet layer
// Two Dense(ReLU) with skip connection
static UnaryBroadcastingModel ResidualNet(size_t outputDim, const DeviceDescriptor& device)
{
    auto W1 = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W1");
    auto W2 = Parameter({ outputDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device, L"W2");
    auto scale1 = Parameter({}, DTYPE, 1.0, device, L"Wscale1");
    auto scale2 = Parameter({}, DTYPE, 1.0, device, L"Wscale2");
    auto b1 = Parameter({ outputDim }, DTYPE, 0.0f, device, L"b1");
    auto b2 = Parameter({ outputDim }, DTYPE, 0.0f, device, L"b2");
    return UnaryModel({ W1, W2, scale1, scale2, b1, b2 }, [=](const Variable& x)
    {
        let h = ReLU(Times(W1, x * scale1) + b1);
        let r = ReLU(Times(W2, h * scale2) + b2 + x);
        return r;
    });
}

// create a Barrier function
static UnaryModel Barrier(const wstring& name = wstring())
{
    static size_t id = 0;
    auto thisId = ++id; // note: don't use 'id' in lambda; it will access the static variable directly
    return [=](const Variable& x) -> Variable
    {
        return BatchSync(x, thisId, name);
    };
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
        let barrier = Barrier(L"Recurrence");
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
        let barrier = Barrier(L"BiRecurrence");
        let splice = Sequence::Map(BinaryModel([=](const Variable& a, const Variable& b) { return Splice({ barrier(a), b }, Axis(0), L"bidi"); }));
        vector<Variable> rFwd, rBwd;
        return BinarySequenceModel({}, { { L"stepFwd", stepFwd },{ L"stepBwd", stepBwd } },
        [=](vector<Variable>& res, const vector<Variable>& inFwd, const vector<Variable>& inBwd) mutable
        {
            fwd(rFwd, inFwd);
            bwd(rBwd, inBwd);
            splice(res, rFwd, rBwd);
            rFwd.clear(); rBwd.clear(); // don't hold references
        });
    }

    static UnaryFoldingModel Fold(const BinaryModel& step, const Variable& initialState)
    {
        let barrier = Barrier(L"Fold");
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

    // TODO: This is somewhat broken presently.
    //static UnarySequenceModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
    //{
    //    let embed = Dynamite::Embedding(embeddingDim, device);
    //    return UnarySequenceModel(embed.Parameters(), {},
    //    [=](vector<Variable>& res, const vector<Variable>& x)
    //    {
    //        return Map(embed);
    //    });
    //}
};

// built-in Softmax requires temp memory, so we use an explicit expression instead
static Variable LogSoftmax(const Variable& z, const Axis& axis = Axis::AllStaticAxes())
{
    //LOG(z);
    //LOG(ReduceLogSum(z, axis, L"smLogDenom"));
    return z - ReduceLogSum(z, axis, L"smLogDenom");
}

// built-in Softmax requires temp memory, so we use an explicit expression instead
static Variable Softmax(const Variable& z, const Axis& axis = Axis::AllStaticAxes())
{
    //LOG(LogSoftmax(z, axis));
    return Exp(LogSoftmax(z, axis), L"sm");
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
    // TODO: reduce ops must be able to drop the axis
    // TODO: Dynamite should rewrite Times() that is really a dot product
    Variable ceLogNumer;
    if (label.IsSparse() && label.Shape().Rank() == 1)
        ceLogNumer = Times(label, z, /*outputRank=*/0, L"ceLogNumer");
    else
        ceLogNumer = ReduceSum(ElementTimes(label, z, L"ceLabel"), axis, L"ceLogNumer");
    let loss = Minus(ReduceLogSum(z, axis, L"ceLogDenom"), ceLogNumer, L"ce");
    //return Reshape(loss, NDShape(), L"ce");
    return loss; // Reshape(loss, NDShape());
}

static inline void as_vector(vector<Variable>& res, const Variable& x)
{
    // 'x' is an entire sequence; last dimension is length
    let len = x.Shape().Dimensions().back();
    res.resize(len);
    for (size_t t = 0; t < len; t++)
        res[t] = Index(x, (int)t, L"as_vector[" + std::to_wstring(t) + L"]");
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
