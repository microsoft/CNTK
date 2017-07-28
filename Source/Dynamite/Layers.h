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
    const vector<Parameter>& AppendParametersTo(vector<Parameter>& res) const
    {
        for (let& kv : m_parameters)
            res.push_back(kv.second);
        for (let& kv : m_parentParameters)
            kv.second->AppendParametersTo(res);
        return res;
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
        extent.pop_back(); // missing extend values default to 1 but do not generate an output axis
        data = data->SliceView(startOffset, extent, true); // slice it
        dims = data->Shape().Dimensions();
    }
    else
    {
        let newShape = NDShape(vector<size_t>(dims.begin(), dims.end() - 1));
        data = data->AsShape(newShape); // and drop the final dimension
    }
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
            auto data = sequences[s];      // NDArrayView
            //data = data->DeepClone(); // otherwise getting a dangling ref with the allPrimitiveFunctions hack
            // return in correct shape
            if (!hasAxis)
            {
                assert(data->Shape().Dimensions().back() == 1);
                data = Index(data, 0); // slice off sample axis (the last in C++)
            }
#if 1 // needed for now since PlainTextDeserializer cannot deliver Dense data, and Dynamite metric blows up on Sparse
            // convert sparse, since currently we cannot GatherBatch() sparse data
            if (data->IsSparse())
            {
                // multiply with  an identity matrix
                auto eye = Eye(data->Shape()[0], data->GetDataType(), data->Device());
                data = NDArrayView::MatrixProduct(false, eye, false, data, false, 1.0, 1);
            }
#endif
            arg[s] = Constant(data);
            //data = data->AsShape(data->Shape()); // (for debugging)
        }
    }
    return res;
}

}; // namespace
