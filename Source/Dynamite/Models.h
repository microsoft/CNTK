//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// experimental/prototypical layers lib in C++

#pragma once

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "API/CNTKLibrary.h"

#include <functional>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <fstream>
#include <stdio.h>

#ifndef let
#define let const auto
#endif

#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced function was removed

#ifndef _MSC_VER // some useful stuff missing on gcc
#include <unistd.h>
#define _dup2 dup2
#define _fileno fileno
static inline std::string utf8path(const wchar_t* s) { return string(s, s + wcslen(s)); }
static inline int _wunlink(const wchar_t* s) { return remove(utf8path(s).c_str()); }
static inline int  _wrename(const wchar_t* a, const wchar_t * b) { return rename(utf8path(a).c_str(), utf8path(b).c_str()); }
static inline FILE* _wfopen(const wchar_t* a, const wchar_t * b) { return fopen(utf8path(a).c_str(), utf8path(b).c_str()); }
static inline FILE* _wpopen(const wchar_t* a, const wchar_t * b) { return popen(utf8path(a).c_str(), utf8path(b).c_str()); }
#define _wifstream(s) ifstream(utf8path((s.c_str())))
#define _wofstream(s) ofstream(utf8path((s.c_str())))
#else
#include <io.h>
#define _wifstream ifstream
#define _wofstream ofstream
#endif

// helper to count API calls
// Call with 0 to get the current count.
#ifdef _MSC_VER // find out how to do this on gcc
__declspec(selectany) // fails on gcc
#else
static
#endif
size_t g_numAPICallsSoFar = 0;
static inline size_t CountAPICalls(size_t n = 1)
{
    g_numAPICallsSoFar += n;
    return g_numAPICallsSoFar;
}

namespace Dynamite {

using namespace CNTK;
using namespace std;

// globally set options
// Use these to set the DataType and Device to be used for any call onwards.
struct LayersOptions
{
    DataType         dataType = DataType::Float;
    DeviceDescriptor device = DeviceDescriptor::UseDefaultDevice();
};

static LayersOptions& CurrentOptions()
{
    static LayersOptions s_currentOptions;
    return s_currentOptions;
};

static inline DeviceDescriptor CurrentDevice()                                  { return CurrentOptions().device; }
static inline void             SetCurrentDevice(const DeviceDescriptor& device)
{
    let succ = DeviceDescriptor::TrySetDefaultDevice(device);
    if (!succ)
        RuntimeError("Failed to set the default device.");
    CurrentOptions().device = device;
}
static inline DataType         CurrentDataType()                                { return CurrentOptions().dataType; }
static inline void             SetCurrentDataType(DataType dataType)            { CurrentOptions().dataType = dataType; }

// debugging helper
static inline NDArrayViewPtr GetValueAsTensor(const Variable& var) { return var.Value(); }
static inline NDArrayViewPtr GetValueAsTensor(const FunctionPtr & fun) { return fun->Output().Value(); }
static inline NDArrayViewPtr GetValueAsTensor(const vector<Variable>& vec) { return (Splice(vec, Axis((int)vec[0].Shape().Rank())))->Output().Value(); }
#define LOG_VAR(var) (GetValueAsTensor(var)->LogToFile(L ## #var, stderr, 10)) // helper to log a value

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

// VariableTuple<N>
template<size_t N> struct VariableTupleType;
template<> struct VariableTupleType<2> { typedef tuple<Variable, Variable> type; };
template<> struct VariableTupleType<3> { typedef tuple<Variable, Variable, Variable> type; };
template<> struct VariableTupleType<4> { typedef tuple<Variable, Variable, Variable, Variable> type; };
template<size_t N>
using VariableTuple = typename VariableTupleType<N>::type;

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
                if (!m_nestedParameters.insert(kv).second)
                    LogicError("parameters must be uniquely named");
        for (const auto& p : parameters)
            if (p.Name().empty())
                LogicError("parameters must be named");
            else
                if (!m_parameters.insert(make_pair(p.Name(), p)).second)
                    LogicError("parameters must be uniquely named");
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
    // TODO: merge the two
    const ModelParametersPtr& NestedPtr(const wstring& name) const
    {
        auto iter = m_nestedParameters.find(name);
        if (iter == m_nestedParameters.end())
            LogicError("no such captured model: %ls", name.c_str());
        return iter->second;
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
            kv.second->LogParameters(prefix + kv.first + L".");
        for (let& kv : m_parameters) // log parameters defined right here
        {
            // operator>>() generates prefixes like ">.>.>.>", which we replace by "[4]" in this case
            auto scope = prefix;
            for (auto spos = scope.find_first_of(L"<>"); spos != wstring::npos; spos = scope.find_first_of(L"<>"))
            {
                auto epos = spos;
                while (epos + 2 < scope.size() && scope[epos + 1] == L'.' && (scope[epos + 2] == L'<' || scope[epos + 2] == L'>'))
                    epos += 2;
                scope.replace(spos, epos - spos + 1, L"[" + to_wstring((epos - spos) / 2 - 1) + L"]");
            }
            let name = scope + kv.first;
            fprintf(stderr, " %-50S : %S\n", name.c_str(), kv.second.AsString().c_str());
            // for debugging, implant the full name. This way, the full name will show up in AutoBatch log output.
            const_cast<Parameter&>(kv.second).DebugUpdateName(name);
        }
        fflush(stderr);
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
    template<typename F>
    TModel(const F& f) : Base(f) {}
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
    const Parameter& operator[](const wstring& name) const { return (*get())[name]; } // TODO: This may not have a test currently.
    const ModelParameters& Nested(const wstring& name) const { return get()->Nested(name); }
    const ModelParametersPtr& NestedPtr(const wstring& name) const { return get()->NestedPtr(name); }
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

// helper to save and restore
static inline
void SaveCheckpoint(const wstring& path, const FunctionPtr& compositeFunction, size_t totalNumTargetsSeen,
                    size_t numWorkers, const MinibatchSourcePtr& minibatchSource, const DistributedLearnerPtr& learner)
{
    // reader state
    Dictionary externalState(/*s_trainingMinibatchSource=*/L"TrainingMinibatchSource", minibatchSource->GetCheckpointState());

    // learner state
    std::vector<DictionaryValue> learnersState{ learner->CreateCheckpoint() };

    // collect state from all workers if running parallel
    DistributedCommunicatorPtr checkpointCommunicator;
    if (numWorkers > 1    ||true)
        checkpointCommunicator = MPICommunicator();

    Dictionary aggregatedState;
    if (checkpointCommunicator)
    {
        const std::wstring internalWorkerStateKey = L"internal_worker_state"; // these are from Serialization.h
        const std::wstring externalWorkerStateKey = L"external_worker_state";
        Dictionary localState(internalWorkerStateKey, Dictionary(), externalWorkerStateKey, externalState);

        // Collect distributed external localState.
        checkpointCommunicator->Barrier();

        std::vector<DictionaryPtr> remoteState;
        checkpointCommunicator->Gather(localState, remoteState, checkpointCommunicator->Workers());

        for (const auto& w : checkpointCommunicator->Workers())
            aggregatedState[std::to_wstring(w.m_globalRank)] = *remoteState[w.m_globalRank];
    }

    if (checkpointCommunicator->CurrentWorker().IsMain())
    {
        // TODO: reuse from Trainer.cpp:
        const std::wstring versionPropertyName = L"Version";
        const std::wstring totalNumTargetsSeenName = L"totalNumTargetsSeen";
        const std::wstring learnersPropertyName = L"Learners";
        const std::wstring externalStatePropertyName = L"ExternalState";
        const std::wstring distributedStatePropertyName = L"DistributedState";
        static const size_t trainerCheckpointVersion = 1;

        Dictionary state(
            versionPropertyName         , trainerCheckpointVersion,
            totalNumTargetsSeenName     , totalNumTargetsSeen,
            externalStatePropertyName   , externalState,
            learnersPropertyName        , learnersState,
            distributedStatePropertyName, aggregatedState);

        // TODO: rename must check return code. To fix this, just move this down to the API and use the functions in Trainer.cpp.
        std::wstring tempModelPath = path + L".tmp";
        std::wstring trainerStateCheckpointFilePath = path + L".ckp";
        std::wstring tempCheckpointPath = trainerStateCheckpointFilePath + L".tmp";
        compositeFunction->Save(tempModelPath);
        state.Save(tempCheckpointPath);
        _wunlink(path.c_str());
        _wunlink(trainerStateCheckpointFilePath.c_str());
        _wrename(tempModelPath.c_str(), path.c_str());
        _wrename(tempCheckpointPath.c_str(), trainerStateCheckpointFilePath.c_str());
    }

    if (checkpointCommunicator)
        // all workers need to sync up after saving model to avoid read-after-write hazard
        // i.e. one worker is in the middle of write while another tries to read
        checkpointCommunicator->Barrier();
        // Note: checkpointCommunicator is destructed at end of this block

}

static inline
void RestoreFromCheckpoint(const wstring& path, const FunctionPtr& compositeFunction, size_t& totalNumTargetsSeen,
                          size_t numWorkers, const MinibatchSourcePtr& minibatchSource, const DistributedLearnerPtr& learner)
{
    // restarting after crash. Note: not checkpointing the reader yet.
    // TODO: The code below is copy-paste from Trainer.cpp, since it is not public. Move this down through the API.
    // TODO: Can we just wrap everything in an actual Trainer instance, and use that?
    //model_fn.RestoreParameters(path);
    std::wstring trainerStateCheckpointFilePath = path + L".ckp";

    // Restore the model's parameters
    compositeFunction->Restore(path);

    // restore remaining state
    Dictionary checkpoint = Dictionary::Load(trainerStateCheckpointFilePath);

    const std::wstring internalWorkerStateKey = L"internal_worker_state"; // these are from Serialization.h
    const std::wstring externalWorkerStateKey = L"external_worker_state";

    // TODO: reuse from Trainer.cpp:
    const std::wstring versionPropertyName = L"Version";
    const std::wstring totalNumTargetsSeenName = L"totalNumTargetsSeen";
    const std::wstring learnersPropertyName = L"Learners";
    const std::wstring externalStatePropertyName = L"ExternalState";
    const std::wstring distributedStatePropertyName = L"DistributedState";

    totalNumTargetsSeen = checkpoint[totalNumTargetsSeenName].Value<size_t>();
    auto learnerState = checkpoint[learnersPropertyName].Value<std::vector<DictionaryValue>>().front().Value<Dictionary>();
    auto externalState = checkpoint[externalStatePropertyName].Value<Dictionary>();

    learner->RestoreFromCheckpoint(learnerState);

    if (numWorkers > 1 || true)
    {
        // this ensures that nobody will start writing to the model/checkpoint files, until
        // everybody is done reading them.
        DistributedCommunicatorPtr checkpointCommunicator = MPICommunicator();
        checkpointCommunicator->Barrier();

        auto mainWorkerId = std::to_wstring(0);
        auto localWorkerId = std::to_wstring(checkpointCommunicator->CurrentWorker().m_globalRank);

        Dictionary distributedState = checkpoint[distributedStatePropertyName].Value<Dictionary>();

        if (!checkpointCommunicator->CurrentWorker().IsMain() && distributedState.Contains(localWorkerId))
        {
            // the checkpoint contains internal state for this worker.
            Dictionary localState = distributedState[localWorkerId].Value<Dictionary>();
            externalState = localState[externalWorkerStateKey].Value<Dictionary>();
        }
    }

    minibatchSource->RestoreFromCheckpoint(externalState[/*s_trainingMinibatchSource=*/L"TrainingMinibatchSource"].Value<Dictionary>());
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

    static int& CurrentMapIndex()
    {
        static int i = -1;
        return i;
    }

    // for binary functions
    static BinarySequenceModel Map(BinaryModel f)
    {
        return BinarySequenceModel({}, { { L"f", f } },
            [=](vector<Variable>& res, const vector<Variable>& x, const vector<Variable>& y)
        {
            assert(y.size() == x.size());
            res.resize(x.size());
            int& i = CurrentMapIndex();
            for (i = 0; i < x.size(); i++)
                res[i] = f(x[i], y[i]);
            i = -1;
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

// function composition
// TODO: Do we need other overloads as well? SequenceModel, and going back and forth?
static inline UnaryModel operator>> (const UnaryModel& before, const UnaryModel& after)
{
    // we use a special name here that is recognized by LogParameters()
    return UnaryModel({}, { { L"<", before },{ L">", after } }, [=](const Variable& x) -> Variable
    {
        return after(before(x));
    });
}

#if 0
// helper to assign the columns of a tensor to a std::vector of column tensors
static inline void as_vector(vector<Variable>& res, const Variable& x)
{
    // 'x' is an entire sequence; last dimension is length
    let len = x.size();
    res.resize(len);
    CountAPICalls(len); // x[t] is a Slice()
    for (size_t t = 0; t < len; t++)
        res[t] = x[t];
}
#endif

}; // namespace

#pragma warning(pop)
