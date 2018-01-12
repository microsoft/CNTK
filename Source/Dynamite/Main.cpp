//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// This implements SequenceClassification.py as an example in CNTK Dynamite.
// Both CNTK Static and CNTK Dynamite are run in parallel to show that both produce the same result.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "Layers.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <deque>

#define let const auto

using namespace CNTK;
using namespace std;

using namespace Dynamite;

#if 0 // these no longer compile, but if needed, can be resurrected with little effort
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

// baseline model for CNTK Static
UnaryModel CreateModelFunction(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim)
{
    return StaticSequential({
        Embedding(embeddingDim),
        StaticSequence::Fold(RNNStep(hiddenDim)),
        Linear(numOutputClasses, ProjectionOptions::stabilize | ProjectionOptions::bias)
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

// CNTK Dynamite model
UnaryModel CreateModelFunctionUnrolled(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim)
{
    auto embed   = Embedding(embeddingDim);
    auto step    = RNNStep(hiddenDim);
    auto zero = Constant({ hiddenDim }, 0.0f, CurrentDevice());
    auto fold = Dynamite::Sequence::Fold(step, zero);
    auto linear  = Linear(numOutputClasses, ProjectionOptions::stabilize | ProjectionOptions::bias);
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
        let ce = Dynamite::CrossEntropyWithSoftmax(z, label);
        //LOG(ce);
        return ce;
    };
    // create a batch mapper (which will eventually allow suspension)
    let batchCriterion = Batch::Map(criterion);
    // for final summation, we create a new lambda (featBatch, labelBatch) -> mbLoss
    vector<Variable> losses;
    return [=](const vector<Variable>& features, const vector<Variable>& labels) mutable
    {
        batchCriterion(losses, features, labels);         // batch-compute the criterion
        let collatedLosses = Splice(losses, Axis(0));     // collate all seq losses
        //LOG(collatedLosses);
        let mbLoss = ReduceSum(collatedLosses, Axis(0));  // aggregate over entire minibatch
        //LOG(mbLoss);
        losses.clear();
        return mbLoss;
    };
}

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
    SetCurrentDevice(device);
    const size_t inputDim         = 2000; // TODO: it's only 1000??
    const size_t embeddingDim     = 500;
    const size_t hiddenDim        = 250;
    const size_t attentionDim     = 20;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    // static model and criterion function
    auto model_fn = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim);
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // dynamic model and criterion function
    auto d_model_fn = CreateModelFunctionUnrolled(numOutputClasses, embeddingDim, hiddenDim);
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
    auto features = InputVariable({ inputDim },         true/*false*/ /*isSparse*/, CurrentDataType(), featuresName);
    auto labels   = InputVariable({ numOutputClasses }, false/*useSparseLabels*/,   CurrentDataType(), labelsName, { Axis::DefaultBatchAxis() });

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
    auto d_learner = SGDLearner(d_parameters, TrainingParameterPerSampleSchedule(0.05));

    auto learner = SGDLearner(FunctionPtr(loss)->Parameters(), TrainingParameterPerSampleSchedule(0.05));
    auto trainer = CreateTrainer(nullptr, loss, loss/*metric*/, { learner });

    // force synchronized GPU operation so we can profile more meaningfully
    // better do this externally: set CUDA_LAUNCH_BLOCKING=1
    //DeviceDescriptor::EnableSynchronousGPUKernelExecution();
    fprintf(stderr, "CUDA_LAUNCH_BLOCKING=%s\n", getenv("CUDA_LAUNCH_BLOCKING"));

    // force-ininitialize the GPU system
    // This is a hack for profiling only.
    // Without this, an expensive CUDA call will show up as part of the GatherBatch kernel
    // and distort the measurement.
    auto t1 = make_shared<NDArrayView>(CurrentDataType(), NDShape({  1,42 }), CurrentDevice());
    auto t2 = make_shared<NDArrayView>(CurrentDataType(), NDShape({ 13,42 }), CurrentDevice());
    auto t3 = make_shared<NDArrayView>(CurrentDataType(), NDShape({ 13,42 }), CurrentDevice());
    t3->NumericOperation({ t1, t2 }, 1.0, 26/*PrimitiveOpType::Plus*/);

    const size_t minibatchSize = 200;  // use 10 for ~3 sequences/batch
    for (size_t repeats = 0; true; repeats++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, CurrentDevice());
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
            FromCNTKMB(args, { minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, { true, false }, DataType::Float, CurrentDevice());
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
        let mbSizeForUpdate = minibatchData[labelStreamInfo].numberOfSamples;
        double loss1;
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "/// ### CNTK Dynamite:  %.6f sec\n");
#if 1       // model update with Dynamite
            mbLoss.Backward(gradients);
            d_learner->Update(gradients, mbSizeForUpdate);
#endif
            loss1 = mbLoss.Value()->AsScalar<float>(); // note: this does the GPU sync
        }
        fprintf(stderr, "Dynamite:    CrossEntropy loss = %.7f\n", loss1 / mbSizeForUpdate);
#endif
#if 1   // static CNTK
        double crit;// = trainer->PreviousMinibatchLossAverage();
        {
            Microsoft::MSR::CNTK::ScopeTimer timer(3, "\\\\\\ ### CNTK Static:    %.6f sec\n");
            trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, CurrentDevice());
            crit = trainer->PreviousMinibatchLossAverage(); // note: this does the GPU sync
        }
        PrintTrainingProgress(trainer, repeats, /*outputFrequencyInMinibatches=*/ 1);
#endif
    }
}
#endif

extern int mt_main(int argc, char *argv[]);
extern void RunDynamiteTests();

#if 0
template<typename F, typename FReturnType>
class wrapper
{
    const F& f;
    const size_t N;
    struct iterator : public std::iterator<std::input_iterator_tag, FReturnType>
    {
        iterator(const F& f, size_t index) : f(f), index(index) {}
        const F& f;
        size_t index;
        FReturnType operator *() const { return f(index); }
        size_t operator-(const iterator& other) const { return other.index - index; }
        iterator operator++() { return iterator(F, index+1); }
    };
public:
    iterator begin() const { return iterator{ f, 0 }; }
    iterator end() const { return iterator{ f, N }; }
    wrapper(size_t N, const F& f) : N(N), f(f) {}
};
template<typename F>
auto create_wrapper(size_t N, const F& f)
{
    return wrapper<F, decltype(f(0))>(N, f);
}
#endif

#if 0
template<typename T, size_t N>
class FixedVectorWithBuffer : public VectorSpan<T>
{
    union // using a union will prevent automatic construction/destruction
    {
        //char buffer[N * sizeof(T)];
        T items[N];
    } u;
    void ConstructBuffer(size_t len, size_t elemSize = sizeof(T))
    {
        if (len >= N) // too large: use dynamic allocation
            bufp = reinterpret_cast<T*>(new char[elemSize * len]); // TODO: How about alignment?
        else // fits: use the built-in buffer
            bufp = &u.items[0];
        endp = bufp; // we will increment it to the correct length during construction
    }
    void ConstructAppendElement(const T& item) { new (endp)(item); endp++; } // we do it this way so that the object is always in proper state
    void ConstructAppendElement(T&& item) { new (endp)(move(item)); endp++; }
    void DestructBuffer()
    {
        if (bufp != &u.items[0]) // we used a dynamically allocated buffer
            delete[] reinterpret_cast<char*>(bufp);
        // we leave bufp and endp dangling, since this is part of destruction
    }
    char* BufAddress() const { return reinterpret_cast<char*>(bufp); }
    char* EndAddress() const { return reinterpret_cast<char*>(endp); }
public:
    // dummy constructor
    FixedVectorWithBuffer() { ConstructBuffer(0); }
    // construct from items
    // TODO: How to avoid conflicting with the constructors below?
    FixedVectorWithBuffer(const T& item) { ConstructBuffer(1); ConstructAppendElement(item); }
    FixedVectorWithBuffer(const T& item, const T& item2) { ConstructBuffer(1); ConstructAppendElement(item); ConstructAppendElement(item2); }
    FixedVectorWithBuffer(T&& item) { ConstructBuffer(1); ConstructAppendElement(move(item)); }
    FixedVectorWithBuffer(T&& item, const T&& item2) { ConstructBuffer(1); ConstructAppendElement(move(item)); ConstructAppendElement(move(item2)); }
    // from iterator pair
    template<typename IteratorType>
    FixedVectorWithBuffer(const IteratorType& beginIter, const IteratorType& endIter)
    {
        ConstructBuffer(endIter - beginIter);
        for (auto iter = beginIter; iter != endIter; ++iter)
            ConstructAppendElement(*iter); // this will increment endp. In case of an exception, endp is valid so that we can destruct.
    }
    // move constructor
    FixedVectorWithBuffer(FixedVectorWithBuffer&& other)
    {
        ConstructBuffer(other.EndAddress() - other.BufAddress(), /*elemSize=*/1); // this construct avoids the division by sizeof(T)
        auto* p = begin();
        for (auto iter = other.begin(); iter != other.end(); ++iter)
        {
            new (p++)(move(*iter));
            *iter.~T(); // destruct other. Hopefully the compiler will understand it is already dead due to move()
        }
        other.DestructBuffer();
        other.bufp = other.endp = &other.u.items[0]; // bring it into defined state. Hopefully the compiler will short-circuit a potential subsequent destructor call
    }
    // from any container
    template<typename ContainerType>
    FixedVectorWithBuffer(const ContainerType& other) : FixedVectorWithBuffer(other.begin(), other.end()) { }
    ~FixedVectorWithBuffer()
    {
        // note: If construction failed half-way, then endp is valid w.r.t. the last successfully placed item
        for (auto iter = begin(); iter != end(); ++iter)
            *iter.~T();
        DestructBuffer();
    }
};

class C : public CNTK::enable_strong_shared_ptr<C>
{
    std::string test;
public:
    C() : test("hello world")
    {
    }
    ~C()
    {
        fprintf(stderr, "");
    }
};

//template<> FixedSizePoolStorage<sizeof FixedSizePoolItem<C>>                                    strong_shared_ptr<C                                   >::Storage::s_storage;
//template<> FixedSizePoolStorage<sizeof (FixedSizePoolItem<OptionalString::SharableString const>)> strong_shared_ptr<OptionalString::SharableString const>::Storage::s_storage;
#endif

int main(int argc, char *argv[])
{
#if 0
    {
        MakeSharedObject1<C>();
        CNTK::OptionalString pp(L"test");
        CNTK::OptionalString ss(pp);
        CNTK::OptionalString qq(move(pp));
        pp;
    }
    using namespace CNTK;
    {
        string a("test");
        string b("test2");
        string c("test3");
        vector<string> vv{ a, b, c };
        FixedVectorWithBuffer<string, 2> xx(move(vv));
        for (let e : xx)
            fprintf(stderr, "%s\n", e.c_str());
    }
    {
        string a("test");
        string b("test2");
        auto b2 = MakeTwoElementVector(a, b);
        auto b1 = MakeOneElementVector(a);
#if 0   // NumericRangeSpan does not compile under nvcc
        vector<string> ab = Transform(NumericRangeSpan<size_t, IntConstant<0>, IntConstant<2>>(), [&](size_t _) -> const string&{ return _ ? b : a; });
        ab.push_back("test3");
        ab.push_back("test4");
        ////vector<int> x = X::NumericRangeSpan<int>(13, 42);
        ////x[5] = 12;
        let& abs = MakeSpan(ab, 1, (size_t)3);
        for (let& e : abs)
            fprintf(stderr, "%s\n", e.c_str());
        let s = Transform(NumericRangeSpan<int, IntConstant<13>, IntConstant<42>>(), [](int _) { return _ + 1; }, [](int _) { return _ + 1; });
        for (let e : s)
            fprintf(stderr, "%d\n", (int)e);
#endif
        struct C { string s; };
        vector<C> c{ { "test" }, C{ "test2" } };
        let cs = MakeSet(Transform(c,
                                   [](const C& _) -> const string&{ return _.s; },
                                   [](const string& _) { return _ + "!"; }));
        fprintf(stderr, "");
        //vector<string> cs(w.begin(), w.end());
        //FixedVectorWithBuffer<string, 2> vec1("s1"s);
        //FixedVectorWithBuffer<string, 2> vec2("s1"s, "s2"s);
        //vector<string> sv(vec2);
    }
    struct T { shared_ptr<int> x, y, z; };
    let e1 = T{ make_shared<int>(13), make_shared<int>(13), make_shared<int>(42) };
    let e2 = T{ make_shared<int>(42), make_shared<int>(13), make_shared<int>(42) };
    auto w = create_wrapper((size_t)2, [&](size_t i) -> const T& { if (i == 0) return e1; else return e2; });
    let b = w.begin();
    let e = w.end();
    vector<T> vec(b, e);
#endif
    try
    {
        //extern int iris_main();
        //iris_main();
        RunDynamiteTests();
#if 1
        return mt_main(argc, argv);
#else
        argc; argv;
        TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
        //TrainSequenceClassifier(DeviceDescriptor::CPUDevice(), true);
        // BUGBUG: CPU currently outputs loss=0??
#endif
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what()), fflush(stderr);
    }
}
