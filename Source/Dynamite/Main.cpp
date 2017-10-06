//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// This implements SequenceClassification.py as an example in CNTK Dynamite.
// Both CNTK Static and CNTK Dynamite are run in parallel to show that both produce the same result.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "PlainTextDeseralizer.h"
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
    return StaticSequential({
        Embedding(embeddingDim, device),
        StaticSequence::Fold(RNNStep(hiddenDim, device)),
        Linear(numOutputClasses, ProjectionOptions::stabilize | ProjectionOptions::bias, device)
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
    auto zero = Constant({ hiddenDim }, 0.0f, device);
    auto fold = Dynamite::Sequence::Fold(step, zero);
    auto linear  = Linear(numOutputClasses, ProjectionOptions::stabilize | ProjectionOptions::bias, device);
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
    auto Wenc = Parameter({ attentionDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device);
    auto Wdec = Parameter({ attentionDim, NDShape::InferredDimension }, DTYPE, GlorotUniformInitializer(), device);
    auto v    = Parameter({ attentionDim }, DTYPE, GlorotUniformInitializer(), device);
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
    const size_t inputDim         = 2000; // TODO: it's only 1000??
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
    auto features = InputVariable({ inputDim },         true/*false*/ /*isSparse*/, DTYPE, featuresName);
    auto labels   = InputVariable({ numOutputClasses }, false/*useSparseLabels*/,   DTYPE, labelsName, { Axis::DefaultBatchAxis() });

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
    auto t1 = make_shared<NDArrayView>(DTYPE, NDShape({  1,42 }), device);
    auto t2 = make_shared<NDArrayView>(DTYPE, NDShape({ 13,42 }), device);
    auto t3 = make_shared<NDArrayView>(DTYPE, NDShape({ 13,42 }), device);
    t3->NumericOperation({ t1, t2 }, 1.0, 26/*PrimitiveOpType::Plus*/);

    const size_t minibatchSize = 200;  // use 10 for ~3 sequences/batch
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
            FromCNTKMB(args, { minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, { true, false }, DataType::Float, device);
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
            trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
            crit = trainer->PreviousMinibatchLossAverage(); // note: this does the GPU sync
        }
        PrintTrainingProgress(trainer, repeats, /*outputFrequencyInMinibatches=*/ 1);
#endif
    }
}

extern int mt_main(int argc, char *argv[]);
extern void RunDynamiteTests();

// a pool for allocating objects of one specific size
template<typename T>
struct FixedSizePoolItem
{
    T item;
    unsigned short blockIndex; // should not require more! Not sure if this will help with some small structs?
};
template<size_t itemByteSize>
class FixedSizePool
{
    static void Assert(bool cond) { if (!cond) LogicError("FixedSizePool: An assertion failed."); }

    // class to store objects of size itemByteSize in lists of char arrays
    template<size_t itemByteSize>
    class Storage
    {
        struct Block
        {
            static const size_t capacity = 3;// 65536; // we reserve this many at a time
            vector<char> bytes = vector<char>(capacity * itemByteSize); // [byte offset]
            vector<bool> used  = vector<bool>(capacity, false);         // [item index]  true if this entry is used
            template<typename T>
            T* TryAllocate(size_t& nextItemIndex)
            {
                Assert(nextItemIndex <= capacity);
                while (nextItemIndex < capacity)
                {
                    if (!used[nextItemIndex])
                    {
                        T* p = (T*)(bytes.data() + nextItemIndex * itemByteSize);
                        used[nextItemIndex] = true; // and mark as allocated
                        nextItemIndex++;
                        return p;
                    }
                    nextItemIndex++;
                }
                return nullptr; // this block is full
            }
            template<typename T>
            void Deallocate(T* p)
            {
                size_t itemIndex = (((char*)p) - bytes.data()) / itemByteSize;
                Assert(itemIndex < capacity);
                Assert(p == (T*)(bytes.data() + itemIndex * itemByteSize));
                Assert(used[itemIndex]);
                used[itemIndex] = false;
            }
        };
        // state of allocation
        vector<shared_ptr<Block>> blocks; // all blocks we have currently allocated
        size_t totalItemsAllocated = 0;   // we have presently this many live objects
        size_t totalItemsReserved = 0;    // we are holding memory sufficient to hold this many
        // state of scan
        size_t currentBlockIndex;         // we are allocating from this block
        size_t nextItemIndex;             // index of next item. If at end of block, this is equal to blockCapacity
        void ResetScan()
        {
            currentBlockIndex = 0;
            nextItemIndex = 0;
        }
    public:
        Storage()
        {
            ResetScan();
            fprintf(stderr, "Scan reset for storage of elements of %d bytes\n", (int)itemByteSize);
        }
        template<typename T>
        T* Allocate()
        {
            if (sizeof(FixedSizePoolItem<T>) != itemByteSize)
                LogicError("FixedSizePoolAllocator: Called for an object of the wrong size.");
            Assert(totalItemsReserved >= totalItemsAllocated);
            fprintf(stderr, "allocate<%s>()  --> %d bytes (%d incl. index)\n", typeid(T).name(), (int)sizeof T, (int)itemByteSize);
            // find next free location
            for (;;)
            {
                // all blocks are full: either reset the scan or grow
                if (currentBlockIndex == blocks.size())
                {
                    if (totalItemsReserved > totalItemsAllocated * 2)
                    {
                        // if we have 50% utilization or below, we start over the scan in our existing allocated space
                        // At 50%, on av. we need to scan 1 extra item to find a free one.
                        ResetScan();
                    }
                    else
                    {
                        // too few free items, we'd scan lots of items to find one: instead use a fresh block
                        if ((short)(currentBlockIndex + 1) != currentBlockIndex + 1)
                            LogicError("FixedSizePoolAllocator: Too many blocks.");
                        blocks.push_back(make_shared<Block>());
                        totalItemsReserved += Block::capacity;
                        // enter the newly created block
                        nextItemIndex = 0;
                    }
                }
                // try to allocate in current block
                T* p = blocks[currentBlockIndex]->TryAllocate<T>(nextItemIndex);
                if (p) // found one in the current block
                {
                    totalItemsAllocated++; // account for it
                    auto* pItem = reinterpret_cast<FixedSizePoolItem<T>*>(p);
                    pItem->blockIndex = currentBlockIndex; // remember which block it came from
                    Assert(pItem->blockIndex == currentBlockIndex); // (overflow)
                    return p;
                }
                // current block is full: advance the scan to the next block
                currentBlockIndex++;
                nextItemIndex = 0;
            }
            LogicError("FixedSizePoolAllocator: Allocation in newly created block unexpectedly failed.");
        }
        template<typename T>
        void Deallocate(T* p)
        {
            fprintf(stderr, "deallocate<%s>()  --> %d bytes (%d incl. index)\n", typeid(T).name(), (int)sizeof T, (int)itemByteSize);
            let* pItem = reinterpret_cast<FixedSizePoolItem<T>*>(p);
            Assert(pItem->blockIndex < blocks.size());
            blocks[pItem->blockIndex]->Deallocate(p);
            Assert(totalItemsAllocated > 0);
            totalItemsAllocated--;
        }
    };
public:
    // say FixedSizePool::get() to get access to a globally shared instance for all pools of the same itemByteSize
    static Storage<itemByteSize>& get() { static Storage<itemByteSize> storage; return storage; }
};

// a C++ allocator that allocates objects of type <T> in FixedSizePool Storage objects shared across all types of the same size
template<typename T>
class FixedSizePoolAllocatorT
{
public: // required stuff
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    template<typename U> struct rebind { typedef FixedSizePoolAllocatorT<U> other; };
    inline pointer address(reference r) { return &r; }
    inline const_pointer address(const_reference r) { return &r; }

public:
    inline explicit FixedSizePoolAllocatorT() {}
    inline ~FixedSizePoolAllocatorT() {}
    inline explicit FixedSizePoolAllocatorT(FixedSizePoolAllocatorT const&) {}
    template<typename U>
    inline explicit FixedSizePoolAllocatorT(FixedSizePoolAllocatorT<U> const&) {}

    inline pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0)
    {
        if (cnt != 1)
            InvalidArgument("FixedSizePoolAllocatorT: Must only be called with a fixed size.");
        auto& storage = FixedSizePool<sizeof(FixedSizePoolItem<T>)>::get();
        return reinterpret_cast<pointer>(storage.Allocate<T>());
    }
    inline void deallocate(pointer p, size_type cnt)
    {
        auto& storage = FixedSizePool<sizeof(FixedSizePoolItem<T>)>::get();
        storage.Deallocate<T>(p);
    }

    inline size_type max_size() const { return std::numeric_limits<size_type>::max() / sizeof(T); }

    inline void construct(pointer p, const T& t) { new(p) T(t); }
    inline void destroy(pointer p) { p->~T(); }

    inline bool operator==(FixedSizePoolAllocatorT const&) { return true; }
    inline bool operator!=(FixedSizePoolAllocatorT const& a) { return !operator==(a); }

public:
    static void Test()
    {
        typedef FixedSizePoolAllocatorT<char> AllocatorUnderTest;
        AllocatorUnderTest alloc;
        vector<int, AllocatorUnderTest> x(1, 1);
        list<int, AllocatorUnderTest> v1;
        v1.push_back(13);
        v1.push_back(42);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.erase(v1.begin()++++);
        v1.erase(v1.begin()++++);
        v1.erase(v1.begin()++++);
        v1.erase(v1.begin()++++);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        v1.push_back(13);
        list<int> v2(v1.begin(), v1.end());
        auto ps = allocate_shared<string>(alloc, "test");
        ps.reset();
        let pi = allocate_shared<int>(alloc, 1968);
    }
};
typedef FixedSizePoolAllocatorT<char> FixedSizePoolAllocator; // turns out, the actual template argument does not matter here, it is never used this way

int main(int argc, char *argv[])
{
    argc; argv;
    {
        FixedSizePoolAllocator::Test();
    }
    exit(0);
    try
    {
        RunDynamiteTests();
#if 1
        return mt_main(argc, argv);
#else
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
