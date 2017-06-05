//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// The functions for automatically-batched evaluation of dynamic graphs (forward and backward) is contained here.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "PrimitiveFunction.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"
#include "PrimitiveFunction.h"
#include "CommonMatrix.h"
#include "Utils.h"

#include <unordered_map>
#include <vector>
#include <string>

#undef LOG_DETAILS   // if defined, log all forward and backward operations
#define LOG_STATS     // if defined, log statistics (#operations)
#undef NO_BATCHED_BACKPROP // if defined, don't do batched backprop

using namespace Microsoft::MSR::CNTK;
using namespace std;

#define BarrierOp NoOp // for now, we use Alias() (=NoOp) to denote a Barrier(). Should become an op in its own right.

#pragma warning (disable: 4456) // until I fixed the shadowing

#define let const auto
#define fail_if(cond, err) (!!(cond) ? (LogicError(__FUNCTION__ ": " err),0) : 0)
#define BreakPoint fprintf(stderr, "") // use this inside a conditional to be able to set a breakpoint in Release code

namespace CNTK
{
    // how graphs work in CNTK V2:
    //  - nodes := PrimitiveFunctions (incl. BlockFunction)
    //  - edges := Variables
    //  - net := CompositeFunction::m_allPrimitiveFunctions; duplicated for all refs to composites
    //  - output node: a node with additional ref to a net, created by calling Output() on a CompositeFunction
    // ownership:
    //  - nodes own edges: Functions hold shared_ptrs to m_inputs[] and m_outputs[]
    //  - edges do NOT own nodes
    //  - net owns full set of nodes
    //  - output node has a strong ref m_outputComposite to the CompositeFunction.
    //    This is injected when calling Output(), i.e. such an Output is really a different type w.r.t. ownership.

    // what we need to do:
    //  - operations that are computed in a batch:
    //     - Slice() ops (PrimitiveFunctions) batch the arguments
    //        - optimizes for the case that the arguments were already batched (they hold a m_lazySlice (pair<batchedOp, sliceIndex>))
    //     - a new PrimitiveFunction executes the batch immediately
    //     - the original operations get their m_value field filled with a slice into the batched op
    //        - this is done lazily; initially, they just remember a pair<batchedOp, sliceIndex>, as m_lazySlice
    //     - the batchedOp in the pair is also kept for batched backprop; it is a strong ref from Variable (cannot be in a cycle)
    //  - hence, we create N+1 new nodes:
    //     - the new batched op
    //     - Splice() for each of the N inputs
    // 'free' ops are always batched together and get executed first

// ===========================================================================
// helper classes
// ===========================================================================

// ---------------------------------------------------------------------------
// NDArrayViewArena -- helper class that implements efficient arena allocation for NDArrayView objects
// ---------------------------------------------------------------------------

class NDArrayViewArena
{
    // allocate a new tensor in a large arena
    // TODO: make the arena static, so that we can carry the allocator over across invocations
    //       Currently, if I do that, program crashes upon termination (unloaded CUDA too early?)
    NDArrayViewPtr m_currentArena;
    size_t m_currentArenaUsed;
    static const size_t ARENASIZE = 64000000; // we allocate in this chunk size
public:
    // allocate an NDArrayView of a given shape, data type, and device
    // The returned memory region is a slice into a much larger NDArrayView; therefore,
    // this operation short-circuits CUDA and is very fast.
    // TODO: once operators exist that emit sparse outputs, we need sparse support here as well
    NDArrayViewPtr NewNDArrayView(const NDShape& shape, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
    {
        //static NDArrayViewPtr m_currentArena; // for now static so that it carries over across invocations, to save the allocation
        //static size_t m_currentArenaUsed;
        let numElements = shape.TotalSize();
        // if too large then plain alloc
        if (numElements > ARENASIZE)
            return make_shared<NDArrayView>(dataType, CNTK::StorageFormat::Dense, shape, device);
        // If arena not large enough then waste its remainder and just allocate a fresh one.
        // This abandons the current arena. This will not cause a memory leak, however:
        // Since the slices into it that were returned before all hold a ref-count to that arena,
        // it will be deallocated automatically as soon the last slice goes away.
        if (!m_currentArena || numElements > (ARENASIZE - m_currentArenaUsed))
        {
            m_currentArena = make_shared<NDArrayView>(dataType, CNTK::StorageFormat::Dense, NDShape{ ARENASIZE }, device);
            m_currentArenaUsed = 0;
        }
        vector<size_t> startOffset{ m_currentArenaUsed };
        vector<size_t> extent{ numElements };
        NDArrayViewPtr region = m_currentArena->SliceView(startOffset, extent);
        m_currentArenaUsed += numElements;
        return region->AsShape(shape);
    }
};

// ---------------------------------------------------------------------------
// RuntimeStatistics -- helper class for collecting runtime statistics, for
// diagnostics and debugging purposes
// ---------------------------------------------------------------------------

struct RuntimeStatistics
{
    // forward
    size_t numOpNodes = 0;
    size_t numLeafNodes = 0;
    size_t numDoneSpliceOps = 0;
    size_t numDoneFreeOps = 0;
    size_t numDoneOtherOps = 0;
    // backward
    size_t numBackpropsToInputs = 0;
    size_t numBatchedBackpropToCalls = 0;
};

// ---------------------------------------------------------------------------
// NonOwningFunctionList, NonOwningFunctionListBuilder -- helper classes:
// linked list over PrimitiveFunction objects, using m_link.
// This is used in auto-batching instead of, say, a std::vector<> or std::set<>
// for performance reasons. It also does not hold shared_ptrs, since those
// have significant runtime overhead. We don't need them, since the lists
// we build here operate on existing structures without allocating additional
// resources to be tracked.
// ---------------------------------------------------------------------------

class NonOwningFunctionList
{
protected:
    PrimitiveFunction* head; // first item or nullptr
    size_t count;            // note: count is only in here for diagnostics; only needed in builder
public:
    NonOwningFunctionList() { clear(); }
    NonOwningFunctionList(PrimitiveFunction* f) : head(f), count(1) { }
    void operator=(const NonOwningFunctionList& other)
    {
        head  = other.head;
        count = other.count;
    }
    NonOwningFunctionList(const NonOwningFunctionList& other)
    {
        *this = other;
    }
    NonOwningFunctionList(NonOwningFunctionList&& other)
    {
        *this = other;
        other.clear();
    }
    PrimitiveFunction* front() const { return head; }
    bool empty() const { return !head; }
    size_t size() const { return count; }
    void clear()
    {
        head = nullptr;
        count = 0;
    }
    class FunctionListIterator
    {
        PrimitiveFunction* iter;
    public:
        FunctionListIterator(PrimitiveFunction* f) : iter(f) { }
        PrimitiveFunction* operator->() const { return iter; }
        PrimitiveFunction& operator*() const { return *iter; } // TODO: This is weird, figure this out
        PrimitiveFunction* operator++() { iter = iter->m_link; return iter; }
        bool operator!=(const FunctionListIterator& other) { return iter != other.iter; }
    };
    FunctionListIterator begin() const { return front(); }
    FunctionListIterator end()   const { return nullptr; }
};
class NonOwningFunctionListBuilder : public NonOwningFunctionList // over PrimitiveFunction, using m_link
{
    PrimitiveFunction* tail; // note: value undefined when list empty
public:
    NonOwningFunctionListBuilder() : NonOwningFunctionList() { }
    NonOwningFunctionListBuilder(PrimitiveFunction* f) : NonOwningFunctionList(f), tail(f) { f->m_link = nullptr; }
    void push_back(PrimitiveFunction* f)
    {
        if (!head)
            head = f;
        else
            tail->m_link = f;
        tail = f;
        count++;
        f->m_link = nullptr;
    }
};

// ===========================================================================
// AutoBatch -- autobatching happening inside here
// The auto-batching related functions are grouped inside a class, since they
// share quite a bit of state.
// ===========================================================================

class Variable::AutoBatch
{
    NDArrayViewArena arena; // helper to allocate NDArrayViews as slices into very large NDArrayView objects
    RuntimeStatistics stats;

    // buffers for building NDArrayViewPtr vectors. Keep as class members to avoid repeated memory allocations.
    vector<NDArrayViewPtr> m_inputValuesBuffer;
    vector<NDArrayViewPtr> m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBufferRaw;
    template<class B> // B=vector<NDArrayViewPtr>
    B& BorrowBuffer(B& buffer, size_t batchSize)
    {
        if (buffer.capacity() < batchSize)
            buffer.reserve(batchSize * 2);
        buffer.resize(batchSize);
        return buffer;
    }

    // =======================================================================
    // forward-related functions
    // =======================================================================

    // predicate whether an op is only taking a view on its input
    // These are considered zero-cost, always batched whole-sale, and always done first.
    static bool IsViewOp(PrimitiveOpType op)
    {
        // if really needed, this can be done as a bit-test
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass         ||
            op == PrimitiveOpType::NoOp         ||
            op == PrimitiveOpType::BarrierOp    ||
            op == PrimitiveOpType::Reshape      ||
            op == PrimitiveOpType::Slice;
    }

    // class to manage the set of ready operations (the schedule)
    class ReadyOps
    {
        NonOwningFunctionListBuilder m_viewOps;
        vector<NonOwningFunctionListBuilder> m_regularOps; // m_regularOps[] is a linked list
        NonOwningFunctionListBuilder m_barrierOps;
        // TODO: This must be turned into something hashable.
        // test whether two PrimitiveFunctions can be executed as a single batched operation
        static bool AreBatchable(const PrimitiveFunction* a, const PrimitiveFunction* b)
        {
            // first it must be the same operation
            let op = a->m_op;
            // free ops always get batched; even if they have different op-codes
            if (IsViewOp(op) && op != PrimitiveOpType::BarrierOp)
                LogicError("should not get here for view ops or barrier ops");
            // op codes must match
            if (op != b->m_op)
                return false;
            // all input dimensions must match (with exception of a few special cases)
            assert(a->m_inputs.size() == b->m_inputs.size());
            for (size_t i = 0; i < a->m_inputs.size(); i++)
            {
                let& ia = a->m_inputs[i];
                let& ib = b->m_inputs[i];
                // there are a few special cases
                if (op == PrimitiveOpType::Times && i == 0)
                {
                    // for Times, the first arg must be the same object, not just the same shape
                    // TODO: a special case is a dot product, which we can write as ReduceSum(ElementTimes(a,b))
                    //       This would require to rewrite the graph though; can we do that?
                    if (ia.m_dataFields != ib.m_dataFields)
                        return false;
                }
                else
                {
                    // shapes must match
                    if (ia.Shape() != ib.Shape())
                        return false;
                }
                // another special case is reduction over all axes
            }
            // attributes must also match
            if (a->m_attributes != b->m_attributes)
                return false;
            // all match: we can batch
            return true;
        }
    public:
        // schedule an operation that has been confirmed ready
        void Schedule(PrimitiveFunction* f)
        {
            let op = f->m_op;
            // we manage three ready sets, since two common kinds are very simple
            if (op == PrimitiveOpType::BarrierOp)
                m_barrierOps.push_back(f);
            else if (IsViewOp(op))
                m_viewOps.push_back(f);
            else
            {
                // this naive implementation just scans linearly
                // scan through all op sets to see if one is batchable with 'f'
                for (auto iter = m_regularOps.begin(); iter != m_regularOps.end(); iter++)
                {
                    if (AreBatchable(f, iter->front()))
                    {
                        iter->push_back(f);
                        return;
                    }
                }
                // none fit: open a new set
                m_regularOps.push_back(NonOwningFunctionListBuilder(f));
            }
        }
        // notify a function that an input has become available; schedule it when all inputs are now available
        void NotifyInputAvailable(PrimitiveFunction* f)
        {
            if (f->m_pendingInputs <= 0)
                LogicError("NotifyInputAvailable: pending inputs already 0 yet we are executing it");
            f->m_pendingInputs--;
            // if it is now ready then schedule it
            if (f->m_pendingInputs == 0)
                Schedule(f);
        }
        // test if no more ready ops
        bool empty() const { return m_viewOps.empty() && m_regularOps.empty() && m_barrierOps.empty(); }
        size_t size() const { return (m_viewOps.size() > 0) +  + (m_barrierOps.size() > 0); }
        size_t numBatchableOpsPending() const { return m_regularOps.size(); }
        // select the next batched op to execute
        NonOwningFunctionList pop_best()
        {
            // try all three queues, in priority order
            if (!m_viewOps.empty()) // view ops always go first
                return move(m_viewOps);
            else if (!m_regularOps.empty()) // regular ops
            {
                auto best = m_regularOps.begin();
                for (auto iter = best + 1; iter != m_regularOps.end(); iter++)
                {
                    if (iter->size() > best->size())
                        best = iter;
                }
                // and remove this one from the list
                NonOwningFunctionList out = *best; // since NonOwningFunctionListBuilder uses unmanaged pointers, we can just copy it
                m_regularOps.erase(best); // TODO: suboptimal complexity; but a list has the same problem. Priority queue?
                return out;
            }
            else
                return move(m_barrierOps); // barriers only get returned when no other op is available
        }
    };
    ReadyOps m_schedule;

    // recursively traverse the tree hanging off a Variable and
    //  - prepare all nodes for batched execution
    //  - schedule all ready operations
    // TODO: Once we are in the main build, change all Function to PrimitiveFunction directly.
    // TODO: What to do with multi-valued functions? Which ones are there? What is Combine(), a barrier?
    // This function assumes that
    //  - it only runs once
    //  - m_value must not have been set (don't call this function if it has)
    //  - m_pendingInputs has been initialized to -1 by the constructor
    void TraverseFunctionTreeForward(const Variable& var)
    {
        let& fields = *var.m_dataFields;
        if (fields.m_value)
            LogicError("TraverseFunctionTreeForward() should not have been called on variables that already have a value.");
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            LogicError("Value() depends on Input or Placeholder, it is not knowable.");
        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
        {
            if (!fields.m_value)
                var.Value(); // this initializes it
            if (!fields.m_value)
                LogicError("Parameter/Constant has no Value??");
            stats.numLeafNodes++;
            return;
        }
        auto& f = *fields.m_ownerFunction.lock();
        if (f.m_pendingInputs != -1) // already visited
            return;
        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = inputs[i];
            auto& fields = *input.m_dataFields;
            // recursively traverse
            if (!fields.m_value)
            {
                TraverseFunctionTreeForward(input);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                {
                    pendingInputs++;
                    // record ourselves as a consumer of the input
                    if (!fields.m_consumers.first.first) // optimized for main case of 1 consumer. No std::vector in that case.
                        fields.m_consumers.first = make_pair(&f, i); // note: we don't need i for forward; can optimize
                    else
                        fields.m_consumers.second.push_back(make_pair(&f, i));
                }
            }
            else
                stats.numLeafNodes++;
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
        stats.numOpNodes++;
    }

    // return the m_value field of a variable, but possibly realizing it lazily if it is an index operation
    static const NDArrayViewPtr& LazilyIndexedValue(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        if (fields.m_value)
            return fields.m_value;
        fail_if(!fields.m_lazyIndex.first, "variable unexpectedly has no value yet, nor is it a slice view into a batched op");
        // the PrimitiveFunction does not own its output, it is a slice view into another
        let& from = LazilyIndexedValue(fields.m_lazyIndex.first->m_outputs[0]);
        let index = fields.m_lazyIndex.second;
        if (index == SIZE_MAX) // special sentinel value that means "don't slice, actually"
            fields.m_value = from;
        else
            fields.m_value = from->IndexLastAxis(index);
        return fields.m_value;
    }

    static void LogFunction(PrimitiveFunction& f, const char* prefix = "", size_t markIndex = SIZE_MAX)
    {
        let& inputs = f.m_inputs;
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        fprintf(stderr, "%s%S%S = %S(", prefix, f.Uid().c_str(), outputShape.AsString().c_str(), f.OpName().c_str());
        for (size_t i = 0; i < inputs.size() && i < 4; i++)
        {
            let& input = inputs[i];
            let& fields = *input.m_dataFields;
            // little helper function to fix up variable names by removing _Output_0
            // TODO: Once we support >1 output, this needs a bit more code.
            let GetVarName = [](const Variable& input) -> wstring
            {
                auto uid = input.Uid();
                if (uid.size() > 9 && wcscmp(uid.c_str() + uid.size() - 9, L"_Output_0") == 0)
                    uid.resize(uid.size() - 9);
                return uid;
            };
            if (fields.m_lazyIndex.first)
            {
                let& input1 = fields.m_lazyIndex.first->m_outputs[0];
                fprintf(stderr, "%s%s%S%S[%d]", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input1).c_str(), input1.Shape().AsString().c_str(), (int)fields.m_lazyIndex.second);
            }
            else
                fprintf(stderr, "%s%s%S%S", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input).c_str(), input.Shape().AsString().c_str());
        }
        if (inputs.size() > 4)
            fprintf(stderr, ", +%d", (int)(inputs.size() - 4));
        fprintf(stderr, ")\n");
    }

    // compute the value of 'f', storing it in the arena (unless 'isFree', which must be set when there is nothing to store)
    const Variable& MemoizeKnowableValueInArena(PrimitiveFunction& f, bool isFree = false)
    {
        if (f.m_outputs.size() != 1)
            LogicError("MemoizeKnowableValueInArena: only functions with 1 output are supported");
        // fetch the NDArrayViewPtrs for all inputs
        let& inputs = f.m_inputs;
        auto& inputValues = BorrowBuffer(m_inputValuesBuffer, inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            inputValues[i] = LazilyIndexedValue(inputs[i]); // (if this is a lazy slice, then now we must resolve it)\
        // allocate the output NDArrayViewPtr in the arena
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        // logging
#ifdef LOG_DETAILS
        LogFunction(f, "[bf] ");
#endif
        auto outValue = isFree ? nullptr : arena.NewNDArrayView(outputShape, inputValues[0]->GetDataType(), inputValues[0]->Device());
        // execute it
        output.m_dataFields->m_value = move(PrimitiveFunction::ComputeKnowableValue(f.m_op, inputValues, f.Attributes(), outputShape, move(outValue), f));
        // stats
        let primitiveOp = f.m_op;
        if (primitiveOp == PrimitiveOpType::StopGradient ||
            primitiveOp == PrimitiveOpType::Pass ||
            primitiveOp == PrimitiveOpType::NoOp ||
            primitiveOp == PrimitiveOpType::Reshape ||
            primitiveOp == PrimitiveOpType::Slice)
            stats.numDoneFreeOps++;
        else if (primitiveOp == PrimitiveOpType::Splice)
            stats.numDoneSpliceOps++;
        else
            stats.numDoneOtherOps++;
        return output;
    }

    static void ResetPendingToIdle(PrimitiveFunction& f)
    {
        if (f.m_pendingInputs != 0)
            LogicError("ResetPendingToIdle: pendingINputs is not 0, so we should not have gotten here");
        f.m_pendingInputs = -1; // unknown
    }

    // temp variables for ExecuteBatchedOpAndUpdateSchedule(); keep outside to reuse the memory allocation
    vector<Variable> m_batchedInputs;
    vector<Variable> m_spliceArgsBuffer;
    size_t m_numBatchedLaunches = 0; // (for statistics only)

    // batch-execute a set of ops that are known to be batchable
    // For every batched operation, this generates a new PrimitiveFunction object for the op itself, and one
    // for a splice operation for each batched inputs.
    // I.e. this is not a full graph transform, but rather a graph augmentation, so that during backprop,
    // we can recover the batched operations, while the original graph does not get modified.
    // Any batched operation will generate its result in a dense tensor with a batch dimension.
    // The consumers of the original ops will get a back-reference in the m_lazyIndex field.
    // If such a result is ever accessed individually, it will lead to a lazy NDArrayView::SliceView() call
    // (but no Splice Function object is used for this).
    // All ops passed to this function must get their m_pendingInputs changed from 0 to -1 (newly created batched ones also will have -1).
    void ExecuteBatchedOpAndUpdateSchedule(NonOwningFunctionList ops) // (note: NonOwningFunctionListBuilder is so small that it is best copied)
    {
        // TODO: need to handle ops that have >1 output, such as Combine(). Just don't batch them ever? Combine() is just a see-through anyway.
        // get a representative op
        auto& f0 = *ops.front();
        let op = f0.m_op;
        let batchSize = ops.size();
        let isFree = IsViewOp(op);
        if (!isFree)
            m_numBatchedLaunches++;
        let numArgs = f0.m_inputs.size();
        // perform the op
        let isTimes = (op == PrimitiveOpType::Times); // is special-cased
        let doNaively =
            isFree ||
            isTimes && f0.m_inputs[1].m_dataFields->m_value && (f0.m_inputs[1].m_dataFields->m_value->IsSparse()) || // can't batch sparse
            op == PrimitiveOpType::Splice ||
            batchSize == 1;
        //fprintf(stderr, "%d %sexecuting %d instances of %S -> %S; %d batchable ops pending\n",
        //        isFree ? -1 : (int)m_numBatchedLaunches,
        //        doNaively ? "" : "batch-",
        //        (int)batchSize, f0.OpName().c_str(), f0.m_outputs[0].Shape().AsString().c_str(),
        //        (int)m_schedule.numBatchableOpsPending());
        if (doNaively)
        {
            // for correctness testing of underlying mechanism, compute them without actual batching
            for (auto op = ops.begin(); op != ops.end(); ++op)
            {
                // execute it
                MemoizeKnowableValueInArena(*op, isFree);
                // reset state
                ResetPendingToIdle(*op);
                // TODO: realize splice ops that are index ops as a m_lazyIndex at this point
                //if (f0.m_op == PrimitiveOpType::Slice)
                //{
                //    // inject the lazyIndex
                //}
#if 0           // test effect of the unbatched sparse times
                if (f0.m_op == PrimitiveOpType::Times)
                    op->ComputeKnowableValue(op->m_op, m_inputValuesBuffer, op->Attributes(), op->m_outputs[0].Shape(), move(out));
#endif
            }
        }
        // execute the batchable operations as a batch
        // Every resulting batched op consists of the following new operations:
        //  - a Splice() or Slice() for each input (e.g. 2 for a binary op)
        //  - a PrimitiveFunction that is the op itself
        //  - m_lazyIndex entries that represent a "virtual" Slice() that is never created as a PrimitiveFunction object to saved mallocs.
        // As for resource management, m_lazyIndex will hold a strong ref to the PrimitiveFunction;
        // and we will hack its m_inputs[].m_outputComposite to hold a strong reference to the Splice() or Slice().
        // (This is a little ugly since m_outputComposite is meant to hold a CompositeFunction, but we misuse it
        // to hold a PrimitiveFunction.)
        else
        {
            // batch all arguments
            // TODO: see if this can be sped up using un-managed pointers (save lots of ref-counting); or even use GatherBatch lambda
            m_batchedInputs.resize(numArgs);
            size_t maxRank = 0;
            size_t i0 = isTimes ? 1 : 0;
            for (size_t i = i0; i < numArgs; i++)
            {
                // determine max rank
                let rank = f0.m_inputs[i].Shape().Rank();
                if (rank > maxRank)
                    maxRank = rank;
            }
            bool anyBatchedInputs = false;
            if (i0 == 1) // Times(): matrix must be identical
                m_batchedInputs[0] = f0.m_inputs[0];
            for (size_t i = i0; i < numArgs; i++)
            {
                // create splice args for this argument
                // allocate buffers to hold the arguments
                auto& spliceInputs = m_spliceArgsBuffer; // TODO rename to gatherInputs and m_gatherArgsBuffer
                assert(spliceInputs.empty()); // previous use must have cleared it
                if (spliceInputs.capacity() < batchSize)
                    spliceInputs.reserve(max(batchSize, 2 * spliceInputs.capacity()));
                // optimization: if all args are consecutive slices, then use a slice view instead
                let* pfields0 = f0.m_inputs[i].m_dataFields.get();
                let& lazyIndex0 = pfields0->m_lazyIndex; // for 'allConsecutiveSlices' test
                let is0LazyIndex = (bool)lazyIndex0.first;
                // loop over all batched ops
                // BUGBUG: How about NoOp? (used for Barrier) Also Alias and Reshape actually
                //         Seems if we can carry on a bacth, we should run them once; otherwise don't batch.
                bool allSame = true;
                bool allConsecutiveSlices = is0LazyIndex && lazyIndex0.second != SIZE_MAX; // to be consecutive, it must be a slice to start with
                size_t j = 0;
                for (auto op = ops.begin(); op != ops.end(); ++op, j++) // create the batched tensors
                {
                    let& input = op->m_inputs[i];
                    let* pfields = input.m_dataFields.get();
                    let& lazyIndex = pfields->m_lazyIndex;
                    // optimization: if all args are the same, then don't batch
                    allSame = allSame &&
                        (pfields == pfields0 ||                 // same
                         (is0LazyIndex && lazyIndex == lazyIndex0)); // or same view  --TODO: could this need to be checked recursively?
                    // optimization: if all args are consecutive slices, then use a slice view instead
                    if (allConsecutiveSlices)
                    {
                        // optimization: if consecutive slices, then recover the original batched tensor
                        allConsecutiveSlices = allConsecutiveSlices &&
                            lazyIndex.first  == lazyIndex0.first    &&
                            lazyIndex.second == lazyIndex0.second + j;
                        // TODO: Per Jon's suggestion, we can be a little loose here. For a variable-length
                        // scenario, we will loose entries in the middle. We can allow to keep a few around
                        // in garbage-in-garbage-out. If, say, there are additional 20% gap values, we just
                        // carry them forward, and ignore them when implanting the result.
                    }
                    // append the input
                    spliceInputs.push_back(input);
                    // note: Variable is just two shared_ptrs, one being NULL; so this is cheap
                    // note: input is a regular Variable with regular ownwership rules (it does not come from inside here)
                }
                // and splice
                if (allSame) // optimized case: all ops share the same operand: no need to batch them
                    // note: we assume strict broadcasting semantics here (if at least one input is actually batched)
                    m_batchedInputs[i] = spliceInputs[0];
                else
                if (allConsecutiveSlices) // they are consecutive: can short-circuit as a slice view
                {
                    let& from  = lazyIndex0.first;
                    let  begin = lazyIndex0.second;
                    let& output = from->m_outputs[0];
                    fail_if(!output.m_dataFields->m_value, "value not yet available??");
                    let& fromDims = output.Shape().Dimensions();
                    let axis = fromDims.size() - 1;
                    if (begin == 0 && j == fromDims[axis]) // full range: just take it
                        m_batchedInputs[i] = output; // note: graph already has a strong ref to output elsewhere
                    else // sub-range: splice it by taking a slice view on the previously spliced batch
                    {
                        // create a new PrimitiveFunction Splice()
                        vector<size_t> outputShape = fromDims; // determine output shape
                        outputShape[axis] = j;
                        auto additionalProperties = Dictionary(); // create additional arguments
                        additionalProperties[L"axis"      /*PrimitiveFunction::AttributeNameAxis*/      ] = Axis((int)axis);
                        additionalProperties[L"beginIndex"/*PrimitiveFunction::AttributeNameBeginIndex*/] = (int)begin;
                        additionalProperties[L"endIndex"  /*PrimitiveFunction::AttributeNameEndIndex*/  ] = (int)(begin + j);
                        let spliceOp = PrimitiveFunction::RawPrimitiveFunction(PrimitiveOpType::Slice, vector<Variable>{ output }, outputShape, move(additionalProperties));
#ifdef LOG_DETAILS
                        spliceOp->m_uid = L"#" + spliceInputs[0].Uid();
#endif
                        // and execute it
                        let& output = MemoizeKnowableValueInArena(*spliceOp, /*isFree=*/true);
                        // and that's our input to the batched operation
                        //m_batchedInputs[i] = output.CompositePreservingCopy(spliceOp);
                        m_batchedInputs[i] = output;
                        m_batchedInputs[i].m_outputComposite = spliceOp;
                    }
                    anyBatchedInputs = true;
                }
                else
                {
                    // create a new PrimitiveFunction Splice()
                    vector<size_t> outputShape; // determine output shape
                    outputShape.reserve(maxRank + 1);
                    outputShape = LazilyIndexedValue(spliceInputs[0])->Shape().Dimensions();
                    outputShape.resize(maxRank, 1);             // pad to maxRank
                    outputShape.push_back(spliceInputs.size()); // and add the batch axis
                    auto additionalProperties = Dictionary(); // create additional arguments
                    additionalProperties[L"axis"/*PrimitiveFunction::AttributeNameAxis*/] = Axis((int)maxRank);
                    let spliceOp = PrimitiveFunction::RawPrimitiveFunction(PrimitiveOpType::Splice, vector<Variable>(spliceInputs), outputShape, move(additionalProperties));
#ifdef LOG_DETAILS
                    spliceOp->m_uid = L"#" + spliceInputs[0].Uid();
#endif
                    // and execute it
                    let& output = MemoizeKnowableValueInArena(*spliceOp);
                    // and that's our input to the batched operation
                    // To make sure we hold a reference to this PrimitiveFunction, inject a strong ref to the spliceOp into the copy of its Output.
                    // Note that we abuse the composite field for a non-composite, which works because it is just a FunctionPtr, and we own it.
                    //m_batchedInputs[i] = output.CompositePreservingCopy(spliceOp);
                    m_batchedInputs[i] = output;
                    m_batchedInputs[i].m_outputComposite = spliceOp;
                    anyBatchedInputs = true;
                }
                // release shared_ptrs asap
                spliceInputs.clear();
            }
            // execute the operation and implant the results
            // BUGBUG: The newly created PrimitiveFunction objects must get their consumer chain set up.
            let& unbatchedOutputShape = f0.m_outputs[0].Shape();
            PrimitiveFunctionPtr batchedOp;
            if (anyBatchedInputs)
            {
                // create a new PrimitiveFunction for the batched op
                // Batched inputs have been prepared in m_batchedInputs[].
                let expectedOutputShape = unbatchedOutputShape.AppendAxis(maxRank, batchSize);
                batchedOp = PrimitiveFunction::RawPrimitiveFunction(f0.m_op, vector<Variable>(m_batchedInputs), expectedOutputShape, Dictionary(f0.Attributes()));
#ifdef LOG_DETAILS
                batchedOp->m_uid = L"*" + f0.Uid();
#endif
            }
            else
            {
                // all inputs identical: compute it only once
                batchedOp = PrimitiveFunction::RawPrimitiveFunction(f0.m_op, vector<Variable>(f0.m_inputs), f0.m_outputs[0].Shape(), Dictionary(f0.Attributes()));
#ifdef LOG_DETAILS
                batchedOp->m_uid = L"." + f0.Uid();
#endif
                // TODO: the following is a little more efficient, but creates a cycle, so we should exclude the lazy index for the first op
                //batchedOp = f0.shared_from_this();
            }
            // execute it
            MemoizeKnowableValueInArena(*batchedOp);
            // implant all results (all as lazy/virtual references through m_lazyIndex)
            size_t j = anyBatchedInputs ? 0 : SIZE_MAX;
            for (auto op = ops.begin(); op != ops.end(); ++op)
            {
                // TODO: review this w.r.t. multi-output functions
                // we remember where we came from for backprop in this case
                auto& fields = *op->m_outputs[0].m_dataFields;
                fields.m_lazyIndex = make_pair(batchedOp, j);
                // semantically, this will compute as fields.m_value = out->IndexLastAxis(j);
                // but it gets deferred to save effort
                if (j != SIZE_MAX) // SIZE_MAX means don't slice
                    j++;
                // TODO: set up batchedOp.m_consumers
                // reset state
                ResetPendingToIdle(*op);
            }
            // release the ref counts on the batched inputs
            m_batchedInputs.clear();
        }

        // update all ops' consumers and schedule them when possible
        // BUGBUG: Consumer chain here should have been migrated to the batched op; and notifed from there.
        for (auto op = ops.begin(); op != ops.end(); ++op)
        {
            for (let& output : op->m_outputs)
            {
                // notify consumers
                auto& fields = *output.m_dataFields;
                auto& c = fields.m_consumers.first; // first consumer (this is a special optimization to avoid a malloc in case of 1 consumer)
                if (c.first)
                    m_schedule.NotifyInputAvailable(c.first);
                for (auto& c : fields.m_consumers.second) // all other consumers
                    m_schedule.NotifyInputAvailable(c.first);
                // clear consumer list (this operation is done)
                fields.m_consumers.first.first = nullptr;
                fields.m_consumers.second.clear();
            }
        }
    }

public:
    // -----------------------------------------------------------------------
    // BatchedForward() -- entry point for auto-batched implementation of PrimitiveFunction::Value()
    // -----------------------------------------------------------------------

    // Value(), computed with automatic batching
    // This routine uses temporary fields that are assumed initialized in a specific way:
    //  - PrimitiveFunction::m_pendingInputs:
    //     - #inputs that still need to be computed before a node's value can be computed
    //     - also used as a 'visited' flag during traversal
    //     - upon entry and exit of this function, this must be -1 (idle)
    //  - Variable::m_consumers:
    //     - set of consumers of this value. Used to count m_pendingInputs.
    //     - must be empty upon entry and exit
    // plus more temp fields:
    //  - PrimitiveFunction::m_link: pointer to next PrimitiveFunction in the same batchable op
    // And it leaves the following:
    //  - m_value: updated as desired
    //    TODO: values not needed by user or gradient should use scratch space
    //  - m_lazyIndex: if a slice or view came from a batched operation, this points to it
    //     - Any newly created batched ops are referenced this way.
    NDArrayViewPtr BatchedForward(const Variable& v)
    {
#ifdef LOG_DETAILS
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr& f) { LogFunction(dynamic_cast<PrimitiveFunction&>(*f), "[r] "); });
#endif
        auto& fields = *v.m_dataFields;
        // if value already there then just return it
        if (fields.m_value)
            return fields.m_value;
        AssertTreeStateForward(v); // (sanity check)
        // mark all nodes w.r.t. how many inputs they are waiting for before being computable
        if (!fields.m_value)
        {
            // prepare and schedule first set
            TraverseFunctionTreeForward(v);
            // compute the entire graph
            while (!m_schedule.empty())
            {
                // select the best amongst the scheduled ops
                auto opBatch = m_schedule.pop_best();
                // execute it, and also update all outputs' values and consumers, and the schedule
                ExecuteBatchedOpAndUpdateSchedule(opBatch);
            }
            assert(fields.m_value);
        }
        AssertTreeStateForward(v); // (sanity check)
#ifdef LOG_STATS
        fprintf(stderr, "BatchedForward: %d forward ops executed besides %d splices and %d views, in nominally %d PrimitiveFunctions on %d known values\n",
                (int)stats.numDoneOtherOps, (int)stats.numDoneSpliceOps, (int)stats.numDoneFreeOps, (int)stats.numOpNodes, (int)stats.numLeafNodes);
        size_t numOpNodes1 = 0;
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr&) { numOpNodes1++; });
        fail_if(numOpNodes1 != stats.numOpNodes, "we did not traverse the graph correctly");
#endif
        return LazilyIndexedValue(v);
    }

    // =======================================================================
    // backward-related functions
    // =======================================================================

    // allocate memory for m_gradient
    // This lazily creates the m_gradient NDArrayView, which may live in a batched op.
    // Returns beta = 0 if gradient was newly created, otherwise 1
    __declspec(noinline) double LazilyCreateLazilyIndexedGradient(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        // if gradient exists then return it
        double beta;
        if (fields.m_gradient)
            beta = 1.0;
        else
        {
            // create new gradient
#ifndef NO_BATCHED_BACKPROP
            // if this op draws from a batched op, then the gradient lives in there as well; we return a view onto it
            if (fields.m_lazyIndex.first)
            {
                let& from  = fields.m_lazyIndex.first;
                let  index = fields.m_lazyIndex.second;
                let& fromOutput = from->m_outputs[0];
                beta = LazilyCreateLazilyIndexedGradient(fromOutput);
                let& fromGradient = fromOutput.m_dataFields->m_gradient;
                if (index == SIZE_MAX) // special sentinel value that means "don't slice, actually"
                    fields.m_gradient = fromGradient;
                else // it's a slice: gradient is a slice view into from's output gradient
                {
                    if (beta == 0.0) // gradient is fresh: explicitly reset all (since we are slicing into the input gradientm, we cannot use the beta mechanism)
                    {
                        fromGradient->SetValue(0.0f);
                        beta = 1.0;
                    }
                    fields.m_gradient = fromGradient->IndexLastAxis(index);
                }
            }
            else
#endif
            {
                // create a new one
                // TODO: allocate parameters as separate objects; and allow user to pass buffers in
                fields.m_gradient = arena.NewNDArrayView(fields.m_shape, fields.m_dataType, fields.m_value->Device());
                beta = 0.0; // has not been initialized (...actually has; but this saves memory round trips)
            }
        }
        return beta;
    }

    // recursively traverse the tree hanging off a Variable and build the m_consumer fields
    // Unlike forward prop, we...
    //  - can skip any branch that does not need a gradient (!m_needsGradient and StopGradient ops).
    //  - short-circuit into batched ops (m_lazyIndex) so that we backprop through them instead
    // All nodes that were traversed have all input's m_consumers set up and their m_pendingInputs set to 0.
    void DetermineConsumersForBackward(const Variable& var)
    {
        auto& fields = *var.m_dataFields;
        fields.m_visited = false; // we use this for backprop control  --TODO: consolidate these

        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
            return; // reached a leaf

        fail_if(!fields.m_value, "variable has no value yet??");
        fail_if(!fields.m_needsGradient, "unexpectedly encountered a node with m_needsGradient=false??");
        fail_if(fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder, "unexpectedly encountered an Input or a Placeholder??");

        // If a variable has the m_lazyIndex field set, it means that its value was not
        // actually computed from its true input; but rather a slice into the result of
        // a batched operation. In that case, we traverse through that batched operation
        // instead. As a consequence, it is the batched operation that will be recorded as the
        // consumer of all inputs of the batched operation, rather than the original
        // unbatched operation. And as a consequence of that, back propagation will use
        // the same batching that was determined in forward computation.
#ifndef NO_BATCHED_BACKPROP
        auto& f = fields.m_lazyIndex.first        // if var was computed via batched op
                ? *fields.m_lazyIndex.first       // then backprop into the batched op
                : *fields.m_ownerFunction.lock(); // otherwise traverse the original op
#else
        auto& f = *fields.m_ownerFunction.lock();
#endif
        //DetermineConsumersForBackward(f);
//#if 1//ndef NO_BATCHED_BACKPROP
//        if(fields.m_lazyIndex.first)
//        {
//            auto& f = *fields.m_lazyIndex.first;
//            DetermineConsumersForBackward(f);
//        }
//        else
//#endif
//        {
//            auto& f = *dynamic_pointer_cast<PrimitiveFunction>(fields.m_ownerFunction.lock());
//            DetermineConsumersForBackward(f);
//        }
    //}
    //void DetermineConsumersForBackward(PrimitiveFunction& f)
    //{
        fail_if(f.m_pendingInputs == -2, "unexpectedly encountered a cyclic graph??"); // graph is cyclic??

        if (f.m_pendingInputs != -1) // already visited
            return;

        fail_if(f.m_op == PrimitiveOpType::StopGradient, "unexpectedly encountered a StopGradient, which should have propagated m_needsGradient=false upwards");

        // we are now in a PrimitiveFunction that should backprop its gradient
        // TODO: implement short-circuiting here
        f.m_pendingInputs = -2; // (temp value to detect cycles; not really needed)

        // determine how many inputs are pending; and also recurse and set up the consumer list
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let* inputp = &inputs[i];
            // if the input is a result of a batched operation, then traverse into that instead
            if (inputp->m_dataFields->m_lazyIndex.first)
                inputp = &inputp->m_dataFields->m_lazyIndex.first->m_outputs[0];
            let& input = *inputp;
            auto& fields = *input.m_dataFields;
            fields.m_visited = false; // TODO: clean this up
            if (!fields.m_needsGradient)
                continue; // skip inputs that receive no gradients
            // this input will receive a gradient; reset it (later, we *accumulate* into it since nodes can receive gradients from multiple consumers)
            // Note that BatchedBackward() returns shared_ptrs to the gradient values, so they won't get lost.
            // BUGBUG: (But they get reallocated over again, and will hold the entire arena!!) BUGBUG!
            // BUGBUG: we must not kill the gradient buffers passed by the user
            fields.m_gradient.reset();
            // record ourselves as a consumer of the input
            // Remember that any input that is a lazyIndex has been redirected to its lazy source,
            // i.e. it is the lazy source that will pull this gradient.
            if (!fields.m_consumers.first.first)
            {
                fields.m_consumers.first = make_pair(&f, i);
                // now process recursively the inputs
                DetermineConsumersForBackward(input);
            }
            else
                fields.m_consumers.second.push_back(make_pair(&f, i));
            stats.numBackpropsToInputs++;
        }
        f.m_pendingInputs = 0; // used as a visited flag
    }

    // helper to batch an array of NDArrayViews of the same shape into a new axis
    // TODO: do this with a lambda so we can go straight into gatherBatchResultDims
    vector<size_t> gatherBatchResultDims;
    NDArrayViewPtr GatherBatchInArena(const vector<NDArrayViewPtr>& inputs)
    {
        let& input0 = *inputs[0];
        let& inputShape = input0.Shape().Dimensions();
        gatherBatchResultDims.assign(inputShape.begin(), inputShape.end());
        let axis = gatherBatchResultDims.size();
        gatherBatchResultDims.push_back(inputs.size());
        auto out = arena.NewNDArrayView(gatherBatchResultDims, input0.GetDataType(), input0.Device());
        return move(NDArrayView::GatherBatch(inputs, (int)axis, move(out)));
    }

    // backprop gradient into 'var' by pulling all of its consumers (recursively)
    // This is the second function that does batching.
    // The vectors for building the lists are class members so that we reuse the malloc.
    vector<pair<PrimitiveFunction*, size_t>> m_placeItemConsumers;    // IndexLastAxis() op  --do we have those actually? Or already short-circuited?
    vector<pair<PrimitiveFunction*, size_t>> m_matrixWeightConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_summandConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_otherConsumers;
    __declspec(noinline) void DetermineAndAddToBucket (const pair<PrimitiveFunction*, size_t>& c)
    {
        let* f = c.first;
        let index = c.second;
        fail_if(f->m_outputs.size() != 1, "for now only functions with a single output are supported"); // (needs some more plumbing to fix this)
        // backprop into Times' matrix argument
        let IsMatrixGradient0Batchable = [](const PrimitiveFunction& f, const PrimitiveFunction& g) -> bool
        {
#if 1
            return false;
#else
            // we compute leftGrad = outGrad @ right^T
            let&   fOutShape = f.m_outputs[0].Shape().Dimensions();
            let&   gOutShape = g.m_outputs[0].Shape().Dimensions();
            let& fRightShape = f.m_inputs[1].Shape().Dimensions();
            let& gRightShape = g.m_inputs[1].Shape().Dimensions();
            let&   leftShape = f.m_inputs[0].Shape().Dimensions();
            let    outRank =   fOutShape.size();
            let   gOutRank =   gOutShape.size();
            let  rightRank = fRightShape.size();
            let gRightRank = gRightShape.size();
            let   leftRank =   leftShape.size();
            fail_if(leftShape != g.m_inputs[0].Shape().Dimensions(), "IsMatrixGradient0Batchable: dimensions of matrix gradient don't match");
            if (outRank != gOutRank || rightRank != gRightRank)
                return false; // rank not matching: stop batching right here (we could do better)
            // the center 'reductionRank' dimensions get reduced over
            if (outRank + rightRank - leftRank != 2) // if 2 then we reduce over a single batch axis
                return false; // this is not a batch gradient; back out
            fail_if(fOutShape.back() != fRightShape.front() || gOutShape.back() != gRightShape.front(), "IsMatrixGradient0Batchable: inner dimensions of matrix gradient don't match");
            // the two gradient ops match if all dimensions except for the batch dim match
            for (size_t k = 0; k < outRank - 1; k++) // check outGrad
                if (fOutShape[k] != gOutShape[k])
                    return false;
            for (size_t k = 1; k < rightRank; k++) // check right
                if (fRightShape[k] != gRightShape[k])
                    return false;
            // gradient is batchable
            return true;
#endif
        };
        // We only collect matrix products with fully matching dimensions.
        if (f->m_op == PrimitiveOpType::Times && index == 0 &&
            (m_matrixWeightConsumers.empty() || (IsMatrixGradient0Batchable(*f, *m_matrixWeightConsumers.back().first))))
            m_matrixWeightConsumers.push_back(c);
        // backprop into either of Plus' arguments
        //else if (f->m_op == PrimitiveOpType::Plus)
        //    return m_summandConsumers;
        // all other
        else
            m_otherConsumers.push_back(c);
    };

    // backprop into weight parameter of a Times op (inputs[0])
    // This can be batched into a single matrix product.
    void BackpropToMatrixWeight(vector<pair<PrimitiveFunction*, size_t>>& consumers)
    {
#if 1
        for (auto& c : consumers)
            BackpropTo(c.first, c.second);
#else
        // We compute
        //  leftGrad = sum_i outGrad_i @ right_i^T
        //           = (concat_i outGrad_i) @ (concat_i right_i)^T
        // where concat_i means to concatenate matrices along their trailing (batch) axis.
        // It has already been verified that all i have the same rank and dimensions except for a single reduction dimension.
        // So concatenate outGrad and right
        let batchSize = m_matrixWeightConsumers.size();
        auto& m_timesOutGrads        = BorrowBuffer(m_inputValuesBuffer,     batchSize);
        auto& m_timesDataRightInputs = BorrowBuffer(m_outputGradientsBuffer, batchSize);
        let& f0 = *m_matrixWeightConsumers.front().first;
        let& input0 = f0.m_inputs[0];
        for (size_t i = 0; i < batchSize; i++)
        {
            let &c = m_matrixWeightConsumers[i];
            fail_if(c.second != 0, "wrong input??");
            m_timesOutGrads       .push_back(c.first->m_outputs[0].m_dataFields->m_gradient);
            m_timesDataRightInputs.push_back(c.first->m_inputs [1].m_dataFields->m_value   );
        }

        auto rightBatch   = GatherBatchInArena(m_timesDataRightInputs); // TODO: we can further factor this and pass a lambda
        auto outGradBatch = GatherBatchInArena(m_timesOutGrads);
        let beta = LazilyCreateLazilyIndexedGradient(input0);
        PrimitiveFunction::BackpropTo(outGradBatch.get(), /*index=*/0, f0.m_op, f0.m_attributes, /*outputValue=*/nullptr, { nullptr, rightBatch.get() }, input0.m_dataFields->m_gradient, beta);
#endif
    }

    // compute a variable's outputs' gradient (var.m_gradient)
    // This operates on the PrimitiveFunction(s) that use this var's output value--its "consumers".
    // A variable knows all of its consumers. This function back-propagates from all consumers
    // into the variable's m_gradient field.
    // A consumer is a specific input of a PrimitiveFunction, specified as (function pointer, input index).
    // In the case of multiple consumers, the gradient is the sum.
    // This recursively traverses the graph upwards via the consumers chain.
    __declspec(noinline) void AggregateGradientFromAllConsumers(const Variable& var)
    {
        let& fields = *var.m_dataFields;
        if (fields.m_visited)
            return;

        auto& c = fields.m_consumers.first;
        // reached a leaf
        if (!c.first)
            return;

        fail_if(!fields.m_needsGradient, "backprop into variable that does not need gradient");

        fields.m_visited = true;

        // recursively realize all consumers' outputs' gradients
        for (let& output : c.first->m_outputs)
            AggregateGradientFromAllConsumers(output);
        for (auto& c : fields.m_consumers.second)
            for (let& output : c.first->m_outputs)
                AggregateGradientFromAllConsumers(output);
        // Now all consumers are ready to propagate into var's m_gradient.
        // The resulting gradient is the sum of all that's backpropped here,
        // and this is the only place where a variable's gradient ever gets aggregated.

        // create var's m_gradient (may be a slice view)
        // For Parameters, m_gradient may already exist; for all others, it must not.
        fail_if(var.Kind() != VariableKind::Parameter && fields.m_gradient, "non-Parameter variable unexpectedly already has a gradient"); // (sanity check; I am not sure actually, maybe too strict)

        // fast path: only one consumer, nothing to batch
        if (fields.m_consumers.second.empty())
        {
            BackpropTo(c.first, c.second);
            return;
        }

        // auto-batched path
        // The forward prop has already batched data. However, for backprop, there can
        // be more batching across the multiple consumer's backprop operations.

        // At this point, we need to execute the BackpropTo() function for every
        // consumer (function, input), and then sum up the result.
        // The combination of the backprop functions and summing up their result
        // may be batchable, as for the matrix product.

        // Since a variable can be consumed by multiple different kinds of PrimitiveFunctions,
        // which each have a different gradient computation, we need to separate them out.
        // At present, we aim for weight gradients that are part of a matrix product
        // or a bias addition.
        // First sort all consumer gradients according to their operation.
        fail_if(!m_otherConsumers.empty(), "consumer bucket lists unexpectedly not cleaned up");
        DetermineAndAddToBucket(c);
        for (auto& c : fields.m_consumers.second)
            DetermineAndAddToBucket(c);

        // matrix-weight bucket
        if (!m_matrixWeightConsumers.empty())
        {
            BackpropToMatrixWeight(m_matrixWeightConsumers);
            m_matrixWeightConsumers.clear();
        }

        // summation bucket
        // ...

        // others bucket
        for (auto& c : m_otherConsumers)
            BackpropTo(c.first, c.second);
        m_otherConsumers.clear();
    }

    // back-propagate all of f's outputs' m_gradients to one input
    // This wraps the PrimitiveFunction's BackpropTo(), interfacing from vectors of Variable to vectors of NDArrayViewPtr.
    // Note that each input that is lazy should redirect into a slice in its lazy source.
    void BackpropTo(PrimitiveFunction* f, size_t index)
    {
#ifdef LOG_DETAILS
        LogFunction(*f, "[bb] ", index);
#endif
        let& inputs =  f->m_inputs;
        auto& input = inputs[index];
        auto& fields = *input.m_dataFields;
        fail_if(!fields.m_needsGradient, "function unexpectedly does not need a gradient");
        // get the TensorViews for everything we may compute the gradient from
        let& outputs = f->m_outputs;
        fail_if(outputs.size() != 1, "only functions with 1 output are currently supported");
        let& outputFields = *outputs[0].m_dataFields;
#ifndef NO_BATCHED_BACKPROP
        fail_if(outputFields.m_lazyIndex.first, "unexpectedly ran into a function that does not own its output"); // we don't backprop through unbatched ops
#endif
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");
        let* outputValue    = outputFields.m_value   .get();
        let* outputGradient = outputFields.m_gradient.get();

        let numInputs = inputs.size();
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, numInputs);
        for (size_t i = 0; i < numInputs; i++)
        {
            let& input1 = inputs[i];
            let& fields = *input1.m_dataFields;
            fail_if(!fields.m_value, "unexpectedly ran into a function that has no m_value yet??");
            inputValues[i] = fields.m_value.get();
        }

        // compute gradients for the desired input
        // Get or create m_gradient as the desired gradient's TensorView.
        // If the input is a lazyIndex, then the gradient is a view into the lazy source.
        let beta = LazilyCreateLazilyIndexedGradient(input);
        // backprop into the input
        PrimitiveFunction::BackpropTo(outputGradient, index, f->m_op, f->m_attributes, outputValue, inputValues, fields.m_gradient, beta, *f);
        stats.numBatchedBackpropToCalls++;
    }

    // helper to verify that the tree is clean
    void AssertTreeStateForward(const Variable& v) const
    {
#if 1
        v;
#else
        let& fields = *v.m_dataFields;
        if (fields.m_consumers.first.first || !fields.m_consumers.second.empty())
            LogicError("AssertTreeStateForward: m_consumers should be empty");
        let owner = fields.m_ownerFunction.lock();
        if (owner)
        {
            if (owner->m_pendingInputs != -1)
                LogicError("AssertTreeStateForward: m_pendingInputs should be -1");
            for (let& input : owner->m_inputs)
                AssertTreeStateForward(input);
        }
#endif
    }

public:
    // -----------------------------------------------------------------------
    // BatchedBackward() -- entry point for auto-batched implementation of PrimitiveFunction::Backward()
    // -----------------------------------------------------------------------

    // implant gradients into all variables
    // Unlike BatchedForward(), this is eager. If you call it twice, it's a completely new computation.
    // If you need multiple gradients, ask for them in a single go.
    void BatchedBackward(const Variable& root, unordered_map<Parameter, NDArrayViewPtr>& gradients)
    {
        if (!root.m_dataFields->m_needsGradient)
            logic_error("BatchedBackward: cannot compute gradient for root with m_needsGradient being False.");
        // BUGBUG: make sure some edge cases are done right:
        //  - root.m_needsGradient=false
        //  - gradients contains root
        //  - root is a m_lazyIndex
        // first get the forward computation, batching, etc. done if not yet
        BatchedForward(root);
        // set up the m_consumer fields, which BatchedBackward() will work off
        DetermineConsumersForBackward(root); // (gotta improve the name of these things)
        // implant the first gradient
        // TODO: allow user to pass in the starting value
        // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
        //if (root.Value()->Shape() != NDShape{})
        //    LogicError("BatchedBackward: root must be a scalar, or root gradient must have been implanted already");
        root.m_dataFields->m_gradient = arena.NewNDArrayView(root.Shape(), root.GetDataType(), root.Value()->Device());
        root.m_dataFields->m_gradient->SetValue(1.0f);
        // if user passed NDArrayViewPtrs for the gradients, then keep using them
        // This way, the same buffers can be recycled.
        for (auto& kv : gradients)
        {
            if (kv.second)
                kv.second->SetValue(0.0f); // BUGBUG: inefficient; better reset to 0 lazily
            kv.first.m_dataFields->m_gradient = kv.second; // (if null then these will be set inside)
        }
        // BUGBUG: how to reset m_pendingInputs when there is no gradient on that path?
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            let& fields = *param.m_dataFields;
            if (!fields.m_consumers.first.first) // if no consumer entry, we did not reach this gradient
                logic_error("BatchedBackward: a requested gradient is not part of root."); // TODO: or could it be due to StopGradient? What if StopGradient is used only sometimes?
            if (!fields.m_needsGradient) // (we could also just leafve the gradient 0)
                logic_error("BatchedBackward: cannot compute gradient for variable with m_needsGradient being False.");
            AggregateGradientFromAllConsumers(param);
        }
        //fprintf(stderr, "Back-propagated through %d functions\n", (int)order.size());
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
            kv.second = kv.first.m_dataFields->m_gradient;
        //AssertTreeStateForward(root); // (sanity check)  --TODO: gotta think this through e.g. nodes for which no gradient is requested
        // WORKAROUND for above. With this, we can at least com,pute more than 1 gradient fro a parameter
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            auto& fields = *param.m_dataFields;
            fields.m_consumers.first.first = nullptr;
            fields.m_consumers.second.clear();
        }
#ifdef LOG_STATS
        fprintf(stderr, "BatchedBackward: %d backprop computations executed in nominal %d post-batching ops\n",
                (int)stats.numBatchedBackpropToCalls, (int)stats.numBackpropsToInputs);
#endif
    }
}; // class

// ===========================================================================
// auto-batching entry points
// ===========================================================================

// this will become Variable::Value()
// Computes lazily the value of a node. Does nothing if called again.
NDArrayViewPtr PrimitiveFunction::BatchedForward() const
{
    auto autoBatcher = Variable::AutoBatch();
    return autoBatcher.BatchedForward(m_outputs[0]);
}

// Perform backprop.
// TODO: CNTK grad() allows to pass multiple roots. Does that ever make sense in this context?
void PrimitiveFunction::BatchedBackward(std::unordered_map<Parameter, NDArrayViewPtr>& gradients) const
{
    auto autoBatcher = Variable::AutoBatch(); // has some internal state
    autoBatcher.BatchedBackward(m_outputs[0], gradients);
}

// non-batched version of BatchedForward()
// This is actually not used except as a fallback for debugging.
void PrimitiveFunction::MemoizeKnowableValue() const
{
    if (m_outputs.size() != 1)
        LogicError("Variable '%S' Value(): Only Variables with one output can compute their Value for now.", AsString().c_str());
    const auto& output = m_outputs[0];
    if (output.m_dataFields->m_value) // already done
        return;
    // get the input values (recursively compute them if needed)
    if (m_inputs.empty())
        LogicError("Variable '%S' Value(): Only Variables with input arguments can compute their Value.", AsString().c_str());
    vector<NDArrayViewPtr> args(m_inputs.size());
    for (size_t i = 0; i < args.size(); i++)
        args[i] = m_inputs[i].Value();
    NDArrayViewPtr out;
    output.m_dataFields->m_value = move(ComputeKnowableValue(m_op, args, m_attributes, output.Shape(), move(out), *this));
}

} // namespace CNTK
