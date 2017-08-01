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

//#define LOG_DETAILS   // if defined, log all forward and backward operations
//#define LOG_STATS     // if defined, log statistics (#operations)
#define NO_BATCHED_FORWARD  // if defined, don't batch forward
//#define NO_BATCHED_BACKPROP // if defined, don't do batched backprop

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
    static NDArrayViewPtr s_currentArena;
    static size_t s_currentArenaUsed;
    static const size_t ARENASIZE = 64000000; // we allocate in this chunk size
public:
    // allocate an NDArrayView of a given shape, data type, and device
    // The returned memory region is a slice into a much larger NDArrayView; therefore,
    // this operation short-circuits CUDA and is very fast.
    // Sparse objects cannot be arena-allocated.
    NDArrayViewPtr NewNDArrayView(const NDShape& shape, const DataType& dataType, StorageFormat format, const DeviceDescriptor& device)
    {
        let numElements = shape.TotalSize();
        // if too large, or sparse, then plain alloc
        if (numElements > ARENASIZE || format != StorageFormat::Dense)
            return make_shared<NDArrayView>(dataType, format, shape, device);
        // If arena not large enough then waste its remainder and just allocate a fresh one.
        // This abandons the current m_arena. This will not cause a memory leak, however:
        // Since the slices into it that were returned before all hold a ref-count to that arena,
        // it will be deallocated automatically as soon the last slice goes away.
        if (!s_currentArena || numElements > (ARENASIZE - s_currentArenaUsed))
        {
            s_currentArena = make_shared<NDArrayView>(dataType, StorageFormat::Dense, NDShape{ ARENASIZE }, device);
            s_currentArenaUsed = 0;
        }
        vector<size_t> startOffset{ s_currentArenaUsed };
        vector<size_t> extent{ numElements };
        NDArrayViewPtr region = s_currentArena->SliceView(startOffset, extent);
        s_currentArenaUsed += numElements;
        return region->AsShape(shape);
    }
};

/*static*/ NDArrayViewPtr NDArrayViewArena::s_currentArena;
/*static*/ size_t NDArrayViewArena::s_currentArenaUsed = 0;

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
    size_t numBackpropGathers = 0;
    size_t numBackpropScatters = 0;
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

// ---------------------------------------------------------------------------
// VisitorTag -- helper for graph traversal
//  - call VisitorTag::Begin()
//  - in traversal: if (VisitorTag::Visited(node.m_visitedTag)) return;
// This does not nest!
// ---------------------------------------------------------------------------

class VisitorTag
{
    static size_t s_nextVisitTag; // unique id for a single non-nested visiting process
    size_t m_visitTag;
public:
    void Begin() // call this at start
    {
        // TODO: to make this thread-safe, use atomic increment
        m_visitTag = s_nextVisitTag++;
    }
    bool Visited(size_t& tag)
    {
        if (tag == m_visitTag)
            return true;
        tag = m_visitTag;
        return false;
    }
};
/*static*/ size_t VisitorTag::s_nextVisitTag = 1;

// ===========================================================================
// AutoBatch -- autobatching happening inside here
// The auto-batching related functions are grouped inside a class, since they
// share quite a bit of state.
// ===========================================================================

class Variable::AutoBatch
{
    NDArrayViewArena m_arena; // helper to allocate NDArrayViews as slices into very large NDArrayView objects
    RuntimeStatistics m_stats;
    VisitorTag m_visitorTag; // helper for managing tree traversal (non-nested)

    // buffers e.g. for building NDArrayViewPtr vectors. Kept as class members to avoid repeated memory allocations.
    vector<NDArrayViewPtr>     m_inputValuesBuffer;
    vector<NDArrayViewPtr>     m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBufferRaw;
    vector<size_t>             m_dimsBuffer;
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

    // predicate whether an op just passes through its input
    // This is used to decide whether we can short-circuit it in Unalias().
    static bool IsAlias(PrimitiveOpType op)
    {
        // if really needed, this can be done as a bit-test
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass ||
            op == PrimitiveOpType::NoOp ||
            op == PrimitiveOpType::BarrierOp;
    }

    // predicate whether an op's gradient is a no-op (just copies the output gradient)
    // These are short-circuited in backprop.
    static bool IsGradientCopyingOp(PrimitiveOpType op, size_t inputIndex)
    {
        // if really needed, this can be done as a bit-test
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass         ||
            op == PrimitiveOpType::NoOp         ||
            op == PrimitiveOpType::BarrierOp    ||
            //op == PrimitiveOpType::Reshape      ||
            op == PrimitiveOpType::Plus         ||
            (op == PrimitiveOpType::Plus && inputIndex == 0);
    }

    // see through no-ops, such as barrier, Pass, or StopGradient
    // Use this for ANY access to PrimitiveFunction::m_inputs EXCEPT when directly getting the shape.
    static const Variable& Unalias(const vector<Variable>& inputs, size_t index)
    {
        let& input = inputs[index];
        let& fields = *input.m_dataFields;
        // lazy index: not an alias
        if (fields.m_lazyIndex.first)
            return input;
        // does not have an owner: not an alias
        let f = fields.Owner();
        if (!f)
            return input;
        // op is not an alias
        if (!IsAlias(f->m_op))
            return input;
        // it is an alias: see right through
        return Unalias(f->m_inputs, 0); // (all aliases are unary functions)
    }

    // class to manage the set of ready operations (the schedule)
    class ReadyOps
    {
        NonOwningFunctionListBuilder m_viewOps;
        vector<NonOwningFunctionListBuilder> m_regularOps; // m_regularOps[] is a linked list
        NonOwningFunctionListBuilder m_barrierOps; // TODO: currently dead
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
            // priority must match (depending on barrier or not)
            if (a->m_priority != b->m_priority)
                return false;
            // some operations have variable number of arguments. Those cannot be batched, e.g. Splice().
            if (a->m_inputs.size() != b->m_inputs.size())
                return false;
            // all input dimensions must match (with exception of a few special cases)
            for (size_t i = 0; i < a->m_inputs.size(); i++)
            {
                // there are a few special cases
                if (op == PrimitiveOpType::Times && i == 0)
                {
                    // for Times, the first arg must be the same object, not just the same shape
                    // TODO: a special case is a dot product, which we can write as ReduceSum(ElementTimes(a,b))
                    //       This would require to rewrite the graph though; can we do that?
                    if (Unalias(a->m_inputs, i).m_dataFields != Unalias(b->m_inputs, i).m_dataFields)
                        return false;
                }
                else
                {
                    // shapes must match
                    if (a->m_inputs[i].Shape() != b->m_inputs[i].Shape())
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
                // BUGBUG: We never get here since we now see through barriers for efficiency...
                m_barrierOps.push_back(f);
            else if (IsViewOp(op))
                m_viewOps.push_back(f);
            else
            {
                // determine the priority. This is for Barriers after short-circuiting NoOps...
                // This is highly inefficient, always reaching through the Owner pointer. Needs a flag.
                int pri = 0; // normal
                for (let& input : f->m_inputs)
                {
                    if (input.IsOutput() && input.OutputOwner()->m_op == PrimitiveOpType::BarrierOp)
                    {
                        pri = -1;
                        break;
                    }
                }
                f->m_priority = pri;
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
                    // barrier is realized through priority
                    int diff = iter->front()->m_priority - best->front()->m_priority;
                    if (diff == 0)
                        diff = (int)iter->size() - (int)best->size();
                    if (diff > 0)
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
    // Caller must call m_visitorTag.Begin() first.
    void RInitForScheduling(const Variable& var)
    {
        auto& fields = *var.m_dataFields;
        // return if already visit
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;
        // some sanity checks
        if (fields.m_value)
            LogicError("RInitForScheduling() should not have been called on variables that already have a value.");
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            LogicError("Value() depends on Input or Placeholder, it is not knowable.");
        // initialize m_consumers chain
        fields.m_consumers.first.first = nullptr;
        fields.m_consumers.second.clear();
        // handle leaves
        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
        {
            if (!fields.m_value)
                var.Value(); // this initializes it
            if (!fields.m_value)
                LogicError("Parameter/Constant has no Value??");
            m_stats.numLeafNodes++;
            return;
        }
        // not a leaf
        auto& f = *fields.m_ownerFunction.lock();
        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = Unalias(inputs, i);
            auto& fields = *input.m_dataFields;
            // recursively traverse
            if (!fields.m_value)
            {
                RInitForScheduling(input);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                {
                    pendingInputs++;
                    // record ourselves as a consumer of the input
                    // Note that RInitForScheduling() will have reset this upon first visit of 'input'.
                    if (!fields.m_consumers.first.first) // optimized for main case of 1 consumer. No std::vector in that case.
                        fields.m_consumers.first = make_pair(&f, i); // note: we don't need i for forward; can optimize
                    else
                        fields.m_consumers.second.push_back(make_pair(&f, i));
                }
            }
            else
                m_stats.numLeafNodes++;
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
        m_stats.numOpNodes++;
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

    static void LogFunction(const PrimitiveFunction& f, const char* prefix = "", size_t markIndex = SIZE_MAX)
    {
        let& inputs = f.m_inputs;
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        auto uid = f.Uid();
        let& name = f.Name();
        if (!name.empty())
            uid = name + L":" + uid;
        fprintf(stderr, "%s%S%S = %S (", prefix, uid.c_str(), outputShape.AsString().c_str(), f.OpName().c_str());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = Unalias(inputs, i);
            let& fields = *input.m_dataFields;
            // little helper function to fix up variable names by removing _Output_0
            // TODO: Once we support >1 output, this needs a bit more code.
            let GetVarName = [](const Variable& input) -> wstring
            {
                auto uid = input.Uid();
                if (uid.size() > 9 && wcscmp(uid.c_str() + uid.size() - 9, L"_Output_0") == 0)
                    uid.resize(uid.size() - 9);
                let& name = input.IsOutput() ? input.Owner()->Name() : input.Name();
                if (!name.empty())
                    uid = name + L":" + uid;
                return uid;
            };
            if (fields.m_lazyIndex.first)
            {
                let& input1 = fields.m_lazyIndex.first->m_outputs[0];
                fprintf(stderr, "%s%s%S%S[%d]", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input1).c_str(), input1.Shape().AsString().c_str(), (int)fields.m_lazyIndex.second);
            }
            else
                fprintf(stderr, "%s%s%S%S", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input).c_str(), input.Shape().AsString().c_str());
            if (i == 4 && inputs.size() > 5) // skip the middle ones
            {
                fprintf(stderr, ", ...+%d", (int)(inputs.size() - 5));
                i = inputs.size() - 2;
            }
        }
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
            inputValues[i] = LazilyIndexedValue(Unalias(inputs, i)); // (if this is a lazy slice, then now we must resolve it)\
        // allocate the output NDArrayViewPtr in the arena
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        // logging
#ifdef LOG_DETAILS
        LogFunction(f, "[bf] ");
#endif
        auto outValue = isFree
            ? nullptr
            : m_arena.NewNDArrayView(outputShape, inputValues[0]->GetDataType(), inputValues[0]->GetStorageFormat(), inputValues[0]->Device());
        // execute it
        output.m_dataFields->m_value = move(PrimitiveFunction::ComputeKnowableValue(f.m_op, inputValues, f.Attributes(), outputShape, move(outValue), f));
        // stats
        let primitiveOp = f.m_op;
        if (primitiveOp == PrimitiveOpType::StopGradient ||
            primitiveOp == PrimitiveOpType::Pass ||
            primitiveOp == PrimitiveOpType::NoOp ||
            primitiveOp == PrimitiveOpType::Reshape ||
            primitiveOp == PrimitiveOpType::Slice)
            m_stats.numDoneFreeOps++;
        else if (primitiveOp == PrimitiveOpType::Splice)
            m_stats.numDoneSpliceOps++;
        else
            m_stats.numDoneOtherOps++;
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
        let isTimes = (op == PrimitiveOpType::Times || op == PrimitiveOpType::TransposeTimes); // is special-cased
#ifdef NO_BATCHED_FORWARD
        auto doNaively = true;
#else
        let doNaively =
            isFree ||
            //(isTimes && f0.m_inputs[1].m_dataFields->m_value && f0.m_inputs[1].m_dataFields->m_value->IsSparse()) || // can't batch sparse
            op == PrimitiveOpType::Splice ||
            batchSize == 1;
#endif
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
        // and we will hack its Unalias(m_inputs,i).m_outputComposite to hold a strong reference to the Splice() or Slice().
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
                m_batchedInputs[0] = Unalias(f0.m_inputs, 0);
            for (size_t i = i0; i < numArgs; i++)
            {
                // create splice args for this argument
                // allocate buffers to hold the arguments
                auto& spliceInputs = m_spliceArgsBuffer; // TODO rename to gatherInputs and m_gatherArgsBuffer
                assert(spliceInputs.empty()); // previous use must have cleared it
                if (spliceInputs.capacity() < batchSize)
                    spliceInputs.reserve(max(batchSize, 2 * spliceInputs.capacity()));
                // optimization: if all args are consecutive slices, then use a slice view instead
                let* pfields0 = Unalias(f0.m_inputs, i).m_dataFields.get();
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
                    let& input = Unalias(op->m_inputs, i);
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
        // mark all nodes w.r.t. how many inputs they are waiting for before being computable
        if (!fields.m_value)
        {
            // prepare and schedule first set
            m_visitorTag.Begin();
            RInitForScheduling(v);
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
        LazilyIndexedValue(v);
#ifdef LOG_STATS
        fprintf(stderr, "BatchedForward: %d forward ops executed besides %d splices and %d views, in nominally %d PrimitiveFunctions on %d known values\n",
                (int)m_stats.numDoneOtherOps, (int)m_stats.numDoneSpliceOps, (int)m_stats.numDoneFreeOps, (int)m_stats.numOpNodes, (int)m_stats.numLeafNodes);
        size_t numOpNodes1 = 0;
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr&) { numOpNodes1++; });
        if (numOpNodes1 != m_stats.numOpNodes)
            fprintf(stderr, "BatchedForward: short-circuited %d aliases\n", (int)(numOpNodes1 - m_stats.numOpNodes));
#endif
        fail_if(fields.m_value->IsSparse() != fields.m_isSparse, "NDArrayView::m_sparse mismatches VariableFields::m_sparse??");
        return fields.m_value;
    }

    // =======================================================================
    // backward-related functions
    // =======================================================================

    static StorageFormat DetermineGradientStorageType(const PrimitiveFunction& f, size_t index)
    {
        //if (f.m_op == PrimitiveOpType::Times && f.m_inputs[1].Shape()[0] == 2000)
        //    fprintf(stderr, "%S --> %d\n", f.m_inputs[1].Shape().AsString().c_str(), (int)f.m_inputs[1].IsSparse());
        // Special case for DENSE * SPARSE -> DENSE, which leads to a SPARSE gradient for input0 (common for embedding).
        if (f.m_op == PrimitiveOpType::Times && index == 0 && f.m_inputs[1].IsSparse())
            return StorageFormat::SparseBlockCol;
        else
            return StorageFormat::Dense;
    }

    // allocate memory for m_gradient
    // This lazily creates the m_gradient NDArrayView, which may live in a batched op.
    // Returns beta = 0 if gradient was newly created, otherwise 1
    double LazilyCreateLazilyIndexedGradient(const Variable& v, StorageFormat format = StorageFormat::Dense)
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
                fields.m_gradient = m_arena.NewNDArrayView(fields.m_shape, fields.m_dataType, format, fields.m_value->Device());
                beta = 0.0; // has not been initialized (random section in arena)
            }
        }
        return beta;
    }

    // determine the input to backprop a gradient into
    // Normally that is Unalias(f.m_inputs, index).
    // However, some gradients are just copies, so we can see through them.
    // This saves memory and allows easier discovery of optimizable patterns.
    // Note that every single time one iterates over m_inputs for gradients, this function must be used.
    // Note: THIS IS NOT LEVERAGED YET, and could be removed if we don't leverage it.
    const Variable& GetShortCircuitedGradientInput(const PrimitiveFunction& f, size_t index)
    {
        let& input = Unalias(f.m_inputs, index);
#ifndef NO_BATCHED_BACKPROP
        // if the input is a result of a batched operation, then traverse into that instead
        if (input.m_dataFields->m_lazyIndex.first)
            return input.m_dataFields->m_lazyIndex.first->m_outputs[0];
#endif
        return input;
    }

    // recursively traverse the tree hanging off a Variable and build the m_consumer fields
    // This propagates in depth-first order like a naive backprop, but only for the purpose
    // of recording consumers of each node.
    // Unlike forward prop, we...
    //  - can skip any branch that does not need a gradient (!m_needsGradient and StopGradient ops).
    //  - short-circuit into batched ops (m_lazyIndex) so that we backprop through them instead
    // All nodes that were traversed have all input's m_consumers set up and their m_pendingInputs set to 0.
    // Caller must call m_visitorTag.Begin() first.
    void RDetermineConsumersForBackward(const Variable& var)
    {
        auto& fields = *var.m_dataFields;

        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
            return; // reached a leaf

        fail_if(!fields.m_value, "variable has no value yet??");
        fail_if(!fields.m_needsGradient, "unexpectedly encountered a node with m_needsGradient=false??");
        fail_if(fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder, "unexpectedly encountered an Input or a Placeholder??");

        // Determine the function that 'var' is an output of.
        // If 'var' has the m_lazyIndex field set, it means that its value was not
        // actually computed from its true owner function; but rather a slice into the result of
        // a batched operation. In that case, we traverse through that batched operation
        // instead. As a consequence, it is the batched operation that will be recorded as the
        // consumer of all inputs of the batched operation, rather than the original
        // unbatched operation. And as a consequence of that, back propagation will use
        // the same batching that was determined in forward computation.
        auto& outFields = *var.m_dataFields;
#ifndef NO_BATCHED_BACKPROP
        auto&f = outFields.m_lazyIndex.first       // if var was computed via batched op
              ? *outFields.m_lazyIndex.first       // then backprop into the batched op
              : *outFields.m_ownerFunction.lock(); // otherwise traverse the original op
#else
        auto&f = outFields.m_ownerFunction.lock();
#endif
        // return if we already visited the function
        if (m_visitorTag.Visited(f.m_visitedTag))
            return;

        fail_if(f.m_op == PrimitiveOpType::StopGradient, "unexpectedly encountered a StopGradient, which should have propagated m_needsGradient=false upwards");

        // recurse and set up the consumer list
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
#if 1
            // Note: Using this function may be misleading.
            // We are not short-circuiting the input here, but rather determine which function
            // it came from.
            let& input = GetShortCircuitedGradientInput(f, i);
#else
            let* inputp = &Unalias(inputs, i);
#ifndef NO_BATCHED_BACKPROP
            // if the input is a result of a batched operation, then traverse into that instead
            if (inputp->m_dataFields->m_lazyIndex.first)
                inputp = &inputp->m_dataFields->m_lazyIndex.first->m_outputs[0];
#endif
            let& input = *inputp;
#endif
            auto& fields = *input.m_dataFields;
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
                RDetermineConsumersForBackward(input);
            }
            else
                fields.m_consumers.second.push_back(make_pair(&f, i));
            m_stats.numBackpropsToInputs++;
        }
        f.m_pendingInputs = 0; // used as a visited flag
    }

    // helper to batch an array of NDArrayViews of the same rank along either the last or into a new axis
    // TODO: do this with a lambda so we can go straight into gatherBatchResultDims
    NDArrayViewPtr GatherBatchInArena(const vector<NDArrayViewPtr>& inputValues, size_t axis, size_t batchDim)
    {
        let& inputValue0 = *inputValues[0];
        let& inputShape = inputValue0.Shape().Dimensions();
        auto& gatherBatchResultDims = BorrowBuffer(m_dimsBuffer, inputShape.size()+1);
        gatherBatchResultDims.assign(inputShape.begin(), inputShape.end());
        fail_if(axis + 1 < gatherBatchResultDims.size(), "axis not trailing??");
        if (axis == gatherBatchResultDims.size())
            gatherBatchResultDims.push_back(inputValues.size());
        else
            gatherBatchResultDims[axis] = batchDim;
        auto out = m_arena.NewNDArrayView(gatherBatchResultDims, inputValue0.GetDataType(), inputValue0.GetStorageFormat(), inputValue0.Device());
        m_stats.numBackpropGathers++;
        return move(NDArrayView::GatherBatch(inputValues, (int)axis, move(out)));
    }

    // back-propagate f's outputs' m_gradient to a specified input
    // This is the standard path for all unbatched ops.
    // This wraps the PrimitiveFunction's BackpropTo(), interfacing from vectors of Variable to vectors of NDArrayViewPtr.
    // Note that each input that is lazy should redirect into a slice in its lazy source.
    void BackpropToUnbatched(PrimitiveFunction* f, size_t index)
    {
#ifdef LOG_DETAILS
        LogFunction(*f, "[bb] ", index);
#endif
        let& inputs =  f->m_inputs;
        auto& input = Unalias(inputs, index);
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
            let& input1 = Unalias(inputs, i);
            let& fields = *input1.m_dataFields;
            fail_if(!fields.m_value, "unexpectedly ran into a function that has no m_value yet??");
            inputValues[i] = fields.m_value.get();
        }

        // compute gradients for the desired input
        // Get or create m_gradient as the desired gradient's TensorView.
        // If the input is a lazyIndex, then the gradient is a view into the lazy source.
        let beta = LazilyCreateLazilyIndexedGradient(input, DetermineGradientStorageType(*f, 0));
        // backprop into the input
        PrimitiveFunction::BackpropTo(outputGradient/*incoming*/, index, f->m_op, f->m_attributes, outputValue, inputValues, fields.m_gradient/*target*/, beta, *f);
        m_stats.numBatchedBackpropToCalls++;
    }

    // backprop into an input of a splice operation
    // This is done as a single CUDA launch into all inputs.
    // This is a little tricky. We backprop into all inputs.
    // So we must make sure we run this only once.
    // The first time this gradient gets puled, we do it for all inputs
    // and remember that this has been done.
    void BackpropToSplice(PrimitiveFunction* f)
    {
        // if we pull this a second time, then don't propagate again
        if (m_visitorTag.Visited(f->m_visitedTag))
            return;
        // fast path: only one input (and not Splice, which we do in bulk)
        if (f->m_inputs.size() == 1)
            return BackpropToUnbatched(f, 0);
        // Considerations:
        //  - For now we do optimize for consecutive inputs, because those would not
        //    have been transformed into a Splice operation in auto-batching.
        //    User's batch ops go here as well; we won't optimize for them for now.
        //    Hence, this is strictly a ScatterBatch operation.
        //  - It is possible that the Splice operation consumed the same input twice.
        //    This is currently handled via atomicAdd(), i.e. will have non-determinism.
        //    (A striding trick minimizes the probability of clashes.)
#if 0
        for (size_t index = 0; index < f->m_inputs.size(); index++)
        {
            BackpropToUnbatched(f, index);
            m_stats.numBatchedBackpropToCalls--; // for now fake the call count to see the potentail impact
        }
#else
#ifdef LOG_DETAILS
        LogFunction(*f, "[bb#] ", SIZE_MAX);
#endif
        // The gradient of Splice is just copying all columns to the respective inputs.
        let& inputs =  f->m_inputs;
        // get the TensorViews for everything we may compute the gradient from
        let& output = f->m_outputs[0];
        let& outputFields = *output.m_dataFields;
#ifndef NO_BATCHED_BACKPROP
        fail_if(outputFields.m_lazyIndex.first, "unexpectedly ran into a function that does not own its output"); // we don't backprop through unbatched ops
#endif
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");
        let& outputGradient = outputFields.m_gradient; // this is the incoming batch of gradients

        let numInputs = inputs.size();
        auto& inputGradients = BorrowBuffer(m_inputValuesBuffer, numInputs); // target locations to propagate the columns to
        bool allBetasZero = true;
        for (size_t i = 0; i < numInputs; i++)
        {
            let& input = Unalias(inputs, i);
            // create the gradient memory for this input
            let beta = LazilyCreateLazilyIndexedGradient(input);
            let& fields = *input.m_dataFields;
            inputGradients[i] = fields.m_gradient;
            // handle inconsistent betas
            if (beta != 0 && allBetasZero)
            {
                // We were running under the assumption that all betas are zero, so we can use beta=0 below.
                // Now we must run with beta 1, and therefore manually reset all pevious ones.
                for (size_t i1 = 0; i1 < i; i1++) // these were all beta=0
                    Unalias(inputs, i1).m_dataFields->m_gradient->SetValue(0.0f);
                allBetasZero = false;
            }
            else if (beta == 0 && !allBetasZero)
                fields.m_gradient->SetValue(0.0f);
        }
        let beta = allBetasZero ? 0.0 : 1.0; // if at least one is not zero, we must run qwith beta=1

        // backprop into all inputs
        NDArrayView::ScatterBatch(outputGradient, inputGradients, beta);
#endif
        m_stats.numBackpropScatters++;
    }

    // backprop into weight parameter of a Times op (Unalias(inputs, 0))
    // This can be batched into a single matrix product.
    void BackpropToMatrixWeight(vector<pair<PrimitiveFunction*, size_t>>& consumers)
    {
#if 0
        for (auto& c : consumers)
            BackpropToUnbatched(c.first, c.second);
#else
        // We compute
        //  leftGrad += sum_i outGrad_i @ right_i^T
        //            = (concat_i outGrad_i) @ (concat_i right_i)^T
        // where concat_i means to concatenate matrices along their trailing (batch) axis.
        // It has already been verified that all i have the same rank and dimensions except for a single reduction dimension.

        // batch all outGrads, and batch all right inputs
        let numBatchItems = m_matrixWeightConsumers.size();
        auto& timesOutGrads        = BorrowBuffer(m_inputValuesBuffer,     numBatchItems);
        auto& timesDataRightInputs = BorrowBuffer(m_outputGradientsBuffer, numBatchItems);
        let& f0 = *m_matrixWeightConsumers.front().first;
#ifdef LOG_DETAILS
        LogFunction(f0, "[bb*] ", 0);
#endif
        let& input0 = Unalias(f0.m_inputs, 0);
        size_t batchDim = 0;
        for (size_t i = 0; i < numBatchItems; i++)
        {
            let &c = m_matrixWeightConsumers[i];
            fail_if(c.second != 0, "wrong input??");
            let& outGrad = c.first->m_outputs[0].m_dataFields->m_gradient;
            let& right = Unalias(c.first->m_inputs, 1).m_dataFields->m_value;
            timesOutGrads       [i] = outGrad;
            timesDataRightInputs[i] = right;
            let numItems = outGrad->Shape().Dimensions().back();
            fail_if(numItems != right->Shape().Dimensions().back(), "batch dimension of two inputs not the same??");
            batchDim += numItems;
        }
        auto outGradBatch = GatherBatchInArena(timesOutGrads       , f0.m_outputs[0].Shape().Rank() - 1, batchDim);
        auto rightBatch   = GatherBatchInArena(timesDataRightInputs, f0.m_inputs[1]. Shape().Rank() - 1, batchDim);

        // backprop into the left input from the batched outGrad and right
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, 2);
        inputValues[0] = nullptr;
        inputValues[1] = rightBatch.get();
        let beta = LazilyCreateLazilyIndexedGradient(input0, DetermineGradientStorageType(f0, 0));
        PrimitiveFunction::BackpropTo(/*outputGradient=*/outGradBatch.get(),      // incoming gradient from top...
                                      /*index=*/0, f0.m_op, f0.m_attributes,      // ...goes through this function...
                                      /*outputValue=*/nullptr, inputValues,       // ...using these values from forward pass...
                                      input0.m_dataFields->m_gradient, beta, f0); // ...into here
        m_stats.numBatchedBackpropToCalls++;
#endif
    }

    // backprop gradient into 'var' by pulling all of its consumers (recursively)
    // This is the second function that does batching.
    // The vectors for building the lists are class members so that we reuse the malloc.
    // This is a subroutine of RAggregateGradientFromAllConsumers().
    vector<pair<PrimitiveFunction*, size_t>> m_spliceConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_matrixWeightConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_summandConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_otherConsumers;
    void DetermineAndAddToBucket (const pair<PrimitiveFunction*, size_t>& c, bool isFirstCall = false)
    {
        if (isFirstCall) // first time pass true here
        {
            m_spliceConsumers      .clear();
            m_matrixWeightConsumers.clear();
            m_summandConsumers     .clear();
            m_otherConsumers       .clear();
        }
        let* f = c.first;
        let index = c.second;
        fail_if(f->m_outputs.size() != 1, "for now only functions with a single output are supported"); // (needs some more plumbing to fix this)
        // backprop into Times' matrix argument
        // BUGBUG: This currently does not capture single time steps that backprop into the same matrix as a batch.
        let IsMatrixGradient0Batchable = [](const PrimitiveFunction& f, const PrimitiveFunction& g) -> bool
        {
#if 0       // use 1 to disable batching of matrix gradients
            return false;
#else
            // we compute leftGrad += outGrad @ right^T
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
            fail_if(leftShape != g.m_inputs[0].Shape().Dimensions(), "dimensions of matrix gradient don't match??");
            if (outRank != gOutRank || rightRank != gRightRank)
                return false; // rank not matching: stop batching right here (we could do better)
            // the center 'reductionRank' dimensions get reduced over
            if (outRank + rightRank - leftRank != 2) // if 2 then we reduce over a single batch axis
                return false; // this is not a batch gradient; back out
            fail_if(fOutShape.back() != fRightShape.back() || gOutShape.back() != gRightShape.back(), "inner dimensions of matrix gradient don't match??");
            // the two gradient ops match if all dimensions except for the batch dim match
            for (size_t k = 0; k < outRank - 1; k++) // check outGrad
                if (fOutShape[k] != gOutShape[k])
                    return false;
            for (size_t k = 0; k < rightRank - 1; k++) // check right
                if (fRightShape[k] != gRightShape[k])
                    return false;
            // gradient is batchable
            return true;
#endif
        };
#ifndef NO_BATCHED_FORWARD
        // splice operation must use scatter
        if (f->m_op == PrimitiveOpType::Splice)
            m_spliceConsumers.push_back(c);
        // matrix product
        // We only collect matrix products with fully matching dimensions.
        else if (f->m_op == PrimitiveOpType::Times && index == 0 &&
            (m_matrixWeightConsumers.empty() || (IsMatrixGradient0Batchable(*f, *m_matrixWeightConsumers.back().first))))
            m_matrixWeightConsumers.push_back(c);
        // backprop into either of Plus' arguments
        //else if (f->m_op == PrimitiveOpType::Plus)
        //    return m_summandConsumers;
        // all other
        else
#endif
            m_otherConsumers.push_back(c);
    };

    // compute a variable's outputs' gradient (var.m_gradient)
    // This operates on the PrimitiveFunction(s) that use this var's output value--its "consumers".
    // A variable knows all of its consumers. This function back-propagates from all consumers
    // into the variable's m_gradient field.
    // A consumer is a specific input of a PrimitiveFunction, specified as (function pointer, input index).
    // In the case of multiple consumers, the gradient is the sum.
    // This recursively traverses the graph upwards via the consumers chain.
    // Caller must call m_visitorTag.Begin() first.
    // This uses Variable::m_visitedTag for traversing the consumer tree upwards.
    // It uses PrimitiveFunction::m_visitedTag for bulk gradient of currently the
    // Splice op, which produces all inputs' gradients in a single go and must therefore only run once.
    __declspec(noinline) void RAggregateGradientFromAllConsumers(const Variable& var)
    {
        let& fields = *var.m_dataFields;
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;

        auto& c = fields.m_consumers.first;
        // reached a "leaf" in the revesre tree; i.e. we hit the root
        if (!c.first)
            return;

        fail_if(!fields.m_needsGradient, "backprop into variable that does not need gradient");

        // recursively realize all consumers' outputs' gradients
        for (let& output : c.first->m_outputs)
            RAggregateGradientFromAllConsumers(output);
        for (auto& c : fields.m_consumers.second)
            for (let& output : c.first->m_outputs)
                RAggregateGradientFromAllConsumers(output);
        // Now all consumers are ready to propagate into var's m_gradient.
        // The resulting gradient is the sum of all that's backpropped here,
        // and this is the only place where a variable's gradient ever gets aggregated.

        // create var's m_gradient (may be a slice view)
        // m_gradient may already exist for Parameters, and when it came through Splice.
        // Because of the latter, we cannot really test this here, and should just remove this check.
        //fail_if(var.Kind() != VariableKind::Parameter && fields.m_gradient, "non-Parameter variable unexpectedly already has a gradient"); // (sanity check; I am not sure actually, maybe too strict)

        // fast path: only one consumer, nothing to batch
        // (with exception of Splice, which has a different optimization)
        if (fields.m_consumers.second.empty() && fields.m_consumers.first.first->m_op != PrimitiveOpType::Splice)
        {
            BackpropToUnbatched(c.first, c.second);
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
        DetermineAndAddToBucket(c, /*isFirstCall=*/true);
        for (auto& c : fields.m_consumers.second)
            DetermineAndAddToBucket(c);

        // splice bucket
        for (auto& c : m_spliceConsumers)
            BackpropToSplice(c.first);

        // matrix-weight bucket
        if (!m_matrixWeightConsumers.empty())
            BackpropToMatrixWeight(m_matrixWeightConsumers);

        // summation bucket
        // ...

        // others bucket
        for (auto& c : m_otherConsumers)
            BackpropToUnbatched(c.first, c.second);
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
            LogicError("BatchedBackward: cannot compute gradient for root with m_needsGradient being False.");
        // BUGBUG: make sure some edge cases are done right:
        //  - root.m_needsGradient=false
        //  - gradients contains root
        //  - root is a m_lazyIndex
        // first get the forward computation, batching, etc. done if not yet
        BatchedForward(root);
        // set up the m_consumer fields, which BatchedBackward() will work off
        m_visitorTag.Begin();
        RDetermineConsumersForBackward(root); // (gotta improve the name of these things)
        // implant the first gradient
        // TODO: allow user to pass in the starting value
        // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
        //if (root.Value()->Shape() != NDShape{})
        //    LogicError("BatchedBackward: root must be a scalar, or root gradient must have been implanted already");
        root.m_dataFields->m_gradient = m_arena.NewNDArrayView(root.Shape(), root.GetDataType(), StorageFormat::Dense, root.Value()->Device());
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
        m_visitorTag.Begin();
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            let& fields = *param.m_dataFields;
            if (!fields.m_consumers.first.first) // if no consumer entry, we did not reach this gradient
                LogicError("BatchedBackward: a requested gradient is not part of root."); // TODO: or could it be due to StopGradient? What if StopGradient is used only sometimes?
            if (!fields.m_needsGradient) // (we could also just leafve the gradient 0)
                LogicError("BatchedBackward: cannot compute gradient for variable with m_needsGradient being False.");
            RAggregateGradientFromAllConsumers(param);
        }
        //fprintf(stderr, "Back-propagated through %d functions\n", (int)order.size());
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
            kv.second = kv.first.m_dataFields->m_gradient;
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            auto& fields = *param.m_dataFields;
            fields.m_consumers.first.first = nullptr;
            fields.m_consumers.second.clear();
        }
#ifdef LOG_STATS
        fprintf(stderr, "BatchedBackward: %d backprop computations besides %d gathers and %d scatters executed in nominal %d post-batching ops\n",
                (int)m_stats.numBatchedBackpropToCalls, (int)m_stats.numBackpropGathers, (int)m_stats.numBackpropScatters, (int)m_stats.numBackpropsToInputs);
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
