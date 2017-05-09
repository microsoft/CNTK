//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"

#include <unordered_map>

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#pragma warning (disable: 4456) // until I fixed the shdowing

#define let const auto

using namespace std;

#define BreakPoint fprintf(stderr, "") // use this inside a conditional to be able to set a breakpoint in Release code

namespace CNTK
{
class Memoize
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

    class NonOwningFunctionList // over Function, using m_link
    {
    protected:
        Function* head;
        size_t count; // note: count is only in here for diagnostics; only needed in builder
    public:
        NonOwningFunctionList() { clear(); }
        NonOwningFunctionList(Function* f) : head(f), count(1) { }
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
        Function* front() const { return head; }
        bool empty() const { return !head; }
        size_t size() const { return count; }
        void clear()
        {
            head = nullptr;
            count = 0;
        }
        class FunctionListIterator
        {
            Function* iter;
        public:
            FunctionListIterator(Function* f) : iter(f) { }
            Function* operator->() const { return iter; }
            Function* operator++() { iter = iter->m_link; return iter; }
            bool operator!=(const FunctionListIterator& other) { return iter != other.iter; }
        };
        FunctionListIterator begin() const { return front(); }
        FunctionListIterator end()   const { return nullptr; }
    };
    class NonOwningFunctionListBuilder : public NonOwningFunctionList // over Function, using m_link
    {
        Function* tail; // note: value undefined when list empty
    public:
        NonOwningFunctionListBuilder() : NonOwningFunctionList() { }
        NonOwningFunctionListBuilder(Function* f) : NonOwningFunctionList(f), tail(f) { f->m_link = nullptr; }
        void append(Function* f)
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

    // class to manage the set of ready operations (the schedule)
    class ReadyOps
    {
        NonOwningFunctionListBuilder m_viewOps;
        vector<NonOwningFunctionListBuilder> m_regularOps; // m_regularOps[] is a linked list
        NonOwningFunctionListBuilder m_barrierOps;
        // TODO: This must be turned into something hashable.
        // test whether two PrimitiveFunctions can be executed as a single batched operation
        static bool AreBatchable(const Function* a, const Function* b)
        {
            // first it must be the same operation
            let op = a->Op();
            // free ops always get batched; even if they have different op-codes
            if (IsViewOp(op) && op != PrimitiveOpType::BarrierOp)
                throw logic_error("should not get here for view ops or barrier ops");
            // op codes must match
            if (op != b->Op())
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
        void Schedule(Function* f)
        {
            let op = f->Op();
            // we manage three ready sets, since two common kinds are very simple
            if (op == PrimitiveOpType::BarrierOp)
                m_barrierOps.append(f);
            else if (IsViewOp(op))
                m_viewOps.append(f);
            else
            {
                // this naive implementation just scans linearly
                // scan through all op sets to see if one is batchable with 'f'
                for (auto iter = m_regularOps.begin(); iter != m_regularOps.end(); iter++)
                {
                    if (AreBatchable(f, iter->front()))
                    {
                        iter->append(f);
                        return;
                    }
                }
                // none fit: open a new set
                m_regularOps.push_back(NonOwningFunctionListBuilder(f));
            }
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
    // This function also sets up the m_notify fields. When it returns, they are all complete.
    void TraverseFunctionTreeForward(const Variable& var)
    {
        let& fields = *var.m_dataFields;
        if (fields.m_value)
            throw logic_error("TraverseFunctionTreeForward() should not have been called on variables that already have a value.");
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            throw logic_error("Value() depends on Input or Placeholder, it is not knowable.");
        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
        {
            var.Value(); // this initializes it
            if (!fields.m_value)
                throw logic_error("Parameter/Constant has no Value??");
            return;
        }
        auto& f = *fields.m_ownerFunction.lock();
        if (f.m_pendingInputs == -2 || fields.m_gradient) // (-2 means we've already run gradient; therefore we must have a value and should not get here)
            throw logic_error("TraverseFunctionTreeForward() should not have been called on variables that already have a gradient.");
        if (f.m_pendingInputs != -1) // already visited
            return;
        // determine how many inputs are pending
        // and also recurse
        // and also set up the consumer list
        size_t pendingInputs = 0;
        for (let& v : f.m_inputs)
        {
            let& fields = *v.m_dataFields;
            // recursively traverse
            if (!fields.m_value)
            {
                TraverseFunctionTreeForward(v);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                {
                    // record ourselves as a consumer of the input--this is used for batching and for Backward()
                    auto fi = fields.m_ownerFunction.lock();
                    if (!fi->m_notify1) // optimized for main case of 1 consumer. No std::vector in that case.
                        fi->m_notify1 = &f;
                    else
                        fi->m_notifyN.push_back(&f);
                    // 
                    pendingInputs++;
                }
            }
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
    }

    //NDArrayViewPtr m_currentArena;
    //size_t m_currentArenaUsed;
    static const size_t ARENASIZE = 64000000; // we allocate in this chunk size

    NDArrayViewPtr Alloc(const NDShape& shape, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
    {
        static NDArrayViewPtr m_currentArena;
        static size_t m_currentArenaUsed;
        let numElements = shape.TotalSize();
        // if too large then plain alloc
        if (numElements > ARENASIZE)
            return make_shared<NDArrayView>(dataType, CNTK::StorageFormat::Dense, shape, device);
        // if arena not large enough then waste its remainder and just allocate a fresh one
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

    // return the m_value field of a variable, but possibly realizing it lazily if it is an index operation
    const NDArrayViewPtr& LazilyIndexedValue(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        if (!fields.m_value)
        {
            let& from = fields.m_lazyIndex.first;
            if (!from)
                throw logic_error("variable unexpectedly has no value yet");
            fields.m_value = from->IndexLastAxis(fields.m_lazyIndex.second);
        }
        return fields.m_value;
    }

    // temp variables for ExecuteBatchedOpAndUpdateSchedule(); keep outside to reuse the memory allocation
    vector<NDArrayViewPtr> m_args;
    vector<NDArrayViewPtr> m_spliceArgsBuffer;
    size_t m_numBatchedLaunches = 0; // (for statistics only)

    // batch-execute a set of ops that are known to be batchable
    void ExecuteBatchedOpAndUpdateSchedule(NonOwningFunctionList ops) // (note: NonOwningFunctionListBuilder is so small that it is best copied)
    {
        // TODO: need to handle ops that have >1 output, such as Combine(). Just don't batch them ever? Combine() is just a see-through anyway.
        // get a representative op
        auto& f0 = *ops.front();
        let op = f0.Op();
        let batchSize = ops.size();
        let isFree = IsViewOp(op);
        if (!isFree)
            m_numBatchedLaunches++;
        let numArgs = f0.m_inputs.size();
        //m_inputs.resize(numArgs);
        m_args.resize(numArgs);
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
            if (op == PrimitiveOpType::Splice && batchSize != 1)
                BreakPoint;
            // for correctness testing of underlying mechanism, compute them without actual batching
            for (auto op = ops.begin(); op != ops.end(); ++op)
            //for (auto op : ops) // TODO: figure this out
            {
                if (op->m_outputs.size() != 1)
                    throw logic_error("only functions with 1 output are supported");
                //m_inputs.resize(op->m_inputs.size());
                m_args.resize(op->m_inputs.size());
                for (size_t i = 0; i < op->m_inputs.size(); i++)
                    m_args[i] = LazilyIndexedValue(op->m_inputs[i]);
                NDArrayViewPtr out = isFree ? nullptr : Alloc(op->m_outputs[0].Shape(), m_args[0]->GetDataType(), m_args[0]->Device()); // arena allocation will happen here
                op->m_outputs[0].m_dataFields->m_value =
                    op->ComputeKnowableValue(op->Op(), m_args, op->Attributes(), op->m_outputs[0].Shape(), move(out));
                // TODO: realize splice ops that are index ops as a m_lazyIndex at this point
                //if (f0.Op() == PrimitiveOpType::Slice)
                //{
                //    // inject the lazyIndex
                //}
#if 0           // test effect of the unbatched sparse times
                if (f0.Op() == PrimitiveOpType::Times)
                    op->ComputeKnowableValue(op->Op(), m_args, op->Attributes(), op->m_outputs[0].Shape(), move(out));
#endif
            }
        }
        else
        {
            // batch all arguments
            size_t maxRank = 0;
            size_t i0 = isTimes ? 1 : 0;
            if (isTimes)
                BreakPoint;
            for (size_t i = i0; i < numArgs; i++)
            {
                // we could even do with un-managed pointers here; would save lots of ref-counting; or even use GatherBatch lambda
                // determine max rank
                let rank = f0.m_inputs[i].Shape().Rank();
                if (rank > maxRank)
                    maxRank = rank;
            }
            bool anyBatchedInputs = false;
            if (isTimes)
                m_args[0] = LazilyIndexedValue(f0.m_inputs[0]); // (should not have to do anything lazy, actually)
            for (size_t i = i0; i < numArgs; i++)
            {
                // create splice args
                // allocate buffers
                auto& spliceArgs = m_spliceArgsBuffer;
                spliceArgs.clear();
                if (spliceArgs.capacity() < batchSize)
                    spliceArgs.reserve(max(batchSize, 2 * spliceArgs.capacity()));
                // optimization: if all args are consecutive slices, then use a slice view instead
                let* pfields0 = f0.m_inputs[i].m_dataFields.get(); // as long as non-NULL, we are consecutive
                if (!pfields0->m_lazyIndex.first) // only if it is an indexed slice, actually
                    pfields0 = nullptr;
                // loop over all batched ops
                size_t j = 0;
                for (auto op = ops.begin(); op != ops.end(); ++op, j++) // create the batched tensors
                {
                    // optimization: if all args are consecutive slices, then use a slice view instead
                    if (pfields0)
                    {
                        let* pfields = op->m_inputs[i].m_dataFields.get();
                        // we continue to be in optimized state
                        if (pfields->m_lazyIndex.first  == pfields0->m_lazyIndex.first &&
                            pfields->m_lazyIndex.second == pfields0->m_lazyIndex.second + j)
                            continue;
                        // nope, chain has been broken: must fix up spliceArgs up to here
                        // This is suboptimal in that we lost the reference to the originating input.
                        // So we cannot remember the realized SliceView there.
                        // TODO: Per Jon's suggestion, we can be a little loose here. For a variable-length
                        // scenario, we will loose entries in the middle. We can allow to keep a few around
                        // in garbage-in-garbage-out. If, say, there are additional 20% gap values, we just
                        // carry them forward, and ignore them when implanting the result.
                        let& from = pfields0->m_lazyIndex.first;
                        let begin = pfields0->m_lazyIndex.second;
                        while (spliceArgs.size() < j)
                            spliceArgs.push_back(from->IndexLastAxis(begin + spliceArgs.size()));
                        pfields0 = nullptr; // no longer in consecutive hypothesis
                    }
                    let& arg = LazilyIndexedValue(op->m_inputs[i]);
                    // optimization: if all args are the same, then don't batch
                    if (spliceArgs.size() == 1 && spliceArgs[0] == arg)
                        continue;
                    // if we thought it is all the same, but then it turned out not to be, we need to fixc it up
                    if (spliceArgs.size() < j)
                        throw logic_error("untested");
                    while (spliceArgs.size() < j)
                        spliceArgs.push_back(spliceArgs.back());
                    // append the arg
                    spliceArgs.push_back(arg);
                }
                // and splice
                if (spliceArgs.size() == 1) // optimized case: all ops share the same operand: no need to batch them
                    m_args[i] = spliceArgs[0];
                else if (pfields0) // they are consecutive: can short-circuit as a slice view
                {
                    let& from = pfields0->m_lazyIndex.first;
                    let begin = pfields0->m_lazyIndex.second;
                    let& fromDims = from->Shape().Dimensions();
                    if (begin == 0 && j == fromDims.back()) // full range: just take it
                        m_args[i] = from;
                    else // sub-range: take a slice veiw
                    {
                        vector<size_t> extent = fromDims;
                        vector<size_t> startOffset(extent.size(), 0);
                        startOffset.back() = begin;
                        extent.back() = j;
                        m_args[i] = from->SliceView(startOffset, extent);
                    }
                    anyBatchedInputs = true;
                }
                else
                {
                    vector<size_t> shape;
                    shape.reserve(maxRank + 1);
                    shape = spliceArgs[0]->Shape().Dimensions();
                    shape.resize(maxRank, 1);
                    shape.push_back(spliceArgs.size());
                    auto out = Alloc(shape, spliceArgs[0]->GetDataType(), spliceArgs[0]->Device());
                    m_args[i] = NDArrayView::GatherBatch(spliceArgs, (int)maxRank, move(out));
                    //m_args[i] = NDArrayView::GatherBatch(spliceArgs, maxRank, out);
                    //fprintf(stderr, "Gathering %d items\n", (int)spliceArgs.size());
                    anyBatchedInputs = true;
                }
            }
            // execute the operation and implant the results
            //for (auto op = ops.begin(); op != ops.end(); ++op)
            //    auto& fields = *op->m_outputs[0].m_dataFields;
            let& unbatchedOutputShape = f0.m_outputs[0].Shape();
            if (!anyBatchedInputs) // short-circuit branch when all inputs were actually the same--we just compute them once
            {
                for (size_t i = 0; i < numArgs; i++)
                    if (m_args[i] != f0.m_inputs[i].m_dataFields->m_value)
                        throw logic_error("fast path unexpectedly got unexpected inputs");
                NDArrayViewPtr out1 = Alloc(unbatchedOutputShape, m_args[0]->GetDataType(), m_args[0]->Device()); // (arena buffer goes here some day)
                auto out = f0.ComputeKnowableValue(f0.Op(), m_args, f0.Attributes(), unbatchedOutputShape, move(out1));
                // implant all results
                for (auto op = ops.begin(); op != ops.end(); ++op)
                    op->m_outputs[0].m_dataFields->m_value = out; // not batched: just duplicate
            }
            else // main branch: computation as a batched operation
            {
                let outputShape = unbatchedOutputShape.AppendAxis(maxRank, batchSize);
                NDArrayViewPtr out1 = Alloc(outputShape, m_args[0]->GetDataType(), m_args[0]->Device()); // (arena buffer goes here some day)
                auto out = f0.ComputeKnowableValue(f0.Op(), m_args, f0.Attributes(), outputShape, move(out1));
                // implant all results
                size_t j = 0;
                for (auto op = ops.begin(); op != ops.end(); ++op)
                {
                    auto& fields = *op->m_outputs[0].m_dataFields;
                    // semantically, this is fields.m_value = out->IndexLastAxis(j);
                    // but it gets deferred to save effort
                    //fields.m_value = out->IndexLastAxis(j);
                    fields.m_lazyIndex = make_pair(out, j); // remember where we came from
                    j++;
                }
            }
            // TODO: for backprop, we need to create a new PrimitiveFunction that executes this operation, or something we can backprop through
        }

        // update all ops' consumers
        for (auto op = ops.begin(); op != ops.end(); ++op)
        {
            // notify first consumer (this is a special optimization)
            auto* f = op->m_notify1;
            if (f)
            {
                if (f->m_pendingInputs <= 0)
                    throw logic_error("pending inputs already 0 yet we are executing it");
                f->m_pendingInputs--;
                // if it is now ready then schedule it
                if (f->m_pendingInputs == 0)
                    m_schedule.Schedule(f);
            }
            // notify all other consumers (this is a special optimization)
            for (auto* f : op->m_notifyN)
            {
                if (f->m_pendingInputs <= 0)
                    throw logic_error("pending inputs already 0 yet we are executing it");
                f->m_pendingInputs--;
                // if it is now ready then schedule it
                if (f->m_pendingInputs == 0)
                    m_schedule.Schedule(f);
            }
        }
    }

    // recursively process the tree *upwards* (through consumer chain)
    // This function assumes:
    //  - m_pendingInputs != -2
    //    BUGBUG: This is a bad one; what if users get a gradient and then continue to build on top?
    Function* TraverseFunctionTreeBackward(const Variable& var, Function* head)
    {
        let& fields = *var.m_dataFields;
        // if already has a gradient (from an earlier call), then skip this branch of the tree
        if (fields.m_gradient)
            return head; // will not be recorded in the nodes list
        auto& f = *fields.m_ownerFunction.lock();
        // if we have already visited this node then done
        if (f.m_pendingInputs == -2)
            return head;
        // mark as visited
        // Note: If users call Backward() multiple times (e.g. for different variables),
        // and this branch of the graph has already been processed then, then we will have
        // a -2 here but also a gradient, therefore we will not reach this test here.
        f.m_pendingInputs = -2;
        // enqueue consumers
        {
            // and recursively traverse the inputs
            //for (let& fc : f.m_notifyN)
            //    head = TraverseFunctionTreeBackward(v, head);


            // this is not working. Parameters' ownerFunction is not what I need here.


            // if no input actually wanted a gradient computed, then we don't need to compute ours either
            //throw logic_error("TraverseFunctionTreeForward() must not be called on variables that already have a value.");
            //if (head == headStart)
            //    return headStart;
            // enqueue ourselves
            f.m_link = head;
            head = &f;
        }
        return head;
    }

public:
    // Value(), computed with automatic batching
    // BUGBUG: The execution should reset the m_pendingInputs values to -1 when done.
    NDArrayViewPtr GetValue(const Variable& v)
    {
        // mark all nodes w.r.t. how many inputs they are waiting for before being computable
        auto& fields = *v.m_dataFields;
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
        return LazilyIndexedValue(v);
    }

    // implant gradients into all variables
    // It can be called multipel times for the same root; but not for different roots
    // (which we won't be able to verify at present--we could remember the gradient root in the Variable).
    void Backward(const CNTK::Variable& root, const std::vector<CNTK::Variable>& variables)
    {
        // traverse the graph from the bottom (the variables to get the gradients for)
        // to form an ordered list of nodes to process
        for (auto& var : variables)
            if (!var.m_dataFields->m_needsGradient)
                logic_error("Backward: cannot compute gradient for variable with m_needsGradient being False.");
            else
                Function* head = TraverseFunctionTreeBackward(var, /*head=*/nullptr);
        // first get the forward computation, batching, etc. done if not yet
        GetValue(root);
        // implant the first gradient if not present yet
        if (!root.m_dataFields->m_gradient)
        {
            if (root.Value()->Shape() != NDShape{})
                throw logic_error("Backward: root must be a scalar, or root gradient must have been implamnted already");
            root.m_dataFields->m_gradient = Alloc(NDShape{}, root.Value()->GetDataType(), root.Value()->Device());
            root.m_dataFields->m_gradient->SetValue(1.0f);
        }
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
    }
}; // class
} // namespace

// this will become Variable::Value()
// Computes lazily the value of a node. Does nothing if called again.
CNTK::NDArrayViewPtr GetValue(const CNTK::Variable& v)
{
#if 0
    // naive version for comparison purposes
    return v.Value();
#else
    auto autoBatcher = CNTK::Memoize();
    return autoBatcher.GetValue(v);
#endif
}

// Perform backprop.
// CNTK grad() allows to pass multiple roots. Does that ever make sense in this context?
void Backward(const CNTK::Variable& root, const std::vector<CNTK::Variable>& variables)
{
    auto autoBatcher = CNTK::Memoize(); // has some internal state
    autoBatcher.Backward(root, variables);
}
