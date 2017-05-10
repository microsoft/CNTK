//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"
#include "PrimitiveFunction.h"

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
            Function& operator*() const { return *iter; } // TODO: This is weird, figure this out
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
        void push_back(Function* f)
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
        void NotifyInputAvailable(Function* f)
        {
            if (f->m_pendingInputs <= 0)
                throw logic_error("NotifyInputAvailable: pending inputs already 0 yet we are executing it");
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
    // This function also sets up the m_consumers fields. When it returns, they are all correctly set up.
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
        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        for (let& v : f.m_inputs)
        {
            auto& fields = *v.m_dataFields;
            // record ourselves as a consumer of the input--this is used for batching and for Backward()
            // TODO: Think this through for Combine(), where m_input==m_outputs.
            if (!fields.m_consumers.first) // optimized for main case of 1 consumer. No std::vector in that case.
                fields.m_consumers.first = &f;
            else
                fields.m_consumers.second.push_back(&f);
            // recursively traverse
            if (!fields.m_value)
            {
                TraverseFunctionTreeForward(v);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                    pendingInputs++;
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
        static NDArrayViewPtr m_currentArena; // for now static so that it carries over across invocations, to save the allocation
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
                    m_args[i] = LazilyIndexedValue(op->m_inputs[i]); // (if this is a lazy slice, then now we must resolve it)
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
                    else // sub-range: take a slice view
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
                    // TODO: review this w.r.t. multi-output functions
                    auto& fields = *op->m_outputs[0].m_dataFields;
                    // semantically, this is fields.m_value = out->IndexLastAxis(j);
                    // but it gets deferred to save effort
                    //fields.m_value = out->IndexLastAxis(j);
                    fields.m_lazyIndex = make_pair(out, j); // remember where we came from
                    j++;
                }
            }
        }

        // update all ops' consumers
        for (auto op = ops.begin(); op != ops.end(); ++op)
        {
            for (let& output : op->m_outputs)
            {
                // notify consumers
                let& fields = *op->m_outputs[0].m_dataFields;
                auto* f = fields.m_consumers.first; // first consumer (this is a special optimization to avoid a malloc in case of 1 consumer)
                if (f)
                    m_schedule.NotifyInputAvailable(f);
                for (auto* f : fields.m_consumers.second) // all other consumers
                    m_schedule.NotifyInputAvailable(f);
            }
        }
    }

    // recursively process the tree *upwards* (through consumer chain)
    // This function assumes:
    //  - m_pendingInputs != -2
    //    BUGBUG: This is a bad one; what if users get a gradient and then continue to build on top?
    void TraverseFunctionTreeBackward(const Variable& var, NonOwningFunctionListBuilder& head)
    {
        let& fields = *var.m_dataFields;
        // if already has a gradient (from an earlier call), then skip this branch of the tree
        if (fields.m_gradient)
            return; // will not be recorded in the nodes list
        // traverse all consumers
        auto* f = fields.m_consumers.first;
        if (f)
            TraverseFunctionTreeBackward(f, head);
        for (auto* f : fields.m_consumers.second)
            TraverseFunctionTreeBackward(f, head);
    }
    // seond half of this fnction
    void TraverseFunctionTreeBackward(Function* f, NonOwningFunctionListBuilder& head)
    {
        // if we have already visited this Function then done
        if (f->m_pendingInputs == -2)
            return;
        // mark as visited
        // Note: If users call Backward() multiple times (e.g. for different variables),
        // and this branch of the graph has already been processed then, then we will have
        // a -2 here but also a gradient, therefore we will not reach this test here.
        f->m_pendingInputs = -2;
        // and recursively traverse the inputs
        // TODO: handle StopGradient here, e.g. return a flag whether we visitied a StopGradient op.
        for (let& output : f->m_outputs)
            TraverseFunctionTreeBackward(output, head);
        // TODO: if no input actually wanted a gradient computed, then we don't need to compute ours either
        // enqueue ourselves
        // Add to the end of the queue *after* we recurse (we do "height" first traversal).
        // Needed because this is called multiple times for multiple roots, and those have to go to the end.
        head.push_back(f);
    }

    // perform back propagation
    static NDArrayViewPtr BackpropTo(const vector<const NDArrayView*>& outputGradients, size_t i,
                                     PrimitiveOpType primitiveOp, const Dictionary& attributes,
                                     const vector<const NDArrayView*>& outputValues, const vector<const NDArrayView*>& inputValues,
                                     NDArrayViewPtr&& gradient, double beta)
    {
        // allocate target gradient if not yet  --TODO: do we even need that?
        if (!gradient)
            gradient = make_shared<NDArrayView>(inputValues[i]->GetDataType(), inputValues[i]->Shape(), inputValues[i]->Device());
        else if (gradient->Shape() != inputValues[i]->Shape())
            LogicError("Variable '%S' Value(): The out buffer passed to op %S does not match outputShape.", L""/*AsString().c_str()*/, L""/*OpName().c_str()*/);
        //switch (primitiveOp)
        //{
        //    // elementwise ops are done outside, we just set the opcode
        //case PrimitiveOpType::Plus:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;                break;
        //}
        return gradient;
    }

    // back-propagate all outputs' m_gradients to all inputs
    // TODO: not all inputs want gradients. How to control this? Needs the same flag as the StopGradient question. Can we use m_needsGradient?
    vector<const NDArrayView*> m_outputValuesBuffer;
    vector<const NDArrayView*> m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBuffer;
    void Backward(Function* f)
    {
        //let* pf = dynamic_cast<PrimitiveFunction*>(f);
        let& outputs = f->m_outputs;
        let& inputs =  f->m_inputs;
        //let numOutputs = outputs.size();
        // get the TensorViews
        m_outputValuesBuffer.clear();
        m_outputGradientsBuffer.clear();
        for (let& output : outputs) // output values and gradients coming from consumer
        {
            let& fields = *output.m_dataFields;
            if (!fields.m_gradient || !fields.m_value)
                throw logic_error("Backward: gradient from consumer unexpectedly unavailable (or value)");
            m_outputValuesBuffer   .push_back(fields.m_value.get());
            m_outputGradientsBuffer.push_back(fields.m_gradient.get());
        }
        m_inputValuesBuffer.clear(); // input values
        for (let& input : inputs)
            m_inputValuesBuffer.push_back(input.m_dataFields->m_value.get());
        // compute gradients for all inputs
        let numInputs = inputs.size();
        for (size_t i = 0; i < numInputs; i++)
        {
            auto& input = inputs[i];
            auto& fields = *input.m_dataFields;
            // BUGBUG: need to set up needsGradient flags based on what variables are selected
            if (!fields.m_needsGradient)
                continue;
            // for now just allocate the gradient, so that we can test the propagation
            auto inputGradient = move(fields.m_gradient);
            let isFirst = !inputGradient;
            if (isFirst)
            {
                inputGradient = Alloc(input.Shape(), input.GetDataType(), input.Value()->Device());
                inputGradient->SetValue(0.0f); // TODO: remove this, replace by beta
            }
            double beta = isFirst ? 0 : 1;
            fields.m_gradient = move(BackpropTo(m_outputGradientsBuffer, i, f->Op(), f->m_attributes, m_outputValuesBuffer, m_inputValuesBuffer, move(inputGradient), beta));
        }
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

    // BUGBUG!!! This is now again operating on the unbatched graph!! Must keep batching info!

    void Backward(const CNTK::Variable& root, const std::vector<CNTK::Variable>& variables)
    {
        // first get the forward computation, batching, etc. done if not yet
        // This will also set up the m_consumers chains, which we rely on.
        GetValue(root);
        // traverse the graph from the bottom (the variables to get the gradients for)
        // to form an ordered list of nodes to process
        NonOwningFunctionListBuilder order;
        for (auto& var : variables)
            if (!var.m_dataFields->m_needsGradient)
                logic_error("Backward: cannot compute gradient for variable with m_needsGradient being False.");
            else
                TraverseFunctionTreeBackward(var, order);
        // implant the first gradient if not present yet
        if (!root.m_dataFields->m_gradient)
        {
            // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
            //if (root.Value()->Shape() != NDShape{})
            //    throw logic_error("Backward: root must be a scalar, or root gradient must have been implanted already");
            root.m_dataFields->m_gradient = Alloc(root.Shape(), root.GetDataType(), root.Value()->Device());
            root.m_dataFields->m_gradient->SetValue(1.0f);
        }
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
        fprintf(stderr, "Back-propagating through %d functions\n", (int)order.size());
        for (auto f = order.begin(); f != order.end(); ++f)
            Backward(&*f);
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
