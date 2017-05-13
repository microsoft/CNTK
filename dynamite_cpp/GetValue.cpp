//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"
#include "PrimitiveFunction.h"
#include "CommonMatrix.h"

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
                LogicError("should not get here for view ops or barrier ops");
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
            var.Value(); // this initializes it
            if (!fields.m_value)
                LogicError("Parameter/Constant has no Value??");
            return;
        }
        auto& f = *fields.m_ownerFunction.lock();
        if (f.m_pendingInputs == -2 || fields.m_gradient) // (-2 means we've already run gradient; therefore we must have a value and should not get here)
            LogicError("TraverseFunctionTreeForward() should not have been called on variables that already have a gradient.");
        if (f.m_pendingInputs != -1) // already visited
            return;
        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        for (let& v : f.m_inputs)
        {
            auto& fields = *v.m_dataFields;
            // recursively traverse
            if (!fields.m_value)
            {
                TraverseFunctionTreeForward(v);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                {
                    pendingInputs++;
                    // record ourselves as a consumer of the input
                    if (!fields.m_consumers.first) // optimized for main case of 1 consumer. No std::vector in that case.
                        fields.m_consumers.first = &f;
                    else
                        fields.m_consumers.second.push_back(&f);
                }
            }
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
    }

    // allocate a new tensor in a large arena
    //NDArrayViewPtr m_currentArena;
    //size_t m_currentArenaUsed;
    static const size_t ARENASIZE = 64000000; // we allocate in this chunk size
    NDArrayViewPtr AllocateTensorInArena(const NDShape& shape, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
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
    static const NDArrayViewPtr& LazilyIndexedValue(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        if (!fields.m_value)
        {
            let& from = fields.m_lazyIndex.first->m_outputs[0].m_dataFields->m_value;
            let index = fields.m_lazyIndex.second;
            if (!from)
                LogicError("variable unexpectedly has no value yet");
            if (index == SIZE_MAX) // special sentinel value that means "don't slice, actually"
                fields.m_value = from;
            else
                fields.m_value = from->IndexLastAxis(index);
        }
        return fields.m_value;
    }

    // compute the value of 'f', storing it in the arena (unless 'isFree', which must be set when there is nothing to store)
    vector<NDArrayViewPtr> m_inputValuesBuffer; // Use a buffer for this that does not get destructed, to reuse the memory allocation.
    const Variable& MemoizeKnowableValueInArena(Function& f, bool isFree = false)
    {
        if (f.m_outputs.size() != 1)
            LogicError("MemoizeKnowableValueInArena: only functions with 1 output are supported");
        // fetch the NDArrayViewPtrs for all inputs
        auto& inputValues = m_inputValuesBuffer;
        let& inputs = f.m_inputs;
        inputValues.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            inputValues[i] = LazilyIndexedValue(inputs[i]); // (if this is a lazy slice, then now we must resolve it)\
        // allocate the output NDArrayViewPtr in the arena
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        auto outValue = isFree ? nullptr : AllocateTensorInArena(outputShape, inputValues[0]->GetDataType(), inputValues[0]->Device());
        // execute it
        output.m_dataFields->m_value = move(f.ComputeKnowableValue(f.Op(), inputValues, f.Attributes(), outputShape, move(outValue)));
        return output;
    }

    // temp variables for ExecuteBatchedOpAndUpdateSchedule(); keep outside to reuse the memory allocation
    vector<Variable> m_batchedInputs;
    vector<Variable> m_spliceArgsBuffer;
    size_t m_numBatchedLaunches = 0; // (for statistics only)

    // batch-execute a set of ops that are known to be batchable
    // For every batched operation, this generates a new Function object for the op itself, and one
    // for a splice operation for each batched inputs.
    // I.e. this is not a full graph transform, but rather a graph augmentation, so that during backprop,
    // we can recover the batched operations, while the original graph does not get modified.
    // Any batched operation will generate its result in a dense tensor with a batch dimension.
    // The consumers of the original ops will get a back-reference in the m_lazyIndex field.
    // If such a result is ever accessed individually, it will lead to a lazy NDArrayView::SliceView() call
    // (but no Splice Function object is used for this).
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
                // TODO: realize splice ops that are index ops as a m_lazyIndex at this point
                //if (f0.Op() == PrimitiveOpType::Slice)
                //{
                //    // inject the lazyIndex
                //}
#if 0           // test effect of the unbatched sparse times
                if (f0.Op() == PrimitiveOpType::Times)
                    op->ComputeKnowableValue(op->Op(), m_inputValuesBuffer, op->Attributes(), op->m_outputs[0].Shape(), move(out));
#endif
            }
        }
        // execute the batchable operations as a batch
        // Every resulting batched op consists of the following new operations:
        //  - a Splice() or Slice() for each input (e.g. 2 for a binary op)
        //  - a PrimitiveFunction that is the op itself
        //  - m_lazyIndex entries that represent a "virtual" Slice() that is never created as a Function object to saved mallocs.
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
                const VariableFields* pfields0 = nullptr;// @@@@@@@bring this back (as let*): f0.m_inputs[i].m_dataFields.get(); // as long as non-NULL, we are consecutive
                if (!pfields0->m_lazyIndex.first) // only if it is an indexed slice, actually
                    pfields0 = nullptr;
                // loop over all batched ops
                size_t j = 0;
                for (auto op = ops.begin(); op != ops.end(); ++op, j++) // create the batched tensors
                {
#if 0
                    // optimization: if all args are consecutive slices, then use a slice view instead
                    if (pfields0)
                    {
                        let* pfields = op->m_inputs[i].m_dataFields.get();
                        // we continue to be in optimized state
                        if (pfields->m_lazyIndex.first  == pfields0->m_lazyIndex.first &&
                            pfields->m_lazyIndex.second == pfields0->m_lazyIndex.second + j)
                            continue;
                        // nope, chain has been broken: must fix up spliceInputs up to here
                        // This is suboptimal in that we lost the reference to the originating input.
                        // So we cannot remember the realized SliceView there.
                        // TODO: Per Jon's suggestion, we can be a little loose here. For a variable-length
                        // scenario, we will loose entries in the middle. We can allow to keep a few around
                        // in garbage-in-garbage-out. If, say, there are additional 20% gap values, we just
                        // carry them forward, and ignore them when implanting the result.
                        let& from = pfields0->m_lazyIndex.first;
                        let begin = pfields0->m_lazyIndex.second;
                        while (spliceInputs.size() < j)
                            spliceInputs.push_back(from->IndexLastAxis(begin + spliceInputs.size())); // ### this is expensive without an Index op
                        pfields0 = nullptr; // no longer in consecutive hypothesis
                    }
#endif
                    let& input = op->m_inputs[i];
                    // optimization: if all args are the same, then don't batch
                    // BUGBUG: Unhandled opportunity: The same input could be used multiple times. E.g. discover with a unique identifier.
                    if (spliceInputs.size() == 1 && spliceInputs[0].m_dataFields.get() == input.m_dataFields.get())
                        continue;
                    // if we thought it is all the same, but then it turned out not to be, we need to fixc it up
                    if (spliceInputs.size() < j)
                        LogicError("untested");
                    while (spliceInputs.size() < j)
                        spliceInputs.push_back(spliceInputs.back());
                    // append the input
                    spliceInputs.push_back(input);
                    // note: Variable is just two shared_ptrs, one being NULL; so this is cheap
                    // note: input is a regular Variable with regular ownwership rules (it does not come from inside here)
                }
                // and splice
                if (spliceInputs.size() == 1) // optimized case: all ops share the same operand: no need to batch them
                    m_batchedInputs[i] = spliceInputs[0];
#if 0           // @@@@@bring this back
                else if (pfields0) // they are consecutive: can short-circuit as a slice view
                {
                    let& from = pfields0->m_lazyIndex.first;
                    let begin = pfields0->m_lazyIndex.second;
                    let& fromDims = from->Shape().Dimensions();
                    if (begin == 0 && j == fromDims.back()) // full range: just take it
                        m_inputValuesBuffer[i] = from;
                    else // sub-range: take a slice view
                    {
                        vector<size_t> extent = fromDims;
                        vector<size_t> startOffset(extent.size(), 0);
                        startOffset.back() = begin;
                        extent.back() = j;
                        m_inputValuesBuffer[i] = from->SliceView(startOffset, extent); // ### Splice(); then execute it through ComputeKnowableValue()
                    }
                    anyBatchedInputs = true;
                }
#endif
                else
                {
                    // create a new Function Splice()
                    vector<size_t> outputShape; // determine output shape
                    outputShape.reserve(maxRank + 1);
                    outputShape = LazilyIndexedValue(spliceInputs[0])->Shape().Dimensions();
                    outputShape.resize(maxRank, 1);             // pad to maxRank
                    outputShape.push_back(spliceInputs.size()); // and add the batch axis
                    auto additionalProperties = Dictionary(); // create additional arguments
                    additionalProperties[L"axis"/*PrimitiveFunction::AttributeNameAxis*/] = Axis((int)maxRank);
                    let spliceOp = Function::RawPrimitiveFunction(PrimitiveOpType::Splice, vector<Variable>(spliceInputs), outputShape, move(additionalProperties));
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
            // BUGBUG: The newly created Function objects must get their consumer chain set up.
            let& unbatchedOutputShape = f0.m_outputs[0].Shape();
            FunctionPtr batchedOp;
            if (anyBatchedInputs)
            {
                // create a new Function for the batched op
                // Batched inputs have been prepared in m_batchedInputs[].
                let expectedOutputShape = unbatchedOutputShape.AppendAxis(maxRank, batchSize);
                batchedOp = Function::RawPrimitiveFunction(f0.Op(), vector<Variable>(m_batchedInputs), expectedOutputShape, Dictionary(f0.Attributes()));
            }
            else
            {
                batchedOp = f0.shared_from_this(); // all inputs identical: compute it only once
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
                auto* f = fields.m_consumers.first; // first consumer (this is a special optimization to avoid a malloc in case of 1 consumer)
                if (f)
                    m_schedule.NotifyInputAvailable(f);
                for (auto* f : fields.m_consumers.second) // all other consumers
                    m_schedule.NotifyInputAvailable(f);
                // clear consumer list (this operation is done)
                fields.m_consumers.first = nullptr;
                fields.m_consumers.second.clear();
            }
        }
    }

    // recursively process the tree *upwards* (through consumer chain)
    // This function assumes:
    //  - m_pendingInputs != -2
    //    BUGBUG: This is a bad one; what if users get a gradient and then continue to build on top?
    // BUGBUG: We must traverse the batched operations, but those don't have the consumer chains set up.
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
    // seond half of this function
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
    static void BackpropTo(const vector<const NDArrayView*>& outputGradients, size_t i,
                           PrimitiveOpType primitiveOp, const Dictionary& attributes,
                           const vector<const NDArrayView*>& outputValues, const vector<const NDArrayView*>& inputValues,
                           const NDArrayViewPtr& gradient, double beta)
    {
#if 0   // TODO: bring this back once we have gradient functions that do not support beta
        if (beta == 0) // TODO: limit this to those ops that do not support beta
        {
            gradient->SetValue(0.0f);
            beta = 1;
        }
#endif
        auto op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // this gets set for 1-argument TensorView ops for execution after the switch()
        auto op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // and this for 2-arg ops; all others execute inside the switch()
        const NDArrayView* arg1 = outputGradients[0];
        const NDArrayView* arg2 = nullptr;
        double alpha = 1;
        // NOTE: For now, this only implements the operators needed for the prototype
        switch (primitiveOp)
        {
            // binary operations with simple TensorView implementation
        case PrimitiveOpType::Plus:           op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
        case PrimitiveOpType::Minus:          op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; alpha = i == 0 ? 1 : -1; break;
        case PrimitiveOpType::ElementTimes:   op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct; arg2 = inputValues[1 - i]; break;
            // Times family
        case PrimitiveOpType::Times:
        case PrimitiveOpType::TransposeTimes:
            arg2 = inputValues[1 - i];
            if (i == 0) // left input
                gradient->MatrixProduct(/*transC=*/primitiveOp == PrimitiveOpType::TransposeTimes,
                                        { const_cast<NDArrayView*>(arg1)->shared_from_this() }, /*transA=*/false,
                                        { const_cast<NDArrayView*>(arg2)->shared_from_this() }, /*transB=*/true, alpha, 0, gradient, beta);
            else // right input
                gradient->MatrixProduct(/*transC=*/false,
                                        { const_cast<NDArrayView*>(arg2)->shared_from_this() }, /*transA=*/primitiveOp != PrimitiveOpType::TransposeTimes,
                                        { const_cast<NDArrayView*>(arg1)->shared_from_this() }, /*transB=*/false, alpha, 0, gradient, beta);
            break;
            // unary operations with simple TensorView implementation
        case PrimitiveOpType::ReLU:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithLinearRectifierDerivativeFromOutput; arg2 = outputValues[0]; break;
            // no-op operations with simple TensorView implementation
            // NOTE: These do not need any data copy if there is only one consumer, which we won't know here. That case will be caught in the batched version.
        case PrimitiveOpType::NoOp:           op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
        case PrimitiveOpType::Reshape:        op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
            // gradients that are copies with broadcasting
        case PrimitiveOpType::ReduceElements:
            {
                const auto& reductionOpName = attributes[L"reductionOpName"/*PrimitiveFunction::AttributeNameReductionOpName*/].Value<wstring>();
                if (reductionOpName == L"Sum"/*PrimitiveFunction::InternalSumReductionOpName*/) // TODO: uncomment these symbols once we have access
                    op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
                else if (reductionOpName == L"LogSum"/*PrimitiveFunction::InternalLogSumReductionOpName*/)
                    gradient->NumericOperation({ const_cast<NDArrayView*>(outputGradients[0])->shared_from_this(),
                                                 const_cast<NDArrayView*>(    inputValues[0])->shared_from_this(),
                                                 const_cast<NDArrayView*>(   outputValues[0])->shared_from_this() },
                                               alpha, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithExpOfDiff,
                                               gradient, beta, Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
                else
                    //  PrimitiveFunction::InternalMeanReductionOpName
                    //  PrimitiveFunction::InternalMaxReductionOpName
                    //  PrimitiveFunction::InternalMinReductionOpName
                    //  PrimitiveFunction::InternalProdReductionOpName
                    LogicError("Variable '%S' Value(): Gradient of reduction op %S not yet implemented.", L""/*AsString().c_str()*/, reductionOpName.c_str());
            }
            break;
            // hard stuff
        case PrimitiveOpType::Splice:
            {
                auto axis = attributes[L"axis"/*PrimitiveFunction::AttributeNameAxis*/].Value<Axis>();
                if (axis.StaticAxisIndex() != arg1->Shape().Rank() -1)
                    LogicError("NDArrayView::GatherBatch: Currently only splicing in a new slowest-changing axis is supported.");
                gradient->NumericOperation({ arg1->IndexLastAxis(i) },
                                           alpha, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta, Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
            }
            break;
        default:
            //fprintf(stderr, "NEEDS: %S\n", PrimitiveOpTypeName(primitiveOp).c_str());
            LogicError("Variable '%S' Value(): Backpropagation for operation %S not implemented yet.", L""/*AsString().c_str()*/, PrimitiveOpTypeName(primitiveOp).c_str());
            //LogicError("Variable '%S' Value(): Backpropagation for non-existent operation %S?", L""/*AsString().c_str()*/, PrimitiveOpTypeName(primitiveOp).c_str());
        }
        // the simple TensorView operations are performed out here
        // TODO: we can eliminate the vector<> by passing a std::function, possibly?
        if (op1Arg != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            gradient->NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this() },
                                       alpha, op1Arg, gradient, beta, Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        else if (op2Args != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            gradient->NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this(), const_cast<NDArrayView*>(arg2)->shared_from_this() },
                                       alpha, op2Args, gradient, beta, Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
    }

    // back-propagate all outputs' m_gradients to all inputs
    // TODO: not all inputs want gradients. How to control this? Needs the same flag as the StopGradient question. Can we use m_needsGradient?
    vector<const NDArrayView*> m_outputValuesBuffer;
    vector<const NDArrayView*> m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBufferRaw;
    void BackpropToInputs(Function* f)
    {
        let& outputs = f->m_outputs;
        let& inputs =  f->m_inputs;
        // get the TensorViews for everything we may compute the gradient from
        m_outputValuesBuffer.clear();
        m_outputGradientsBuffer.clear();
        for (let& output : outputs) // output values and gradients coming from consumer
        {
            let& fields = *output.m_dataFields;
            if (!fields.m_gradient)
                LogicError("BackpropToInputs: gradient from consumer unexpectedly unavailable");
            m_outputValuesBuffer   .push_back(LazilyIndexedValue(output).get());
            m_outputGradientsBuffer.push_back(fields.m_gradient.get());
        }
        m_inputValuesBufferRaw.clear(); // input values
        for (let& input : inputs)
            m_inputValuesBufferRaw.push_back(LazilyIndexedValue(input).get()); // inefficient!! But fine for now, I just want correctness.
        // compute gradients for all inputs
        let numInputs = inputs.size();
        for (size_t i = 0; i < numInputs; i++)
        {
            auto& input = inputs[i];
            auto& fields = *input.m_dataFields;
            // BUGBUG: need to set up needsGradient flags based on what variables are selected
            if (!fields.m_needsGradient)
                continue;
            // get or create the desired gradient's TensorView
            auto inputGradient = fields.m_gradient;
            let isFirst = !inputGradient;
            if (isFirst) // first time: allocate the gradient memory
                inputGradient = AllocateTensorInArena(input.Shape(), input.GetDataType(), input.Value()->Device());
            double beta = isFirst ? 0 : 1;
            // backprop into the input
            BackpropTo(m_outputGradientsBuffer, i, f->Op(), f->m_attributes, m_outputValuesBuffer, m_inputValuesBufferRaw, inputGradient, beta);
            if (isFirst)
                fields.m_gradient = inputGradient;
        }
    }

    // helper to verify that the tree is clean
    void AssertTreeStateGetValue(const Variable& v, bool post) const
    {
        let& fields = *v.m_dataFields;
        if (fields.m_consumers.first || !fields.m_consumers.second.empty())
            LogicError("AssertTreeStateGetValue: m_consumers should be empty");
        let owner = fields.m_ownerFunction.lock();
        if (owner)
        {
            if (!post)  // TODO: remove this once we figured out how to reset m_pendingInputs
            if (owner->m_pendingInputs != -1)
                LogicError("AssertTreeStateGetValue: m_pendingInputs should be -1");
            for (let& input : owner->m_inputs)
                AssertTreeStateGetValue(input, post);
        }
    }

public:
    // Value(), computed with automatic batching
    // This routine uses temporary fields that are assumed initialized in a specific way:
    //  - Function::m_pendingInputs:
    //     - #inputs that still need to be computed before a node's value can be computed
    //     - also used as a 'visited' flag during traversal
    //     - upon entry and exit of this function, this must be -1
    //       BUGBUG: It would break some logic to reset it to -1. Solve this.
    //  - Variable::m_consumers:
    //     - set of consumers of this value. Used to count m_pendingInputs.
    //     - must be empty upon entry and exit
    // plus more temp fields:
    //  - m_link: pointer to next Function in the same batchable op
    // And it leaves the following:
    //  - m_value: updated
    //  - m_lazyIndex: if a slice or view came from a batched operation, this points to it
    //     - Any newly created batched ops are referenced this way.
    NDArrayViewPtr GetValue(const Variable& v)
    {
        AssertTreeStateGetValue(v, false); // (sanity check)
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
        AssertTreeStateGetValue(v, true); // (sanity check)
        return LazilyIndexedValue(v);
    }

    // implant gradients into all variables
    // It can be called multipel times for the same root; but not for different roots
    // (which we won't be able to verify at present--we could remember the gradient root in the Variable).

    // BUGBUG!!! This is now again operating on the unbatched graph!! Must keep batching info!

    void Backward(const Variable& root, unordered_map<Parameter, NDArrayViewPtr>& gradients)
    {
        // first get the forward computation, batching, etc. done if not yet
        // This will also set up the m_consumers chains, which we rely on.
        GetValue(root);
        // traverse the graph from the bottom (the variables to get the gradients for)
        // to form an ordered list of nodes to process
        NonOwningFunctionListBuilder order;
        for (auto& kv : gradients)
            if (!kv.first.m_dataFields->m_needsGradient)
                logic_error("Backward: cannot compute gradient for variable with m_needsGradient being False.");
            else
            {
                // we must destroy the gradient of parameters since those are not constructed anew each time
                kv.first.m_dataFields->m_gradient.reset();
                TraverseFunctionTreeBackward(kv.first, order);
            }
        // implant the first gradient if not present yet
        if (!root.m_dataFields->m_gradient)
        {
            // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
            //if (root.Value()->Shape() != NDShape{})
            //    LogicError("Backward: root must be a scalar, or root gradient must have been implanted already");
            root.m_dataFields->m_gradient = AllocateTensorInArena(root.Shape(), root.GetDataType(), root.Value()->Device());
            root.m_dataFields->m_gradient->SetValue(1.0f);
        }
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
        fprintf(stderr, "Back-propagating through %d functions\n", (int)order.size());
        for (auto f = order.begin(); f != order.end(); ++f)
            BackpropToInputs(&*f);
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
        {
            kv.second = kv.first.m_dataFields->m_gradient;
            // and make sure we don't get into an incorrect consumer chain
            // BUGBUG: This is brittle. Forward sets this up, and we won't know whether we are called multiple times (we shouldn't though).
            kv.first.m_dataFields->m_consumers.first = nullptr;
            kv.first.m_dataFields->m_consumers.second.clear();
        }
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
void Backward(const CNTK::Variable& root, std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr>& gradients)
{
    auto autoBatcher = CNTK::Memoize(); // has some internal state
    autoBatcher.Backward(root, gradients);
}
