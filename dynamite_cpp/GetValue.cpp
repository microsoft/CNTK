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
#define fail_if(cond, err) (!!(cond) ? (LogicError(__FUNCTION__ ": " err),0) : 0)
#define BreakPoint fprintf(stderr, "") // use this inside a conditional to be able to set a breakpoint in Release code

using namespace std;

namespace CNTK
{
// perform back propagation
// Gradient must have been allocated to the correct shape already.
// If beta == 0 then gradient can be uninitialized memory.
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
            NDArrayView::MatrixProduct(/*transC=*/primitiveOp == PrimitiveOpType::TransposeTimes,
                                      { const_cast<NDArrayView*>(arg1)->shared_from_this() }, /*transA=*/false,
                                      { const_cast<NDArrayView*>(arg2)->shared_from_this() }, /*transB=*/true, alpha, 0, gradient, beta);
        else // right input
            NDArrayView::MatrixProduct(/*transC=*/false,
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
                NDArrayView::NumericOperation({ const_cast<NDArrayView*>(outputGradients[0])->shared_from_this(),
                                                const_cast<NDArrayView*>(    inputValues[0])->shared_from_this(),
                                                const_cast<NDArrayView*>(   outputValues[0])->shared_from_this() }, alpha,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithExpOfDiff,
                                              gradient, beta,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
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
            NDArrayView::NumericOperation({ arg1->IndexLastAxis(i) }, alpha,
                                          Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta,
                                          Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        }
        break;
    case PrimitiveOpType::Slice:
        {
            auto axis       = attributes[L"axis"      /*PrimitiveFunction::AttributeNameAxis*/      ].Value<Axis>();
            auto beginIndex = attributes[L"beginIndex"/*PrimitiveFunction::AttributeNameBeginIndex*/].Value<int>();
            auto endIndex   = attributes[L"endIndex"  /*PrimitiveFunction::AttributeNameEndIndex*/  ].Value<int>();
            auto extent = gradient->Shape().Dimensions();
            auto startOffset = vector<size_t>(extent.size(), 0);
            auto axisIndex = axis.StaticAxisIndex();
            if (startOffset[axisIndex] != beginIndex || extent[axisIndex] != endIndex - beginIndex)
            {
                // backprop into a slice of 'gradient'
                if (beta == 0) // if beta = 0 then we must explicitly initialize the entire gradient matrix, not just the slice
                    gradient->SetValue(0.0f);
                startOffset[axisIndex] = beginIndex;
                extent[axisIndex] = endIndex - beginIndex;
                NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this() }, alpha,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient->SliceView(startOffset, extent), beta,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
            }
            else
                op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; // full slice actually: just copy (like a NoOp)
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
        NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this() }, alpha,
                                      op1Arg, gradient, beta,
                                      Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
    else if (op2Args != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
        NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this(), const_cast<NDArrayView*>(arg2)->shared_from_this() }, alpha,
                                      op2Args, gradient, beta,
                                      Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
}

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

    // allocate a new tensor in a large arena
    // TODO: move this function up since it is ahred between fo2ward and backward
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

    // ===== forward =====

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
            if (!fields.m_value)
                var.Value(); // this initializes it
            if (!fields.m_value)
                LogicError("Parameter/Constant has no Value??");
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
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
    }

    // return the m_value field of a variable, but possibly realizing it lazily if it is an index operation
    static const NDArrayViewPtr& LazilyIndexedValue(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        if (fields.m_value)
            return fields.m_value;
        fail_if(!fields.m_lazyIndex.first, "variable unexpectedly has no value yet, nor is it a slice view into a batched op");
        // the Function does not own its output, it is a slice view into another
        let& from = LazilyIndexedValue(fields.m_lazyIndex.first->m_outputs[0]);
        let index = fields.m_lazyIndex.second;
        if (index == SIZE_MAX) // special sentinel value that means "don't slice, actually"
            fields.m_value = from;
        else
            fields.m_value = from->IndexLastAxis(index);
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
        // logging
#if 1
        fprintf(stderr, "%S%S = %S(", f.Uid().c_str(), outputShape.AsString().c_str(), f.OpName().c_str());
        for (size_t i = 0; i < inputs.size() && i < 4; i++)
        {
            let& input = inputs[i];
            let& fields = *input.m_dataFields;
            if (fields.m_lazyIndex.first)
            {
                let& input1 = fields.m_lazyIndex.first->m_outputs[0];
                fprintf(stderr, "%s%S%S[%d]", (i == 0) ? "" : ", ", input1.Uid().c_str(), input1.Shape().AsString().c_str(), (int)fields.m_lazyIndex.second);
            }
            else
                fprintf(stderr, "%s%S%S", (i == 0) ? "" : ", ", input.Uid().c_str(), input.Shape().AsString().c_str());
        }
        if (inputs.size() > 4)
            fprintf(stderr, ", +%d", (int)(inputs.size() - 4));
        fprintf(stderr, ")\n");
#endif
        auto outValue = isFree ? nullptr : AllocateTensorInArena(outputShape, inputValues[0]->GetDataType(), inputValues[0]->Device());
        // execute it
        output.m_dataFields->m_value = move(f.ComputeKnowableValue(f.Op(), inputValues, f.Attributes(), outputShape, move(outValue)));
        return output;
    }

    static void ResetPendingToIdle(Function& f)
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
    // For every batched operation, this generates a new Function object for the op itself, and one
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
                // reset state
                ResetPendingToIdle(*op);
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
                        // create a new Function Splice()
                        vector<size_t> outputShape = fromDims; // determine output shape
                        outputShape[axis] = j;
                        auto additionalProperties = Dictionary(); // create additional arguments
                        additionalProperties[L"axis"      /*PrimitiveFunction::AttributeNameAxis*/      ] = Axis((int)axis);
                        additionalProperties[L"beginIndex"/*PrimitiveFunction::AttributeNameBeginIndex*/] = (int)begin;
                        additionalProperties[L"endIndex"  /*PrimitiveFunction::AttributeNameEndIndex*/  ] = (int)(begin + j);
                        let spliceOp = Function::RawPrimitiveFunction(PrimitiveOpType::Slice, vector<Variable>{ output }, outputShape, move(additionalProperties));
                        spliceOp->m_uid = L"#" + spliceInputs[0].Uid();
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
                    // create a new Function Splice()
                    vector<size_t> outputShape; // determine output shape
                    outputShape.reserve(maxRank + 1);
                    outputShape = LazilyIndexedValue(spliceInputs[0])->Shape().Dimensions();
                    outputShape.resize(maxRank, 1);             // pad to maxRank
                    outputShape.push_back(spliceInputs.size()); // and add the batch axis
                    auto additionalProperties = Dictionary(); // create additional arguments
                    additionalProperties[L"axis"/*PrimitiveFunction::AttributeNameAxis*/] = Axis((int)maxRank);
                    let spliceOp = Function::RawPrimitiveFunction(PrimitiveOpType::Splice, vector<Variable>(spliceInputs), outputShape, move(additionalProperties));
                    spliceOp->m_uid = L"#" + spliceInputs[0].Uid();
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
                batchedOp->m_uid = L"*" + f0.Uid();
            }
            else
            {
                // all inputs identical: compute it only once
                batchedOp = Function::RawPrimitiveFunction(f0.Op(), vector<Variable>(f0.m_inputs), f0.m_outputs[0].Shape(), Dictionary(f0.Attributes()));
                batchedOp->m_uid = L"." + f0.Uid();
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

    // ===== backward =====

    // lazily create m_gradient, which may live in a batched op
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
#undef NO_BATCHED_BACKPROP
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
                fields.m_gradient = AllocateTensorInArena(fields.m_shape, fields.m_dataType, fields.m_value->Device());
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

#ifndef NO_BATCHED_BACKPROP
        if(fields.m_lazyIndex.first)
        {
            auto& f = *fields.m_lazyIndex.first;
            DetermineConsumersForBackward(f);
        }
        else
#endif
        {
            auto& f = *fields.m_ownerFunction.lock();
            DetermineConsumersForBackward(f);
        }
    }
    void DetermineConsumersForBackward(Function& f)
    {
        fail_if(f.m_pendingInputs == -2, "unexpectedly encountered a cyclic graph??"); // graph is cyclic??

        if (f.m_pendingInputs != -1) // already visited
            return;

        fail_if(f.Op() == PrimitiveOpType::StopGradient, "unexpectedly encountered a StopGradient, which should have propagated m_needsGradient=false upwards");

        // we are now in a Function that should backprop its gradient
        // TODO: implement short-circuiting here
        f.m_pendingInputs = -2; // (temp value to detect cycles; not really needed)

        // determine how many inputs are pending; and also recurse and set up the consumer list
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let* inputp = &inputs[i];
            if (inputp->m_dataFields->m_lazyIndex.first)
                inputp = &inputp->m_dataFields->m_lazyIndex.first->m_outputs[0];
            let& input = *inputp;
            auto& fields = *input.m_dataFields;
            fields.m_visited = false; // TODO: clean this up
            if (!fields.m_needsGradient)
                continue; // skip inputs that receive no gradients
            // this input will receive a gradient; reset it (later, we *accumulate* into it since nodes can receive gradients from multiple consumers)
            // Note that Backward() returns shared_ptrs to the gradient values, so they won't get lost.
            // BUGBUG: (But they get reallocated over again, and will hold the entire arena!!) BUGBUG!
            // BUGBUG: we must not kill the gradient buffers passed by the user
            fields.m_gradient.reset();
            // record ourselves as a consumer of the input
            if (!fields.m_consumers.first.first)
                fields.m_consumers.first = make_pair(&f, i);
            else
                fields.m_consumers.second.push_back(make_pair(&f, i));
            // now process recursively the inputs
            DetermineConsumersForBackward(input);
        }
        f.m_pendingInputs = 0; // used as a visited flag
    }

    // backprop gradient into 'var' by pulling all of its consumers (recursively)
    // This function assumes:
    //  - m_pendingInputs != -2
    //    BUGBUG: This is a bad one; what if users get a gradient and then continue to build on top?
    __declspec(noinline) void BackwardFromAllConsumers(const Variable& var)
    {
        let& fields = *var.m_dataFields;
        fail_if(!fields.m_needsGradient, "backprop into variable that does not need gradient");
        if (fields.m_visited)
            return;
        fields.m_visited = true;
        // have all consumers to push gradients from them into var
        // TODO: do we need the index? Answer is yes for batched backprop
        // TODO: additional optimization goes here (GatherBatch, weight updates)
        auto& c = fields.m_consumers.first;
        if (c.first)
        {
            // (the following test would require the rediect through lazy indexc)
            //fail_if(c.first->m_inputs[c.second].m_dataFields != var.m_dataFields, "input is not the right variable??");
            BackwardToOneInput(c.first, c.second);
        }
        for (auto& c : fields.m_consumers.second)
        {
            //fail_if(c.first->m_inputs[c.second].m_dataFields != var.m_dataFields, "input is not the right variable??");
            BackwardToOneInput(c.first, c.second);
        }
    }
    // second half of above function
    // Backprop from a consumer recursively into its n-th input.
    void BackwardToOneInput(Function* f, size_t index)
    {
        // get all gradients incoming from consumer's consumers
        for (let& output : f->m_outputs)
            BackwardFromAllConsumers(output);
        // perform the backprop operation
        BackpropTo(f, index);
    }

    // back-propagate all outputs' m_gradients to all inputs
    // by pulling gradients from all m_consumers
    vector<const NDArrayView*> m_outputValuesBuffer;
    vector<const NDArrayView*> m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBufferRaw;
    void BackpropTo(Function* f, size_t index)
    {
        let& inputs =  f->m_inputs;
        auto& input = inputs[index];
        auto& fields = *input.m_dataFields;
        // BUGBUG: need to set up needsGradient flags based on what variables are selected
        if (!fields.m_needsGradient)
            return;
        // get the TensorViews for everything we may compute the gradient from
        let& outputs = f->m_outputs;
        m_outputValuesBuffer.clear();
        m_outputGradientsBuffer.clear();
        for (let& output : outputs) // output values and gradients coming from consumer
        {
            let& fields = *output.m_dataFields;
#ifndef NO_BATCHED_BACKPROP
            fail_if(fields.m_lazyIndex.first, "unexpectedly ran into a function that does not own its output"); // we don't backprop through unbatched ops
#endif
            fail_if(!fields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
            fail_if(!fields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");
            m_outputValuesBuffer   .push_back(fields.m_value.get());
            m_outputGradientsBuffer.push_back(fields.m_gradient.get());
        }
        m_inputValuesBufferRaw.clear(); // input values
        for (let& input1 : inputs)
        {
            let& fields = *input1.m_dataFields;
            fail_if(!fields.m_value, "unexpectedly ran into a function that has no m_value yet??");
            m_inputValuesBufferRaw.push_back(fields.m_value.get());
        }
        // compute gradients for the desired input
        // get or create the desired gradient's TensorView
        let beta = LazilyCreateLazilyIndexedGradient(input);
        // backprop into the input
        CNTK::BackpropTo(m_outputGradientsBuffer, index, f->Op(), f->m_attributes, m_outputValuesBuffer, m_inputValuesBufferRaw, fields.m_gradient, beta);
    }

    // helper to verify that the tree is clean
    void AssertTreeStateGetValue(const Variable& v) const
    {
        let& fields = *v.m_dataFields;
        if (fields.m_consumers.first.first || !fields.m_consumers.second.empty())
            LogicError("AssertTreeStateGetValue: m_consumers should be empty");
        let owner = fields.m_ownerFunction.lock();
        if (owner)
        {
            if (owner->m_pendingInputs != -1)
                LogicError("AssertTreeStateGetValue: m_pendingInputs should be -1");
            for (let& input : owner->m_inputs)
                AssertTreeStateGetValue(input);
        }
    }

public:
    // Value(), computed with automatic batching
    // This routine uses temporary fields that are assumed initialized in a specific way:
    //  - Function::m_pendingInputs:
    //     - #inputs that still need to be computed before a node's value can be computed
    //     - also used as a 'visited' flag during traversal
    //     - upon entry and exit of this function, this must be -1 (idle)
    //  - Variable::m_consumers:
    //     - set of consumers of this value. Used to count m_pendingInputs.
    //     - must be empty upon entry and exit
    // plus more temp fields:
    //  - Function::m_link: pointer to next Function in the same batchable op
    // And it leaves the following:
    //  - m_value: updated as desired
    //    TODO: values not needed by user or gradient should use scratch space
    //  - m_lazyIndex: if a slice or view came from a batched operation, this points to it
    //     - Any newly created batched ops are referenced this way.
    NDArrayViewPtr GetValue(const Variable& v)
    {
        AssertTreeStateGetValue(v); // (sanity check)
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
        AssertTreeStateGetValue(v); // (sanity check)
        return LazilyIndexedValue(v);
    }

    // implant gradients into all variables
    // Unlike GetValue(), this is eager. If you call it twice, it's a completely new computation.
    // If you need multiple gradients, ask for them in a single go.

    // BUGBUG!!! This is now again operating on the unbatched graph!! Must keep batching info!

    void Backward(const Variable& root, unordered_map<Parameter, NDArrayViewPtr>& gradients)
    {
        if (!root.m_dataFields->m_needsGradient)
            logic_error("Backward: cannot compute gradient for root with m_needsGradient being False.");
        // BUGBUG: make sure some edge cases are done right:
        //  - root.m_needsGradient=false
        //  - gradients contains root
        //  - root is a m_lazyIndex
        // first get the forward computation, batching, etc. done if not yet
        GetValue(root);
        // set up the m_consumer fields, which Backward() will work off
        DetermineConsumersForBackward(root); // (gotta improve the name of these things)
        // implant the first gradient
        // TODO: allow user to pass in the starting value
        // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
        //if (root.Value()->Shape() != NDShape{})
        //    LogicError("Backward: root must be a scalar, or root gradient must have been implanted already");
        root.m_dataFields->m_gradient = AllocateTensorInArena(root.Shape(), root.GetDataType(), root.Value()->Device());
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
                logic_error("Backward: a requested gradient is not part of root."); // TODO: or could it be due to StopGradient? What if StopGradient is used only sometimes?
            if (!fields.m_needsGradient) // (we could also just leafve the gradient 0)
                logic_error("Backward: cannot compute gradient for variable with m_needsGradient being False.");
            BackwardFromAllConsumers(param);
        }
        //fprintf(stderr, "Back-propagated through %d functions\n", (int)order.size());
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
            kv.second = kv.first.m_dataFields->m_gradient;
        //AssertTreeStateGetValue(root); // (sanity check)  --TODO: gotta think this through e.g. nodes for which no gradient is requested
        // WORKAROUND for above. With this, we can at least com,pute more than 1 gradient fro a parameter
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            auto& fields = *param.m_dataFields;
            fields.m_consumers.first.first = nullptr;
            fields.m_consumers.second.clear();
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
