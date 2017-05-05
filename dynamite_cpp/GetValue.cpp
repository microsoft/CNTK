//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"
//#include "CompositeFunction.h"

#include <deque>
#include <unordered_map>

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#define let const auto

using namespace std;

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

    // the Key determines which operations are batchable with each other (those with the same Key)
    typedef string Key;
    struct Key1 : public string
    {
        Key1(const Function& f)
        {
            assign("xxx");
        }
    };

    // class to manage the set of ready operations (the schedule)
    class ReadyOps
    {
        vector<deque<Function*>> m_allOps;
        static bool IsBatchableWith(const Function* a, const Function* b)
        {
            return a->Op() == b->Op(); // ... for now (won't actually work)
        }
    public:
        // schedule an operation that has been confirmed ready
        void Schedule(Function* fp)
        {
            // this naive implementation just scans linearly
            // scan through all op sets to see if one is batchable with 'fp'
            for (auto iter = m_allOps.begin(); iter != m_allOps.end(); iter++)
            {
                if (IsBatchableWith(fp, iter->front()))
                {
                    iter->push_back(fp);
                    return;
                }
            }
            // none fit: open a new set
            m_allOps.push_back(deque<Function*>{ fp });
        }
        bool empty() const { return m_allOps.empty(); }
        void pop_best(deque<Function*>& out)
        {
            // TODO: add priority for Barrier()
            auto best = m_allOps.begin();
            for (auto iter = best+1; iter != m_allOps.end(); iter++)
                if (iter->size() > best->size())
                    best = iter;
            // and remove this one from the list
            out = move(*best);
            m_allOps.erase(best); // TODO: suboptimal complexity; a list in a self-allocated container would do
        }
    };
    ReadyOps m_schedule;

    // traverse the tree hanging off a Variable and
    //  - prepare all nodes for batched execution
    //  - schedule all ready operations
    // TODO: Once we are in the main build, change all Function to PrimitiveFunction directly.
    void RTraverseOwner(const Variable& v)
    {
        let& fields = *v.m_dataFields;
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            throw logic_error("Value() depends on Input or Placeholder, it is not knowable.");
        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
        {
            if (!fields.m_value) // force-initialize Parameters
                v.Value();
            if (!fields.m_value) // TODO: need to do this first
                throw logic_error("Parameter/Constant has no Value??");
            return;
        }
        auto& f = *fields.m_ownerFunction.lock();
        if (f.m_pendingInputs != -1) // already visited
            return;
        // determine how many inputs are pending
        // and also recurse
        size_t pendingInputs = 0;
        for (let& v : f.m_inputs)
        {
            let& fields = *v.m_dataFields;
            if (!fields.m_value)
            {
                RTraverseOwner(v);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                {
                    // no need for anything ref-counted since this is a local temp variable
                    v.m_dataFields->m_ownerFunction.lock()->m_notify.push_back(&f);
                    pendingInputs++;
                }
            }
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
    }

    vector<Variable> m_inputs;
    vector<NDArrayViewPtr> m_args;
    size_t m_numBatches = 0;

    // batch-execute a set of ops that are known to be batchable
    void ExecuteBatchedAndUpdateSchedule(const deque<Function*>& ops)
    {
        // TODO: need to handle ops that have >1 output, such as Combine(). Just don't batch them ever? Combine() is just a see-through anyway.
        // get a representative op
        let& f0 = *ops.front();
        fprintf(stderr, "%d executing %d instances of %S\n", (int)++m_numBatches, (int)ops.size(), f0.OpName().c_str());
        m_inputs.resize(f0.m_inputs.size());
        m_args.resize(f0.m_inputs.size());
#if 1
        // for correctness testing of underlying mechanism, compute them without actual batching
        for (let& op : ops)
        {
            if (op->m_outputs.size() != 1)
                throw logic_error("only functions with 1 output are supported");
            m_inputs.resize(op->m_inputs.size());
            m_args.resize(op->m_inputs.size());
            for (size_t i = 0; i < op->m_inputs.size(); i++)
            {
                m_args[i] = op->m_inputs[i].m_dataFields->m_value;
                if (!m_args[i])
                    throw logic_error("input unexpectedly not available");
            }
            auto out = NDArrayViewPtr();
            op->m_outputs[0].m_dataFields->m_value =
                op->ComputeKnowableValue(op->Op(), m_args, op->Attributes(), op->m_outputs[0].Shape(), move(out));
        }
#else
        // batch all arguments
        // create a new PrimitiveFunction that executes this operation
        // compute its Value   --TODO: we need a lower-level executor that just takes the op and the inputs' TensorViews?
        auto out = NDArrayViewPtr();
        f0.ComputeKnowableValue(f0.Op(), args, f0.Attributes(), shape, move(out));
        // distribute the results to ops[]
#endif
        // update all ops' consumers
        for (let& op : ops)
        {
            for (let& f : op->m_notify)
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

public:
    // Value(), computed with automatic batching
    NDArrayViewPtr operator()(const Variable& v)
    {
        // mark all nodes w.r.t. how many inputs they are waiting for before being computable
        let& fields = *v.m_dataFields;
        if (!fields.m_value)
        {
            // prepare and schedule first set
            RTraverseOwner(v);
            // compute the entire graph
            deque<Function*> opBatch;
            while (!m_schedule.empty())
            {
                // select the best amongst the scheduled ops
                m_schedule.pop_best(opBatch);
                // execute it, and also update all outputs' values and consumers, and the schedule 
                ExecuteBatchedAndUpdateSchedule(opBatch);
            }
            v.Value(); // old code--will disappear
        }
        return fields.m_value;
    }

    Memoize()
    {
    }
}; // class
} // namespace

CNTK::NDArrayViewPtr GetValue(const CNTK::Variable& v)
{
#if 0
    // naive version
    return v.Value();
#else
    auto getValue = CNTK::Memoize();
    return getValue(v);
#endif
}
