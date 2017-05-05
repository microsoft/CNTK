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

    unordered_map<Key, deque<Function*>> m_scheduledOps; // set of ready operations, batched

    // schedule an operation that has been confirmed ready
    void Schedule(Function* fp)
    {
        auto key = Key("*fp");
        m_scheduledOps[move(key)].push_back(fp);
    }

    // traverse the tree hanging off a Variable and
    //  - prepare all nodes for batched execution
    //  - schedule all ready operations
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
            Schedule(&f); // add to ready set
    }

    vector<Variable> m_args;

    // batch-execute a set of ops that are known to be batchable
    void ExecuteBatchedAndUpdateSchedule(const deque<Function*>& ops)
    {
        // get a representative op
        let& f0 = *ops.front();
        // batch all arguments
        m_args.resize(f0.m_inputs.size());
        // create a new PrimitiveFunction that executes this operation
        // compute its Value   --TODO: we need a lower-level executor that just takes the op and the inputs' TensorViews?
        // distribute the results to ops[]
        // update all ops' consumers
    }

public:
    NDArrayViewPtr operator()(const Variable& v)
    {
        // mark all nodes w.r.t. how many inputs they are waiting for before being computable
        let& fields = *v.m_dataFields;
        if (!fields.m_value)
        {
            // prepare and schedule first set
            RTraverseOwner(v);
            // process everything
            while (!m_scheduledOps.empty())
            {
                // select the best amongst the scheduled ops
                // TODO: add priority for Barrier()
                deque<Function*>* bestOp = nullptr;
                for (auto& kv : m_scheduledOps)
                    if (!bestOp || kv.second.size() > bestOp->size())
                        bestOp = &kv.second;
                // execute it, and also update all outputs' values and consumers, and the schedule 
                ExecuteBatchedAndUpdateSchedule(*bestOp);
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
#if 1
    // naive version
    return v.Value();
#else
    auto getValue = CNTK::Memoize();
    return getValue(v);
#endif
}
