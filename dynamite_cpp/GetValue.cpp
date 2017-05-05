//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"
//#include "CompositeFunction.h"

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
    //  - operations that compute batchable ops must be replaced
    //  - those are PrimitiveFunctions
    //  - a set of batchable PrimitiveFunctions get replaced by different primitives, in-place,
    //    that simply Slice() the result of the batched operation
    //  - the new batched operation is a Primitive of the same original type
    //  - its inputs are Splice() over the original inputs
    //  - hence, we create N+1 new nodes:
    //     - the new batched op
    //     - Splice() for each of the N inputs
    //  - those must be kept in the composite
    //    Technically in ALL composites that may refer to any of these primitive functions.
    //    Hence, we must clean up the ownership mess.
    //    *** Ownership must move into the primitives. ***

public:
    static NDArrayViewPtr GetValue(Variable v)
    {
        auto& composite = *(CompositeFunction*)v.m_outputComposite.get();
        // must be a composite due to the ownership problem
        if (!v.m_outputComposite)
            LogicError("Value must be called on a composite");
        //auto& nodeSet = composite.m_allPrimitiveFunctions;
        // nodeSet must be augmented with all newly created nodes
        let& fields = *v.m_dataFields;
        fields.m_value;
        return v.Value();
    }
};
}

CNTK::NDArrayViewPtr GetValue(const CNTK::Variable& v)
{
#if 0
    // naive version
    return v.Value();
#else
    return CNTK::Memoize::GetValue(const_cast<CNTK::Variable&>(v));
#endif
}
