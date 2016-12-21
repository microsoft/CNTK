//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ComputationNode.h"
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

template <class ElemType>
bool AreEqual(const ElemType* a, const ElemType* b, const size_t count, const float threshold);

// Minimalistic version of input node used to avoid dependency to other nodes.
template <class ElemType>
class DummyNodeTest : public ComputationNode<ElemType>
{
public:
    typedef ComputationNode<ElemType> Base;

    UsingComputationNodeMembersBoilerplate;

    static const std::wstring TypeName();

    DummyNodeTest(DEVICEID_TYPE deviceId, size_t minibatchSize, SmallVector<size_t> sampleDimensions,
                  std::vector<ElemType>& data);

    DummyNodeTest(DEVICEID_TYPE deviceId, const wstring& name);

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& /*fr*/) override
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange& /*fr*/) override
    {
    }

    Matrix<ElemType>& GetGradient();

    void SetMinibatch(size_t minibatchSize, SmallVector<size_t> sampleDimensions, std::vector<ElemType>& data);
};
} } } }