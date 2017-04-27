//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"
#include "Sequences.h"
#include "ScriptableObjects.h"
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

static const std::ptrdiff_t SentinelValueIndicatingUnspecifedSequenceBeginIdx = std::numeric_limits<std::ptrdiff_t>::min();

template <class ElemType> class DelayedValueNodeState;

// -----------------------------------------------------------------------
// DelayedValueNodeBase (input [, initialState]) -- abstract base class for PastValueNode and FutureValueNode to hold all shared code
// The two differ in the step direction, some loop directions, and sequence-boundary flags.
// -----------------------------------------------------------------------

// TODO: 'direction' is really too general. signOfTimeOffset?
template <class ElemType, int direction /*-1 for Past/left-to-right or +1 for Future/right-to-left*/ /*, MinibatchPackingFlags SequenceStart_or_End/*-Start or -End*/>
class DelayedValueNodeBase : public ComputationNode<ElemType>, public IRecurrentNode, public ILateAttachingNode, public IStatefulNode
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers; using Base::OperationName;
    typedef std::shared_ptr<DelayedValueNodeState<ElemType>> DelayedNodeStatePtr;

private:
    TensorView<ElemType> GetMaskTensor(size_t rank, const FrameRange& fr) const;

protected:
    DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring& name, ElemType fixedInitialStateScalarValue, const TensorShape& sampleLayout, size_t timeStep);
    DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring& name) :
        DelayedValueNodeBase(deviceId, name, (ElemType)DEFAULT_HIDDEN_ACTIVATION, TensorShape(), 0)
    {
    }
    DelayedValueNodeBase(const ScriptableObjects::IConfigRecordPtr configp) :
        DelayedValueNodeBase(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"defaultHiddenActivation"), configp->Get(L"shape"), configp->Get(L"timeStep"))
    {
        // We do NOT attach the inputs, as we cannot resolve them without causing a circular reference.
        // Instead, we capture them in a lambda, which will be called by ComputationNetwork during the build process through LateAttachInputs() below.
        // This is a contract between ComputationNetwork and this specific node type.
        m_attachInputsFn = [this, configp]() // This is the lambda to complete the process. Note that config captured as a shared_ptr.
        {
            AttachInputs(GetInputsFromConfig(configp)); // this is executed by network builder while iterating the nodes
        };
    }
    virtual void /*ILateAttachingNode::*/ LateAttachInputs() override final
    {
        m_attachInputsFn();
        m_attachInputsFn = []()
        {
            LogicError("LateAttachingNode::AttachInputs: must only be called once");
        };
    }

public:
    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override;
    virtual void Load(File& fstream, size_t modelVersion) override;
    virtual void Save(File& fstream) const override;
    virtual void UpdateFunctionMBSize() override;
    virtual void BeginForwardProp() override;
    virtual void ForwardProp(const FrameRange& fr) override;
    virtual void EndForwardProp() override;
    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override;
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;
    virtual int /*IRecurrentNode::*/ GetRecurrenceSteppingDirection() const override { return -direction; }
    virtual NodeStatePtr /*IStatefulNode::*/ ExportState() override;
    virtual void /*IStatefulNode::*/ ImportState(const NodeStatePtr& pImportedState) override;
    int TimeStep() const { return m_timeStep; }
    ElemType InitialActivationValue() const { return m_initialStateValue; }

protected:
    ElemType m_initialStateValue;                           // starting value for hidden activation vector at boundary
    int m_timeStep;                                         // delay in frames (typ. 1)

    function<void()> m_attachInputsFn;                      // for late expansion of inputs (scripting)

    vector<ElemType> m_inputInvalidMatrixTemp;              // [j] CPU-side buffer for constructing the mask matrix
    vector<bool> m_inputAnySeqValid, m_inputAllSeqValid;    // [t] denotes whether there are any valid frames at a time step, and if all are valid

    shared_ptr<Matrix<ElemType>> m_initialStateValueMatrix; // potentially GPU-side versions
    shared_ptr<Matrix<ElemType>> m_inputInvalidMatrix;      // [0,j] contains 1 if matrix column belongs to an frame with boundary condition or a gap frame
    shared_ptr<Matrix<ElemType>> m_zeroMatrix;              // constant [1]-dimensional 0 used for backprop  --TODO: could use a static map[deviceId]
    shared_ptr<Matrix<ElemType>> m_packedIndexMatrix;       // index mapping for DoGatherColumnsOf() in case of per-sequence initial state

    shared_ptr<Matrix<ElemType>> m_delayedValue;            // saves the activation of the previous step that this node points to
    MBLayoutPtr m_delayedActivationMBLayout;                // layout for m_delayedValue
};

#define UsingDelayedValueNodeMembers        \
    UsingComputationNodeMembersBoilerplate; \
    using Base::m_initialStateValue;        \
    using Base::m_delayedValue;             \
    using Base::m_timeStep;

// -----------------------------------------------------------------------
// PastValueNode (input) -- delay node
// TODO: Can this just be a typedef?
// -----------------------------------------------------------------------

template <class ElemType>
class PastValueNode : public DelayedValueNodeBase<ElemType, -1 /*, MinibatchPackingFlags::SequenceStart*/>
{
    typedef DelayedValueNodeBase<ElemType, -1 /*, MinibatchPackingFlags::SequenceStart*/> Base; UsingDelayedValueNodeMembers;
    static const std::wstring TypeName() { return L"PastValue"; }

public:
    PastValueNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    PastValueNode(DEVICEID_TYPE deviceId, const wstring& name, ElemType fixedInitialStateScalarValue, const TensorShape& sampleLayout, size_t timeStep)
        : Base(deviceId, name, fixedInitialStateScalarValue, sampleLayout, timeStep)
    {
    }
    PastValueNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& sampleLayout, size_t timeStep)
        : Base(deviceId, name, (ElemType)0, sampleLayout, timeStep)
    {
    }
    PastValueNode(DEVICEID_TYPE deviceId, const wstring& name, ElemType fixedInitialStateScalarValue, size_t numRows, size_t timeStep)
        : PastValueNode(deviceId, name, fixedInitialStateScalarValue, TensorShape(numRows), timeStep)
    {
    }
    PastValueNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp)
    {
    }
};

// -----------------------------------------------------------------------
// FutureValueNode (input) -- delay node in future direction
// -----------------------------------------------------------------------

// get value from future (used in the bi-directional models)
template <class ElemType>
class FutureValueNode : public DelayedValueNodeBase<ElemType, +1 /*, MinibatchPackingFlags::SequenceEnd*/>
{
    typedef DelayedValueNodeBase<ElemType, +1 /*, MinibatchPackingFlags::SequenceEnd*/> Base; UsingDelayedValueNodeMembers;
    static const std::wstring TypeName() { return L"FutureValue"; }

public:
    FutureValueNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    FutureValueNode(DEVICEID_TYPE deviceId, const wstring& name, ElemType fixedInitialStateScalarValue, const TensorShape& sampleLayout, size_t timeStep)
        : Base(deviceId, name, fixedInitialStateScalarValue, sampleLayout, timeStep)
    {
    }
    FutureValueNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& sampleLayout, size_t timeStep)
        : Base(deviceId, name, (ElemType)0, sampleLayout, timeStep)
    {
    }
    FutureValueNode(DEVICEID_TYPE deviceId, const wstring& name, ElemType fixedInitialStateScalarValue, size_t numRows, size_t timeStep)
        : FutureValueNode(deviceId, name, fixedInitialStateScalarValue, TensorShape(numRows), timeStep)
    {
    }
    FutureValueNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp)
    {
    }
};

}}}
