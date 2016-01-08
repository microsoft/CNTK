//
// <copyright file="RecurrentNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "TensorShape.h"
#include "ComputationNode.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>
#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // ShiftNode (input, fromOffset, boundaryValue, dim=-1, offsetRange=1, multiOffsetDim=0) -- delay and rolling window
    //
    // This shifts the input by (-fromOffset) steps. In other words, output(t) will be input(t+fromOffset).
    // E.g. for fromOffset=-1, this gives the past value.
    // This node has quite some options that make it powerful for many use cases.
    //
    // This node can be used in a recurrent loop. This requires special handling by the ComputationNetwork,
    // for both execution (sequential execution) and creation (avoiding circular references).
    // TODO: When outside a recurrent loop and used with frame randomization, this will communicate to the reader
    // that additional frames are needed, which will then return a frame range. TODO: This will not match
    // the labels, which are still 1 frame. Think through which dimension this should go in.
    //
    // Values shifted in from beyond sequence boundaries will be copied from boundaryValue.
    // Normally, this is a scalar Constant(). However, it can be any node, which will be indexed from the end
    // (e.g. for fromOffset=-1, the last frame of boundaryValue will be used). This can implement
    // sequence-to-sequence models. Broadcasting is supported, so it can be e.g. a single output-dimension vector
    // applied to all sequences.
    //
    // To delay (past value), use negative fromOffset. To access future value, use positive fromOffset.
    //
    // To pull in multiple offsets, use offsetRange>1. This will pull in offsetRange consecutive offsets starting
    // with fromOffset. This implements a rolling window. A new dimension will be inserted at multiOffsetDim
    // (default 0 means after the last sample dimension). Special considerations:
    //  - If the boundaryValue is not wide enough, the sequence will be dropped (e.g. if you pull in 5 history frames,
    //    but the sequence in boundaryValue only has 4 samples).
    //  - If the current time step (offset 0) is included in the range (e.g. fromOffset=-1, offsetRange=3) then
    //    this node cannot participate in a recurrence.
    //
    // By default, this shifts over the time dimension, but you can choose to shift over any
    // sample tensor dimension instead using 'dim' (-1 stands for time). This will only work, however,
    // when all involved nodes are implemented using the tensor library. Nodes implemented using
    // Matrix slices can only support iterating over time.
    //
    // The fromOffset can also be a tensor, e.g. (1,1). In that case, iteration will be over multiple
    // consecutive dimensions. offsetRange must have the same number of dimensions.
    //
    // If the boundaryValue has 0 elements, the sequence will be trimmed (frames reaching beyond the boundary
    // are dropped). This will initially not be implemented for the time dimension (as it would require
    // change of MBLayout).
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ShiftNode : public ComputationNode<ElemType>, public ILateAttachingNode, public IStatefulNode,  public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Shift"; }
    private:
    protected:
        ShiftNode(DEVICEID_TYPE deviceId, const wstring & name, const TensorShape & fromOffset, int shiftDimension, const TensorShape & offsetRange, int expandDimension) :
            Base(deviceId, name), m_fromOffsetBegin(fromOffset.GetDims()),
            m_shiftDimension(shiftDimension), m_expandDimension(expandDimension),
            m_insertExpandShapeAt(SIZE_MAX/*uninitialized at this point*/)
        {
            // determine m_fromOffsetEnd from fromOffset/offsetRange
            for (size_t k = 0; k < m_fromOffsetBegin.size(); k++)
                m_fromOffsetEnd.push_back(m_fromOffsetBegin[k] + (k < offsetRange.size() ? offsetRange[k] : 1));
            CreateMatrixIfNull(m_value);
            SetDims(TensorShape(), 0);  // empty for now
        }
        ShiftNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ShiftNode(deviceId, name, TensorShape(1), -1, TensorShape(1), 0)
        { }
        ShiftNode(const ScriptableObjects::IConfigRecordPtr configp) :
            ShiftNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"fromOffset"), configp->Get(L"dim"), configp->Get(L"offsetRange"), configp->Get(L"multiOffsetDim"))
        {
            // We do NOT attach the inputs, as we cannot resolve the main input without causing a circular reference.
            // Instead, we capture them in a lambda, which will be called by ComputationNetwork during the build process through LateAttachInputs() below.
            // This is a contract between ComputationNetwork and this specific node type.
            // (TODO: We could force-evaluate the boundary input here.)
            m_attachInputsFn = [this, configp]()   // This is the lambda to complete the process. Note that config captured as a shared_ptr.
            {
                AttachInputs(GetInputsFromConfig(configp));    // this is executed by network builder while iterating the nodes
            };
        }
        virtual void /*ILateAttachingNode::*/LateAttachInputs() override final
        {
            m_attachInputsFn();
            m_attachInputsFn = [](){ LogicError("LateAttachingNode::AttachInputs: must only be called once"); };
        }
    public:
        void Save(File& fstream) const
        {
            Base::Save(fstream);

            TensorShape(m_fromOffsetBegin).Save(fstream);
            TensorShape(m_fromOffsetEnd).Save(fstream);
            fstream << m_shiftDimension;
            fstream << m_expandDimension;
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);

            m_fromOffsetBegin = TensorShape().Load(fstream).GetDims();
            m_fromOffsetEnd   = TensorShape().Load(fstream).GetDims();
            fstream >> m_shiftDimension;
            fstream >> m_expandDimension;
        }

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            assert(inputIndex == 0); inputIndex;
            fr;
        }

        virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
        virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override {return false; }

        virtual void EndForwardProp() override        // called after last iteration step of ForwardProp()
        {
            Base::EndForwardProp();

            // In BPTT, we carry over left-to-right state across minibatches.
            // TODO: package up the state using ExportState(). Then in BeginForwardProp() bring it back. In-between, the packages can be moved around.
        }

        // This function assumes BeginForwardProp/EndForwardProp() to be called before/after the iteration loop.
        // TODO: In the future, there may be value for one more way of handling the boundary condition: Fill as 'NoInput'. Then we can use this to implement rolling windows (albeit inefficiently). Would require to unshare the layout.
        virtual void ForwardProp(const FrameRange & fr) override
        {
            fr;
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            assert(m_inputs.size() == 1);
            ComputationNodeBase::Validate(isFinalValidationPass);

            // MBLayout is just inherited
            m_pMBLayout = Input(0)->GetMBLayout();
            if (isFinalValidationPass && !m_pMBLayout)
                InvalidArgument("%ls %ls operation must operate on data (must have an MB Layout).", NodeName().c_str(), OperationName().c_str());
            InferMBLayoutFromInputsForStandardCase();

            // determine expandShape--empty if no multiple offsets; otherwise the 1 or more dimensions that need to be added at m_expandDimension
            m_expandShape.clear();
            for (size_t k = 0; k < m_fromOffsetBegin.size(); k++)
            {
                size_t dim = m_fromOffsetEnd[k] - m_fromOffsetBegin[k];
                if (dim > 1)
                {
                    m_expandShape.resize(k, 1);
                    m_expandShape.push_back(dim);
                }
            }
            if (!m_expandShape.empty())
                m_expandShape.resize(m_fromOffsetBegin.size(), 1);  // pad ones to end
            // now it either matches the dimensions to insert, or is empty if none to append

            auto inputSampleLayout = Input(0)->GetSampleLayout();
            auto inputDims = inputSampleLayout.GetDims();
            if (m_expandDimension < 0)
                InvalidArgument("%ls %ls operation: Specified insertion location %d refers to a time dimension, but this is not allowed.", 
                                NodeName().c_str(), OperationName().c_str(), m_expandDimension);
            m_insertExpandShapeAt = m_expandShape.empty() ? 0 : (m_expandDimension > 0 ? m_expandDimension - 1 : inputDims.size());
            if (m_insertExpandShapeAt >= inputDims.size())
                InvalidArgument("%ls %ls operation: Specified insertion location %d beyond end of input sample layout [%s].", 
                                NodeName().c_str(), OperationName().c_str(), m_expandDimension, string(inputSampleLayout).c_str());
            SmallVector<size_t> dims;
            dims.append(inputDims.begin(), inputDims.begin() + m_insertExpandShapeAt);
            dims.append(m_expandShape.begin(), m_expandShape.end());
            dims.append(inputDims.begin() + m_insertExpandShapeAt, inputDims.end());
            auto sampleLayout = TensorShape(dims);

            SetDims(sampleLayout, 0);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ShiftNode<ElemType>>(nodeP);
                node->m_fromOffsetBegin     = m_fromOffsetBegin;
                node->m_fromOffsetEnd       = m_fromOffsetEnd;
                node->m_shiftDimension      = m_shiftDimension;
                node->m_expandDimension     = m_expandDimension;
                node->m_expandShape         = m_expandShape;
                node->m_insertExpandShapeAt = m_insertExpandShapeAt;
                node->m_state               = m_state;
            }
        }

        class ShiftNodeState : public INodeState
        {
            Matrix<ElemType> m_delayedActivation;       // saves the activation of the previous step that this node points to
        };
        typedef std::shared_ptr<ShiftNodeState> ShiftNodeStatePtr;

        // state export/import
        // This is done with a shared_ptr. The moment state is exported, the internal state is cleared; ownership is transferred to the exporting entity.
        // This way, the next invocation does not overwrite the exported state, but is required to create a new one if needed.
        // On the other hand, once imported, the state object is owned by the node and will be overwritten with the next state.
        virtual NodeStatePtr ExportState() { return std::move(m_state); }
        virtual void ImportState(NodeStatePtr && state) override
        {
            m_state = dynamic_pointer_cast<ShiftNodeState>(state);
            if (state && !m_state)
                LogicError("ImportState: Wrong state object passed (wrong type).");
        }
    protected:
        // parameters remembered from construction
        SmallVector<size_t> m_fromOffsetBegin;      // offset to pull from; first offset in case of offset range
        SmallVector<size_t> m_fromOffsetEnd;        // end of offset range
        int m_shiftDimension;                       // dimension to shift (default: time)
        int m_expandDimension;                      // in case of offset range, this is where a new dimension will be inserted

        // derived params set up in Validate()
        SmallVector<size_t> m_expandShape;          // offsetEnd-offsetBegin if >1 offset in any dimension; empty otherwise
        size_t m_insertExpandShapeAt;               // at which dimension to insert (internal 0-based index)

        ShiftNodeStatePtr m_state;                  // saves the activation of the previous step that this node points to

        function<void()> m_attachInputsFn;          // for late expansion of inputs (scripting)
    };

    // -----------------------------------------------------------------------
    // DelayedValueNodeState -- helper class for exporting/importing state from/to DelayedValueNodes.
    // This is used for sub-minibatching in case of truncated BPTT.
    // -----------------------------------------------------------------------

    template<class ElemType>
    class DelayedValueNodeState: public INodeState
    {
        public:
            DelayedValueNodeState(int deviceID) :
                m_cachedActivity((size_t)0, (size_t)0, deviceID), 
                m_delayedActivationMBLayout(nullptr), 
                m_isEmpty(true)
            { }
            void CacheDelayedMBLayout(const MBLayoutPtr& pMBLayout)
            {
                m_delayedActivationMBLayout = make_shared<MBLayout>();
                m_delayedActivationMBLayout->CopyFrom(pMBLayout);
            }
            void CacheState(const Matrix<ElemType>& cachedActivity)
            {
                m_cachedActivity.SetValue(cachedActivity); 
                m_isEmpty = false; 
            }
            void ExportDelayedMBLayout(MBLayoutPtr& pMBLayout)
            {
                pMBLayout->CopyFrom(m_delayedActivationMBLayout); 
            }
            bool IsEmpty()
            {
                return m_isEmpty; 
            }
            const Matrix<ElemType>& ExportCachedActivity()
            {
                return m_cachedActivity; 
            }
            ~DelayedValueNodeState(){}
            
        protected:
            Matrix<ElemType>    m_cachedActivity; // 1 column per parallel sequence 
            // MBLayoutPtr         m_shiftedMBLayout;   
            // Currently, we only support saving state for m_timeStep == 1
            // there is no need for this m_shiftedMBLayout if m_timeStep == 1
            MBLayoutPtr         m_delayedActivationMBLayout; 
            bool                m_isEmpty;      // in some case 
            // (e.g., at the boundary of sentence end or begin/full utterance mode), we don't need to store state (but we do need to need know m_delayedActivationMBLayout)
    };

    // -----------------------------------------------------------------------
    // DelayedValueNodeBase (input) -- abstract base class for PastValueNode and FutureValueNode to hold all shared code
    // The two differ in the step direction, some loop directions, and sequence-boundary flags.
    // This is an old node which will be replaced by ShiftNode (with Past/FutureValueNode being emulated).
    //
    // This is planned:
    //  - carrying over state at sentence boundaries from other nodes (for s2s)
    //  - ranges of neighbor frames as a secondary tensor dimension (i.e. can be used to implement a rolling window)
    //  - full support/efficiency of non-recurrent use (in which case the range can be from negative to positive, e.g. a symmetric rolling window)
    //  - denoting which tensor dimension to loop over (this may not be completed, but I will plant a seed)
    //  - support for Yongqiang’s sub-minibatching with BPTT (export/import state)
    //  - more efficient storage of carried-over state (only store the needed frames, not a full copy of the previous MB as currently; which will on the other hand also allow windows that reach back beyond a minibatch)
    // -----------------------------------------------------------------------

    // TODO: 'direction' is really too general. signOfTimeOffset?
    template<class ElemType, int direction/*-1 for Past/left-to-right or +1 for Future/right-to-left*/  /*, MinibatchPackingFlags SequenceStart_or_End/*-Start or -End*/>
    class DelayedValueNodeBase : public ComputationNode<ElemType>, public
                                 ILateAttachingNode, public IStatefulNode,  public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        typedef std::shared_ptr<DelayedValueNodeState<ElemType>> DelayedNodeStatePtr; 
        static const std::wstring TypeName() { return L"DelayedValue"; }
    private:
        void Init(const TensorShape & sampleLayout, ElemType initialActivationValue)
        {
            m_initialActivationValue = initialActivationValue;
            m_timeStep = 1;
            CreateMatrixIfNull(m_value);
            SetDims(sampleLayout, 0);
            m_value->SetValue(m_initialActivationValue);        // is this needed?
        }
    protected:
        DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_delayedActivation(deviceId)
        {
            Init(TensorShape(), (ElemType)DEFAULT_HIDDEN_ACTIVATION);
        }
        DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, const TensorShape & sampleLayout, size_t timeStep) :
            Base(deviceId, name),
            m_delayedActivation(deviceId)
        {
            Init(sampleLayout, initialActivationValue);
            m_timeStep = (int)timeStep; // TODO: pass this to Init() instead as well
        }
        DelayedValueNodeBase(const ScriptableObjects::IConfigRecordPtr configp) :
            DelayedValueNodeBase(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"defaultHiddenActivation"), configp->Get(L"shape"), configp->Get(L"timeStep"))
        {
            // We do NOT attach the inputs, as we cannot resolve them without causing a circular reference.
            // Instead, we capture them in a lambda, which will be called by ComputationNetwork during the build process through LateAttachInputs() below.
            // This is a contract between ComputationNetwork and this specific node type.
            m_attachInputsFn = [this, configp]()   // This is the lambda to complete the process. Note that config captured as a shared_ptr.
            {
                AttachInputs(GetInputsFromConfig(configp));    // this is executed by network builder while iterating the nodes
            };
        }
        virtual void /*ILateAttachingNode::*/LateAttachInputs() override final
        {
            m_attachInputsFn();
            m_attachInputsFn = [](){ LogicError("LateAttachingNode::AttachInputs: must only be called once"); };
        }
    public:
        void Save(File& fstream) const
        {
            Base::Save(fstream);

            fstream << m_timeStep;
            fstream << GetNumRows() << GetNumCols();

            fstream << m_initialActivationValue;
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            // the node has already been initialized e.g. w.r.t. direction
            Base::Load(fstream, modelVersion);

            fstream >> m_timeStep;

            size_t rows, cols;
            fstream >> rows >> cols;

            // Note: Do we need load cols for delay node? I just set to zero to see if there is any problem.
            SetDims(TensorShape(rows), 0);          // tensor shape will be overwritten in Validate()  --TODO: We should serialize it here.
            m_delayedActivation.Resize(rows, 0);    // Note: If we try to access history in first minibatch, we shall crash. It would be a consequence of a missing sentence-begin flag

            if (modelVersion >= CNTK_MODEL_VERSION_2)
                fstream >> m_initialActivationValue;
        }

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            assert(inputIndex == 0); inputIndex;

            // special case: DelayedValueNodes may be used outside of loops
            // TODO: this should be a bulk operation; this implementation is a quick hack
            int dir = direction;    // (this avoids a 'conditional expression is constant' warning)
            if (fr.IsAllFrames())
            {
                // recursive call to ourselves
                FrameRangeIteration range(m_pMBLayout, -dir);
                for (auto t = range.rbegin(); t != range.rend(); t++)   // note: reverse iterator
                    BackpropTo(inputIndex, t);
                return;
            }

            // we backpropagated into the delayed frame
            FrameRange frDelayed = fr.WithTimeOffset(direction * m_timeStep);

            // if delayed input is within valid time range then add its gradient
            size_t t = fr.t();
            int t_delayed = (int)(t + direction * m_timeStep);  // this might end up outside the current window
            if (t_delayed >= 0 && t_delayed < GetNumTimeSteps())
            {
                // Boundary frames must not propagate. Gaps must also not propagate.
                // if there is a boundary in this frame, we treat each stream separately; otherwise we do all in one go
                //assert(m_pShiftedMBLayout->Is(t, SequenceStart_or_End | MinibatchPackingFlags::NoFeature) ==
                //       m_pMBLayout->IsGap(fr) || m_pMBLayout->IsBeyondStartOrEnd(frDelayed));
                if (m_pMBLayout->IsGap(fr) || m_pMBLayout->IsBeyondStartOrEnd(frDelayed)) // true if at least one parallel sequence has a boundary or gap
                {
                    size_t mNbr = m_pMBLayout->GetNumParallelSequences();
                    for (size_t id = 0; id < mNbr; id++)
                    {
                        //assert(m_pShiftedMBLayout->Is(id, t, SequenceStart_or_End | MinibatchPackingFlags::NoFeature) ==
                        //       m_pMBLayout->IsGap(fr.Sequence(id)) || m_pMBLayout->IsBeyondStartOrEnd(frDelayed.Sequence(id)));
                        if (!(m_pMBLayout->IsGap(fr.Sequence(id)) || m_pMBLayout->IsBeyondStartOrEnd(frDelayed.Sequence(id))))    // don't propagate boundary frames or gaps
                        {
                            Matrix<ElemType> frm = GradientFor(fr.Sequence(id));
                            // TODO: use delayed FrameRange here as well
                            //Matrix<ElemType> to = Input(0)->GradientFor(FrameRange(m_pMBLayout, t_delayed).Sequence(id));
                            Matrix<ElemType> to = Input(0)->GradientFor(frDelayed.Sequence(id));
                            to += frm;
                        }
                    }
                }
                else    // operate on entire time step in one go (over all parallel sequences)
                {
                    Matrix<ElemType> frm = GradientFor(fr);
                    // TODO: use something like fr.WithDelay(t) instead, instead of recreating FrameRanges
                    //Matrix<ElemType> to = Input(0)->GradientFor(FrameRange(m_pMBLayout, t_delayed));
                    Matrix<ElemType> to = Input(0)->GradientFor(frDelayed);
                    to += frm;
                }
            }
        }

        virtual bool OutputUsedInComputingInputNodesGradients() const override
        {
            // The DelayedValueNode does not require its output value for computing
            // the gradients of its input nodes
            return false;
        }

        virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
        {
            // The DelayedValueNode does not require any of it's input's values for computing
            // the gradients of its input nodes
            UNREFERENCED_PARAMETER(childIndex);
            return false;
        }

        virtual void EndForwardProp() override        // called after last iteration step of ForwardProp()
        {
            // In BPTT, we carry over left-to-right state across minibatches.
            // It is kept in m_delayedActivation, m_delayedActivationMBLayout.
            // This could be optimized as follows:
            //  - only keep the required number of frames (m_timeStep)
            //  - we don't need to keep anything in full-sequence mode
            //  - we don't need to keep anything if all sequences are closed (sentence end)
            //    This condition includes full-sequence mode.
            // TODO: Can we optimize this and only copy if there is a sequence spanning across the end of the MB? And add a check to BeginForwardProp() to make sure we got one if there is a boundary at the start?
            m_delayedActivation = Input(0)->Value();
            if (!m_delayedActivationMBLayout) m_delayedActivationMBLayout = make_shared<MBLayout>();
            m_delayedActivationMBLayout->CopyFrom(m_pMBLayout);

            Base::EndForwardProp();
        }

        // This function assumes BeginForwardProp/EndForwardProp() to be called before/after the iteration loop.
        // TODO: In the future, there may be value for one more way of handling the boundary condition: Fill as 'NoInput'. Then we can use this to implement rolling windows (albeit inefficiently). Would require to unshare the layout.
        virtual void ForwardProp(const FrameRange & fr) override
        {
            assert(m_pMBLayout);

            // special case: DelayedValueNodes may be used outside of loops
            // TODO: this should be a bulk operation; this implementation is a quick hack
            int dir = direction;    // (this avoids a 'conditional expression is constant' warning)
            if (fr.IsAllFrames())
            {
                // recursive call to ourselves
                FrameRangeIteration range(m_pMBLayout, -dir);
                for (auto t = range.begin(); t != range.end(); t++)
                    ForwardProp(t);
                return;
            }

            // we forward prop from the previous frame to this frame
            FrameRange frDelayed = fr.WithTimeOffset(direction * m_timeStep);

            VerifyDims(Input(0));

            size_t T = GetNumTimeSteps();
            size_t T_delayedActivation = m_delayedActivationMBLayout ? m_delayedActivationMBLayout->GetNumTimeSteps() : 0;  // (note: should never happen in full-sequence mode)

            // compute logical position of delayed value
            assert(m_timeStep > 0);

            size_t t = fr.t();
            int t_delayed = (int)(t + direction * m_timeStep);  // this might end up outside the current window

            Matrix<ElemType> inp;   // ((DEVICEID_TYPE)m_value.GetDeviceId());

            // if any sequence at this time step has a boundary flag, then process one by one
            // TODO: Would there be an efficiency gain from grouping consecutive sequences with identical flags?
            //assert(m_pShiftedMBLayout->Is(t, SequenceStart_or_End) == m_pMBLayout->IsBeyondStartOrEnd(frDelayed));
            if (m_pMBLayout->IsBeyondStartOrEnd(frDelayed))
            {
                for (size_t id = 0; id < GetNumParallelSequences(); id++)
                {
                    if (m_pMBLayout->IsGap(fr.Sequence(id)))    // if output is in a gap then don't bother filling it
                        continue;

                    Matrix<ElemType> out = ValueFor(fr.Sequence(id));

                    //assert(m_pShiftedMBLayout->Is(id, t, SequenceStart_or_End) == m_pMBLayout->IsBeyondStartOrEnd(frDelayed.Sequence(id)));
                    if (m_pMBLayout->IsBeyondStartOrEnd(frDelayed.Sequence(id)))
                        out.SetValue(m_initialActivationValue);     // crossed a boundary
                    else    // not a boundary: just copy the delayed value
                    {
                        // inside the sequence: access delayed value
                        if (t_delayed < 0)
                            inp = DataWithMBLayoutFor(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed + T_delayedActivation).Sequence(id), m_delayedActivationMBLayout); // delay reaches in previous minibatch
                        else if (t_delayed >= T)
                            inp = DataWithMBLayoutFor(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed - T).Sequence(id), m_delayedActivationMBLayout); // delay reaches in previous minibatch
                        else
                            inp = Input(0)->ValueFor(frDelayed.Sequence(id));
                            //inp = Input(0)->ValueFor(FrameRange(m_pMBLayout, t_delayed).Sequence(id));

                        out.SetValue(inp);
                    }
                }
            }
            else        // frame has no boundary flags: use ValueFor directly (still may have a gap here)
            {
                Matrix<ElemType> out = ValueFor(fr);

                if (t_delayed < 0)
                    inp = DataWithMBLayoutFor(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed + T_delayedActivation), m_delayedActivationMBLayout);
                else if (t_delayed >= T)
                    inp = DataWithMBLayoutFor(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed - T), m_delayedActivationMBLayout);
                else
                    inp = Input(0)->ValueFor(frDelayed);
                    //inp = Input(0)->ValueFor(FrameRange(m_pMBLayout, t_delayed));

                out.SetValue(inp);
            }
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DelayedValueNodeBase<ElemType, direction/*, SequenceStart_or_End*/>>(nodeP);
                node->m_timeStep = m_timeStep;
                node->m_initialActivationValue = m_initialActivationValue;
                node->m_delayedActivation = m_delayedActivation;
                if (m_delayedActivationMBLayout)
                    (node->m_delayedActivationMBLayout = make_shared<MBLayout>())->CopyFrom(m_delayedActivationMBLayout);
                else
                    node->m_delayedActivationMBLayout = nullptr;
            }
        }

        //========================================
        // implement the IStatefulNode interface
        //========================================

        virtual NodeStatePtr ExportState()
        {
            NodeStatePtr pExportedState;
            size_t nT = m_pMBLayout->GetNumTimeSteps();
            size_t nU = m_pMBLayout->GetNumParallelSequences();
            int dir = direction; 
            if (m_timeStep != 1)
            {
                // not support yet; give user a hint 
                RuntimeError("Currently importing/exporting state info for timeStep>1 is not supported. Contact erw@microsoft.com for more detail");
            }
            if (dir == -1) // we look into past 
            {
#if 0
                bool   allAtBoundary = true;
                // if the current last frames are all sentence end or no feature , there is no need to carry on state info
                if (m_pMBLayout->Is(FrameRange(nT-1), MinibatchPackingFlags::SequenceEnd | MinibatchPackingFlags::NoFeature))
                {
                    for (size_t u = 0; u < nU; u++)
                    {
                        if (!m_pMBLayout->Is(FrameRange(nT - 1).Sequence(u), MinibatchPackingFlags::SequenceEnd | MinibatchPackingFlags::NoFeature))
                        {
                            allAtBoundary = false;
                            break;
                        }
                    }
                }
                else
                {
                    allAtBoundary = false; 
                }

                if (allAtBoundary)
#endif
                if (!m_pMBLayout->HasSequenceBeyondEnd())       // only need to export state if anything crosses the MB boundary
                {
                    auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId); 
                    pState->CacheDelayedMBLayout(m_delayedActivationMBLayout); 
                    // return an empty one 
                }
                else
                {
                    auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
                    pState->CacheState(m_delayedActivation.ColumnSlice((nT - 1)*nU, nU)); 
                    pState->CacheDelayedMBLayout(m_delayedActivationMBLayout); 
                    pExportedState = pState; 
                }
            }
            if (dir == 1) // we look into future 
            {
#if 0
                // TODO: check whether all at boundary and don't carry state if it is the case 
                size_t nT = m_pMBLayout->GetNumTimeSteps(); 
                size_t nU = m_pMBLayout->GetNumParallelSequences(); 
                bool allAtBoundary = true; 
                if (m_pMBLayout->Is(FrameRange(nullptr, 0), MinibatchPackingFlags::NoFeature | MinibatchPackingFlags::SequenceStart))
                {
                    for (size_t u = 0; u < nU; u++)
                    {
                        if (!m_pMBLayout->Is(FrameRange(nullptr, 0).Sequence(u), MinibatchPackingFlags::SequenceStart | MinibatchPackingFlags::NoFeature))
                        {
                            allAtBoundary = false; 
                            break;
                        }
                    }
                }
                if (allAtBoundary)
#endif
                if (!m_pMBLayout->HasSequenceBeyondBegin())       // only need to export state if anything crosses the MB boundary
                {
                    auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId); 
                    pState->CacheDelayedMBLayout(m_delayedActivationMBLayout); 
                    pExportedState = pState; 
                }
                else
                {
                    auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
                    pState->CacheState(m_delayedActivation.ColumnSlice((nT-1)*nU, nU));
                    pState->CacheDelayedMBLayout(m_delayedActivationMBLayout);
                    pExportedState = pState;
                }
            }
            if (dir != -1 && dir != 1)
            {
                RuntimeError("Unrecognized direction in DelayedValueNodeBase");
            }
            return pExportedState;
        }
        virtual void ImportState(NodeStatePtr && pImportedState) override
        {
            DelayedNodeStatePtr pState = dynamic_pointer_cast<DelayedValueNodeState<ElemType>> (pImportedState); 

            if (!pState)
                RuntimeError("Expecting DelayValueNodeState after down casting"); 

            pState->ExportDelayedMBLayout(m_delayedActivationMBLayout);  // pstate copy to m_delayedActivationMBLayout
            if (pState->IsEmpty())
            {
                return;
            }

            const Matrix<ElemType>& delayedActivation = pState->ExportCachedActivity();
            size_t nT = m_delayedActivationMBLayout->GetNumTimeSteps();
            size_t nU = m_delayedActivationMBLayout->GetNumParallelSequences();

            int dir = direction;
            if (dir == -1) // looking backward 
            {
                m_delayedActivation.SetColumnSlice(delayedActivation, (nT - 1)*nU, nU);
            }
            if (dir == 1)
            {
                //m_delayedActivation.CopyColumnsStrided(delayedActivation, nU, 1, nT);
                m_delayedActivation.SetColumnSlice(delayedActivation, 0, nU);
            }
            if (dir != -1 && dir == 1)
            {// it is really a compile error ? 
                RuntimeError("Unrecognized direction in DelayedValueNodeBase");
            }

        }
    protected:

        ElemType m_initialActivationValue;          // starting value for hidden activation vector at boundary
        Matrix<ElemType> m_delayedActivation;       // saves the activation of the previous step that this node points to
        MBLayoutPtr m_delayedActivationMBLayout;    // layout for m_delayedActivation
        int      m_timeStep;                        // delay in frames (typ. 1)
        function<void()> m_attachInputsFn;          // for late expansion of inputs (scripting)
    };

#define UsingDelayedValueNodeMembers UsingComputationNodeMembersBoilerplate; \
    using Base::m_initialActivationValue; using Base::m_delayedActivation; using Base::m_timeStep;

    // -----------------------------------------------------------------------
    // PastValueNode (input) -- delay node
    // TODO: Can this just be a typedef?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class PastValueNode : public DelayedValueNodeBase<ElemType, -1 /*, MinibatchPackingFlags::SequenceStart*/>
    {
        typedef DelayedValueNodeBase<ElemType, -1/*, MinibatchPackingFlags::SequenceStart*/> Base; UsingDelayedValueNodeMembers;
        static const std::wstring TypeName() { return L"PastValue"; }
    public:
        PastValueNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }
        PastValueNode(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, const TensorShape & sampleLayout, size_t timeStep) :
            Base(deviceId, name, initialActivationValue, sampleLayout, timeStep)
        { }
        PastValueNode(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, size_t numRows, size_t timeStep) :
            PastValueNode(deviceId, name, initialActivationValue, TensorShape(numRows), timeStep)
        { }
        PastValueNode(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp)
        { }
    };

    template class PastValueNode<float>; 
    template class PastValueNode<double>;

    // -----------------------------------------------------------------------
    // FutureValueNode (input) -- delay node in future direction
    // -----------------------------------------------------------------------

    // get value from future (used in the bi-directional models)
    template<class ElemType>
    class FutureValueNode : public DelayedValueNodeBase<ElemType, +1 /*, MinibatchPackingFlags::SequenceEnd*/>
    {
        typedef DelayedValueNodeBase<ElemType, +1 /*, MinibatchPackingFlags::SequenceEnd*/> Base; UsingDelayedValueNodeMembers;
        static const std::wstring TypeName() { return L"FutureValue"; }
    public:
        FutureValueNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }
        FutureValueNode(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, const TensorShape & sampleLayout, size_t timeStep) :
            Base(deviceId, name, initialActivationValue, sampleLayout, timeStep)
        { }
        FutureValueNode(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, size_t numRows, size_t timeStep) :
            FutureValueNode(deviceId, name, initialActivationValue, TensorShape(numRows), timeStep)
        { }
        FutureValueNode(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp)
        { }
    };

    template class FutureValueNode<float>;
    template class FutureValueNode<double>;

}}}
