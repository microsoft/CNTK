//
// <copyright file="RecurrentNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

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

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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
    // -----------------------------------------------------------------------

    // TODO: 'direction' is really too general. signOfTimeOffset?
    template<class ElemType, int direction/*-1 for Past/left-to-right or +1 for Future/right-to-left*/  /*, MinibatchPackingFlags SequenceStart_or_End/*-Start or -End*/>
    class DelayedValueNodeBase : public ComputationNode<ElemType>, public
                                 ILateAttachingNode, public IStateFulNode,  public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        typedef std::shared_ptr<DelayedValueNodeState<ElemType>> DelayedNodeStatePtr; 
        static const std::wstring TypeName() { return L"DelayedValue"; }
    private:
        void Init(size_t row_size, size_t col_size, ElemType initialActivationValue = (ElemType)DEFAULT_HIDDEN_ACTIVATION)
        {
            m_initialActivationValue = initialActivationValue;
            m_timeStep = 1;
            CreateMatrixIfNull(m_value);
            SetDims(row_size, col_size);
            m_isHistoryCarryOverManagedExternally = false;      // used for PairNetworkNode/PastValueNode combination
        }
    protected:
        DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_delayedActivation(deviceId)
        {
            Init(1, 1);
        }
        DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, size_t row_size, size_t col_size, size_t timeStep) :
            Base(deviceId, name),
            m_delayedActivation(deviceId)
        {
            Init(row_size, col_size, initialActivationValue);

            m_timeStep = (int)timeStep;

            m_value->SetValue(m_initialActivationValue);
        }
        DelayedValueNodeBase(const ScriptableObjects::IConfigRecordPtr configp) :
            DelayedValueNodeBase(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"defaultHiddenActivation"), configp->Get(L"rows"), configp->Get(L"cols"), configp->Get(L"timeStep"))
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
            // the node has already been initialized e.g. w.r.t. direction and sequence flags
            Base::Load(fstream, modelVersion);

            fstream >> m_timeStep;

            size_t rows, cols;
            fstream >> rows >> cols;

            // Note: Do we need load cols for delay node? I just set to zero to see if there is any problem.
            SetDims(rows, 0);
            m_delayedActivation.Resize(rows, 0);    // Note: If we try to access history in first minibatch, we shall crash. It would be a consequence of a missing sentence-begin flag

            if (modelVersion >= CNTK_MODEL_VERSION_2)
                fstream >> m_initialActivationValue;
        }

#if 0
    private:
        // cache a post-processed version of m_pMBLayout (depends on the actual minibatch)
        // This post-processed layout has its bits spread out over m_timeStep, to help detect if we'd hop across a boundary.
        void CacheMBLayout()
        {
            if (m_timeStep <= 0)
                LogicError("timeStep should be 1 or larger");

            m_pShiftedMBLayout->CopyFrom(m_pMBLayout);      // it gets modified below
            if (m_timeStep == 1)
                return;

#if 1
            LogicError("CacheMBLayout: m_timeStep > 1 temporarily disabled until MBLayout update completed.");
#else
            // modify m_pShiftedMBLayout
            // If two utterances are packed together (S: start, E: end, N: no input) and we need to get values 2 steps in the past
            //    S X X X E S X X X X E N N
            // then this becomes
            //    S S X X E S S X X X E N N

            size_t numSeq = GetNumParallelSequences();

            // each row has a number to indicate how many values should be reset for that utterance
            // TODO: This algorithm is not obvious and should be explained. E.g. how come it is direction independent?
            vector<int> numResetLeft(numSeq, 0);
            for (size_t i = 0; i < GetNumTimeSteps(); i++)   // i = frame index (time)
            {
                if (m_pMBLayout->Is(i, SequenceStart_or_End | MinibatchPackingFlags::NoFeature))
                {
                    // we set timeStep-1 elements following it to be SequenceStart until met NoInput
                    for (size_t j = 0; j < numSeq; j++)        // j = stream
                    {
                        // we use & since ((int) MinibatchPackingFlags::SequenceStart) may come with NoLabel
                        if (m_pMBLayout->Is(j, i, SequenceStart_or_End))
                            numResetLeft[j] = m_timeStep;
                        else if (m_pMBLayout->Is(j, i, MinibatchPackingFlags::NoFeature))
                            numResetLeft[j] = 0;
                    }
                }

                // now set the sequence-boundary flag
                for (size_t j = 0; j < numSeq; j++)
                {
                    if (numResetLeft[j]-- > 0)
                    {
                        m_pShiftedMBLayout->Mask(j, i, MinibatchPackingFlags::NoLabel); // keep only this flag
                        m_pShiftedMBLayout->Set(j, i, SequenceStart_or_End);            // now implant the boundary flag
                    }
                }
            }
#endif
        }
    public:
#endif

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

        virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const
        {
            // The DelayedValueNode does not require any of it's input's values for computing
            // the gradients of its input nodes
            UNREFERENCED_PARAMETER(childIndex);
            return false;
        }

        //virtual void BeginForwardProp() override      // called before first iteration step of ForwardProp()
        //{
        //    Base::BeginForwardProp();
        //    CacheMBLayout();
        //}

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
            if (!m_isHistoryCarryOverManagedExternally) // means it's externally managed (for PairNetworkNode)
            {
                m_delayedActivation = Input(0)->Value();
                if (!m_delayedActivationMBLayout) m_delayedActivationMBLayout = make_shared<MBLayout>();
                m_delayedActivationMBLayout->CopyFrom(m_pMBLayout);
            }

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

        // this function is only used for PairNetworkNode (on PastValueNode)
        // BUGBUG: Need to transfer the layout as well. PairNetworkNode will go away.
        bool GetHistory(Matrix<ElemType>& hist, bool)
        {
            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            hist.SetValue(Input(0)->Value());

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
            return true;
        }

        // this function is only used for PairNetworkNode (on PastValueNode)
        void SetHistory(const Matrix<ElemType>& hist)
        {
            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_delayedActivation.SetValue(hist);
            m_isHistoryCarryOverManagedExternally = true;

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);

            // need a layout as well
            // ForwardProp() expects it to have the same number of parallel sequences.
            if (!m_delayedActivationMBLayout) m_delayedActivationMBLayout = make_shared<MBLayout>();
            m_delayedActivationMBLayout->Init(GetNumParallelSequences(), hist.GetNumCols() / GetNumParallelSequences());
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
                node->m_isHistoryCarryOverManagedExternally = false;
            }
        }

        //========================================
        // implement the IStateFulNode interface
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
        virtual void ImportState(const NodeStatePtr& pImportedState) override
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
        //MBLayoutPtr m_pShiftedMBLayout;             // individual sentence boundary information     --TODO: do we actually need this separate variable?
        bool m_isHistoryCarryOverManagedExternally; // for PastValueNode only
        function<void()> m_attachInputsFn;          // for late expansion of inputs (scripting)
    };

#define UsingDelayedValueNodeMembers UsingComputationNodeMembersBoilerplate; \
    using Base::m_initialActivationValue; using Base::m_delayedActivation; using Base::m_timeStep; \
    /*using Base::m_pShiftedMBLayout;*/ using Base::m_isHistoryCarryOverManagedExternally;

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
        PastValueNode(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, size_t row_size, size_t col_size, size_t timeStep) :
            Base(deviceId, name, initialActivationValue, row_size, col_size, timeStep)
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

    //get value from future (used in the bi-directional models)
    template<class ElemType>
    class FutureValueNode : public DelayedValueNodeBase<ElemType, +1 /*, MinibatchPackingFlags::SequenceEnd*/>
    {
        typedef DelayedValueNodeBase<ElemType, +1 /*, MinibatchPackingFlags::SequenceEnd*/> Base; UsingDelayedValueNodeMembers;
        static const std::wstring TypeName() { return L"FutureValue"; }
    public:
        FutureValueNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }
        FutureValueNode(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, size_t row_size, size_t col_size, size_t timeStep) :
            Base(deviceId, name, initialActivationValue, row_size, col_size, timeStep)
        { }
        FutureValueNode(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp)
        { }
    };

    template class FutureValueNode<float>;
    template class FutureValueNode<double>;

}}}
