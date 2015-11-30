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
    // The following defines a state of a delay node which is going to be exported to others (saving for the next minibatch)
    // -----------------------------------------------------------------------
    template<class ElemType>
    class DelayedValueNodeState: public INodeState
    {
               
        public:
            DelayedValueNodeState(int deviceID) :
                m_cachedActivity((size_t)0, (size_t)0, deviceID), m_delayedActivationMBLayout(nullptr), m_isEmpty(true)
            {

            }
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
    template<class ElemType, int direction/*-1 for Past/left-to-right or +1 for Future/right-to-left*/, MinibatchPackingFlags SequenceStart_or_End/*-Start or -End*/>
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
            CreateMatrixIfNull(m_functionValues);
            SetDims(row_size, col_size);
            //m_delayedActivation.Resize(row_size, col_size);     // TODO: relevance of col_size? Why not timeStep?
            m_isHistoryCarryOverManagedExternally = false;      // used for PairNetworkNode/PastValueNode combination
        }
    protected:
        DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_delayedActivation(deviceId), m_pShiftedMBLayout(make_shared<MBLayout>())
        {
            Init(1, 1);
        }
        DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring & name, ElemType initialActivationValue, size_t row_size, size_t col_size, size_t timeStep) :
            Base(deviceId, name),
            m_delayedActivation(deviceId), m_pShiftedMBLayout(make_shared<MBLayout>())
        {
            Init(row_size, col_size, initialActivationValue);

            m_timeStep = (int)timeStep;

            m_functionValues->SetValue(m_initialActivationValue);
            //m_delayedActivation.SetValue(m_initialActivationValue);

            //m_gradientValues->Resize(row_size, col_size);
            //m_gradientValues->SetValue(0.0f);
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
        void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);

            fstream << m_timeStep;
            fstream << GetNumRows() << GetNumCols();

            fstream << m_initialActivationValue;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            // the node has already been initialized e.g. w.r.t. direction and sequence flags
            Base::LoadFromFile(fstream, modelVersion);

            fstream >> m_timeStep;

            size_t rows, cols;
            fstream >> rows >> cols;

            // Note: Do we need load cols for delay node? I just set to zero to see if there is any problem.
            SetDims(rows, 0);
            m_delayedActivation.Resize(rows, 0);    // Note: If we try to access history in first minibatch, we shall crash. It would be a consequence of a missing sentence-begin flag

            if (modelVersion >= CNTK_MODEL_VERSION_2)
                fstream >> m_initialActivationValue;
        }

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
        }
    public:

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            assert(inputIndex == 0); inputIndex;

            // special case: DelayedValueNodes may be used outside of loops
            // TODO: this should be a bulk operation; this implementation is a quick hack
            int dir = direction;    // (this avoids a 'conditional expression is constant' warning)
            if (frameRange.IsAllFrames())
            {
                // recursive call to ourselves
                FrameRangeIteration range(m_pMBLayout, -dir);
                for (auto t = range.rbegin(); t != range.rend(); t++)   // note: reverse iterator
                    ComputeInputPartial(inputIndex, t);
                return;
            }

            size_t t = frameRange.t();

            // if delayed input is within valid time range then add its gradient
            int t_delayed = (int)t + direction * m_timeStep;
            if (t_delayed >= 0 && t_delayed < GetNumTimeSteps())
            {
                // if there is a boundary in this frame, we treat each stream separately; otherwise we do all in one go
                if (m_pShiftedMBLayout->Is(t, SequenceStart_or_End | MinibatchPackingFlags::NoFeature)) // true if at least one parallel sequence has a boundary or gap
                {
                    size_t mNbr = m_pMBLayout->GetNumParallelSequences();
                    for (size_t id = 0; id < mNbr; id++)
                    {
                        if (!m_pShiftedMBLayout->Is(id, t, SequenceStart_or_End | MinibatchPackingFlags::NoFeature))    // don't propagate boundary frames or gaps
                        {
                            Matrix<ElemType> frm = GradientSlice(frameRange.Sequence(id));
                            Matrix<ElemType> to = Inputs(0)->GradientSlice(FrameRange(m_pMBLayout, t_delayed).Sequence(id));
                            to += frm;
                        }
                    }
                }
                else    // operate on entire time step in one go (over all parallel sequences)
                {
                    Matrix<ElemType> frm = GradientSlice(frameRange);
                    Matrix<ElemType> to = Inputs(0)->GradientSlice(FrameRange(m_pMBLayout, t_delayed)); // TODO: we need to be able to create a FrameRange with a delta, not like this
                    to += frm;
                }
            }
        }

        virtual void OnEvaluateBeginIteration() override      // called before first iteration step of EvaluateThisNode()
        {
            Base::OnEvaluateBeginIteration();
            CacheMBLayout();
        }

        virtual void OnEvaluateEndIteration() override        // called after last iteration step of EvaluateThisNode()
        {
            // In BPTT, we carry over left-to-right state across minibatches.
            // It is kept in m_delayedActivation, m_delayedActivationMBLayout.
            // This could be optimized as follows:
            //  - only keep the required number of frames (m_timeStep)
            //  - we don't need to keep anything in full-sequence mode
            //  - we don't need to keep anything if all sequences are closed (sentence end)
            //    This condition includes full-sequence mode.
            // BUGBUG: The code does not check for skipping across an utterance boundary in the previosu minibatch. Only affects BPTT with timeStep > 1.
            //         Maybe we should carry over the shifted layout (cf. CacheMBLayout())? We could just move the pointer. But the code must be adapted as well.
            if (!m_isHistoryCarryOverManagedExternally) // means it's externally managed (for PairNetworkNode)
            {
                m_delayedActivation = Inputs(0)->FunctionValues();
                if (!m_delayedActivationMBLayout) m_delayedActivationMBLayout = make_shared<MBLayout>();
                m_delayedActivationMBLayout->CopyFrom(m_pMBLayout);
            }

            Base::OnEvaluateEndIteration();
        }

        // This function assumes OnEvaluateBegin/EndIteration() to be called before/after the iteration loop.
        // TODO: In the future, there may be value for one more way of handling the boundary condition: Fill as 'NoInput'. Then we can use this to implement rolling windows (albeit inefficiently). Would require to unshare the layout.
        virtual void EvaluateThisNode(const FrameRange & frameRange) override
        {
            assert(m_pMBLayout);

            // special case: DelayedValueNodes may be used outside of loops
            // TODO: this should be a bulk operation; this implementation is a quick hack
            int dir = direction;    // (this avoids a 'conditional expression is constant' warning)
            if (frameRange.IsAllFrames())
            {
                // recursive call to ourselves
                FrameRangeIteration range(m_pMBLayout, -dir);
                for (auto t = range.begin(); t != range.end(); t++)
                    EvaluateThisNode(t);
                return;
            }

            size_t t = frameRange.t();

            VerifyDims(Inputs(0));

            size_t T = GetNumTimeSteps();
            size_t T_delayedActivation = m_delayedActivationMBLayout ? m_delayedActivationMBLayout->GetNumTimeSteps() : 0;  // (note: should never happen in full-sequence mode)

            // compute logical position of delayed value
            assert(m_timeStep > 0);

            int t_delayed = (int)(t + direction * m_timeStep);  // this might end up outside the current window

            Matrix<ElemType> inp;   // ((DEVICEID_TYPE)m_functionValues.GetDeviceId());

            // if any sequence at this time step has a boundary flag, then process one by one
            // TODO: Would there be an efficiency gain from grouping consecutive sequences with identical flags?
            if (m_pShiftedMBLayout->Is(t, SequenceStart_or_End))
            {
                for (size_t id = 0; id < GetNumParallelSequences(); id++)
                {
                    Matrix<ElemType> out = ValueSlice(frameRange.Sequence(id));

                    if (m_pShiftedMBLayout->Is(id, t, SequenceStart_or_End))
                        out.SetValue(m_initialActivationValue);     // crossed a boundary
                    else    // not a boundary: just copy the delayed value
                    {
                        // inside the sequence: access delayed value
                        if (t_delayed < 0)
                            inp = DataSliceWithMBLayout(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed + T_delayedActivation).Sequence(id), m_delayedActivationMBLayout); // delay reaches in previous minibatch
                        else if (t_delayed >= T)
                            inp = DataSliceWithMBLayout(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed - T).Sequence(id), m_delayedActivationMBLayout); // delay reaches in previous minibatch
                        else
                            inp = Inputs(0)->ValueSlice(FrameRange(m_pMBLayout, t_delayed).Sequence(id));

                        out.SetValue(inp);
                    }
                }
            }
            else        // frame has no boundary flags: use ValueSlice directly (still may have a gap here)
            {
                Matrix<ElemType> out = ValueSlice(frameRange);

                if (t_delayed < 0)
                    inp = DataSliceWithMBLayout(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed + T_delayedActivation), m_delayedActivationMBLayout);
                else if (t_delayed >= T)
                    inp = DataSliceWithMBLayout(m_delayedActivation, FrameRange(m_delayedActivationMBLayout, t_delayed - T), m_delayedActivationMBLayout);
                else
                    inp = Inputs(0)->ValueSlice(FrameRange(m_pMBLayout, t_delayed));

                out.SetValue(inp);
            }

            //MaskMissingValuesColumnsToZero(t);  // fix gaps if any  --TODO: make this take a FrameRange
            // TODO: why is masking needed here? We should never carry over data from those into valid regions, right?
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateUnaryMap(isFinalValidationPass);
        }

        // this function is only used for PairNetworkNode (on PastValueNode)
        // BUGBUG: Need to transfer the layout as well. PairNetworkNod will go away.
        bool GetHistory(Matrix<ElemType>& hist, bool)
        {
            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            hist.SetValue(Inputs(0)->FunctionValues());

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
            // EvaluateThisNode() expects it to have the same number of parallel sequences.
            if (!m_delayedActivationMBLayout) m_delayedActivationMBLayout = make_shared<MBLayout>();
            m_delayedActivationMBLayout->Init(GetNumParallelSequences(), hist.GetNumCols() / GetNumParallelSequences(), true/*sequential*/);
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<DelayedValueNodeBase<ElemType, direction, SequenceStart_or_End>>(nodeP);
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
                bool   allAtBoundary = true;
                // if the current last frames are all sentence end or no feature , there is no need to carry on state info
                if (m_pMBLayout->Is(nT-1, MinibatchPackingFlags::SequenceEnd | MinibatchPackingFlags::NoFeature))
                {
                    for (size_t u = 0; u < nU; u++)
                    {
                        if (!m_pMBLayout->Is(u, nT - 1, MinibatchPackingFlags::SequenceEnd | MinibatchPackingFlags::NoFeature))
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
                {
                    auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId); 
                    pState->CacheDelayedMBLayout(m_delayedActivationMBLayout); 
                    // return an empty one 
                }
                else
                {
                    auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
                    //pState->CacheState(FunctionValues().Reshaped(nD*nU, nT).RowSlice(nD*(nT - 1), nD));
                    pState->CacheState(m_delayedActivation.ColumnSlice((nT - 1)*nU, nU)); 
                    pState->CacheDelayedMBLayout(m_delayedActivationMBLayout); 
                    pExportedState = pState; 
                }
            }
            if (dir == 1) // we look into future 
            {
                // TODO: check whether all at boundary and don't carry state if it is the case 
                size_t nT = m_pMBLayout->GetNumTimeSteps(); 
                size_t nU = m_pMBLayout->GetNumParallelSequences(); 
                bool allAtBoundary = true; 
                if (m_pMBLayout->Is(0, MinibatchPackingFlags::NoFeature | MinibatchPackingFlags::SequenceStart))
                {
                    for (size_t u = 0; u < nU; u++)
                    {
                        if (!m_pMBLayout->Is(u, 0, MinibatchPackingFlags::SequenceStart | MinibatchPackingFlags::NoFeature))
                        {
                            allAtBoundary = false; 
                            break;
                        }
                    }
                }

                if (allAtBoundary)
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
        virtual void ImportState(const NodeStatePtr& pImportedState)
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
        MBLayoutPtr m_pShiftedMBLayout;             // individual sentence boundary information     --TODO: do we actually need this separate variable?
        bool m_isHistoryCarryOverManagedExternally; // for PastValueNode only
        function<void()> m_attachInputsFn;          // for late expansion of inputs (scripting)
    };

#define UsingDelayedValueNodeMembers UsingComputationNodeMembersBoilerplate; \
    using Base::m_initialActivationValue; using Base::m_delayedActivation; using Base::m_timeStep; \
    using Base::m_pShiftedMBLayout; using Base::m_isHistoryCarryOverManagedExternally;

    // =======================================================================
    // -----------------------------------------------------------------------
    // PastValueNode (input) -- delay node
    // TODO: Can this just be a typedef?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class PastValueNode : public DelayedValueNodeBase<ElemType, -1, MinibatchPackingFlags::SequenceStart>
    {
        typedef DelayedValueNodeBase<ElemType, -1, MinibatchPackingFlags::SequenceStart> Base; UsingDelayedValueNodeMembers;
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
    class FutureValueNode : public DelayedValueNodeBase<ElemType, +1, MinibatchPackingFlags::SequenceEnd>
    {
        typedef DelayedValueNodeBase<ElemType, +1, MinibatchPackingFlags::SequenceEnd> Base; UsingDelayedValueNodeMembers;
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

    // -----------------------------------------------------------------------
    // LSTMNode (obs, inputGate, forgetGate, outputGate, memoryCellWgt)
    // deprecated early implementation of LSTM operating on minibatches directly
    //  - input(0) : child with dimension [inputdim x T]
    //  - input(1) : input gate [outputdim x [inputdim + outputdim + 2]] bi, Wxi, Whi, Wci
    //  - input(2) : forget gate [outputdim x [inputdim + outputdim + 2]] for bf, Wxf, Whf, Wcf
    //  - input(3) : output gate [outputdim x [inputdim + outputdim + 2]] for bo, Wxo, Who, and Wco
    //  - input(4) : memory cell weight [outputdim x [inputdim + outputdim + 1]] for bc, Wxc, and Whc 
    //  - output : dimension [outputdim x T]
    // -----------------------------------------------------------------------

    /**
    LSTM specific node. This node uses matrix operations to have LSTM functionality. 
    It avoids using general recurrent loop operations in the network operations in ComputationNetwork.

    Developed by Kaisheng Yao
    Used in the following works:
    K. Yao, G. Zweig, "Sequence to sequence neural net models for graphone to phoneme conversion", in Interspeech 2015
    */
    template<class ElemType>
    class LSTMNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<5>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"LSTM"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(LSTMNode);
        LSTMNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name),
            m_State(deviceId), m_PastState(deviceId),
            m_PastOutput(deviceId), m_Gi(deviceId), m_Gf(deviceId), m_Go(deviceId), grdToObs(deviceId), grdToInputGate(deviceId),
            grdToForgetGate(deviceId), grdToOutputGate(deviceId), grdToCellWgt(deviceId), tanhObs(deviceId),
            tanhState(deviceId), m_tempMatrix(deviceId),
            mSlicePrevState(deviceId), mSlicePrevOutput(deviceId),
            grdBeforeInputGate(deviceId),
            grdBeforeForget(deviceId), grdBeforeGo(deviceId), grdToCell(deviceId),
            grdBeforeTanhInputGate(deviceId), m_obs_error_from_future_minibatch(deviceId),
            m_state_error_from_future_minibatch(deviceId), mLastState(deviceId), mLastOutput(deviceId),
            m_inputDim(0),
            m_outputDim(0),
            m_use_errors_from_future_minibatch(false),
            m_DefaultState((ElemType)DEFAULT_HIDDEN_ACTIVATION)
        {
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_inputDim << m_outputDim;
            fstream << m_DefaultState;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            if (modelVersion == 2)
                fstream >> m_inputDim >> m_outputDim;
            fstream >> m_DefaultState;
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<LSTMNode<ElemType>>(nodeP);
                node->m_inputDim = m_inputDim;
                node->m_outputDim = m_outputDim;

                node->m_State = m_State;  // hidden state activity
                node->m_PastState = m_PastState; // state activity in the previous minibatch
                node->m_PastOutput = m_PastOutput; // output in the previou minibatch 

                node->m_Gi = m_Gi;     // input gate activity
                node->m_Gf = m_Gf;     // forget gate activity
                node->m_Go = m_Go;     // output gate activity

                node->mSlicePrevOutput = mSlicePrevOutput;
                node->mSlicePrevState = mSlicePrevState;

                node->m_use_errors_from_future_minibatch = m_use_errors_from_future_minibatch;

                node->m_DefaultState = m_DefaultState;
            }
        }

        virtual void ComputeInputPartialNonLooping(size_t inputIndex) override
        {
            if (inputIndex > 4)
                InvalidArgument("LSTM operation only takes five inputs.");

            size_t nT = Inputs(0)->GetNumCols();
            size_t inputDim = Inputs(0)->GetNumRows();
            size_t outputDim = Inputs(1)->GetNumRows();

            if (m_GradientComputed == false)
            {
                if (GetNumCols() != GradientValues().GetNumCols() ||
                    GetNumRows() != GradientValues().GetNumRows())
                {
                    RuntimeError("LSTMNode::GradientValue size doesn't match to the function value size");
                }

                // reset gradients
                grdToObs.Resize(inputDim, nT); grdToObs.SetValue(0);
                grdToInputGate.Resize(Inputs(1)->GetNumRows(), Inputs(1)->GetNumCols()); grdToInputGate.SetValue(0);
                grdToForgetGate.Resize(Inputs(2)->GetNumRows(), Inputs(2)->GetNumCols()); grdToForgetGate.SetValue(0);
                grdToOutputGate.Resize(Inputs(3)->GetNumRows(), Inputs(3)->GetNumCols()); grdToOutputGate.SetValue(0);
                grdToCellWgt.Resize(Inputs(4)->GetNumRows(), Inputs(4)->GetNumCols()); grdToCellWgt.SetValue(0);

                Matrix<ElemType> slicePrevOutput(m_deviceId), slicePrevState(m_deviceId);
                Matrix<ElemType> grdToPrevOutput(m_deviceId), grdToPrevState(m_deviceId);
                Matrix<ElemType> stateError(m_deviceId);
                slicePrevState.Resize(outputDim, GetNumParallelSequences());
                slicePrevOutput.Resize(outputDim, GetNumParallelSequences());
                slicePrevOutput.SetValue(0);

                stateError.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());

                grdToPrevOutput.Resize(slicePrevOutput.GetNumRows(), slicePrevOutput.GetNumCols());
                grdToPrevState.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());
                grdToPrevOutput.SetValue(0);
                grdToPrevState.SetValue(0);

                for (int timeIdxInSeq = nT - GetNumParallelSequences(); timeIdxInSeq >= 0; timeIdxInSeq -= GetNumParallelSequences())
                {
                    FrameRange frameRange(m_pMBLayout, timeIdxInSeq);
                    Matrix<ElemType> sliceObs = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceOutput = ValueSlice(frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceState = DataSlice(m_State, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));

                    Matrix<ElemType> sliceGi = DataSlice(m_Gi, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceGf = DataSlice(m_Gf, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceGo = DataSlice(m_Go, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));

                    Matrix<ElemType> sliceTanhState = DataSlice(tanhState, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceTanhObs = DataSlice(tanhObs, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));

                    Matrix<ElemType> error = GradientSlice(frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout));

                    Matrix<ElemType> grdToObsSlice(this->m_deviceId);


#ifdef DEBUG_DECODER
                    fprintf(stderr, "original output error [%ld] norm = %.8e\n", timeIdxInSeq, error.FrobeniusNorm());
#endif

                    PrepareThisErrorsBeforeBackProp(timeIdxInSeq, nT, error, stateError, grdToPrevOutput, grdToPrevState,
                                                    m_obs_error_from_future_minibatch, m_state_error_from_future_minibatch, GetNumParallelSequences(), &m_pMBLayout->GetM());

#ifdef DEBUG_DECODER
                    fprintf(stderr, "output error [%ld] norm = %.8e\n", timeIdxInSeq, error.FrobeniusNorm());
                    fprintf(stderr, "state error [%ld] norm = %.8e\n", timeIdxInSeq, stateError.FrobeniusNorm());
#endif

                    grdToPrevOutput.Resize(slicePrevOutput.GetNumRows(), slicePrevOutput.GetNumCols());
                    grdToPrevState.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());
                    grdToPrevOutput.SetValue(0);
                    grdToPrevState.SetValue(0);

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, FunctionValues(), m_State, m_PastOutput, m_PastState, GetNumParallelSequences(), m_DefaultState, &m_pMBLayout->GetM());

                    ComputeInputGradientWrtGates(
                        error,
                        sliceObs,
                        grdToObsSlice,
                        Inputs(1)->FunctionValues(),
                        grdToInputGate,
                        Inputs(2)->FunctionValues(),
                        grdToForgetGate,
                        Inputs(3)->FunctionValues(),
                        grdToOutputGate,
                        Inputs(4)->FunctionValues(),
                        grdToCellWgt,
                        mSlicePrevOutput,
                        mSlicePrevState,
                        stateError,
                        sliceState,
                        sliceTanhState,
                        sliceTanhObs,
                        sliceGi,
                        sliceGf,
                        sliceGo,
                        grdToPrevOutput,
                        grdToPrevState,
                        m_tempMatrix
                    );
                    DataSlice(grdToObs, frameRange/*TODO: delete this:*/.Check(timeIdxInSeq, GetNumParallelSequences(), m_pMBLayout)).SetValue(grdToObsSlice);

                    PrepareErrors(timeIdxInSeq, grdToPrevOutput, grdToPrevState, GetNumParallelSequences(), &m_pMBLayout->GetM());
                }
#ifdef DEBUG_DECODER
                fprintf(stderr, "after error prop b_c norm = %.8e\n", Inputs(4)->FunctionValues().ColumnSlice(0, 1).FrobeniusNorm());
#endif
                m_obs_error_from_future_minibatch = grdToPrevOutput;
                m_state_error_from_future_minibatch = grdToPrevState;


#ifdef DEBUG_DECODER
                fprintf(stderr, "pass error to encoder error = %.4e state error = %.4e\n", m_obs_error_from_future_minibatch.FrobeniusNorm(), m_state_error_from_future_minibatch.FrobeniusNorm());
#endif
                m_GradientComputed = true;
            }

            if (inputIndex == 0)  //derivative with regard to the observation
            {
                if (Inputs(inputIndex)->GradientValues().HasNoElements())
                    Inputs(inputIndex)->GradientValues().SetValue(grdToObs);
                else
                    Inputs(inputIndex)->GradientValues() += grdToObs;
            }

            if (inputIndex == 1)
            {
                if (Inputs(inputIndex)->GradientValues().HasNoElements())
                    Inputs(inputIndex)->GradientValues().SetValue(grdToInputGate);
                else
                    Inputs(inputIndex)->GradientValues() += grdToInputGate;
            }

            if (inputIndex == 2)
            {
                if (Inputs(inputIndex)->GradientValues().HasNoElements())
                    Inputs(inputIndex)->GradientValues().SetValue(grdToForgetGate);
                else
                    Inputs(inputIndex)->GradientValues() += grdToForgetGate;
            }

            if (inputIndex == 3)
            {
                if (Inputs(inputIndex)->GradientValues().HasNoElements())
                    Inputs(inputIndex)->GradientValues().SetValue(grdToOutputGate);
                else
                    Inputs(inputIndex)->GradientValues() += grdToOutputGate;
            }

            if (inputIndex == 4)
            {
                if (Inputs(inputIndex)->GradientValues().HasNoElements())
                    Inputs(inputIndex)->GradientValues().SetValue(grdToCellWgt);
                else
                    Inputs(inputIndex)->GradientValues() += grdToCellWgt;
            }
#ifdef DEBUG_DECODER
            fprintf(stderr, "LSTM gradient[%d] norm = %.8e\n", inputIndex, Inputs(inputIndex)->GradientValues().FrobeniusNorm());
#endif

        }

        static void WINAPI GradientOfTanh(const Matrix<ElemType>& functionValues,
            const Matrix<ElemType>& gradientOut,
            Matrix<ElemType>& inputGradientValues,
            Matrix<ElemType>& extTmp)
        {
            Matrix<ElemType> mTmp(inputGradientValues.GetDeviceId());
            extTmp.AssignElementProductOf(functionValues, functionValues); // v .* v
            mTmp.AssignDifferenceOf(1, extTmp); // 1-v^2
            if (inputGradientValues.GetNumRows() != functionValues.GetNumRows() ||
                inputGradientValues.GetNumCols() != functionValues.GetNumCols())
                LogicError("LSTMNode::GradientOfTanh : inputGradientValues need to be pre-allocated!");
            inputGradientValues.AddElementProductOf(gradientOut, mTmp); //  d .* ((1-v) .* v))
        }

        static void WINAPI ComputeInputGradientWrtGates(
            const Matrix<ElemType>& outGrd,  // the error to h_t from upper layer
            const Matrix<ElemType> & obs,
            Matrix<ElemType> &grdToObs,
            const Matrix<ElemType>& mInputGate,
            Matrix<ElemType> &grdToInputGate,
            const Matrix<ElemType> &mForgetGate,
            Matrix<ElemType> &grdToForgetGate,
            const Matrix<ElemType> &mOutputGate,
            Matrix<ElemType>& grdToOutputGate,
            const Matrix<ElemType> &mCellWgt,
            Matrix<ElemType> &grdToCellWgt,
            const Matrix<ElemType>& prevOutput,
            const Matrix<ElemType>& prevState,
            const Matrix<ElemType>& stateError,  // the error propagated to cell from t+1
            const Matrix<ElemType> &state,
            const Matrix<ElemType> &tanhState,
            const Matrix<ElemType> & tanhBeforeApplyingInputGating,
            const Matrix<ElemType> &gi,
            const Matrix<ElemType> &gf,
            const Matrix<ElemType> &go,
            Matrix<ElemType> &grdToPrevOutput,
            Matrix<ElemType> &grdToPrevState,
            Matrix<ElemType> & tmpMat
            )
        {
            int inputDim = obs.GetNumRows();
            int outputDim = mOutputGate.GetNumRows();

            assert(grdToPrevOutput.FrobeniusNorm() == 0);
            assert(grdToPrevState.FrobeniusNorm() == 0);
            assert(state.FrobeniusNorm() > 0);
            Matrix<ElemType> Who = mOutputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wco = mOutputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxo = mOutputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWho = grdToOutputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWco = grdToOutputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxo = grdToOutputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobo = grdToOutputGate.ColumnSlice(0, 1);

            Matrix<ElemType> Whf = mForgetGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wcf = mForgetGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxf = mForgetGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhf = grdToForgetGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWcf = grdToForgetGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxf = grdToForgetGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobf = grdToForgetGate.ColumnSlice(0, 1);

            Matrix<ElemType> Wxc = mCellWgt.ColumnSlice(1, inputDim);
            Matrix<ElemType> Whc = mCellWgt.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWxc = grdToCellWgt.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhc = grdToCellWgt.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdTobc = grdToCellWgt.ColumnSlice(0, 1);

            Matrix<ElemType> Whi = mInputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wci = mInputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxi = mInputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhi = grdToInputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWci = grdToInputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxi = grdToInputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobi = grdToInputGate.ColumnSlice(0, 1);

            // error backpropagate to output gate
            Matrix<ElemType> grdToGo(tmpMat.GetDeviceId()), gradientOfSigmoid(tmpMat.GetDeviceId());
            Matrix<ElemType> grdBeforeGo(tmpMat.GetDeviceId()), grdBeforeInputGate(tmpMat.GetDeviceId());
            Matrix<ElemType> grdToCell(tmpMat.GetDeviceId());

            tmpMat.AssignElementProductOf(outGrd, tanhState);  // error to o_t
            gradientOfSigmoid.AssignSigmoidDerivativeOf(go);
            grdBeforeGo.AssignElementProductOf(tmpMat, gradientOfSigmoid);  // error before softmax
#ifdef DEBUG_DECODER
            fprintf(stderr, "output gate error = %.4e\n", grdBeforeGo(0, 0));
#endif
            Matrix<ElemType>::MultiplyAndAdd(Who, true, grdBeforeGo, false, grdToPrevOutput);  // error to previous output
            Matrix<ElemType>::MultiplyAndAdd(Wxo, true, grdBeforeGo, false, grdToObs);      // error to observation 
            tmpMat = grdBeforeGo;
            tmpMat.ColumnElementMultiplyWith(Wco);
            grdToCell = tmpMat;                                                            // error to memory cell

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeGo, false, prevOutput, true, grdToWho); // gradient to Who
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeGo, false, obs, true, grdToWxo); // gradient to Wxo
            tmpMat.AssignInnerProductOf(grdBeforeGo, state, false);
            grdToWco += tmpMat;                    // to Wco
            for (size_t i = 0; i < grdBeforeGo.GetNumCols(); i++)
            {
                grdTobo += grdBeforeGo.ColumnSlice(i, 1);  // gradient to bo
            }

            grdToGo.AssignElementProductOf(outGrd, go);  // error to tanh
            GradientOfTanh(tanhState, grdToGo, grdToCell, tmpMat); // error to memory cell
            grdToCell += stateError; // add error to memory cell from t+1
#ifdef DEBUG_DECODER
            fprintf(stderr, "previous state[0] = %.4e norm = %.4e\n", prevState(0, 0), prevState.FrobeniusNorm());
            fprintf(stderr, "state error = %.4e\n", grdToCell(0, 0));
            fprintf(stderr, "state error norm = %.4e\n", grdToCell.FrobeniusNorm());
#endif
            // error backpropagate to memory cells
            grdToPrevState.AssignElementProductOf(gf, grdToCell);  // error to previous memory cell
            // be careful, need to double check if errors are missing

            Matrix<ElemType> grdBeforeForget(tmpMat.GetDeviceId());
            tmpMat.AssignElementProductOf(prevState, grdToCell);  // error to f_t
            gradientOfSigmoid.AssignSigmoidDerivativeOf(gf);
            grdBeforeForget.AssignElementProductOf(gradientOfSigmoid, tmpMat); // error before forget gate
#ifdef DEBUG_DECODER
            fprintf(stderr, "forget gate error = %.4e\n", grdBeforeForget(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whf, true, grdBeforeForget, false, grdToPrevOutput);  // error to previous output
            tmpMat = grdBeforeForget;
            tmpMat.ColumnElementMultiplyWith(Wcf);
            grdToPrevState += tmpMat;                                                            // error to previous state

            Matrix<ElemType>::MultiplyAndAdd(Wxf, true, grdBeforeForget, false, grdToObs);  // error to observation

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeForget, false, prevOutput, true, grdToWhf); // gradient to Whf
            tmpMat.AssignInnerProductOf(grdBeforeForget, prevState, false);
            grdToWcf += tmpMat;                                                             // gradient to Wcf

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeForget, false, obs, true, grdToWxf); // gradient to Wxf
            for (size_t i = 0; i < grdBeforeForget.GetNumCols(); i++)
                grdTobf += grdBeforeForget.ColumnSlice(i, 1);                                                    // gradient to bf

            // error backpropagate to input gate
            tmpMat.AssignElementProductOf(tanhBeforeApplyingInputGating, grdToCell);
            gradientOfSigmoid.AssignSigmoidDerivativeOf(gi);
            grdBeforeInputGate.AssignElementProductOf(gradientOfSigmoid, tmpMat); // error before input gate
#ifdef DEBUG_DECODER
            fprintf(stderr, "input gate error = %.4e\n", grdBeforeInputGate(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whi, true, grdBeforeInputGate, false, grdToPrevOutput);  // error to previous output
            tmpMat = grdBeforeInputGate;
            tmpMat.ColumnElementMultiplyWith(Wci);
            grdToPrevState += tmpMat;                                                            // error to previous state

#ifdef DEBUG_DECODER
            fprintf(stderr, "to previous state error = %.4e\n", grdToPrevState(0, 0));
            fprintf(stderr, "to previous state error norm = %.4e\n", grdToPrevState.FrobeniusNorm());
#endif
            Matrix<ElemType>::MultiplyAndAdd(Wxi, true, grdBeforeInputGate, false, grdToObs);  // error to observation

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeInputGate, false, prevOutput, true, grdToWhi); // gradient to Whi
            tmpMat.AssignInnerProductOf(grdBeforeInputGate, prevState, false);
            grdToWci += tmpMat;                                                             // gradient to Wci
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeInputGate, false, obs, true, grdToWxi); // gradient to Wxi
            for (size_t i = 0; i < grdBeforeInputGate.GetNumCols(); i++)
                grdTobi += grdBeforeInputGate.ColumnSlice(i, 1);                                                  // gradient to bi

            // error backpropagate to inputs
            Matrix<ElemType> grdTmp2(tmpMat.GetDeviceId());
            Matrix<ElemType> grdBeforeTanhInputGate(tmpMat.GetDeviceId());
            grdTmp2.AssignElementProductOf(gi, grdToCell);
            grdBeforeTanhInputGate.Resize(tanhBeforeApplyingInputGating.GetNumRows(), tanhBeforeApplyingInputGating.GetNumCols());
            GradientOfTanh(tanhBeforeApplyingInputGating, grdTmp2, grdBeforeTanhInputGate, tmpMat); // error to memory cell
            Matrix<ElemType>::MultiplyAndAdd(Wxc, true, grdBeforeTanhInputGate, false, grdToObs);  // error to observation
#ifdef DEBUG_DECODER
            fprintf(stderr, "to observation error = %.4e\n", grdToObs(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whc, true, grdBeforeTanhInputGate, false, grdToPrevOutput);  // error to previous output
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeTanhInputGate, false, obs, true, grdToWxc); // gradient to Wxc

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeTanhInputGate, false, prevOutput, true, grdToWhc); // gradient to Whc
            for (size_t i = 0; i < grdBeforeTanhInputGate.GetNumCols(); i++)
                grdTobc += grdBeforeTanhInputGate.ColumnSlice(i, 1);                                                    // gradient to bc

        }

        /**
        get the segmentation information, SENTENECE_BEGIN, ((int) MinibatchPackingFlags::None), ((int) MinibatchPackingFlags::NoInput) 
        for time at t and stream of streamid
        */
        int GetSegInfo(size_t t, size_t streamid)
        {
            if (streamid >= GetNumParallelSequences())
                LogicError("GetSegInfo: stream id %d is larger than the number of streams %d", (int)streamid, (int)GetNumParallelSequences());

            size_t nT = Inputs(0)->GetNumCols();
            if (t >= nT)
                LogicError("GetSegInfo: time %d times is larger than the total number of observations %d", (int)t, (int)nT);

            int utt_t = (int)t / GetNumParallelSequences();
            auto thisCol = m_pMBLayout->GetFrame(utt_t).first;
            thisCol.Reshape(1, GetNumParallelSequences());
            return (int) thisCol.ColumnSlice(streamid, 1).Get00Element();
        }

        /**
        save the last hidden layer activity and output
        */
        void SaveLastStateActity()
        {
            size_t nT = Inputs(0)->GetNumCols();
            size_t outputDim = Inputs(1)->GetNumRows();
            
            // save the hidden activities and output for the next minibatch
            mLastOutput.Resize(outputDim, GetNumParallelSequences());
            mLastState.Resize(outputDim, GetNumParallelSequences());

            for (size_t i = 0; i < GetNumParallelSequences(); i++)
            {
                for (int t = nT - GetNumParallelSequences() + i; t >= 0; t -= GetNumParallelSequences())
                {
                    if (GetSegInfo(t, i) == ((int) MinibatchPackingFlags::None))
                    {
                        mLastOutput.ColumnSlice(i, 1).SetValue(FunctionValues().ColumnSlice(t, 1));
                        mLastState.ColumnSlice(i, 1).SetValue(m_State.ColumnSlice(t, 1));
                        break;
                    }
                }
            }
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            size_t nT = Inputs(0)->GetNumCols();
            size_t outputDim = Inputs(1)->GetNumRows();

            {
                SetDims(outputDim, nT);
                FunctionValues().SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_State.Resize(outputDim, nT);
                m_State.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Gi.Resize(outputDim, nT);
                m_Gi.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Gf.Resize(outputDim, nT);
                m_Gf.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Go.Resize(outputDim, nT);
                m_Go.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhState.Resize(outputDim, nT);
                tanhState.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhObs.Resize(outputDim, nT);
                tanhObs.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 

                if (m_PastState.IsEmpty() || m_PastState.GetNumCols() != GetNumParallelSequences())
                {
                    m_PastState.Resize(outputDim, GetNumParallelSequences());
                    m_PastState.SetValue(m_DefaultState);
                }
                if (m_PastOutput.IsEmpty() || m_PastOutput.GetNumCols() != GetNumParallelSequences())
                {
                    m_PastOutput.Resize(outputDim, GetNumParallelSequences());
                }

#ifdef DEBUG_DECODER
                if (m_PastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past output norm = %.8e\n", this->NodeName().c_str(), m_PastOutput.FrobeniusNorm());
                if (m_PastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past state norm = %.8e\n", this->NodeName().c_str(), m_PastState.FrobeniusNorm());
#endif

                for (size_t timeIdxInSeq = 0; timeIdxInSeq < nT; timeIdxInSeq += GetNumParallelSequences())
                {
                    FrameRange frameRange(m_pMBLayout, timeIdxInSeq);
                    Matrix<ElemType> sliceObs = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceOutput = ValueSlice(frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceState = DataSlice(m_State, frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));

                    Matrix<ElemType> sliceGi = DataSlice(m_Gi, frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceGf = DataSlice(m_Gf, frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceGo = DataSlice(m_Go, frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));

                    Matrix<ElemType> sliceTanhState = DataSlice(tanhState, frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));
                    Matrix<ElemType> sliceTanhInput = DataSlice(tanhObs, frameRange/*TODO: delete this:*/.Check(frameRange.t(), GetNumParallelSequences(), m_pMBLayout));

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, FunctionValues(), m_State, m_PastOutput, m_PastState, GetNumParallelSequences(), m_DefaultState, &m_pMBLayout->GetM());

                    EvaluateThisNodeS(Inputs(1)->FunctionValues(), Inputs(2)->FunctionValues(), Inputs(3)->FunctionValues(), Inputs(4)->FunctionValues(),
                            sliceObs, mSlicePrevOutput, mSlicePrevState, sliceOutput, sliceState, sliceGi, sliceGf, sliceGo, sliceTanhState, sliceTanhInput, m_tempMatrix);
                }

                // save the hidden activities and output for the next minibatch
                SaveLastStateActity();

#ifdef DEBUG_DECODER
                if (mLastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls last output norm = %.8e\n", this->NodeName().c_str(), mLastOutput.FrobeniusNorm());
                if (mLastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls last state norm = %.8e\n", this->NodeName().c_str(), mLastState.FrobeniusNorm());
#endif

#ifdef DEBUG_DECODER
                ElemType tmpnorm = FunctionValues().FrobeniusNorm();
                if (ISCLOSE(tmpnorm, 0.834251, 0.002))
                    fprintf(stderr, "check!");
                fprintf(stderr, "LSTM function norm = %.8e\n", tmpnorm);
                for (size_t i = 0; i < 5; i++)
                    fprintf(stderr, "LSTM input[%d] norm = %.8e ", i, Inputs(i)->FunctionValues().FrobeniusNorm());
                fprintf(stderr, "\n");
#endif

                m_GradientComputed = false;
            }
        }

        /**
        Prepare history for LSTMnode

        This function returns state and output from the previous time instance. For recurrent network, the initial state needs to be set in the case of sentence begining, which is carried over from sentenceBegin. In case of sentence begining, the state activity is set to an initial value. The sentenceBegin has element of ((int) MinibatchPackingFlags::SequenceStart), ((int) MinibatchPackingFlags::None) and ((int) MinibatchPackingFlags::NoInput), which are 0, 1, and -1, respectively. 
        To compute the initial value, we use
        prevState = sentenceBegin * delayedActivation + ~sentenceBegin * initialStateValue
        and ~sentenceBegin is computed as -1*(sentenceBegin - 1), assuming that sentenceBegin is either 0 or 1. For example, when sentenceBegin == 1, ~sentenceBegin == 0. 
        The previous-time output doesn't have initial value, so it is computed as 
        prevOutput = sentenceBegin * pastOutput

        */
        // prepare prevstate and prevoutput
        static void WINAPI PrepareHistory(
            size_t timeIdxInSeq,
            Matrix<ElemType> & slicePrevOutput,
            Matrix<ElemType> & slicePrevState,
            const Matrix<ElemType> & output,
            const Matrix<ElemType> & state,
            const Matrix<ElemType> & pastOutput,
            const Matrix<ElemType> & pastState,
            size_t nsamples, const ElemType & initStateValue, const Matrix<float>* sentenceBegin)
        {
            size_t nRow = pastOutput.GetNumRows();
            size_t nStream = sentenceBegin->GetNumRows();

            assert(nStream == nsamples);

            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            if (slicePrevOutput.IsEmpty() || slicePrevOutput.GetNumRows() != nRow || slicePrevOutput.GetNumCols() != nsamples)
                slicePrevOutput.Resize(nRow, nsamples);
            if (slicePrevState.IsEmpty() || slicePrevState.GetNumRows() != nRow || slicePrevState.GetNumCols() != nsamples)
                slicePrevState.Resize(nRow, nsamples);

            if (sentenceBegin->GetNumRows() != nsamples)
                LogicError("Number of rows should be the same as the number of data streams");

            Matrix<float> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nStream, nStream);
            // will reset to 0 if sentence begining at a position is 0
            // will keep the output if it is not the sentence begining
            colBegin.InplaceTruncateBottom(((int) MinibatchPackingFlags::SequenceStart));
            colBegin.InplaceTruncateTop(((int) MinibatchPackingFlags::None));
#if 1
            initStateValue; pastState; pastOutput; state; output;
            LogicError("PrepareHistory: finish this");
#else
            // BUGBUG: we need to upcast float to double here
            colSeg.SetDiagonalValue(colBegin);

            Matrix<ElemType> newPrevOutput(colBegin.GetDeviceId());
            Matrix<ElemType> newPrevState(colBegin.GetDeviceId());
            if (utt_t == 0)
            {
                // this is the begining of this minibatch
                Matrix<ElemType>::Multiply(pastOutput.ColumnSlice(0, nsamples), false, colSeg, false, newPrevOutput);
                Matrix<ElemType>::Multiply(pastState.ColumnSlice(0, nsamples), false, colSeg, false, newPrevState);
            }
            else
            {
                // this is in the minibatch
                FrameRange frameRange(timeIdxInSeq, nsamples);
                Matrix<ElemType>::Multiply(DataSlice(output, frameRange/*TODO: delete the next two parameters*/, frameRange.t() - nsamples, nsamples), false, colSeg, false, newPrevOutput);
                Matrix<ElemType>::Multiply(DataSlice(state, frameRange/*TODO: delete the next two parameters*/, frameRange.t() - nsamples, nsamples), false, colSeg, false, newPrevState);
            }

            Base::SetToInitStateValueForResetSeg(sentenceBegin->ColumnSlice(utt_t, 1), nStream, initStateValue, newPrevState);

            slicePrevOutput.ColumnSlice(0, nsamples).SetValue(newPrevOutput);
            slicePrevState.ColumnSlice(0, nsamples).SetValue(newPrevState);
#endif
        }

        // prepare prevstate and prevoutput
        void PrepareThisErrorsBeforeBackProp(
            size_t timeIdxInSeq,
            size_t nT, // number of columns
            Matrix<ElemType> & error,
            Matrix<ElemType> & stateError,
            const Matrix<ElemType>& grdToPrevOutput,
            const Matrix<ElemType>& grdToPrevState,
            const Matrix<ElemType>& obs_error_from_future_minibatch,
            const Matrix<ElemType>& state_error_from_future_minibatch,
            size_t nsamples, const Matrix<float>* sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            int total_utt_t = (int)floor(nT / nsamples);

            error += grdToPrevOutput;
            stateError = grdToPrevState;

            if (m_use_errors_from_future_minibatch)
            {
                for (size_t utt_id = 0; utt_id < nsamples; utt_id++)
                {
                    // if uses errors from future minibatch
                    if ((GetSegInfo(timeIdxInSeq, utt_id) == ((int) MinibatchPackingFlags::None) && utt_t == total_utt_t - 1) // last time 
                        || (utt_t < total_utt_t - 1 && GetSegInfo(timeIdxInSeq, utt_id) == ((int) MinibatchPackingFlags::None) && GetSegInfo(timeIdxInSeq + nsamples, utt_id) == ((int) MinibatchPackingFlags::NoInput)) // future observation is no observation
                        )
                    {
                        error.ColumnSlice(utt_id, 1) += obs_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                        stateError.ColumnSlice(utt_id, 1) += state_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                    }
                }
            }


#if 1
            sentenceBegin;
            LogicError("PrepareThisErrorsBeforeBackProp: finish this");
#else
            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            colBegin.InplaceTruncateBottom(((int) MinibatchPackingFlags::NoInput));
            colBegin.InplaceTruncateTop(((int) MinibatchPackingFlags::SequenceStart));
            colBegin += fabs((ElemType)((int) MinibatchPackingFlags::NoInput)); // raise this so that -1 -> 0 and therefore 
            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nsamples, nsamples);
            colSeg.SetDiagonalValue(colBegin);

            // times the errors with the mask
            Matrix<ElemType> newOutputError(colBegin.GetDeviceId());
            Matrix<ElemType> newStateError(colBegin.GetDeviceId());

            Matrix<ElemType>::Multiply(error, false, colSeg, false, newOutputError);
            Matrix<ElemType>::Multiply(stateError, false, colSeg, false, newStateError);
            
            error.ColumnSlice(0, nsamples).SetValue(newOutputError);
            stateError.ColumnSlice(0, nsamples).SetValue(newStateError);
#endif
        }

        // prepare prevstate and prevoutput
        static void WINAPI PrepareErrors(
            size_t timeIdxInSeq,
            Matrix<ElemType> & errors,
            Matrix<ElemType> & stateError,
            size_t nsamples, const Matrix<float>* sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
#if 1
            errors; stateError; utt_t;
            LogicError("PrepareErrors: finish this");
#else
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            // will reset to 0 if sentence begining at a posiiton is 0
            // will keep the output if it is not the sentence begining
            colBegin.InplaceTruncateBottom(((int) MinibatchPackingFlags::SequenceStart));
            colBegin.InplaceTruncateTop(((int) MinibatchPackingFlags::None));

            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nsamples, nsamples);
            colSeg.SetDiagonalValue(colBegin);

            // times the errors with the mask
            Matrix<ElemType> newOutputError(colBegin.GetDeviceId());
            Matrix<ElemType> newStateError(colBegin.GetDeviceId());

            Matrix<ElemType>::Multiply(errors, false, colSeg, false, newOutputError);
            Matrix<ElemType>::Multiply(stateError, false, colSeg, false, newStateError);

            errors.ColumnSlice(0, nsamples).SetValue(newOutputError);
            stateError.ColumnSlice(0, nsamples).SetValue(newStateError);
#endif
        }

        /*TODO: merge with call site*/void EvaluateThisNodeS(
            const Matrix<ElemType>& mInputGate,
            const Matrix<ElemType> &mForgetGate, const Matrix<ElemType> &mOutputGate,
            const Matrix<ElemType> &mCellWgt,
            const Matrix<ElemType> &obs,
            const Matrix<ElemType>& prevOutput,
            const Matrix<ElemType>& prevState,
            Matrix<ElemType> &output,
            Matrix<ElemType> &state,
            Matrix<ElemType> &gi,
            Matrix<ElemType> &gf,
            Matrix<ElemType> &go,
            Matrix<ElemType> &tanhState,
            Matrix<ElemType> &tanhObs,
            Matrix<ElemType> &tmp)
        {
            int inputDim = obs.GetNumRows();
            int outputDim = mOutputGate.GetNumRows();

            // for input gate
            Matrix<ElemType>::Multiply(mInputGate.ColumnSlice(1, inputDim), false, obs, false, gi);
            Matrix<ElemType>::MultiplyAndAdd(mInputGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, gi);
            gi += mInputGate.ColumnSlice(0, 1);
            tmp = prevState;
            tmp.ColumnElementMultiplyWith(mInputGate.ColumnSlice(1 + inputDim + outputDim, 1));
            gi += tmp;
            gi.AssignSigmoidOf(gi);

            // for forget gate
            Matrix<ElemType>::Multiply(mForgetGate.ColumnSlice(1, inputDim), false, obs, false, gf);
            Matrix<ElemType>::MultiplyAndAdd(mForgetGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, gf);
            gf += mForgetGate.ColumnSlice(0, 1);
            tmp = prevState;
            tmp.ColumnElementMultiplyWith(mForgetGate.ColumnSlice(1 + inputDim + outputDim, 1));
            gf += tmp;
            gf.AssignSigmoidOf(gf);

            // for cell state
            Matrix<ElemType>::Multiply(mCellWgt.ColumnSlice(1, inputDim), false, obs, false, state);
            Matrix<ElemType>::MultiplyAndAdd(mCellWgt.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, state);
            state += mCellWgt.ColumnSlice(0, 1);
#ifdef DEBUG_DECODER
//            fprintf(stderr, "W_xc norm = %.8e\n", mCellWgt.ColumnSlice(1, inputDim).FrobeniusNorm());
//            fprintf(stderr, "W_hc norm = %.8e\n", mCellWgt.ColumnSlice(1 + inputDim, outputDim).FrobeniusNorm());
//            fprintf(stderr, "b_c norm = %.8e\n", mCellWgt.ColumnSlice(0, 1).FrobeniusNorm());
#endif
            tanhObs.AssignTanhOf(state);
            state.AssignElementProductOf(gi, tanhObs);
            state.AddElementProductOf(gf, prevState);

            // for output gate
            Matrix<ElemType>::Multiply(mOutputGate.ColumnSlice(1, inputDim), false, obs, false, go);
            Matrix<ElemType>::MultiplyAndAdd(mOutputGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, go);
            go += mOutputGate.ColumnSlice(0, 1);
            tmp = state;
            tmp.ColumnElementMultiplyWith(mOutputGate.ColumnSlice(1 + inputDim + outputDim, 1));
            go += tmp;
            go.AssignSigmoidOf(go);

            // to return output
            tanhState.AssignTanhOf(state);
            output.AssignElementProductOf(go, tanhState);
        }


        // input(0) : child with dimension [inputdim x T]
        // input(1) : input gate [outputdim x [inputdim + outputdim + 2]] bi, Wxi, Whi, Wci
        // input(2) : forget gate [outputdim x [inputdim + outputdim + 2]] for bf, Wxf, Whf, Wcf
        // input(3) : output gate [outputdim x [inputdim + outputdim + 2]] for bo, Wxo, Who, and Wco
        // input(4) : memory cell weight [outputdim x [inputdim + outputdim + 1]] for bc, Wxc, and Whc 
        // output : dimension [outputdim x T]
        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();

            if (Inputs(0)->FunctionValues().GetMatrixType() == SPARSE)
                LogicError("LSTMNode: input to LSTM has to be dense matrix. Consider adding a project layer using lookuptable before LSTM node. ");

#if 0
            // TODO: use dynamic_pointer_cast instead
            if (Inputs(1)->OperationName() != OperationNameOf(LearnableParameter) ||
                Inputs(2)->OperationName() != OperationNameOf(LearnableParameter) ||
                Inputs(3)->OperationName() != OperationNameOf(LearnableParameter) ||
                Inputs(4)->OperationName() != OperationNameOf(LearnableParameter))
                LogicError("LSTM validation: need to have learnable parameters ");
#endif

            //if (Inputs(0)->GetNumRows() == 0)
            //    LogicError("LSTM validation: input size is zero!");

            //if (Inputs(1)->GetNumRows() == 0 ||
            //    Inputs(2)->GetNumRows() == 0 ||
            //    Inputs(3)->GetNumRows() == 0 ||
            //    Inputs(4)->GetNumRows() == 0)
            //    LogicError("LSTM validation : parameter size is zero!");

            size_t nindim = Inputs(0)->GetNumRows();
            size_t noutdim = Inputs(1)->GetNumRows();
            size_t nT = Inputs(0)->GetNumCols();
            size_t nCol = nindim + noutdim + 2;
            if (isFinalValidationPass)
            {
                if (Inputs(1)->GetNumCols() != nCol)
                {
                    LogicError("LSTM validation : dimension mismatched between child and inputGate");
                }
                if (Inputs(2)->GetNumCols() != nCol)
                {
                    LogicError("LSTM validation : dimension mismatched between child and forgetGate");
                }
                if (Inputs(3)->GetNumCols() != nCol)
                {
                    LogicError("LSTM validation : dimension mismatched between child and outputGate");
                }

                if (noutdim != Inputs(2)->GetNumRows() ||
                    noutdim != Inputs(3)->GetNumRows() ||
                    noutdim != Inputs(4)->GetNumRows())
                {
                    LogicError("LSTM validation: output dimension mismatched!");
                }
            }

            SetDims(noutdim, nT);
            FunctionValues().SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
        }

        bool UnitTest()
        {
            {
                size_t nT = 3;
                size_t nInput = 2;
                size_t nHidden = 3;
                size_t nOutput = 3;

                // backup 
                Matrix<ElemType> f0(m_deviceId), f1(m_deviceId), f2(m_deviceId), f3(m_deviceId), f4(m_deviceId), func(m_deviceId), f5(m_deviceId);
                Matrix<ElemType> target(m_deviceId);
                Matrix<ElemType> giWeight, ghWeight, goWeight;
                ElemType initStateValue = m_DefaultState;
                auto pMBLayout = make_shared<MBLayout>();
                pMBLayout->Init(1, nT, true);
                //Matrix<float> & boundary = pMBLayout->m_sentenceBoundaryFlags;
                //vector<MinibatchPackingFlags> & minibatchPackingFlags = pMBLayout->m_minibatchPackingFlags;
                //boundary.ColumnSlice(0, 1).SetValue(((int) MinibatchPackingFlags::SequenceStart));
                //minibatchPackingFlags[1] = MinibatchPackingFlags::SequenceStart;
                pMBLayout->Set(0, 1, MinibatchPackingFlags::SequenceStart); // TODO: strange--start at frame[1] instead of [0]?
                Base::LinkToMBLayout(pMBLayout);

                f0 = Inputs(0)->FunctionValues();
                f1 = Inputs(1)->FunctionValues();
                f2 = Inputs(2)->FunctionValues();
                f3 = Inputs(3)->FunctionValues();
                f4 = Inputs(4)->FunctionValues();
                func = FunctionValues();

                target.Resize(nOutput, nT);
                for (size_t i = 0; i < nT; i++)
                    target(0, i) = 1;

                Inputs(0)->SetDims(nInput, nT);
                Inputs(0)->FunctionValues().SetValue(ConstOnes(nInput, nT, m_deviceId));
                Inputs(0)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(1)->SetDims(nHidden, nInput + nOutput + 2);
                Inputs(1)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(2)->SetDims(nHidden, nInput + nHidden + 2);
                Inputs(2)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(3)->SetDims(nOutput, nInput + nHidden + 2);
                Inputs(3)->FunctionValues().SetValue((ElemType)0.1);
                Inputs(4)->SetDims(nOutput, nHidden + nInput + 1);
                Inputs(4)->FunctionValues().SetValue((ElemType)0.1);
                SetDims(nOutput, nT);

                m_DefaultState = 0.0;
                EvaluateThisNode(FrameRange(m_pMBLayout));

                // check with expected values
                if (!ISCLOSE(FunctionValues()(0, 0), 0.0335975, EPSILON) ||
                    !ISCLOSE(FunctionValues()(0, 1), 0.05485132, EPSILON) ||
                    !ISCLOSE(FunctionValues()(0, 2), 0.06838435, EPSILON) ||
                    !(FunctionValues()(0, 0) == FunctionValues()(1, 0)))
                    throw("LSTMNode forward computation error");

                
                    FunctionValues().TransferToDeviceIfNotThere( m_deviceId, true);

                GradientValues().Resize(nOutput, nT);
                GradientValues().SetValue(1.0);
                for (size_t i = 0; i < 5; i++)
                {
                    Inputs(i)->GradientValues().Resize(Inputs(i)->GetNumRows(), Inputs(i)->GetNumCols());
                    Inputs(i)->GradientValues().SetValue(0);
                }
                for (size_t i = 0; i < 5; i++)
                    ComputeInputPartial(i, FrameRange(m_pMBLayout));

                // check with expected values
                if (!ISCLOSE(Inputs(1)->GradientValues()(0, 0), 0.07843818, EPSILON) // bi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 1), 0.00784382, EPSILON)  // Wxi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 3), 0.00192997, EPSILON)  // Whi
                    || !ISCLOSE(Inputs(1)->GradientValues()(0, 6), 0.00362767, EPSILON)  // Wci
                    )
                    throw("LSTMNode gradient error on input gates");
                if (!ISCLOSE(Inputs(2)->GradientValues()(0, 0), 0.02738655, EPSILON)  // bf
                    || !ISCLOSE(Inputs(2)->GradientValues()(0, 1), 0.00273866, EPSILON)  // Wxf
                    || !ISCLOSE(Inputs(2)->GradientValues()(0, 3), 0.00120922, EPSILON)  // Whf
                    || !ISCLOSE(Inputs(2)->GradientValues()(0, 6), 0.00227184, EPSILON)  // Wcf
                    )
                    throw("LSTMNode gradient error on forget gates");
                if (!ISCLOSE(Inputs(3)->GradientValues()(0, 0), 0.07801557, EPSILON)  // bo
                    || !ISCLOSE(Inputs(3)->GradientValues()(0, 1), 0.00780156, EPSILON)  // Wxo
                    || !ISCLOSE(Inputs(3)->GradientValues()(0, 3), 0.00268089, EPSILON)  // Who
                    || !ISCLOSE(Inputs(3)->GradientValues()(0, 6), 0.00809852, EPSILON)  // Wco
                    )
                    throw("LSTMNode gradient error on output gates");
                if (!ISCLOSE(Inputs(4)->GradientValues()(0, 0), 1.3075038, EPSILON)  // bc
                    || !ISCLOSE(Inputs(4)->GradientValues()(0, 1), 0.13075038, EPSILON)  // Wxc
                    || !ISCLOSE(Inputs(4)->GradientValues()(0, 3), 0.03080355, EPSILON)  // Whc
                    )
                    throw("LSTMNode gradient error on memory cells");

                for (size_t i = 0; i < 5; i++)
                {
                    
                        Inputs(i)->GradientValues().TransferToDeviceIfNotThere( m_deviceId, true);
                }
                m_DefaultState = initStateValue;
            }

            fprintf(stderr, "LSTMNode unit test passed!\n");
            return true;
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false);
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);
            fstream << L"Input[Width:" << m_inputDim << L"]  \n" ; 
            fstream << L"Hidden[Width:" << m_outputDim << L"]    Output[Width:" << m_outputDim << L"]  \n";
        }
    public:
        bool GetHistory(Matrix<ElemType>& hist, bool bLastTime)
        {
            size_t tRow = m_PastOutput.GetNumRows();
            size_t tCol = m_PastOutput.GetNumCols();
            size_t rCol = m_PastState.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            if (bLastTime)
            {
                hist.ColumnSlice(0, tCol).SetValue(mLastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(mLastState);
            }
            else{
                hist.ColumnSlice(0, tCol).SetValue(m_PastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(m_PastState);
            }

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
            return true;
        }

        void SetHistory(const Matrix<ElemType>& hist)
        {
            size_t tRow = hist.GetNumRows();
            size_t tCol = hist.GetNumCols();
            size_t eCols = tCol / 2;

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_PastOutput.Resize(tRow, eCols);
            m_PastState.Resize(tRow, eCols);
            m_PastOutput.SetValue(hist.ColumnSlice(0, eCols));
            m_PastState.SetValue(hist.ColumnSlice(eCols, eCols));

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

        virtual void GetErrorsToPreviousMinibatch(Matrix<ElemType>& hist)
        {
            size_t tRow = m_obs_error_from_future_minibatch.GetNumRows();
            size_t tCol = m_obs_error_from_future_minibatch.GetNumCols();
            size_t rCol = m_state_error_from_future_minibatch.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();

            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            hist.ColumnSlice(0, tCol).SetValue(m_obs_error_from_future_minibatch);
            hist.ColumnSlice(tCol, rCol).SetValue(m_state_error_from_future_minibatch);

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

        virtual void SetErrorsFromFutureMinibatch(Matrix<ElemType>& hist)
        {
            size_t tCol = hist.GetNumCols();
            size_t rCol = tCol / 2;

            DEVICEID_TYPE device = hist.GetDeviceId();

            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_obs_error_from_future_minibatch.SetValue(hist.ColumnSlice(0, rCol));
            m_state_error_from_future_minibatch.SetValue(hist.ColumnSlice(rCol, rCol));

            m_use_errors_from_future_minibatch = true;

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

    protected:
        size_t m_inputDim;
        size_t m_outputDim;

        Matrix<ElemType> m_State;  // hidden state activity
        Matrix<ElemType> m_PastState; // state activity in the previous minibatch
        Matrix<ElemType> m_PastOutput; // output in the previou minibatch 

        Matrix<ElemType> mLastState; // last state activity 
        Matrix<ElemType> mLastOutput; // last output 

        Matrix<ElemType> m_Gi;     // input gate activity
        Matrix<ElemType> m_Gf;     // forget gate activity
        Matrix<ElemType> m_Go;     // output gate activity

        Matrix<ElemType> grdToObs, grdToInputGate, grdToForgetGate, grdToOutputGate, grdToCellWgt;
        Matrix<ElemType> tanhState, tanhObs;

        Matrix<ElemType> m_tempMatrix; // temp matrix for speed-up

        bool     m_GradientComputed; // true if LSTM node has computed gradients, set to false if forward computation is just finished 

        Matrix<ElemType> mSlicePrevOutput, mSlicePrevState;

        Matrix<ElemType> grdBeforeInputGate, grdBeforeForget, grdBeforeGo, grdToCell, grdBeforeTanhInputGate;

    public:
        // errors from future minibatch
        Matrix<ElemType> m_obs_error_from_future_minibatch;
        Matrix<ElemType> m_state_error_from_future_minibatch;
        bool m_use_errors_from_future_minibatch;

        ElemType m_DefaultState;

    };

    template class LSTMNode<float>;
    template class LSTMNode<double>;

}}}
