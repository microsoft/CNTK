//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "ComputationNode.h"
#include "Sequences.h"
#include "Matrix.h"
#include "TensorShape.h"
#include "RecurrentNodes.h"

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType, int direction>
DelayedValueNodeBase<ElemType, direction>::DelayedValueNodeBase(DEVICEID_TYPE deviceId, const wstring& name,
                                                                ElemType initialState, const TensorShape& sampleLayout,
                                                                size_t timeStep) :
    Base(deviceId, name),
    m_initialStateValueMatrix(make_shared<Matrix<ElemType>>(deviceId)),
    m_inputInvalidMatrix(make_shared<Matrix<ElemType>>(deviceId)),
    m_zeroMatrix(make_shared<Matrix<ElemType>>(deviceId)),
    m_packedIndexMatrix(make_shared<Matrix<ElemType>>(deviceId)),
    m_delayedValue(make_shared<Matrix<ElemType>>(deviceId))
{
    m_initialStateValue = initialState;
    m_timeStep = 1;
    CreateMatrixIfNull(m_value);
    SetDims(sampleLayout, HasMBLayout() /*false at this point*/);
    m_initialStateValueMatrix->Resize(1, 1);
    m_initialStateValueMatrix->SetValue(m_initialStateValue);
    m_zeroMatrix->Resize(1, 1);
    m_zeroMatrix->SetValue((ElemType)0);
    m_timeStep = (int)timeStep;
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const /*override*/
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<DelayedValueNodeBase<ElemType, direction /*, SequenceStart_or_End*/>>(nodeP);
        node->m_timeStep = m_timeStep;
        node->m_initialStateValue = m_initialStateValue;
        node->m_initialStateValueMatrix->SetValue(m_initialStateValue);
        node->m_delayedValue->SetValue(*m_delayedValue);
        if (m_delayedActivationMBLayout)
            (node->m_delayedActivationMBLayout = make_shared<MBLayout>())->CopyFrom(m_delayedActivationMBLayout);
        else
            node->m_delayedActivationMBLayout = nullptr;
    }
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::Load(File& fstream, size_t modelVersion) /*override*/
{
    // the node has already been initialized e.g. w.r.t. direction
    Base::Load(fstream, modelVersion);

    fstream >> m_timeStep;

    if (modelVersion > CNTK_MODEL_VERSION_3)
    {
        TensorShape sampleLayout;
        sampleLayout.Load(fstream);
        SetDims(sampleLayout, HasMBLayout() /*may be true on reload (roll-back)*/);
    }
    else
    {
        size_t rows, colsDummy;
        fstream >> rows >> colsDummy;
        // legacy format: if #rows matches then assume current tensor shape is up to date
        // BUGBUG: This fails for non-column tensors. It should be sufficient to set
        //         these to 0 and rely on Validate(), but some unknown nodes in the loop don't do that right.
        SetDims(TensorShape(rows), HasMBLayout() /*may be true on reload (roll-back)*/); // tensor shape will be overwritten in Validate()
    }
    m_delayedValue->Resize(m_sampleLayout.GetNumElements(), 0); // Note: If we try to access history in first minibatch, we shall crash. It would be a consequence of a missing sentence-begin flag

    if (modelVersion >= CNTK_MODEL_VERSION_2)
    {
        fstream >> m_initialStateValue;
        m_initialStateValueMatrix->SetValue(m_initialStateValue);
    }
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::Save(File& fstream) const /*override*/
{
    Base::Save(fstream);

    fstream << m_timeStep;
#if CURRENT_CNTK_MODEL_VERSION > CNTK_MODEL_VERSION_3
    m_sampleLayout.Save(fstream);
#else
    fstream << GetSampleLayout().GetNumElements() << (size_t)0; // used to be (rows,cols); no need since inferred in Validate(), and wrong for non-matrix tensors
#endif

    fstream << m_initialStateValue;
}

// called before first iteration step of ForwardProp()
// We prepare the mask mask matrix here.
template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType, direction>::BeginForwardProp() /*override*/
{
    Base::BeginForwardProp();

    // The following is specifically for handling of truncated sequences fed using the V2 API where
    // the only information passed by the user is whether a given sequence starts in the current minibatch or not.
    // The Recurrent nodes however currently require the actual begin time-index in the past to be specified
    // which is actually not required since this info can be obtained from the saved m_delayedActivationMBLayout.
    // So the V2 library just sets the sequence tBegin to be SentinelValueIndicatingUnspecifedSequenceBeginIdx if the sequence
    // does not begin in the current MB and we patch the actual tBegin in by obtaining the info from the saved m_delayedActivationMBLayout.

    // Collect the trailing sequences in each parallel stream in the previous MB
    // TODO: Switch to a vector instead of an unordered_map
    std::unordered_map<size_t, MBLayout::SequenceInfo> trailingSequencesOfLastMB;
    if (m_delayedActivationMBLayout)
    {
        const auto& prevMBSequences = m_delayedActivationMBLayout->GetAllSequences();
        for (const auto& sequenceInfo : prevMBSequences)
        {
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
            {
                auto insertPos = trailingSequencesOfLastMB.insert({ sequenceInfo.s, sequenceInfo });
                if (!insertPos.second)
                {
                    if (trailingSequencesOfLastMB[sequenceInfo.s].tBegin < sequenceInfo.tBegin)
                        trailingSequencesOfLastMB[sequenceInfo.s] = sequenceInfo;
                }
            }
        }
    }

    bool anySequencesPatched = false;
    std::vector<MBLayout::SequenceInfo> patchedMBSequences = m_pMBLayout->GetAllSequences();
    for (auto& patchedSequenceInfo : patchedMBSequences)
    {
        if (patchedSequenceInfo.seqId != GAP_SEQUENCE_ID)
        {
            if (patchedSequenceInfo.tBegin == SentinelValueIndicatingUnspecifedSequenceBeginIdx)
            {
                if (trailingSequencesOfLastMB.find(patchedSequenceInfo.s) == trailingSequencesOfLastMB.end())
                    LogicError("No matching sequence found in the saved previous MBLayout to determine the real tBegin from for a truncated sequence in the current MBLayout");

                anySequencesPatched = true;
                patchedSequenceInfo.tBegin = trailingSequencesOfLastMB[patchedSequenceInfo.s].tBegin - m_delayedActivationMBLayout->GetNumTimeSteps();
            }
        }
    }

    // Now reconstruct the MBLayout with patched sequences if needed
    if (anySequencesPatched)
    {
        auto newMBLayout = make_shared<MBLayout>();
        newMBLayout->Init(m_pMBLayout->GetNumParallelSequences(), m_pMBLayout->GetNumTimeSteps());
        for (auto sequence : patchedMBSequences)
            newMBLayout->AddSequence(sequence);

        m_pMBLayout->MoveFrom(newMBLayout);
    }

    m_inputInvalidMatrix->SetValue(0);

    // --- create the mask for invalid sequences
    // The mask stores for every time step of every sequence whether that location is invalid; that is, when
    //  - the delayed time crosses a boundary, or
    //  - the current time is in a gap
    // Forward and backprop will exclude invalid frames.
    // TODO: in forward, we don't actually care if we propagate into a gap; could avoid a few unnecessary conditional copies towards the end
    m_inputAnySeqValid.assign(GetNumTimeSteps(), false); // start with assumptions which we update in the loop below
    m_inputAllSeqValid.assign(GetNumTimeSteps(), true);
    m_inputInvalidMatrixTemp.assign(m_inputInvalidMatrix->GetNumCols(), 0);

    let S = GetNumParallelSequences();
    int dir = direction; // (this avoids a 'conditional expression is constant' warning)
    FrameRangeIteration range(m_pMBLayout, -dir);
    for (auto fr = range.begin(); fr != range.end(); fr++)
    {
        let t = fr.t();
        FrameRange frDelayed = fr.WithTimeOffset(direction * m_timeStep);

        // first check in bulk for the frame--if all frames are good (most frequent case), take the quick path
        if (!m_pMBLayout->IsBeyondStartOrEnd(frDelayed) &&
            !m_pMBLayout->IsGap(fr))
        {
            m_inputAnySeqValid[t] = true;                  // no special case: just copy all
            assert(m_inputAllSeqValid[t]);
            continue;
        }

        // determine each sequence for the current frame that has a boundary condition or is a gap:
        for (size_t s = 0; s < S; s++)
        {
            // source frame is either invalid or valid (or target frame is a gap, in which case we consider everything valid)
            if (!m_pMBLayout->IsBeyondStartOrEnd(frDelayed.Sequence(s)) &&
                !m_pMBLayout->IsGap(fr.Sequence(s)))
            {
                m_inputAnySeqValid[t] = true;
            }
            else
            {
                m_inputAllSeqValid[t] = false;
                let j = fr.t() * S + s;
                m_inputInvalidMatrixTemp[j] = 1; // invalid: exclude this in copy/backprop
            }
        }
    }
    // move to GPU
    // TODO: move this to the MBLayout where this can be done together with the creation of the other mask and is likely to further improve performance.
    m_inputInvalidMatrix->SetValue(1, m_inputInvalidMatrixTemp.size(), m_deviceId, m_inputInvalidMatrixTemp.data(), matrixFlagNormal);

    // --- create the packed index in case of per-sequence initial state
    // In this case, we use Gather() to select the respective columns from the input state.
    if (GetNumInputs() > 1 && InputRef(1).HasMBLayout())
    {
        // ForwardProp() will first propagate a constant zero into the boundary frames.
        // Gather() is then used with beta=1 to add the correct boundary value into those,
        // while all other frames get added a 0.
        // We must initialize the packed index as follows:
        //  - packed index must have shape of 'this'
        //  - packed index values are:
        //     - matrix column indices into the initial state
        //        - if initial-state sequence has >1 steps, then index from back
        //        - if 1 step, then broadcast that to all
        //     - or -1 for non-boundary entires

        // our own output MB layout
        let& outMBLayout = GetMBLayout();
        vector<ElemType> buf(outMBLayout->GetNumCols(), -1); // -1 means non-boundary column
        // MB layout of initial state
        let& inMBLayout = InputRef(1).GetMBLayout();
        // loop over all sequences (of the initial state)
        let& inSequences = inMBLayout->GetAllSequences();
        for (size_t i = 0; i < inSequences.size(); i++)
        {
            let& inSeq = inSequences[i];
            if (inSeq.seqId == GAP_SEQUENCE_ID)
                continue;
            // find corresponding output sequence
            let& outSeq = outMBLayout->FindMatchingSequence(inSequences, i);
            let Tout = outSeq.GetNumTimeSteps(); // length of this output sequence
            let Tin  =  inSeq.GetNumTimeSteps(); // length of initial state's sequence. 1 means broadcasting in case m_timeStep > 1.
            // unless we are broadcasting, we will need m_timeStep values from the initial-state sequence
            if (Tin != 1 && Tin < m_timeStep)
                InvalidArgument("%ls %ls operation requires second argument (initialState) sequences to be either of length 1 or at least as long as the timestep (%d).", NodeName().c_str(), OperationName().c_str(), (int)m_timeStep);
            // fill packed-index array
            for (size_t dt = 0; dt < m_timeStep && dt < Tout; dt++) // we have this many boundary frames in each sequence
            {
                // For example, for m_timeStep = 2 and PastValue direction, and an initial-state length of 13,
                // the index sequence would be [11, 12, -1, -1, ...].
                // If the initial-state length is 1, we will broadcast and instead get [0, 0, -1, -1, ...].
                let tout =                 direction < 0 ?                    dt : Tout      - 1 - dt;  // step index of boundary frame in output
                let tin  = Tin == 1 ? 0 : (direction < 0 ? Tin - m_timeStep + dt : m_timeStep- 1 - dt); // and where in the initial state it comes from (Tin=1: broadcasting)
                // now this must be mapped to matrix column indices relative to the respective MBLayout
                let jout = outMBLayout->GetColumnIndex(outSeq, tout);
                let jin  =  inMBLayout->GetColumnIndex(inSeq,  tin);
                buf[jout] = (ElemType)jin;
            }
        }
        m_packedIndexMatrix->SetValue(1, outMBLayout->GetNumCols(), m_packedIndexMatrix->GetDeviceId(), buf.data(), MatrixFormat::matrixFormatColMajor);
    }
}

// update temporaries' column dimensions from MBLayout
// We reallocate the mask mask matrix here.
template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType, direction>::UpdateFunctionMBSize() /*override*/
{
    Base::UpdateFunctionMBSize();

    // resize the temporaries to their proper sizes
    // TODO: Are we sharing memory correctly? (no big deal as these are small; yet would be nice)
    m_inputInvalidMatrix->Resize(1, GetMBLayout()->GetNumCols());
    if (GetNumInputs() > 1 && InputRef(1).HasMBLayout())
        m_packedIndexMatrix->Resize(1, GetMBLayout()->GetNumCols());
}

// retrieve the mask tensor for the current frame
// This function mimics GetTensorSliceFor().
template<class ElemType, int direction>
/*private*/ TensorView<ElemType> DelayedValueNodeBase<ElemType, direction>::GetMaskTensor(size_t rank, const FrameRange& fr) const
{
    // tensorShape of m_inputInvalidMatrix is [1 x S x T]
    auto tensorShape = TensorShape(1);
    tensorShape.AppendInPlace(rank++, GetMBLayout()->GetNumParallelSequences());
    tensorShape.AppendInPlace(rank++, GetMBLayout()->GetNumTimeSteps());

    let slice = TensorSliceWithMBLayoutFor(tensorShape.GetDims(), fr, GetMBLayout());
    tensorShape.NarrowTo(slice);

    return TensorView<ElemType>(m_inputInvalidMatrix, tensorShape);
}

// This function assumes EndForwardProp() to be called after the iteration loop.
template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::ForwardProp(const FrameRange& fr) /*override*/
{
    assert(m_pMBLayout);

    // special case: DelayedValueNodes may be used outside of loops
    // TODO: this should be a bulk operation; this implementation is a quick hack
    int dir = direction; // (this avoids a 'conditional expression is constant' warning)
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

    // compute logical position of delayed value
    assert(m_timeStep > 0);

    // source tensor --considering truncated BPTT
    size_t rank = DetermineElementwiseTensorRank();
    TensorView<ElemType> src;
    int t_delayed = (int)(fr.t() + direction * m_timeStep); // this might end up outside the current window
    if (t_delayed < 0) // handle special case of truncated BPTT
    {
        if (!m_inputAnySeqValid[fr.t()])
            ; // none valid: leave it uninitialized
        else if (!m_delayedValue->IsEmpty()) // truncated BPTT
        {
            // truncated BPTT carry-over
            size_t T_delayedActivation = m_delayedActivationMBLayout ? m_delayedActivationMBLayout->GetNumTimeSteps() : 0; // (note: should never happen in full-sequence mode)
            auto tensorShape = GetTensorShape(rank);
            auto slice = TensorSliceWithMBLayoutFor(tensorShape.GetDims(), FrameRange(m_delayedActivationMBLayout, t_delayed/*<0*/ + T_delayedActivation), m_delayedActivationMBLayout);
            tensorShape.NarrowTo(slice);
            src = TensorView<ElemType>(m_delayedValue, tensorShape);
        }
        else
            LogicError("The delay node tries to access past values that are out of bound, possibly because there is no sentence start marker in the MBLayout.");
    }
    else if (t_delayed >= GetNumTimeSteps())
    {
        if (!m_inputAnySeqValid[fr.t()])
            ; // none valid: leave it uninitialized
        else  // truncated BPTT goes left-to-right only
            LogicError("The delay node tries to access future values that are out of bound, possibly because there is no sentence end marker in the MBLayout.");
    }
    else // regular case
        src = InputRef(0).ValueTensorFor(rank, frDelayed);

    // target tensor
    auto tgt = ValueTensorFor(rank, fr);

    // init value tensor (in case of constant, this is a [1] tensor with broadcasting)
    auto init = GetNumInputs() == 1 || InputRef(1).HasMBLayout()
        ? TensorView<ElemType>(m_initialStateValueMatrix, TensorShape(1)) // old form or per-sequence: initial state given as C++ constant
        : InputRef(1).ValueTensorFor(rank, FrameRange());                 // initial state given as a tensor

    // now perform the copy operation
    // In case of per-sequence state, we first pretend we have a constant init value of 0,
    // and then add over it the actual per-sequence state in a Gather operation with beta=1.
    if (m_inputAllSeqValid[fr.t()]) // all frames are valid: copy as one tensor-copy operation
        tgt.AssignCopyOf(src);
    else if (m_inputAnySeqValid[fr.t()]) // some are valid, some are not: use a OpCond to select 'src' for valid and 'init' for invalid frames
        tgt.AssignCondOf(GetMaskTensor(rank, fr), init, src); // assign either input or init value, based on the mask
    else // no frame is valid: initialize from init value
        tgt.AssignCopyOf(init);
    if (!m_inputAllSeqValid[fr.t()] && GetNumInputs() > 1 && InputRef(1).HasMBLayout())    // implant per-sequence state
    {
        let& idx   =             DataFor(*m_packedIndexMatrix, fr);  // column indices that guide the copy operation
        auto tgt2  =             ValueFor                      (fr); // output goes here
        let& init2 = InputRef(1).Value();                            // source data is the initial state. Not sliced, but we only copy parts.
        tgt2.DoGatherColumnsOf(/*beta=*/1, idx, init2, /*alpha=*/1); // beta=1 so that we add to what we previously initialized to 0
    }
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::EndForwardProp() /*override*/ // called after last iteration step of ForwardProp()
{
    // In truncated BPTT, we carry over left-to-right state across minibatches.
    // It is kept in m_delayedValue, m_delayedActivationMBLayout.
    // This could be optimized as follows:
    //  - only keep the required number of frames (m_timeStep)
    //  - we don't need to keep anything in full-sequence mode
    //  - we don't need to keep anything if all sequences are closed (sentence end)
    //    This condition includes full-sequence mode.
    // TODO: Can we optimize this and only copy if there is a sequence spanning across the end of the MB? And add a check to BeginForwardProp() to make sure we got one if there is a boundary at the start?
    m_delayedValue->SetValue(InputRef(0).Value());
    if (!m_delayedActivationMBLayout)
        m_delayedActivationMBLayout = make_shared<MBLayout>();
    m_delayedActivationMBLayout->CopyFrom(m_pMBLayout);
    // Perf BUGBUG: ^^ This copies a matrix from CPU to GPU at each MB; we should short-circuit it

    Base::EndForwardProp();
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::/*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) /*override*/
{
    // input 1 (initial state) is done in bulk
    if (inputIndex == 1)    
    {
        size_t rank = DetermineElementwiseTensorRank();

        if (!InputRef(1).HasMBLayout())
        {
            MaskMissingGradientColumnsToZero(fr); // we backprop invalid frames, including gaps; so zero them out

            auto src =                      GradientTensorFor(rank, fr); // incoming gradient from top
            auto tgt = InputRef(inputIndex).GradientTensorFor(rank, FrameRange()); // outgoing gradient to initial state
            TensorView<ElemType> zero(m_zeroMatrix, TensorShape(1));

            tgt.AddCondOf(GetMaskTensor(rank, fr), src, zero); // when back-propping into initial state, we swap the args and propagate the invalid ones
            // This will drag along the gaps as well, hence we mask them to zero above. --TODO : this is not optimal.
            // Alternative is a targeted copy using indices. Also needed to support initial state from nodes with time dimension.
        }
        else // per-sequence initial state uses Scatter() instead
        {
            // In this case, we really only back-prop values we have.
            // Non-determinism note:
            // If timeStep > 1 and initial state sequences are broadcasting, we will have a reduction.
            // That reduction may be non-deterministic.
            // In the regular case (timeStep = 1), there will be no non-determinism.
            let&  idx  =                 DataFor(*m_packedIndexMatrix, fr); // column indices that guide the copy operation
            let&  src  =             GradientFor                      (fr); // gradient as received from top = source
            auto& init = InputRef(1).Gradient();                            // target is the initial state. Not sliced, but we only copy parts.
            init.DoScatterColumnsOf(/*beta=*/1, idx, src, /*alpha=*/1);
        }
    }
    else if (inputIndex == 0)
    {
        // special case: DelayedValueNodes may be used outside of loops
        // TODO: this should be a bulk operation; this implementation is a quick hack
        if (fr.IsAllFrames())
        {
            // recursive call to ourselves
            int dir = direction; // (this avoids a 'conditional expression is constant' warning)
            FrameRangeIteration range(m_pMBLayout, -dir);
            for (auto t = range.rbegin(); t != range.rend(); t++) // note: reverse iterator
                BackpropTo(inputIndex, t);
            return;
        }

#if 0   // this should be removed; keep it around for a while in case we find it fail
        // move the target matrix to the target device, since below it is accessed as slices which cannot move
        // TODO: change below accesses to TensorView, then this is no longer needed. This is now the case, but need to test it.
        // TODO: we seem to already use TensorView, so this thing may no longer be needed. Too scary to remove.
        if (InputRef(0).NeedsGradient()) // (if not needs gradient then gradient matrix does not exist and therefore cannot be moved)
            InputRef(0).Gradient().TransferToDeviceIfNotThere(m_deviceId, /*isBeingMoved=*/ true);
#endif

        // if delayed input is within valid time range then add its gradient
        FrameRange frDelayed = fr.WithTimeOffset(direction * m_timeStep); // target frame
        if (!m_pMBLayout->IsBeyondMinibatch(frDelayed)) // only propagate if our target is inside the minibatch
        {
            size_t rank = DetermineElementwiseTensorRank();

            auto src =                      GradientTensorFor(rank, fr); // incoming gradient from top
            auto tgt = InputRef(inputIndex).GradientTensorFor(rank, frDelayed); // target is outgoing gradient to input
            TensorView<ElemType> zero(m_zeroMatrix, TensorShape(1));

            if (m_inputAllSeqValid[fr.t()]) // all valid: just jam it over in one go
                tgt.AddCopyOf(src);
            else if (m_inputAnySeqValid[fr.t()]) // some are valid, some are not: use a OpCond tgt select 'src' for valid and 'zero' for invalid frames
                tgt.AddCondOf(GetMaskTensor(rank, fr), zero, src); // now add either source or zero value, based on the mask
            else // none valid: nothing to back-prop
                ;
        }
    }
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::/*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) /*override*/
{
    if (m_inputs.size() == 1) // old form: initial activation passed as a C++ value at construction time
        ValidateUnaryMap(isFinalValidationPass);
    else if (m_inputs.size() == 2) // initial activation passed as a second input
        ValidateBinaryZip(isFinalValidationPass, /*allowBroadcast=*/true);
    else
        InvalidArgument("%ls %ls operation accepts one or two inputs.", NodeName().c_str(), OperationName().c_str());

    if (isFinalValidationPass && !Input(0)->HasMBLayout())
        InvalidArgument("%ls %ls operation requires the main (first) input to have a dynamic axis.", NodeName().c_str(), OperationName().c_str());

    // if we have a per-sequence initial state, we leverage a scalar init value of 0 in the computation
    if (isFinalValidationPass && GetNumInputs() > 1 && Input(1)->HasMBLayout() && m_initialStateValue != 0)
        InvalidArgument("%ls %ls operation requires the scalar initial value to be 0 if the second input (initial state) has a dynamic axis.", NodeName().c_str(), OperationName().c_str());
}

template<class ElemType, int direction>
/*virtual*/ NodeStatePtr DelayedValueNodeBase<ElemType, direction>::/*IStatefulNode::*/ ExportState() /*override*/
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
        if (!m_pMBLayout->HasSequenceBeyondEnd()) // only need to export state if anything crosses the MB boundary
        {
            auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
            pState->CacheDelayedMBLayout(m_delayedActivationMBLayout);
            // return an empty one
        }
        else
        {
            auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
            pState->CacheState(m_delayedValue->ColumnSlice((nT - 1) * nU, nU));
            pState->CacheDelayedMBLayout(m_delayedActivationMBLayout);
            pExportedState = pState;
        }
    }
    else if (dir == 1) // we look into future
    {
        if (!m_pMBLayout->HasSequenceBeyondBegin()) // only need to export state if anything crosses the MB boundary
        {
            auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
            pState->CacheDelayedMBLayout(m_delayedActivationMBLayout);
            pExportedState = pState;
        }
        else
        {
            auto pState = make_shared<DelayedValueNodeState<ElemType>>(m_deviceId);
            pState->CacheState(m_delayedValue->ColumnSlice((nT - 1) * nU, nU));
            pState->CacheDelayedMBLayout(m_delayedActivationMBLayout);
            pExportedState = pState;
        }
    }
    else
    {
        LogicError("Unrecognized direction in DelayedValueNodeBase");
    }
    return pExportedState;
}

template<class ElemType, int direction>
/*virtual*/ void DelayedValueNodeBase<ElemType,direction>::/*IStatefulNode::*/ ImportState(const NodeStatePtr& pImportedState) /*override*/
{
    DelayedNodeStatePtr pState = dynamic_pointer_cast<DelayedValueNodeState<ElemType>>(pImportedState);

    if (!pState)
        LogicError("Expecting DelayValueNodeState after downcasting");

    pState->ExportDelayedMBLayout(m_delayedActivationMBLayout); // pstate copy to m_delayedActivationMBLayout
    if (pState->IsEmpty())
    {
        return;
    }

    const Matrix<ElemType>& delayedActivation = pState->ExportCachedActivity();
    size_t nT = m_delayedActivationMBLayout->GetNumTimeSteps();
    size_t nU = m_delayedActivationMBLayout->GetNumParallelSequences();

    int dir = direction;
    if (dir == -1) // looking backward
        m_delayedValue->SetColumnSlice(delayedActivation, (nT - 1) * nU, nU);
    else if (dir == 1)
        m_delayedValue->SetColumnSlice(delayedActivation, 0, nU);
    else
        LogicError("Unrecognized direction in DelayedValueNodeBase");
}

// instantiate the classes that derive from the above
template class PastValueNode<float>;
template class PastValueNode<double>;

template class FutureValueNode<float>;
template class FutureValueNode<double>;

// -----------------------------------------------------------------------
// DelayedValueNodeState -- helper class for exporting/importing state from/to DelayedValueNodes.
// This is used for sub-minibatching in case of truncated BPTT.
// -----------------------------------------------------------------------

template <class ElemType>
class DelayedValueNodeState : public INodeState
{
public:
    DelayedValueNodeState(int deviceID)
        : m_cachedActivity((size_t) 0, (size_t) 0, deviceID),
          m_delayedActivationMBLayout(nullptr),
          m_isEmpty(true)
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
    ~DelayedValueNodeState()
    {
    }

protected:
    Matrix<ElemType> m_cachedActivity; // 1 column per parallel sequence
    // MBLayoutPtr         m_shiftedMBLayout;
    // Currently, we only support saving state for m_timeStep == 1
    // there is no need for this m_shiftedMBLayout if m_timeStep == 1
    MBLayoutPtr m_delayedActivationMBLayout;
    bool m_isEmpty; // in some case
                    // (e.g., at the boundary of sentence end or begin/full utterance mode), we don't need to store state (but we do need to need know m_delayedActivationMBLayout)
};

#ifdef COMING_SOON

// -----------------------------------------------------------------------
// ShiftNode (input, fromOffset, boundaryValue, dim=-1) -- delay and rolling window
//
// This shifts the input by (-fromOffset) steps. In other words, output(t) will be input(t+fromOffset).
// E.g. for fromOffset=-1, this gives the past value.
// This node has quite some options that make it powerful for many use cases.
//
// This node can be used in a recurrent loop. This requires special handling by the ComputationNetwork,
// for both execution (sequential execution) and creation (avoiding circular references).
//
// To delay (past value), use negative fromOffset. To access future value, use positive fromOffset.
//
// Values shifted in from beyond sequence boundaries will be copied from boundaryValue.
// Normally, this is a scalar Constant(). However, it can be any node, where the last (left-to-right iteration)
// or first (right-to-left) frame will be used (broadcast to all boundary frames). This can implement the basic
// sequence-to-sequence model.
//
// By default, this shifts over the time dimension, but you can choose to shift over any
// sample tensor dimension instead using 'dim' (-1 stands for time). This will only work, however,
// when all involved nodes are implemented using the tensor library. Nodes implemented using
// Matrix slices can only support iterating over time.
//
// TODO (this is still unfinished):
//  - backprop into boundary node
//  - backprop with packed sequences
//  - import/export for sub-minibatching
// -----------------------------------------------------------------------

template <class ElemType>
class ShiftNode : public ComputationNode<ElemType>, public IRecurrentNode, public ILateAttachingNode, public IStatefulNode, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Shift";
    }

public:
    enum BoundaryMode : int // how to fill frames at boundaries
    {
        reachAcross = -1, // go across the boundary: use boundaryValue
        duplicate = 0     // duplicate frame at boundary, e.g. duplicate first frame. Non-recurrent mode only.
    };
    ShiftNode(DEVICEID_TYPE deviceId, const wstring& name, int fromOffset, BoundaryMode boundaryMode, int shiftDimParam)
        : Base(deviceId, name), m_fromOffset(fromOffset), m_boundaryMode(boundaryMode), m_shiftDimParam(shiftDimParam), m_shiftDim(SIZE_MAX), m_state(deviceId)
    {
        CreateMatrixIfNull(m_value);
    }
    ShiftNode(DEVICEID_TYPE deviceId, const wstring& name)
        : ShiftNode(deviceId, name, 1, BoundaryMode::reachAcross, -1)
    {
    }
    ShiftNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ShiftNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"fromOffset"), (BoundaryMode)(int) configp->Get(L"boundaryMode"), configp->Get(L"dim"))
    {
        // We do NOT attach the inputs, as we cannot resolve the main input without causing a circular reference.
        // Instead, we capture them in a lambda, which will be called by ComputationNetwork during the build process through LateAttachInputs() below.
        // This is a contract between ComputationNetwork and this specific node type.
        // (TODO: We could force-evaluate the boundary input here.)
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
    void Save(File& fstream) const
    {
        Base::Save(fstream);
        fstream << m_fromOffset << m_boundaryMode << m_shiftDimParam;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_fromOffset >> m_boundaryMode >> m_shiftDimParam;
    }

    virtual void BeginForwardProp() override // called after last iteration step of ForwardProp()
    {
        Base::BeginForwardProp();

        // TODO: If we have a truncated-BPTT state then verify that the sequence indices match with m_state->m_sequences, and the tensor dimensions.

        // in case of trimming, narrow the layout
        // We actually do not drop content, only reduce the range of sequences.
        // This is meant to optimize for the case where we have multiple sequences concatenated while trimming a small amount only.
    }

    virtual void EndForwardProp() override // called after last iteration step of ForwardProp()
    {
        Base::EndForwardProp();

        // In truncated BPTT, we carry over left-to-right state across minibatches.
        // The necessary frames are stored in m_state->m_delayedValue.

        if (GetMBLayout()->HasSequenceBeyondEnd()) // only if layout has any sequence that has ends beyond this minibatch
        {
        }
        else
            m_state.clear();
    }

private:
    typedef std::pair<SmallVector<int>, SmallVector<int>> SliceBounds; // slice bounds for dimension k are [first[k], second[k]) (think STL begin/end)

    // helper to shift dimension 'm_shiftDim' of SliceBounds by an offset (a common operation below)
    SliceBounds ShiftDim(const SliceBounds& in, int shiftBy) const
    {
        SliceBounds result = in;
        result.first[m_shiftDim] += shiftBy;
        result.second[m_shiftDim] += shiftBy;
        return result;
    }

    // helper to typecast dimensions from a TensorShape into a signed-int array
    static SmallVector<int> ToIntDims(const TensorShape& shape)
    {
        SmallVector<int> dimsSigned;
        dimsSigned.append(shape.GetDims().begin(), shape.GetDims().end()); // we need the bounds as signed integers as they may shift into negative ranges
        return dimsSigned;
    }

    // determine shapes and slices to move
    // This is used for both forward and backprop.
    // 'In' below refers to Input(0) where 'Out' refers to the output of *this.
    void DetermineSlices(size_t rank, const FrameRange& fr,
                         TensorShape& inShape, TensorShape& outShape,                     // our MB's shape
                         SliceBounds& inSliceLogical, SliceBounds& outSliceLogical) const // the logical ranges to shift
    {
        // get the slice bounds for the given FrameRange
        outShape = GetTensorShape(rank); // describes the full tensor including sequence and time dimensions
        inShape = Input(0)->GetTensorShape(rank);

        // determine the logical in and out slices
        // This may now have bounds that fall outside, which we need to split off next.
        outSliceLogical = TensorSliceWithMBLayoutFor(ToIntDims(outShape), fr, GetMBLayout());
        inSliceLogical = TensorSliceWithMBLayoutFor(ToIntDims(inShape), fr.WithTimeOffset(m_fromOffset), GetMBLayout()); // apply the offset
    }

    // determine stripes to move w.r.t. main storage and from/to state
    // For efficiency:
    //  - this function assumes that the return values have been freshly constructed (it won't reset them)
    //  - it may return a slice with end < begin which indicates an empty slice
    void PartitionSlices(const SliceBounds& inSliceLogical, const SliceBounds& outSliceLogical, // the move we want to make
                         int T,                                                                 // our actual size
                         SliceBounds& inSliceMain, SliceBounds& outSliceMain,                   // the part that goes main-to-main
                         SliceBounds& inSliceState, SliceBounds& outSliceState) const           // the part that goes from/to state
    {
        inSliceMain = inSliceLogical;
        outSliceMain = outSliceLogical;
        if (inSliceMain.first[m_shiftDim] < 0)
        {
            assert(inSliceMain.second[m_shiftDim] < T);
            if (!m_state.empty()) // truncated BPTT case
            {
                // determine range that lives in state
                SliceBounds inSliceOutside = inSliceMain; // beginning falls to the left of the MB
                if (inSliceOutside.second[m_shiftDim] > 0)
                    inSliceOutside.second[m_shiftDim] = 0; // trim end; e.g. [-2,97) -> [-2,0), but [-2,-1) remains
                // now inSliceOutside represents only the region that falls outside

                // map to dimensions of our saved state
                SliceBounds inSliceState = ShiftDim(inSliceOutside, m_state.m_shape[m_shiftDim]);
                // E.g. for offset = -4, m_state will be 4 elements, so [-2,0) -> [2,4), and [-2,-1) -> [2,3)

                // map to target dimensions
                SliceBounds outSliceState = ShiftDim(inSliceOutside, -m_fromOffset);
                assert(inSliceState == outSliceState); // (when we fall out on the left, both must be the same)
            }
            // else: no truncated BPTT means we must have a proper boundary. So don't write those values here, they will be initialized with boundary values below.

            // and trim main (if 'from' is entirely outside, such as in the common single-frame case, we get begin >= end)
            outSliceMain.first[m_shiftDim] += -inSliceMain.first[m_shiftDim];
            inSliceMain.first[m_shiftDim] += -inSliceMain.first[m_shiftDim];
            assert(inSliceMain.first[m_shiftDim] == 0);
        }
        else if (inSliceMain.second[m_shiftDim] > T)
        {
            if (!m_state.empty())
            {
                // determine range to get from state
                SliceBounds inSliceOutside = inSliceMain;
                if (inSliceOutside.first[m_shiftDim] < T)
                    inSliceOutside.first[m_shiftDim] = T; // trim end; e.g. [2,102) -> [100,102), but [101,102) remains
                // now inSliceOutside is where we should copy from, with indices completely out of bounds

                // map to dimensions of our saved state
                SliceBounds inSliceState = ShiftDim(inSliceOutside, -T);
                // E.g. for offset = 4, m_state will be 4 elements, so [100,102) -> [0,2), and [101,102) -> [1,2)

                // map to target dimensions
                SliceBounds outSliceState = ShiftDim(inSliceOutside, T - m_fromOffset);
                // E.g. [0,2) -> [96,98), and [1,2) -> [97,98)
            }
            // and trim main (if 'from' is entirely outside, such as in the common single-frame case, we get begin >= end)
            outSliceMain.first[m_shiftDim] -= (inSliceMain.second[m_shiftDim] - T);
            inSliceMain.second[m_shiftDim] -= (inSliceMain.second[m_shiftDim] - T);
            assert(inSliceMain.second[m_shiftDim] == T);
        }
    }

    // get a sliced TensorView on a Matrix given a shape and a slice
    TensorView<ElemType> DataTensorFor(Matrix<ElemType>& data, TensorShape shape /*original shape of 'data'*/, SliceBounds slice)
    {
        shape.NarrowTo(slice);
        return TensorView<ElemType>(data, shape);
    }

    // determine FrameRange objects that describe the boundary frames of the sequence for the output, for the case of iterating over time.
    void DetermineBoundaryToFrameRange(const FrameRange& fr, const MBLayout::SequenceInfo& toSeqInfo, // range we operate on and current sequence under consideration
                                       size_t T, FrameRange& frTo) const                              // ourselves (output)
    {
        // get FrameRange to write to in our output
        frTo = fr.Sequence(toSeqInfo.s); // clip to this one sequence only
        if (frTo.IsAllFrames())          // whole batch: narrow to the boundary range
        {
            auto steps = min((size_t) abs(m_fromOffset), toSeqInfo.GetNumTimeSteps());
            frTo = frTo.WithTimeStep(m_fromOffset < 0 ? toSeqInfo.tBegin : toSeqInfo.tEnd - steps).WithTimeRange(steps); // all frames to be filled in this sequence
            LogicError("This code path has never been tested.");                                                         // remove this once we have
        }
        // frTo now describes the frame range that needs to be filled from the boundary node
    }

    // determine FrameRange objects that describe the boundary frames of the sequence
    // This version is for the case of iterating over time.
    void DetermineBoundaryFrameRanges(const FrameRange& fr, const MBLayout::SequenceInfo& toSeqInfo, // range we operate on and current sequence under consideration
                                      const ComputationNodeBasePtr& fromNode, FrameRange& frFrom,    // boundary node
                                      size_t T, FrameRange& frTo) const                              // ourselves (output)
    {
        // get FrameRange to write to in our output
        DetermineBoundaryToFrameRange(fr, toSeqInfo, T, frTo);
        // frTo now describes the frame range that needs to be filled from the boundary node

        // create a FrameRange for the boundary node to read from
        // Boundary data is always a single frame.
        frFrom = frTo.WithLayout(fromNode->GetMBLayout()).WithTimeRange(1).AllowBroadcast(); // start with this, next update time step and possibly toSeqInfo index
        bool clamp = m_boundaryMode == BoundaryMode::duplicate;
        if (clamp) // get frames from our own input
            frFrom = frFrom.WithTimeStep(m_fromOffset < 0 ? toSeqInfo.tBegin : toSeqInfo.tEnd - 1);
        else if (!fromNode->HasMBLayout())   // get frames from separate node that is not data
            frFrom = frFrom.WithTimeStep(0); // Validate() has ensured that input is one column
        else                                 // get frames from separate node that is data
        {
            if (fromNode->GetMBLayout() != GetMBLayout())
                frFrom = frFrom.Sequence(fromNode->GetMBLayout()->FindSequence(toSeqInfo.seqId).seqId); // get matching sequence entry in boundary node
            const auto& fromSeqInfo = fromNode->GetMBLayout()->GetAllSequences()[frFrom.seqIndex];
            frFrom = frFrom.WithTimeStep(m_fromOffset > 0 ? fromSeqInfo.tBegin : fromSeqInfo.tEnd - 1);
        }
    }

    // determine FrameRange objects that describe the boundary frames of the sequence
    // This version is for the case of iterating over a non-time dimension (which is non-ragged).
    void DetermineBoundaryFrameRanges(const FrameRange& fr,                                                 // range we operate on (parameter to ForwardProp() and BackpropTo())
                                      const ComputationNodePtr& fromNode, size_t fromT, FrameRange& frFrom, // boundary node
                                      size_t T, FrameRange& frTo) const                                     // ourselves (output)
    {
        // get FrameRange to fill in our output
        frTo = fr;
        if (frTo.IsAllFrames())
        {
            auto steps = std::min((size_t) abs(m_fromOffset), T);
            frTo = frTo.WithTimeStep(m_fromOffset < 0 ? 0 : T - steps);
        }

        // get tensor to fill from
        frFrom = frTo.WithTimeRange(1).AllowBroadcast(); // start with this, next will in time step and possibly update the layout
        bool clamp = m_boundaryMode == BoundaryMode::duplicate;
        if (clamp)
            frFrom = frFrom.WithTimeStep(m_fromOffset < 0 ? 0 : fromT - 1); // (no need to update layout as it is the same)
        else
            frFrom = frFrom.WithTimeStep(m_fromOffset > 0 ? 0 : fromT - 1).WithLayout(fromNode->GetMBLayout());
    }

    // perform op on all sequences that get boundary frames filled in a range that intersects with our output range
    template <class OpFn>
    void ForAllBoundaryIntersectingSequences(const FrameRange& fr, const SliceBounds& outSlice, size_t T, const OpFn& opFn)
    {
        if (fr.IsAllFrames() || GetMBLayout()->IsBeyondStartOrEnd(fr.WithTimeOffset(m_fromOffset))) // short-cut test whether there is anything to do
        {
            auto ts = outSlice.first[m_shiftDim];
            auto te = outSlice.second[m_shiftDim];
            // iterate over all sequences in this batch and handle all that overlap with the target region
            for (auto toSeqInfo : GetMBLayout()->GetAllSequences())
            {
                // reduce to boundary frames
                if (m_fromOffset < 0)
                    toSeqInfo.tEnd = min(toSeqInfo.tEnd, (size_t) max(toSeqInfo.tBegin - m_fromOffset, (ptrdiff_t) 0));
                else
                    toSeqInfo.tBegin = max(toSeqInfo.tBegin, (ptrdiff_t) toSeqInfo.tEnd - m_fromOffset);

                // if no overlap then skip
                if (toSeqInfo.tEnd <= ts || toSeqInfo.tBegin >= te)
                    continue;

                // clip sequence to [ts,te)
                if (toSeqInfo.tBegin < ts)
                    toSeqInfo.tBegin = ts;
                if (toSeqInfo.tEnd > te)
                    toSeqInfo.tEnd = te;

                // action to perform
                opFn(toSeqInfo);
            }
        }
    }

    // perform the copy (forward) or add (backprop) operation
    void Propagate(const ComputationNodePtr& fromNode, TensorShape fromShape, const FrameRange& frFrom, TensorShape toShape, const FrameRange& frTo, bool isForward, ElemType backwardSign)
    {
        auto fromSlice = TensorSliceWithMBLayoutFor(ToIntDims(fromShape), frFrom, fromNode->GetMBLayout());
        auto toSlice = TensorSliceWithMBLayoutFor(ToIntDims(toShape), frTo, GetMBLayout());

        fromShape.NarrowTo(fromSlice);
        toShape.NarrowTo(toSlice);

        if (isForward)
        {
            auto from = TensorView<ElemType>(fromNode->Value(), fromShape);
            auto to = TensorView<ElemType>(Value(), toShape);
            to.AssignCopyOf(from);
        }
        else
        {
            auto from = TensorView<ElemType>(fromNode->Gradient(), fromShape);
            auto to = TensorView<ElemType>(Gradient(), toShape);
            from.AddCopyOf(to, backwardSign); // sign = -1 to subtract
        }
    }

    // perform propagation of bounary frames (either copy from or backprop to)
    void PropagateBoundaryFrames(const FrameRange& fr, size_t rank, const SliceBounds& inSliceLogical, const TensorShape& outShape, const SliceBounds& outSliceLogical, bool isForward)
    {
        // get node to fill from and its dimensions
        // We fill either from the provided boundary node or from ourselves (BoundaryMode::duplicate = clamp).
        bool clamp = m_boundaryMode == BoundaryMode::duplicate;
        ComputationNodePtr fromNode = clamp ? Input(0) : // duplicating our own boundary frame
                                          Input(1);      // pulling in a frame from another node or a constant
        auto fromShape = fromNode->GetTensorShape(rank);

        auto T = outShape[m_shiftDim];                   // upper bound of iteration dimension
        assert(fr.seqIndex == SIZE_MAX);                 // (can't run loops over individual sequences)
        assert(fr.IsAllFrames() || fr.m_timeRange == 1); // (we only support full range or single frames; otherwise we'd have to narrow it to the intersection with this sequence)
        bool isTimeIteration = m_shiftDim >= rank;

        // if iterating in time, we must pay attention to sequence boundaries inside the batch
        if (isTimeIteration)
        {
            ForAllBoundaryIntersectingSequences(fr, outSliceLogical, T, [&](const MBLayout::SequenceInfo& toSeqInfo)
                                                {
                                                    // determine FrameRanges for from and to
                                                    FrameRange frFrom, frTo;
                                                    DetermineBoundaryFrameRanges(fr, toSeqInfo, fromNode, frFrom, T, frTo);

                                                    // copy/backprop
                                                    Propagate(fromNode, fromShape, frFrom, outShape, frTo, isForward, +1);
                                                });
        }
        // iterating over fixed sample-shape dimensions
        else if (!isTimeIteration && (inSliceLogical.first[m_shiftDim] < 0 || inSliceLogical.second[m_shiftDim] >= T))
        {
            // get bounds
            auto fromT = fromShape[m_shiftDim]; // upper bound of iteration dimension in boundary node (may match or broadcast)
            FrameRange frFrom, frTo;
            DetermineBoundaryFrameRanges(fr, fromNode, fromT, frFrom, T, frTo);

            // copy/backprop
            Propagate(fromNode, fromShape, frFrom, outShape, frTo, isForward, +1);
            LogicError("This code path has never been tested."); // remove this once we have
        }
    }

public:
    virtual void ForwardProp(const FrameRange& fr) override
    {
        // for (size_t xx = 0; xx < 3; xx++)   // for testing the strange slow-down
        {
            if (fr.GetIterationDimension() != m_shiftDimParam)
                LogicError("ShiftNode::ForwardProp(): FrameRange not iterating over user-specified dimension.");

#ifdef _DEBUG
            // for debugging, invalidate the output region, so we will catch if we missed to update something
            ValueFor(fr).Invalidate();
#endif

            // STEP 1: whole-sale copy a shifted version of the input to the output
            //  - consider the saved parts from the last minibatch as part of the input at dimensions beyond the bounds
            //  - ignore boundary conditions at this point (will be fixed subsequently)
            // When iterating over time, this will copy a little too much in case of multiple concatenated sequences within a single parallel sequence.

            // get the logical ranges we want to shift
            TensorShape inShape, outShape;               // expanded tensor shapes of input and output
            SliceBounds inSliceLogical, outSliceLogical; // the logical ranges to shift
            size_t rank = DetermineElementwiseTensorRank();
            DetermineSlices(rank, fr, inShape, outShape, inSliceLogical, outSliceLogical);

            // now copy the two stripes--one that is main-to-main, and one that pulls in data from previous state (truncated BPTT only)
            // This correctly handles if input is a tensor with strides. This is currently not the case, but may be if we support in-place.

            SliceBounds inSliceMain, outSliceMain;   // main-to-main
            SliceBounds inSliceState, outSliceState; // from state
            PartitionSlices(inSliceLogical, outSliceLogical, outShape[m_shiftDim], inSliceMain, outSliceMain, inSliceState, outSliceState);

            if (!inSliceState.first.empty() && inSliceState.second[m_shiftDim] > inSliceState.first[m_shiftDim])
            {
                // Note: If all sequences begin at the start of the range, this would copy invalid values which would be overwrittten below.
                // This is prevented in that m_state will be set to empty in the previous MB if all sequences ended, which will in turn return an empty slice.
                auto from = DataTensorFor(m_state.m_delayedValue, m_state.m_shape, inSliceState);
                auto to = DataTensorFor(Value(), outShape, outSliceState);
                to.AssignCopyOf(from);
            }
            if (inSliceMain.second[m_shiftDim] > inSliceMain.first[m_shiftDim])
            {
                auto from = DataTensorFor(Input(0)->Value(), inShape, inSliceMain);
                auto to = DataTensorFor(Value(), outShape, outSliceMain);
                to.AssignCopyOf(from);
            }
            // We have now pulled anything from within the logical bounds.
            // Any frame that pulls from outside contains invalid values (either not initialized or copied from incorrect source), which must be fixed next.

            // STEP 2: fix up the boundary conditions
            //  - fill in all frames that are too close to boundary and must be filled from context (recurrent) or by replication (non-recurrent only)
            //    The above may already have written (wrong) values in there, or not written anything at all yet.

            PropagateBoundaryFrames(fr, rank, inSliceLogical, outShape, outSliceLogical, /*isForward=*/true);
        }
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        //     if (!fr.IsAllFrames())   // for measuring speed
        //         return;
        TensorShape inShape, outShape;               // expanded tensor shapes of input and output
        SliceBounds inSliceLogical, outSliceLogical; // the logical ranges to shift
        size_t rank = DetermineElementwiseTensorRank();
        DetermineSlices(rank, fr, inShape, outShape, inSliceLogical, outSliceLogical);

        // propagate into boundary
        // If the boundary is a scalar constant, then this will not be called.
        // Note: This will typically be called outside the loop, so in case of delay > 1, we get a minor benefit from doing it in bulk.
        if (inputIndex == 1)
        {
            PropagateBoundaryFrames(fr, rank, inSliceLogical, outShape, outSliceLogical, /*isForward=*/false);
        }

        // propagate into input
        else if (inputIndex == 0)
        {
            // STEP 1a: backprop all we got, including invalid ones. Inner boundary frames that we shouldn't have propagated, we later subtract again.
            SliceBounds inSliceMain, outSliceMain;   // main-to-main
            SliceBounds inSliceState, outSliceState; // from state   --dummy
            auto T = outShape[m_shiftDim];           // upper bound of iteration dimension
            PartitionSlices(inSliceLogical, outSliceLogical, T, inSliceMain, outSliceMain, inSliceState, outSliceState);

            if (inSliceMain.second[m_shiftDim] > inSliceMain.first[m_shiftDim])
            {
                Input(0)->MaskMissingGradientColumnsToZero(fr); // zero out gaps, which will leak (note: we really only need to zero out gaps close enough to boundaries)
                auto from = DataTensorFor(Input(0)->Gradient(), inShape, inSliceMain);
                auto to = DataTensorFor(Gradient(), outShape, outSliceMain);
                from.AddCopyOf(to);

                // We have now propagated anything from within the logical bounds.
                // In the case of packing we will have propagated incorrectly propagated across boundaries.
                // We will now subtract the incorrectly leaked gradient frames out again.
                // (We also propagated from gaps, but those have already been reset to 0, so those require no correction.)
                // E.g. shifting by -1
                //  |X X X X X|Y Y Y|G G G   output gradient
                //  |X X X X|Y Y Y|G G G     Input(0) gradient
                //           ^               incorrect leak: must subtract out
                //                 ^     ^   no need to correct since already 0
                //    |<----------------->|  output gradient range we must consider = outSliceMain
                // (Maybe a better way would be to copy around the frames that we should not copy.)

                // STEP 1b: fix up the frames that we incorrectly propagated
                // Only happens for time iterations, only at inner boundaries.
                bool isTimeIteration = m_shiftDim >= rank;
                if (isTimeIteration)
                {
                    ForAllBoundaryIntersectingSequences(fr, outSliceMain /*already clipped*/, T, [&](const MBLayout::SequenceInfo& toSeqInfo)
                                                        {
                                                            // determine FrameRanges for from and to
                                                            FrameRange frTo;
                                                            DetermineBoundaryToFrameRange(fr, toSeqInfo, T, frTo);
                                                            FrameRange frFrom = frTo.WithTimeOffset(m_fromOffset);
                                                            assert((int) frFrom.timeIdxInSeq + frFrom.m_timeOffset >= 0 && (int) frFrom.timeIdxInSeq + frFrom.m_timeOffset + (int) frFrom.m_timeRange <= (int) T);

                                                            // copy/backprop
                                                            Propagate(shared_from_this(), inShape, frFrom, outShape, frTo, /*isForward=*/false, -1 /*subtract*/);
                                                        });
                }
            }
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        assert(m_inputs.size() == 2);
        ComputationNodeBase::Validate(isFinalValidationPass);

        // MBLayout is just inherited
        m_pMBLayout = Input(0)->GetMBLayout();
        if (isFinalValidationPass && !m_pMBLayout)
            InvalidArgument("%ls %ls operation must operate on data (must have an MB Layout).", NodeName().c_str(), OperationName().c_str());
        if (isFinalValidationPass && !Input(1)->GetMBLayout() && Input(1)->GetSampleMatrixNumCols() != 1)
            InvalidArgument("%ls %ls operation requires the boundary node to have one column.", NodeName().c_str(), OperationName().c_str());

        // as is the sample layout
        SetDims(Input(0));

        // determine the dimension that is to be shifted (convert user-specified as a zero-based index)
        if (isFinalValidationPass)
        {
            size_t rank = DetermineElementwiseTensorRank();
            auto valueShape = GetTensorShape(rank); // bounds of the Value()
            m_shiftDim = m_shiftDimParam > 0 ? m_shiftDimParam - 1 /*regular dimensions are specified as 1-based*/ : valueShape.size() + m_shiftDimParam /*-1 for time dimension*/;
        }
    }

    // special interface for use by loop detection
    virtual int /*IRecurrentNode::*/ GetRecurrenceSteppingDirection() const override
    {
        if (m_boundaryMode != BoundaryMode::reachAcross) // duplicating boundary frames cannot be done with recurrence
            return 0;
        else if (m_fromOffset < 0)
            return +1;
        else if (m_fromOffset > 0)
            return -1;
        else
            return 0;
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ShiftNode<ElemType>>(nodeP);
            node->m_fromOffset = m_fromOffset;
            node->m_boundaryMode = m_boundaryMode;
            node->m_shiftDimParam = m_shiftDimParam;
            node->m_shiftDim = m_shiftDim;
            node->m_state = m_state;
        }
    }

    class ShiftNodeState : public INodeState
    {
    public:
        Matrix<ElemType> m_delayedValue;                   // saves the activation of the previous step that this node points to
        TensorShape m_shape;                               // tensor shape that describes m_delayedValue
        vector<MBLayout::SequenceInfo> m_delayedSequences; // and associated sequence info. This is only used for consistency checking (it must match).
        ShiftNodeState(DEVICEID_TYPE deviceId)
            : m_delayedValue(deviceId)
        {
        }
        bool empty() const
        {
            return m_delayedSequences.empty();
        }
        void clear()
        {
            m_delayedValue.Resize(0, 0);
            m_shape = TensorShape();
            m_delayedSequences.clear();
        }
    };
    typedef std::shared_ptr<ShiftNodeState> ShiftNodeStatePtr;

    // state export/import
    // This is done with a shared_ptr. The current state is exported, the internal state is cleared.
    // Ownership of members is logically transferred to the exporting entity.
    // Physically, however, since we often transfer between CPU and GPU, activation data is merely copied,
    // and the GPU or CPU object resized to (0,0) without giving up the memory.
    virtual NodeStatePtr ExportState() // TODO: can we instead pass the shared_ptr object in? So we don't need to create a new one all the time? Or should we still take ownership of the ptr?
    {
        auto state = make_shared<ShiftNodeState>(CPUDEVICE);
        state->m_delayedValue.SetValue(m_state.m_delayedValue); // note: this will transfer from GPU to CPU
        m_state.m_delayedValue.Resize(0, 0);
        state->m_shape = std::move(m_state.m_shape);
        state->m_delayedSequences = std::move(m_state.m_delayedSequences);
        return state;
    }
    virtual void ImportState(const NodeStatePtr& statep) override
    {
        ShiftNodeStatePtr state = dynamic_pointer_cast<ShiftNodeState>(statep);
        if (!state)
            LogicError("ImportState: Wrong state object passed (wrong type).");
        m_state.m_delayedValue.SetValue(state->m_delayedValue); // note: this will transfer from CPU to GPU
        state->m_delayedValue.Resize(0, 0);
        m_state.m_shape = std::move(state->m_shape);
        m_state.m_delayedSequences = std::move(state->m_delayedSequences);
    }
    m_inputInvalidSequences
protected:
    // parameters remembered from construction
    int m_fromOffset;            // offset to pull from
    BoundaryMode m_boundaryMode; // how to fill at the boundary (reach across or duplicate)
    int m_shiftDimParam;         // dimension to shift (default: time)

    size_t m_shiftDim; // m_shiftDimParam matched to the real tensor index

    ShiftNodeState m_state; // state that is carried over across evaluations
    // Note: The version held by this node lives in the GPU, whereas the versions being exported carry CPU-side copies

    function<void()> m_attachInputsFn; // for late expansion of inputs (scripting)
};

#endif

}}}
