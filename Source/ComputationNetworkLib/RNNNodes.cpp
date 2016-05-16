//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"
#include "TensorView.h"
#include "RNNNodes.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <utility>
#include <assert.h>


namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// RNNNode
// -----------------------------------------------------------------------

template<class ElemType>
RNNNode<ElemType>::RNNNode(DEVICEID_TYPE deviceId, const wstring& name)
    : Base(deviceId, name),
    m_rnnParameters(0, 0, 0, L"LSTM"),
    m_BackwardDataCalledYet(false)
{
}

// This constructor helps with BrainScript integration
template<class ElemType>
RNNNode<ElemType>::RNNNode(const ScriptableObjects::IConfigRecordPtr configp)
    : Base(configp->Get(L"deviceId"), L"<placeholder>"), 
    m_rnnParameters(configp->Get(L"bidirectional"), configp->Get(L"numLayers"), configp->Get(L"hiddenSize"), configp->Get(L"rnnMode")),
    m_BackwardDataCalledYet(false)
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
}

template<class ElemType>
void RNNNode<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);
    m_rnnParameters.Write(fstream);
}

template<class ElemType>
void RNNNode<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);
    m_rnnParameters.Read(fstream);
}

template<class ElemType>
TensorView<ElemType> RNNNode<ElemType>::TensorHelper(int inputIndex/*-1 for output*/, bool gradient/*instead of value*/, const FrameRange& fr)
{
    auto input = inputIndex < 0 ? this : Input(inputIndex).get();
    return gradient ? input->GradientTensorFor(SIZE_MAX, fr) : input->ValueTensorFor(SIZE_MAX, fr);
}

template<class ElemType>
void RNNNode<ElemType>::TransposeHelper(const MatrixBasePtr matX, const TensorShape &shapeX, MatrixBasePtr matY, TensorShape &shapeY)
{
    shapeY = shapeX;
    shapeY.SwapDimsInPlace(1, 2);

    TensorView<ElemType> Y(matY, TensorShape(shapeY.GetDims()));
    TensorView<ElemType> X(matX, shapeY);
    Y.AssignCopyOf(X);
    shapeY = Y.GetShape();
};

template<class ElemType>
void RNNNode<ElemType>::ForwardProp(const FrameRange& fr)
{
    // ComputationNode derived classes are guaranteed to have a MBLayout
    if (!this->HasMBLayout())
    {
        LogicError("RNNNode must operate on minibatches");
    }

    // The parameters are stored in a column matrix
    Matrix<ElemType>& paramW = Input(1)->Value();

    // Detect frame mode. Bugbug: should never revisit this decision after the first data comes through
    MBLayoutPtr mb = this->GetMBLayout();
    bool frameMode = (mb->GetNumTimeSteps() == 1) ? true : false;

    if (frameMode)
    {
        TensorView<ElemType> outputY = ValueTensorFor(SIZE_MAX, fr);

        // For windowed LSTM, CNTK is providing data with the second dimension being time-like and the third dimension
        // being minibatch index. CuDnn expects the second dimension to be minibatch index, and the third dimension
        // to be time-like. This sequence of operations creates a transposed copy of the data in m_transposedInput
        // and shapeXT

        m_transposedInput->Resize(Input(0)->Value());
        TransposeHelper(Input(0)->ValuePtr(), Input(0)->GetTensorSliceFor(SIZE_MAX, fr), m_transposedInput, shapeXT);

        // Similarly, we will eventually need to transpose the output. Generate the necessary shape here, and do
        // the transposition after RNNForward() returns.

        // ensure enough storage.
        m_transposedOutput->Resize(this->Value());

        // create the necessary shape.
        shapeYT = TensorShape(this->GetTensorSliceFor(SIZE_MAX, fr));
        // this swap results in a shape with swapped dimensions, but also swapped strides
        shapeYT.SwapDimsInPlace(1, 2);
        // this copy is necessary so that the strides are dense.
        shapeYT = TensorShape(shapeYT.GetDims());

        //TODO: make this work for numSequencesForFrame style call
        //vector<size_t> numSequencesForFrame(1, mb->GetActualNumSamples());
        //m_transposedOutput->RNNForward(*m_transposedInput, shapeXT, paramW, shapeYT, m_rnnParameters, *m_reserve, *m_workspace);

        // No one uses shapeY, but it is necessary
        TensorShape shapeY;
        TransposeHelper(m_transposedOutput, TensorShape(shapeYT.GetDims()), this->ValuePtr(), shapeY);
        m_BackwardDataCalledYet = false;
    }
    else
    {
        vector<size_t> numSequencesForFrame;
        // ensure enough storage - due to blank frames we don't need, this could be bigger than strictly necessary
        m_transposedOutput->Resize(this->Value());
        shapeXT = TensorShape(Input(0)->GetTensorSliceFor(SIZE_MAX, fr));
        shapeYT = TensorShape(this->GetTensorSliceFor(SIZE_MAX, fr));

        this->PackSequencesForCuDNN(Input(1)->Value(), *m_transposedInput, numSequencesForFrame);
        m_transposedOutput->RNNForward(*m_transposedInput, shapeXT, paramW, shapeYT, m_rnnParameters, numSequencesForFrame, *m_reserve, *m_workspace);
        this->UnpackSequencesFromCuDNN(*m_transposedOutput, this->Value());
    }
}

template<class ElemType>
void RNNNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr)
{
    // ensure BackwardData is the first method called, as required by CuDnn API
    if (!m_BackwardDataCalledYet)
    {
        Matrix<ElemType>& paramW = Input(1)->Value();

        // To obey the data layout constraints of CuDnn, we take the derivative we're given,
        // and transpose it before feeding to the interface.
        m_transposedDOutput->Resize(this->Gradient());
        TransposeHelper(this->GradientPtr(), this->GetTensorSliceFor(SIZE_MAX, fr), m_transposedDOutput, shapeYT);

        // Ensure enough space for the result
        m_transposedDInput->Resize(Input(1)->Gradient());

        // Do the work
        m_transposedOutput->RNNBackwardData(*m_transposedDOutput, shapeYT, paramW, *m_transposedDInput, shapeXT, m_rnnParameters, *m_reserve, *m_workspace);
        m_BackwardDataCalledYet = true;
    }
    if (inputIndex == 1) // parameters
    {
        Matrix<ElemType>& paramDW = Input(1)->Gradient();
        m_transposedOutput->RNNBackwardWeights(*m_transposedInput, shapeXT, *m_transposedOutput, shapeYT, paramDW, m_rnnParameters, *m_reserve, *m_workspace);
    }
    else if (inputIndex == 0) // data
    {
        // all of the work was done above, where RNNBackwardData is called. Now, just place a transposed result.
        TensorShape tmp;
        TransposeHelper(m_transposedDInput, shapeXT, Input(0)->GradientPtr(), tmp);
    }
}

template<class ElemType>
void RNNNode<ElemType>::Validate(bool isFinalValidationPass)
{
    // N.B.: I need both of these lines.
    Base::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    // get tensor shapes
    auto dimsA = Input(1)->GetSampleLayout().GetDims(); // data
    auto dimsB = Input(0)->GetSampleLayout().GetDims(); // parameters

    // validate and infer
    if (isFinalValidationPass || (dimsA.size() > 0 && dimsB.size() > 0)) // only if we got at least some input dimensions to work with or need to wrap up
    {
        // now determine result dimensions
        auto dimsC = dimsB;
        // output dims - bugbug: this is hard-coded for bidirectional models
        dimsC[0] = (m_rnnParameters.m_bidirectional ? 2 : 1) * m_rnnParameters.m_hiddenSize;

        // N.B. - this is the magical call, the reason for the function
        // dimensions would be outputRank * numSamples * minibatch * time.
        // This call establishes outputRank * numSamples, the rest will be filled in
        // dynamically though the MBLayout.
        SetDims(TensorShape(dimsC), HasMBLayout());

    }
};

template<class ElemType>
void RNNNode<ElemType>::PackSequencesForCuDNN(const Matrix<ElemType>& src, Matrix<ElemType>& dst, vector<size_t>& numSequencesForFrame)
{
    MBLayoutPtr mb = this->GetMBLayout();
    if (mb->HasSequenceBeyondBegin())
        RuntimeError("Invalid MBLayout: Only whole-utterance processing is supported");
    if (mb->HasSequenceBeyondEnd())
        RuntimeError("Invalid MBLayout: Only whole-utterance processing is supported");

    // retrieve only the non-gap sequences
    vector<MBLayout::SequenceInfo> seq;
    std::copy_if(
        mb->GetAllSequences().begin(),
        mb->GetAllSequences().end(),
        back_inserter<vector<MBLayout::SequenceInfo>>(seq),
        [](const MBLayout::SequenceInfo& x) { return x.seqId != GAP_SEQUENCE_ID; });

    // sequenceOrder[i] will eventually be the i'th longest sequence,
    // after sorting from longest to shortest. Ties are broken by the sequence id.
    size_t numSequences = seq.size();
    vector<size_t> sequenceOrder(numSequences);
    for (size_t j = 0; j<numSequences; j++)
        sequenceOrder[j] = j;
    sort(sequenceOrder.begin(), sequenceOrder.end(), [&](size_t a, size_t b)
    {
        // sort in decreasing order of length
        if (seq[a].GetNumTimeSteps() > seq[b].GetNumTimeSteps())
            return true;
        // break ties with increasing seqId
        else if (seq[a].GetNumTimeSteps() == seq[b].GetNumTimeSteps())
            return seq[a].seqId < seq[b].seqId;
        return false;
    }
    );

    size_t maxSeqLength = seq[sequenceOrder[0]].GetNumTimeSteps();

    // a count of how many sequnces are packed for a particular frame.
    // reset to zero, and compute from current layout information
    // this information is useful when creating the tensor descriptors for CuDNN.
    numSequencesForFrame.resize(maxSeqLength);
    fill(numSequencesForFrame.begin(), numSequencesForFrame.end(), 0L);

    // make sure the index is on CPU so we can use SetValue()
    m_packingIndex->TransferToDeviceIfNotThere(-1, true, false, false);

    // Reserve one element for every valid sample. DoGatherColumnsOf() requires it to be a row vector
    m_packingIndex->Resize(1, mb->GetActualNumSamples());

    size_t dst_frame = 0;
    for (size_t fr = 0; fr < maxSeqLength; fr++)
    {
        for (size_t j = 0; j < numSequences && seq[sequenceOrder[j]].GetNumTimeSteps()>fr; j++)
        {
            m_packingIndex->SetValue(0, dst_frame++, (ElemType)mb->GetColumnIndex(seq[sequenceOrder[j]], fr));
            numSequencesForFrame[fr]++;
        }
    }

    // this->gather(beta,idx,a,alpha) operation is defined as
    // *this[:,j] = a[:,idx[j]] * alpha + *this[:,j] * beta
    dst.DoGatherColumnsOf(0.0, *(this->m_packingIndex), src, 1.0);
}
template<class ElemType>
void RNNNode<ElemType>::UnpackSequencesFromCuDNN(const Matrix<ElemType>& src, Matrix<ElemType>& dst)
{
    // this->scatter(beta,ndx,a,alpha) operation is defined as
    // *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
    dst.DoScatterColumnsOf(0.0, *(this->m_packingIndex), src, 1.0);
}


template class RNNNode<float>;
template class RNNNode<double>;

} } }
