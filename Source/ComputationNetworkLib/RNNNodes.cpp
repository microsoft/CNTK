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

vector<size_t> numSequencesForFrame;
// -----------------------------------------------------------------------
// OptimizedRNNStackNode
// -----------------------------------------------------------------------

template<class ElemType>
OptimizedRNNStackNode<ElemType>::OptimizedRNNStackNode(DEVICEID_TYPE deviceId, const wstring& name)
    : Base(deviceId, name),
    m_rnnAttributes(0, 0, 0, L"lstm", -1),
    m_BackwardDataCalledYet(false)
{
}

// This constructor helps with BrainScript integration
template<class ElemType>
OptimizedRNNStackNode<ElemType>::OptimizedRNNStackNode(const ScriptableObjects::IConfigRecordPtr configp)
    : Base(configp->Get(L"deviceId"), L"<placeholder>"), 
    m_rnnAttributes(configp->Get(L"bidirectional"), configp->Get(L"numLayers"), configp->Get(L"hiddenDims"), configp->Get(L"recurrentOp"), configp->Get(L"axis")),
    m_BackwardDataCalledYet(false)
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
}

template<class ElemType>
OptimizedRNNStackNode<ElemType>::OptimizedRNNStackNode(DEVICEID_TYPE deviceId, const std::wstring& name, bool bidirectional, size_t numLayers, size_t hiddenSize, const std::wstring& recurrentOp)
    : Base(deviceId, name),
    m_rnnAttributes(bidirectional, numLayers, hiddenSize, recurrentOp, -1),
    m_BackwardDataCalledYet(false)
{
}

template<class ElemType>
/*virtual*/ void OptimizedRNNStackNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const /*override*/
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<OptimizedRNNStackNode<ElemType>>(nodeP);
        node->m_rnnAttributes = m_rnnAttributes;
    }
}

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);
    m_rnnAttributes.Write(fstream);
}

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);
    bool isLegacyVersion = modelVersion < CNTK_MODEL_VERSION_14; // (to support an internal legacy version)
    m_legacySwapInputsPending = isLegacyVersion;
    m_rnnAttributes.Read(fstream, /*readAxis=*/ !isLegacyVersion);
}

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::TransposeHelper(const MatrixBasePtr matX, const TensorShape &shapeX, MatrixBasePtr matY, TensorShape &shapeY)
{
    // This function transposes the second and third axes of the input (X), creating a transposed copy in the output (Y).
    //
    // In 'frame mode', CNTK will present the data with the final two axes being the recurrent axis followed by the 
    // 'minibatch axis'. CUDNN expects these to be in the reverse order, which is accomplished by TransposeHelper().

    shapeY = shapeX;
    shapeY.SwapDimsInPlace(1, 2);

    TensorView<ElemType> Y(matY, TensorShape(shapeY.GetDims()));
    TensorView<ElemType> X(matX, shapeY);
    Y.AssignCopyOf(X);
    shapeY = Y.GetShape();
};

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::ForwardProp(const FrameRange& fr)
{
    // The parameters are stored in a column matrix
    Matrix<ElemType>& paramW = InputRef(0).Value();

    MBLayoutPtr mb = GetMBLayout();
    if (m_rnnAttributes.IsSpatialRecurrence())
    {
        TensorView<ElemType> outputY = ValueTensorFor(SIZE_MAX, fr);

        // ensure enough storage.
        m_transposedOutput->Resize(           Value());
        m_transposedInput->Resize(InputRef(1).Value());

        // For windowed LSTM, CNTK is providing data with the second dimension being time-like and the third dimension
        // being minibatch index. CuDnn expects the second dimension to be minibatch index, and the third dimension
        // to be time-like. This sequence of operations creates a transposed copy of the data in m_transposedInput
        // and shapeXT

        TransposeHelper(InputRef(1).ValuePtr(), InputRef(1).GetTensorSliceFor(SIZE_MAX, fr), m_transposedInput, shapeXT);

        // Similarly, we will eventually need to transpose the output. Generate the necessary shape here, and do
        // the transposition after RNNForward() returns.

        // create the necessary shape.
        shapeYT = TensorShape(this->GetTensorSliceFor(SIZE_MAX, fr));
        // this swap results in a shape with swapped dimensions, but also swapped strides
        shapeYT.SwapDimsInPlace(1, 2);
        // this copy is necessary so that the strides are dense.
        shapeYT = TensorShape(shapeYT.GetDims());

        // create a vector with the correct number of timesteps(shapeXT[2]) containing the sequence count (shapeXT[1])
        numSequencesForFrame = vector<size_t>(shapeXT[2], shapeXT[1]);
        m_transposedOutput->RNNForward(*m_transposedInput, paramW, shapeXT[0], shapeYT[0], numSequencesForFrame, m_rnnAttributes, *m_reserve, *m_workspace);

        // No one uses shapeY, but it is necessary
        TensorShape shapeY;
        TransposeHelper(m_transposedOutput, TensorShape(shapeYT.GetDims()), this->ValuePtr(), shapeY);
    }
    else
    {
        shapeXT = TensorShape(InputRef(1).GetTensorSliceFor(SIZE_MAX, fr));
        shapeYT = TensorShape(          GetTensorSliceFor(SIZE_MAX, fr));

        // This changes the data from "minibatch paking" in InputRef(0).Value() to "dense CuDNN packing" in m_transposedInput
        this->PackSequencesForCuDNN(InputRef(1).Value(), *m_transposedInput, numSequencesForFrame);

        // ensure enough storage
        m_transposedOutput->Resize(this->Value().GetNumRows(), m_transposedInput->GetNumCols());

        m_transposedOutput->RNNForward(*m_transposedInput, paramW, shapeXT[0], shapeYT[0], numSequencesForFrame, m_rnnAttributes, *m_reserve, *m_workspace);
        this->UnpackSequencesFromCuDNN(*m_transposedOutput, this->Value());
    }
    m_BackwardDataCalledYet = false;
}

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr)
{
    MBLayoutPtr mb = this->GetMBLayout();

    // ensure BackwardData is the first method called, as required by CuDnn API
    if (!m_BackwardDataCalledYet)
    {
        Matrix<ElemType>& paramW = InputRef(0).Value();

        if (m_rnnAttributes.IsSpatialRecurrence())
        {
            // To obey the data layout constraints of CuDnn, we take the derivative we're given,
            // and transpose it before feeding to the interface.
            m_transposedDOutput->Resize(this->Gradient());
            TransposeHelper(this->GradientPtr(), this->GetTensorSliceFor(SIZE_MAX, fr), m_transposedDOutput, shapeYT);
        }
        else
        {
            m_transposedDOutput->DoGatherColumnsOf(0.0, *(this->m_packingIndex), this->Gradient(), 1.0);
        }

        // Ensure enough space for the result
        m_transposedDInput->Resize(InputRef(1).GetSampleLayout().GetNumElements(), m_transposedDOutput->GetNumCols());

        // Do the work
        m_transposedOutput->RNNBackwardData(*m_transposedDOutput, paramW, *m_transposedDInput, m_rnnAttributes, *m_reserve, *m_workspace);
        m_BackwardDataCalledYet = true;
    }
    if (inputIndex == 0) // parameters
    {
        Matrix<ElemType>& paramDW = InputRef(0).Gradient();
        m_transposedOutput->RNNBackwardWeights(*m_transposedInput, *m_transposedOutput, paramDW, m_rnnAttributes, *m_reserve, *m_workspace);
    }
    else if (inputIndex == 1) // data
    {
        // all of the work was done above, where RNNBackwardData is called. Now, just unpack the result.
        if (m_rnnAttributes.IsSpatialRecurrence())
        {
            TensorShape tmp;
            TransposeHelper(m_transposedDInput, shapeXT, InputRef(1).GradientPtr(), tmp);
        }
        else
        {
            InputRef(1).Gradient().DoScatterColumnsOf(1.0, *(this->m_packingIndex), *m_transposedDInput, 1.0, /*idxHaveDups*/ false);
        }
    }
}

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::Validate(bool isFinalValidationPass)
{
    // support an internal legacy version
    if (m_legacySwapInputsPending)
    {
        ::swap(m_inputs[0], m_inputs[1]);
        m_legacySwapInputsPending = false;
    }
    // N.B.: I need both of these lines.
    Base::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    // get tensor shapes
    let& shapeA = Input(0)->GetSampleLayout(); // parameters
    let& shapeB = Input(1)->GetSampleLayout(); // data
    auto dimsA = shapeA.GetDims();
    auto dimsB = shapeB.GetDims();

    // data rank must match spatial/temporal recurrence mode
    if (isFinalValidationPass &&
        dimsB.size() != (m_rnnAttributes.IsSpatialRecurrence() ? 2 : 1))
    {
        InvalidArgument("%ls: Input [%s] must have rank 1 for axis=-1 and rank 2 for axis=2.", NodeDescription().c_str(), string(shapeB).c_str());
    }

    // ComputationNode derived classes are guaranteed to have a MBLayout
    if (isFinalValidationPass && !HasMBLayout())
        InvalidArgument("%ls: Input [%s] must operate on minibatches.", NodeDescription().c_str(), string(shapeB).c_str());

    // validate and infer
    if (isFinalValidationPass || (dimsA.size() > 0 && dimsB.size() > 0)) // only if we got at least some input dimensions to work with or need to wrap up
    {
        // now determine result dimensions
        auto dimsC = dimsB;

        // output dims
        dimsC[0] = (m_rnnAttributes.m_bidirectional ? 2 : 1) * m_rnnAttributes.m_hiddenSize;

        // infer input size
        // Note: Output dim is second axis, so say initOutputRank=-1 in the Parameters{} definition.
        if (dimsA.size() == 2)
        {
            let numParameters = m_rnnAttributes.GetNumParameters(shapeB.GetNumElements());
            Input(0)->ValidateInferInputDimsFrom(TensorShape(numParameters.first, numParameters.second));
        }

        // N.B. - this is the magical call, the reason for the function
        // dimensions would be outputRank * numSamples * minibatch * time.
        // This call establishes outputRank * numSamples, the rest will be filled in
        // dynamically though the MBLayout.
        SetDims(TensorShape(dimsC), HasMBLayout());
    }
};

template<class ElemType>
void OptimizedRNNStackNode<ElemType>::PackSequencesForCuDNN(const Matrix<ElemType>& src, Matrix<ElemType>& dst, vector<size_t>& numSequencesForFrame2)
{
    MBLayoutPtr mb = this->GetMBLayout();
    if (mb->HasSequenceBeyondBegin())
        RuntimeError("Invalid MBLayout: Only whole-utterance processing is supported");
#if 0
    BUGBUG: Disable this check to mask a problem with the way EvalReader creates segments.
    if (mb->HasSequenceBeyondEnd())
        RuntimeError("Invalid MBLayout: Only whole-utterance processing is supported");
#endif

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
    });

    size_t maxSeqLength = seq[sequenceOrder[0]].GetNumTimeSteps();
    // BUGBUG: This forces the sequences to fit, due to a very bad convention in the evaldll interface.
    if (maxSeqLength > mb->GetNumTimeSteps())
        maxSeqLength = mb->GetNumTimeSteps();

    // a count of how many sequnces are packed for a particular frame.
    // reset to zero, and compute from current layout information
    // this information is useful when creating the tensor descriptors for CuDNN.
    numSequencesForFrame2.resize(maxSeqLength);
    fill(numSequencesForFrame2.begin(), numSequencesForFrame2.end(), 0L);

    // make sure the index is on CPU so we can use SetValue()
    // 
    m_packingIndex->TransferToDeviceIfNotThere(-1, true/*isBeingMoved*/, false/*emptyTransfer*/, false/*updatePreferredDevice*/);

    // Reserve one element for every valid sample. DoGatherColumnsOf() requires it to be a row vector
    m_packingIndex->Resize(1, mb->GetActualNumSamples());

    size_t dst_frame = 0;
    for (size_t fr = 0; fr < maxSeqLength; fr++)
    {
        for (size_t j = 0; j < numSequences && seq[sequenceOrder[j]].GetNumTimeSteps()>fr; j++)
        {
            m_packingIndex->SetValue(0, dst_frame++, (ElemType)mb->GetColumnIndex(seq[sequenceOrder[j]], fr));
            numSequencesForFrame2[fr]++;
        }
    }

    // this->gather(beta,idx,a,alpha) operation is defined as
    // *this[:,j] = a[:,idx[j]] * alpha + *this[:,j] * beta
    dst.DoGatherColumnsOf(0.0, *(this->m_packingIndex), src, 1.0);
}
template<class ElemType>
void OptimizedRNNStackNode<ElemType>::UnpackSequencesFromCuDNN(const Matrix<ElemType>& src, Matrix<ElemType>& dst)
{
    // this->scatter(beta,ndx,a,alpha) operation is defined as
    // *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
    dst.DoScatterColumnsOf(0.0, *(this->m_packingIndex), src, 1.0, /*idxHaveDups*/ false);
}


template class OptimizedRNNStackNode<float>;
template class OptimizedRNNStackNode<double>;
template class OptimizedRNNStackNode<half>;

}}}
