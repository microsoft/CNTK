//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReshapingNodes.cpp -- collection of nodes that reshape or sub-sample matrices leading to layout changes
//

#include "Basics.h"
#include "ReshapingNodes.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "Sequences.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <stack>
#include <unordered_map>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ReduceElements (op, axis=, input)
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const /*override*/
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<ReduceElementsNode<ElemType>>(nodeP);
        node->m_axis        = m_axis;
        node->m_operation   = m_operation;
        node->m_reductionOp = m_reductionOp;
        node->m_scale       = m_scale;
        node->m_keepDimensions = m_keepDimensions;
    }
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    Base::Load(fstream, modelVersion);
    fstream >> m_axis >> m_operation;
    if (modelVersion >= CNTK_MODEL_VERSION_24)
        fstream >> m_keepDimensions;
    else
        m_keepDimensions = DefaultKeepDimensionsSetting(m_axis);

    ValidateOp();
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::Save(File& fstream) const /*override*/
{
    Base::Save(fstream);
    fstream << m_axis << m_operation; // note: we serialize the string and not the opcode, since opcodes may change
    fstream << m_keepDimensions;
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::ForwardProp(const FrameRange& fr) /*override*/
{
    // We are mixing two kinds of operations here; elementwise and whole-batch or sequence reduction (ReduceAllAxes()).
    // In the latter case, we must mimic the behaviour of ComputationNodeNonLooping.
    if ((ReduceAllAxes() || ReduceSequenceAxis() || ReduceBatchAxis()) && !fr.IsAllFrames())
        LogicError("%ls: %s node should never be in a loop when reducing over all static and dynamic axes or just the sequence axis.", Base::NodeDescription().c_str(), typeid(*this).name());

    const auto frInput = (ReduceAllAxes() || ReduceBatchAxis()) ? FrameRange(InputRef(0).GetMBLayout()) : fr; // can't use 'fr' for ReduceAllAxes() and ReduceBatchAxis() as it refers to the result (same as for training criteria)

    // when reducing all, we must mask gaps
    if (ReduceAllAxes() || ReduceBatchAxis())
    {
        InputRef(0).MaskMissingValueColumnsTo(frInput, NeutralValue(m_reductionOp));
        if (IsMean())
        {
            // In mean reduction for all axes or batch axis, we need to carefully compute the scaling factor
            auto actual_samples = InputRef(0).HasMBLayout() ? InputRef(0).GetMBLayout()->GetActualNumSamples() : 1;
            m_scale = ElemType(1.0 / actual_samples);
            if (ReduceAllAxes())
                m_scale /= ElemType(GetInputSampleLayout(0).GetNumElements());
        }
    }

    // Create a new layout if we are reducing the sequence axis
    if (ReduceSequenceAxis())
    {
        auto inputMBLayout = InputRef(0).GetMBLayout();
        if (inputMBLayout->HasSequenceBeyondBegin() || inputMBLayout->HasSequenceBeyondEnd())
            LogicError("%ls: %s node cannot perform sequence axis reduction for truncated sequence.", Base::NodeDescription().c_str(), typeid(*this).name());

        GetMBLayout()->InitAsFrameMode(inputMBLayout->GetNumSequences());
        UpdateFunctionValuesSize();
    }
    // get the args
    size_t rank = DetermineElementwiseTensorRank();
    TensorView<ElemType> input;
    if (ReduceSequenceAxis())
    {
        ElemType gapPadValue = NeutralValue(m_reductionOp);
        input = ComputationNode<ElemType>::Unpack(GetSampleLayout(), InputRef(0).Value(), InputRef(0).GetMBLayout(), m_tempUnpackedData, m_tempScatterIndices, m_tempMask, /*batchMajor=*/ true, &gapPadValue);
    }
    else
        input = InputRef(0).ValueTensorFor(rank, frInput);

    auto result = ReduceAllAxes() ? TensorView<ElemType>(ValuePtr(), TensorShape(1)) : ValueTensorFor(rank, fr);

    switch (m_reductionOp)
    {
    case ElementWiseOperator::opArgmin:
    case ElementWiseOperator::opArgmax:
        result.DoArgReductionOpOf(input, m_reductionOp);
        break;
    default:
        // the actual operation is a Copy with reduction, where the magic is in the reduction op
        // For "Mean", m_scale is 1/#elements, and 1 otherwise.
        result.DoUnaryOpOf(0, input, m_scale, ElementWiseOperator::opCopy, m_reductionOp);
    }
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr) /*override*/
{
    assert(inputIndex == 0), inputIndex;

    if (ReduceSequenceAxis())
    {
        // Broadcast along the sequence
        auto result = ValueFor(fr);
        ComputationNode<ElemType>::BroadcastToPacked(Gradient(), GetMBLayout(), /*beta =*/ 1, InputRef(0).Gradient(), FrameRange(InputRef(0).GetMBLayout()), m_tempGatherIndices);
    }
    else
    {
        const auto frInput = (ReduceAllAxes() || ReduceBatchAxis()) ? FrameRange(InputRef(0).GetMBLayout()) : fr; // can't use 'fr' for ReduceAllAxes() as it refers to the result (same as for training criteria)
                                                                                        // get the args
        size_t rank = DetermineElementwiseTensorRank();
        auto sliceOutputGrad = ReduceAllAxes() ? TensorView<ElemType>(GradientPtr(), TensorShape(1)) : GradientTensorFor(rank, fr); // propagate from this one...
        auto sliceInputGrad = InputRef(0).GradientTensorFor(rank, frInput); // ...to this one

        // gradients are not as simple as passing an op-code, unfortunately
        switch (m_reductionOp)
        {
        case ElementWiseOperator::opSum:
            // "Sum":  broadcast the gradient
            // "Mean": same as "Sum" with scaling by 1/#dims
            sliceInputGrad.AddCopyOf(sliceOutputGrad, m_scale);
            break;

        case ElementWiseOperator::opLogSum:
        {
            auto input = InputRef(inputIndex).ValueTensorFor(rank, frInput);
            auto output = ValueTensorFor(rank, fr.AllowBroadcast());
            // Let: f(x, y, z) = log(exp x + exp y + exp z)
            // For the derivative we get:
            // df / dx = exp(x)/exp(f)
            //         = exp(x - f)
            sliceInputGrad.AddElementwiseProductWithExpOfDiffOf(sliceOutputGrad, input, output);
        }
        break;

        case ElementWiseOperator::opMin:
        case ElementWiseOperator::opMax:
        {
            auto input = InputRef(inputIndex).ValueTensorFor(rank, frInput);
            auto output = ValueTensorFor(rank, fr.AllowBroadcast());

            // POTENTIAL PROBLEM:
            // For ReduceMin/Max there are combinations of input values where the gradient is not defined because the function has an edge at these points.
            // E.g. for ReduceMin this is the case when the minimum input value is attained by several inputs at the same time.
            // In these cases there is no correct gradient.The question is if this could lead to any problems.
            // Let's look at two scenarios where this might happen:
            //
            // * Scenario 1: The input comes from a layer of nodes like e.g. ReLU and some of them might operate in the regime where they clip to a constant value.
            //   In this case it's not a problem that the input gradient is kind of bad as the derivative of the concerning input nodes will be zero anyway.
            //
            // * Scenario 2: The input data is directly coming from training data. Here bad gradients don't matter as we wouldn't wan't to propagate gradients to the training data.
            //
            // So as we don't have a better solution yet and it probably doesn't have impact let's stay with the current solution.
            // Also note that for Clip , Min, Max and ReLU we have the same kind of problem.
            sliceInputGrad.AddCopyIfEqualOf(input, output, sliceOutputGrad);
        }
        break;
        case ElementWiseOperator::opElementwiseProduct:
        {
            auto input  = InputRef(inputIndex).ValueTensorFor(rank, frInput);
            auto output =                      ValueTensorFor(rank, fr.AllowBroadcast());
            sliceInputGrad.AddElementwiseProductWithQuotientOf(sliceOutputGrad, output, input);
            break;
        }
        case ElementWiseOperator::opArgmin:
        case ElementWiseOperator::opArgmax:
            break;

            // more coming
        }
    }
}

template <class ElemType>
/*virtual*/ bool ReduceElementsNode<ElemType>::OutputUsedInComputingInputNodesGradients() const /*override*/
{
    switch (m_reductionOp)
    {
    case ElementWiseOperator::opSum:                   return false;
    case ElementWiseOperator::opLogSum:                return true;
    case ElementWiseOperator::opMin:                   return true;
    case ElementWiseOperator::opMax:                   return true;
    case ElementWiseOperator::opElementwiseProduct:    return true;
    case ElementWiseOperator::opArgmin:                return false;
    case ElementWiseOperator::opArgmax:                return false;
    }
    LogicError("Should not get here.");
}

template <class ElemType>
/*virtual*/ bool ReduceElementsNode<ElemType>::InputUsedInComputingInputNodesGradients(size_t inputIndex) const /*override*/
{
    switch (m_reductionOp)
    {
    case ElementWiseOperator::opSum:                   return false;
    case ElementWiseOperator::opLogSum:                return true;
    case ElementWiseOperator::opMin:                   return true;
    case ElementWiseOperator::opMax:                   return true;
    case ElementWiseOperator::opElementwiseProduct:    return true;
    case ElementWiseOperator::opArgmin:                return false;
    case ElementWiseOperator::opArgmax:                return false;
    }
    LogicError("Should not get here.");
}

// map the operation specified as a string to an ElementWiseOperator value.
template <class ElemType>
void ReduceElementsNode<ElemType>::ValidateOp()
{
    m_reductionOp = ReductionOpEnumValue(m_operation);
}

template <class ElemType>
/*virtual*/ void ReduceElementsNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    // validate the opcode (in case we got instantiated empty and never updated)
    ValidateOp();
    m_scale = (ElemType)1;
    if (ReduceAllAxes())
        Base::ValidateUnaryReduce(isFinalValidationPass, m_keepDimensions);
    else if (ReduceSequenceAxis())
    {
        Base::Validate(isFinalValidationPass);

        // we generate its own MBLayout
        if (isFinalValidationPass && !Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation can perform sequence axis reduction only on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());

        if ((m_operation != L"Sum") && (m_operation != L"Plus"))
            InvalidArgument("%ls %ls operation can perform sequence axis reduction only for the 'sum' reduction operation, specified operation %ls.", NodeName().c_str(), OperationName().c_str(), m_operation.c_str());

        if (!m_pMBLayout)
        {
            m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
            m_pMBLayout->SetUniqueAxisName(ComputationNodeBase::DefaultNoSequenceAxisName);
        }

        SetDims(Input(0)->GetSampleLayout(), HasMBLayout());
    }
    else if (ReduceBatchAxis())
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && !Input(0)->HasMBLayout())
            InvalidArgument("%ls %ls operation can perform batch axis reduction only on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());

        SetDims(Input(0)->GetSampleLayout(), false);
    }
    else
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        let shape = Input(0)->GetSampleLayout();
        auto dims = shape.GetDims();
        size_t reducedDim = 0; // (init to keep compiler happy)
        if (ReduceAllStaticAxes())
        {
            reducedDim = shape.GetNumElements();
            dims = m_keepDimensions ? SmallVector<size_t>(shape.GetRank(), 1) : SmallVector<size_t>({ 1 }); // entire sample is reduced to a scalar
        }
        else if (m_axis - 1 >= 0 && m_axis - 1 < dims.size())
        {
            reducedDim = dims[m_axis - 1];
            // one axis is reduced to a scalar
            if (m_keepDimensions)
                dims[m_axis - 1] = 1;
            else
            {
                SmallVector<size_t> reducedDims(dims.size() - 1);
                for (size_t i = 0, j = 0; i < dims.size(); ++i)
                {
                    if (i == (m_axis - 1))
                        continue;

                    reducedDims[j] = dims[i];
                    j++;
                }
                dims = reducedDims;
            }
        }
        else if (isFinalValidationPass)
            InvalidArgument("The shape of %ls [%s] has no axis %d", NodeDescription().c_str(), string(shape).c_str(), m_axis);

        // for "Mean", we must divide by #elements
        if (isFinalValidationPass && IsMean())
            m_scale = (ElemType)(1.0 / reducedDim);

        SetDims(TensorShape(dims), Input(0)->HasMBLayout());
    }
}

template class ReduceElementsNode<float>;
template class ReduceElementsNode<double>;

// -----------------------------------------------------------------------
// Where(bitVector) -- extract indices of non-0 values in a sequence
// -----------------------------------------------------------------------

// wrapper class to pass MBLayout sequence vector to PackSequences()
struct SequenceLengthVector
{
    typedef vector<vector<size_t>> SequenceVector;
    typedef MBLayout::SequenceInfo SequenceInfo;
    const SequenceVector& m_sequenceVector;        // vector of sequences (to get sequence length)
    const vector<SequenceInfo>& m_sequenceInfo;    // original sequence info (for seqId)
    SequenceLengthVector(const vector<SequenceInfo>& sequenceInfo, const SequenceVector& sequenceVector) : m_sequenceInfo(sequenceInfo), m_sequenceVector(sequenceVector) { }
    size_t size() const { return m_sequenceInfo.size(); }
    MBLayout::SequenceInfo operator[](size_t i) const // return a descriptor of the new sequence
    {
        SequenceInfo seq;
        seq.seqId = m_sequenceInfo[i].seqId;
        seq.s = i;
        seq.tBegin = 0;
        seq.tEnd = m_sequenceVector[i].size();
        return seq;
    }
    void operator=(const SequenceLengthVector&) = delete;
};

// TODO: Where should the MBLayout be created--in BeginForwardProp() or ForwardProp()?
//       BeginForwardProp() should generally have no access to the actual values,
//       while ForwardProp() might be too late. We may have to define the semantics here.
// BUGBUG: This is the first node with value-dependent MBLayout. It resizes Value(), which we otherwise always do before.
template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    // gather all sequences
    let& inMBLayout = InputRef(0).GetMBLayout();
    let& input = InputRef(0).Value();
    let& sequences = inMBLayout->GetAllSequences();
    auto& indexSequences = m_indexSequenceBuffer;
    if (indexSequences.size() < sequences.size())
        indexSequences.resize(sequences.size());
    for (size_t i = 0; i < sequences.size(); i++)
    {
        let& seq = sequences[i];
        if (seq.seqId == GAP_SEQUENCE_ID)
            continue;
        auto& indexSequence = indexSequences[i];
        // create index map for one sequence
        // this is the condition check that this node performs; the meat
        indexSequence.clear();
        double desiredCount = 0.0;
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
        {
            double delta = input(0, inMBLayout->GetColumnIndex(seq, t)); // how many frames the current time step should expand into
            desiredCount += delta; // this is now how many frames we should have
            // use a margin against round-off errors, so that we get non-binary ratios like 1/3 and 1/5 right
            // This really means generate a frame if too few, unless we are within machine accuracy of the target.
            // The assumption is that the delta has this error, while accumulation (in double) has no error.
            ElemType relativeMargin = 1 - std::numeric_limits<ElemType>::epsilon();
            while ((indexSequence.empty() && desiredCount > 0)  // no margin for the first frame (always include unless flag is 0)
                   || indexSequence.size() < desiredCount * relativeMargin)
                indexSequence.push_back(t);
        }
        // Note: The above accesses m_value directly on the CPU, putting it into BOTH state, possibly for other consumers as well.
    }
    input.CollapseDataLocation(); // BUGBUG: Move back, since BOTH state is broken at present.
    // create a new MBLayout
    let& outMBLayout = GetMBLayout();
    outMBLayout->InitAsPackedSequences(SequenceLengthVector(sequences, indexSequences), /*temp*/m_placementBuffer, /*temp*/m_rowAllocationsBuffer);
    // copy to output
    vector<ElemType> buf(outMBLayout->GetNumCols(), numeric_limits<ElemType>::quiet_NaN()); // STL cannot easily avoid initializing, so we might as well init with NaN for gaps
    let size = min(sequences.size(), outMBLayout->GetAllSequences().size()); // no non-gap sequence has an index beyond this
    for (size_t i = 0; i < size; i++)
    {
        let& seq = outMBLayout->GetAllSequences()[i];
        if (seq.seqId == GAP_SEQUENCE_ID) // gaps will keep the NaN
            continue;
        let& indexSequence = indexSequences[i];
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
            buf[outMBLayout->GetColumnIndex(seq, t)] = (ElemType)indexSequence[t];
    }
    // there may be dangling gaps at the end. Take the opportunity to verify this.
    for (size_t i = size; i < sequences.size(); i++)
        assert(sequences[i].seqId == GAP_SEQUENCE_ID);
    for (size_t i = size; i < outMBLayout->GetAllSequences().size(); i++)
        assert(outMBLayout->GetAllSequences()[i].seqId == GAP_SEQUENCE_ID);
    // the result will be kept in CPUDEVICE, since most likely we will access it again in PackedIndexNode
    Value().TransferToDeviceIfNotThere(CPUDEVICE, /*isBeingMoved=*/ true, /*emptyTransfer=*/ true, /*updatePreferredDevice=*/ true);
    Value().SetValue(1, outMBLayout->GetNumCols(), CPUDEVICE, buf.data(), MatrixFormat::matrixFormatColMajor);
}

template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/) /*override*/
{
    // we cannot backprop through a condition
    return;
}

template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);
    // we generate its own MBLayout
    if (isFinalValidationPass && !Input(0)->HasMBLayout())
        InvalidArgument("%ls %ls operation can only operate on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());
    if (!m_pMBLayout)
    {
        m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
        m_pMBLayout->SetUniqueAxisName(m_dynamicAxisName);
    }
    // we map scalars to scalars
    if (isFinalValidationPass && Input(0)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls %ls operation can only operate on scalar input.", NodeName().c_str(), OperationName().c_str());
    SetDims(TensorShape(1), true);
}

template class WhereNode<float>;
template class WhereNode<double>;

// -----------------------------------------------------------------------
// PackedIndexNode(targetObject, indexSequence) -- map sequence
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void PackedIndexNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    let& sourceMBLayout = InputRef(SOURCEDATA).GetMBLayout(); // only used for index conversion
    let& indexMBLayout  = InputRef(INDEXDATA).GetMBLayout();
    let&  index  = InputRef(INDEXDATA).Value(); // per-seq index values that are to be mapped
    auto& result =                     Value(); // packed index values as mapped to sourceData's layout
    // loop over sourceSequences
    // Input matrix contains time indices for each sequence that refer to frames inside that sequence.
    // We replace every per-sequence index by the resolved column index w.r.t. the same MBLayout.
    let& sourceSequences = sourceMBLayout->GetAllSequences();
    for (size_t i = 0; i < sourceSequences.size(); i++)
    {
        let& sourceSeq = sourceSequences[i];
        if (sourceSeq.seqId == GAP_SEQUENCE_ID)
            continue;
        let& indexSeq = indexMBLayout->FindMatchingSequence(sourceSequences, i); // find corresponding entry in indexMBLayout
        for (size_t tIndex = 0; tIndex < indexSeq.GetNumTimeSteps(); tIndex++)   // map all index values in index sequence
        {
            let jIndex  = indexMBLayout->GetColumnIndex(indexSeq, tIndex);    // map time index to actual location in the matrix storage object
            let tSource = (size_t)index(0, jIndex);                           // the new time location (relative to source sequence)
            let jSource = sourceMBLayout->GetColumnIndex(sourceSeq, tSource); // map new time index as well. This performs a range check.
            result(0, jIndex) = (ElemType)jSource;
        }
    }
    // Note: maybe this is no longer needed, now that we do the same inside UpdateFunctionValueSize() for all nodes.
    result.CollapseDataLocation(); // BUGBUG: Move back, since BOTH state is broken at present.
}

template <class ElemType>
/*virtual*/ void PackedIndexNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/) /*override*/
{
    // we cannot backprop through a condition
    // Can we?
    return;
}

template <class ElemType>
/*virtual*/ void PackedIndexNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);

    // inherit both MBLayout and sample dimension (scalar) from indexData
    // Because we map (per-seq) index sequence to (packed) index sequence. Target is only for index calculation.
    m_pMBLayout = Input(INDEXDATA)->GetMBLayout();
    if (isFinalValidationPass && (!Input(INDEXDATA)->HasMBLayout() || !Input(SOURCEDATA)->HasMBLayout()))
        LogicError("%ls %ls operation requires both inputs to be minibatch data (must have MBLayouts).", NodeName().c_str(), OperationName().c_str());

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls %ls operation requires the second argument (indexData) to be a scalar sequence.", NodeName().c_str(), OperationName().c_str());

    SetDims(Input(INDEXDATA)->GetSampleLayout(), HasMBLayout());
}

template class PackedIndexNode<float>;
template class PackedIndexNode<double>;

// -----------------------------------------------------------------------
// GatherPackedNode(packedIndex, sourceData) -- gather operation
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    InputRef(INDEXDATA).MaskMissingValueColumnsTo(FrameRange(InputRef(INDEXDATA).GetMBLayout()), -1); // indicates an invalid column to Gather/Scatter
    let&  index  = InputRef(INDEXDATA) .Value(); // column indices to copy from
    let&  source = InputRef(SOURCEDATA).Value(); // source data to copy

#ifdef _MSC_VER
    auto& outputValuePtrRef = ValuePtrRef();
#else
    auto& outputValuePtrRef = this->template ValuePtrRef();
#endif
    if ((source.GetMatrixType() == SPARSE) && (outputValuePtrRef->GetMatrixType() != SPARSE))
        outputValuePtrRef = std::make_shared<Matrix<ElemType>>(outputValuePtrRef->GetNumRows(),
                                                               outputValuePtrRef->GetNumCols(),
                                                               outputValuePtrRef->GetPreferredDeviceId(),
                                                               source.GetMatrixType(),
                                                               source.GetFormat());

    auto& output = Value(); // output goes here
    output.DoGatherColumnsOf(/*beta=*/0, index, source, /*alpha=*/1);
}

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::BackpropToNonLooping(size_t inputIndex) /*override*/
{
    if (inputIndex == SOURCEDATA)
    {
        let&  index          = InputRef(INDEXDATA) .Value();    // column indices to copy from
        auto& sourceGradient = InputRef(SOURCEDATA).Gradient(); // source to propagate the gradient intpu
        auto& outputGradient =                      Gradient(); // output gradient to propagate
        sourceGradient.DoScatterColumnsOf(/*beta=*/1, index, outputGradient, /*alpha=*/1);
    }
}

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);

    // inherit MBLayout from indexData
    m_pMBLayout = Input(INDEXDATA)->GetMBLayout();
    if (isFinalValidationPass && (!Input(INDEXDATA)->HasMBLayout()))
        LogicError("%ls requires first argument (index data) to have a time dimension.", NodeDescription().c_str());

    bool sourceHasTimeDimension = Input(SOURCEDATA)->HasMBLayout();

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls requires the first argument (index data) to be a scalar time sequence.", NodeDescription().c_str());

    // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
    if (sourceHasTimeDimension)
        SetDims(Input(SOURCEDATA)->GetSampleLayout(), HasMBLayout());
    else
    {
        SmallVector<size_t> layout = { 1 }; // Scalar
        if (Input(SOURCEDATA)->GetSampleLayout().GetRank() > 1)
        {
            auto srcLayout = Input(SOURCEDATA)->GetSampleLayout().GetDims();
            layout.assign(srcLayout.begin(), srcLayout.end() - 1);
        }
        SetDims(TensorShape(layout), HasMBLayout());
    }
}

template class GatherPackedNode<float>;
template class GatherPackedNode<double>;

// -----------------------------------------------------------------------
// ScatterPackedNode(layoutData, packedIndex, sourceData) -- scatter operation
// -----------------------------------------------------------------------

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::ForwardPropNonLooping() /*override*/
{
    if (*InputRef(INDEXDATA).GetMBLayout() != *InputRef(SOURCEDATA).GetMBLayout())
        InvalidArgument("%ls %ls operation requires the minibatch layout of index and source data to be the same.", NodeName().c_str(), OperationName().c_str());
    InputRef(INDEXDATA).MaskMissingValueColumnsTo(FrameRange(InputRef(INDEXDATA).GetMBLayout()), -1); // indicates an invalid column to Gather/Scatter
    let&  index  = InputRef(INDEXDATA) .Value(); // column indices to copy from
    let&  source = InputRef(SOURCEDATA).Value(); // source data to copy

#ifdef _MSC_VER
    auto& outputValuePtrRef = ValuePtrRef();
#else
    auto& outputValuePtrRef = this->template ValuePtrRef();
#endif
    if ((source.GetMatrixType() == SPARSE) && (outputValuePtrRef->GetMatrixType() != SPARSE))
        outputValuePtrRef = std::make_shared<Matrix<ElemType>>(outputValuePtrRef->GetNumRows(),
                                                               outputValuePtrRef->GetNumCols(),
                                                               outputValuePtrRef->GetPreferredDeviceId(),
                                                               source.GetMatrixType(),
                                                               source.GetFormat());

    auto& output =                      Value(); // output goes here
    output.DoScatterColumnsOf(/*beta=*/0, index, source, /*alpha=*/1);
}

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::BackpropToNonLooping(size_t inputIndex) /*override*/
{
    if (inputIndex == SOURCEDATA)
    {
        let&  index          = InputRef(INDEXDATA).Value();     // column indices to copy from
        auto& sourceGradient = Input(SOURCEDATA)->Gradient(); // source to propagate the gradient input
        auto& outputGradient =                      Gradient(); // output gradient to propagate
        sourceGradient.DoGatherColumnsOf(/*beta=*/1, index, outputGradient, /*alpha=*/1);
    }
}

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    ComputationNodeBase::Validate(isFinalValidationPass);

    // inherit MBLayout from layoutData (that's the only thing we use it for)
    m_pMBLayout = Input(LAYOUTDATA)->GetMBLayout();
    if (isFinalValidationPass && (!Input(LAYOUTDATA)->HasMBLayout() || !Input(INDEXDATA)->HasMBLayout() || !Input(SOURCEDATA)->HasMBLayout()))
        LogicError("%ls %ls operation requires all inputs to be minibatch data (must have MBLayouts).", NodeName().c_str(), OperationName().c_str());

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls %ls operation requires the second argument (indexData) to be a scalar sequence.", NodeName().c_str(), OperationName().c_str());

    // TODO: We also know that indexData and sourceData must have the same MBLayout. But that is checked at runtime.

    // inherit tensor dimension from sourceData
    SetDims(Input(SOURCEDATA)->GetSampleLayout(), HasMBLayout());
}

template class ScatterPackedNode<float>;
template class ScatterPackedNode<double>;

// -----------------------------------------------------------------------
// CropNode -- crop operation, crops first input according to shape of second
//             input at offsets which are directly given or automatically calculated.
// -----------------------------------------------------------------------

template <class ElemType>
CropNode<ElemType>::CropNode(DEVICEID_TYPE deviceId, const wstring& name)
    : Base(deviceId, name), m_xOffset(numeric_limits<double>::max()), m_yOffset(numeric_limits<double>::max())
{
}

template <class ElemType>
CropNode<ElemType>::CropNode(size_t offsetX, size_t offsetY, DEVICEID_TYPE deviceId, const wstring& name)
    : CropNode(deviceId, name)
{
    m_xOffset = (double)(offsetX);
    m_yOffset = (double)(offsetY);
}

template <class ElemType>
CropNode<ElemType>::CropNode(const ScriptableObjects::IConfigRecordPtr configp)
    : CropNode(configp->Get(L"deviceId"), L"<placeholder>")
{
    // We may have 2 or 4 node inputs, check that and attach them.
    const auto inputs = GetInputsFromConfig(configp);
    if (inputs.size() != 2 && inputs.size() != 4)
        LogicError("Crop node must have 2 or 4 node inputs.");

    AttachInputs(inputs);

    // Here we have 3 possibilities:
    // 1. 2 input nodes -> auto crop calculation without equivalence nodes
    // 2. 2 input nodes + 2 parameters -> manual crop with given offsets
    // 3. 4 inputs -> auto crop calculation with equivalence nodes

    if (inputs.size() == 2)
    {
        // We have 2 input nodes related to cropping (no equivalence node inputs given). Check if we have offsets
        // directly given.
        if (configp->Exists(L"yOffset") && configp->Exists(L"xOffset"))
        {
            // We have manual crop with given offsets (option 2. above). Save given offsets.
            m_xOffset = configp->Get(L"xOffset");
            m_yOffset = configp->Get(L"yOffset");
        }
        // else: Offsets not given (option 1. above), we have automatic crop calculation without equivalence nodes.
    }
    // else: We have 4 node inputs (option 3. above), we have automatic crop calculation with equivalence nodes.
}

template <class ElemType>
void CropNode<ElemType>::Validate(bool isFinalValidationPass)
{
    Base::Validate(isFinalValidationPass);
    InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

    // Here we need to determine output dimensions which are same as dimensions of second input.
    TensorShape inputShape0 = Input(0)->GetSampleLayout();
    TensorShape inputShape1 = Input(1)->GetSampleLayout();

    SmallVector<size_t> inDims = inputShape0.GetDims();
    SmallVector<size_t> inDimsCropped = inputShape1.GetDims();

    // We assume we have at least two dimensions (first two are to be cropped).
    if (inDims.size() < 2 || inDimsCropped.size() < 2)
        RuntimeError("Crop input samples must have at least two dimensions.");

    // Output dimensions are equal to input dimensions with first two axis copied from cropped dimensions.
    SmallVector<size_t> outDims = inDims;
    outDims[0] = inDimsCropped[0];
    outDims[1] = inDimsCropped[1];

    // Set output dimensions.
    SetDims(TensorShape(outDims), HasMBLayout());

    if (isFinalValidationPass)
    {
        // In final validation pass we compute crop offsets if needed.
        ComputeCropOffsets();

        // Cropped input must be large enough to allow cropping at given offset.
        if (inDims[0] < outDims[0] + m_xOffset)
            RuntimeError("Input is small to be cropped along x dimension in crop node.");

        if (inDims[1] < outDims[1] + m_yOffset)
            RuntimeError("Input is small to be cropped along y dimension in crop node.");
    }
}

template <class ElemType>
void CropNode<ElemType>::ForwardProp(const FrameRange& /*fr*/)
{
    // Our offsets must be initialized here.
    if (m_xOffset == numeric_limits<double>::max() || m_yOffset == numeric_limits<double>::max())
        LogicError("Crop offsets not initialized in ForwardProp.");

    // Retrieve input and output views for the values. Input and output views are tensor views
    // that define parts of first input and output that we operate on (we copy input from input view
    // to output).
    CroppedIOViews ioViews = CreateIOViews(&ComputationNode<ElemType>::ValuePtr);

    // Copy values from cropped input to output.
    ioViews.outputView.AssignCopyOf(ioViews.inputViewCropped);
}

template <class ElemType>
void CropNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& /*fr*/)
{
    // We propagate gradients just to the cropped input.
    if (inputIndex == 0)
    {
        // Reset input gradients to ensure that non-cropped parts do not affect backprop.
        Input(0)->Gradient().SetValue(0);

        // Retrieve input and output views for the gradients. Input and output views are tensor views
        // that define parts of first input and output that we operate on (we copy gradients from output view
        // to input view).
        CroppedIOViews ioViews = CreateIOViews(&ComputationNode<ElemType>::GradientPtr);

        // Copy gradients from output to cropped input.
        ioViews.inputViewCropped.AddCopyOf(ioViews.outputView);
    }
}

template <class ElemType>
void CropNode<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);

    fstream << m_xOffset;
    fstream << m_yOffset;
}

template <class ElemType>
void CropNode<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);

    fstream >> m_xOffset;
    fstream >> m_yOffset;
}

template <class ElemType>
void CropNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const wstring& newName, const CopyNodeFlags flags) const
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<CropNode<ElemType>>(nodeP);
        node->m_xOffset = m_xOffset;
        node->m_yOffset = m_yOffset;
    }
}

template <class ElemType>
typename CropNode<ElemType>::CroppedIOViews CropNode<ElemType>::CreateIOViews(MatrixGetter matrixGetter)
{
    // Get the shapes of the inputs.
    TensorShape inputShape0 = Input(0)->GetTensorShape(Input(0)->GetSampleLayout().GetRank());
    TensorShape inputShape1 = Input(1)->GetTensorShape(Input(1)->GetSampleLayout().GetRank());

    // Calculate cropped shape of the input.
    TensorShape inputShapeCropped = inputShape0;
    inputShapeCropped.NarrowTo(0, (size_t)(m_xOffset), (size_t)(m_xOffset) + inputShape1.GetDim(0));
    inputShapeCropped.NarrowTo(1, (size_t)(m_yOffset), (size_t)(m_yOffset) + inputShape1.GetDim(1));

    // Get output shape.
    TensorShape outputShape = GetTensorShape(GetSampleLayout().GetRank());
    // Cropped input and output dimensions must be same.
    if (inputShapeCropped.GetDims() != outputShape.GetDims())
        LogicError("Cropped input and output must have same rank.");

    // Create proper views using calculated shapes.
    return CroppedIOViews(this, matrixGetter, inputShapeCropped, outputShape);
}

// ComputeCropOffsets computes offsets to be used for cropping if manual offsets are absent. The offsets are computed
// by traversing the network graph and finding common ancestor of crop node inputs. Once ancestor is found affine transform
// is computed along the paths from first and second input to common ancestor. Complete transform from one input to other it
// finally calculated composing these two transforms. Translate components of final transform define crop offsets.
template <class ElemType>
void CropNode<ElemType>::ComputeCropOffsets()
{
    // Helper method for traversing the tree and calculating node transforms.
    // For currNode, calculates coordinate maps of its inputs based on known coordinate maps of its outputs.
    // nodeToTransformMap contains coordinate maps for all nodes traversed so far, and is updated by this function.
    // Traversal stack contains all nodes traversed so far. Inputs of currNode are pushed to traversal stack so that their
    // inputs can be processed later on.
    auto ProcessInputs = [](ComputationNodeBase* currNode, stack<ComputationNodeBase*>& traversalStack, unordered_map<ComputationNodeBase*, SpaceTransform>& nodeToTransformMap)
    {
        if (!currNode->Is<TransformerNode>())
            RuntimeError("Node does not support affine transform for cropping.");

        auto transformerNode = currNode->As<TransformerNode>();
        // Go over the nodes inputs.
        for (size_t i = 0; i < currNode->GetNumInputs(); i++)
        {
            // Check if input-output transform is supported on the node.
            if (transformerNode->SupportsTransformOnInput(i))
            {
                // Transform is supported, take the input.
                ComputationNodeBase* currInput = currNode->GetInputs()[i].get();
                // Take node transform from input to output.
                const SpaceTransform& nodeTransform = transformerNode->GetTransformForInput(i);
                // Calculate composite transform from node input to crop node.
                SpaceTransform nodeToCropTransform = nodeToTransformMap.find(currNode)->second.Compose(nodeTransform);

                // Check if we already visited this input node.
                auto it = nodeToTransformMap.find(currInput);
                if (it == nodeToTransformMap.end())
                {
                    // We have not visited this node before. Add it to the transform map and to traversal stack to continue
                    // traversing its children.
                    nodeToTransformMap.insert(make_pair(currInput, nodeToCropTransform));
                    traversalStack.push(currInput);
                }
                else
                {
                    // We have been here before, check that transforms along two different paths are same.
                    if (it->second != nodeToCropTransform)
                    {
                        // Different transforms along two different paths, should never happen.
                        RuntimeError("Different transforms along different paths in Crop node.");
                    }
                }
            }
        }
    };

    if (m_xOffset != numeric_limits<double>::max() && m_yOffset != numeric_limits<double>::max())
    {
        // Offsets are already available, skip compute.
        return;
    }

    // Used to keep nodes while traversing the network graph.
    stack<ComputationNodeBase*> traversalStack;
    // Maps node to transform between its output and crop node.
    unordered_map<ComputationNodeBase*, SpaceTransform> nodeToCropInput0TransformMap;
    unordered_map<ComputationNodeBase*, SpaceTransform> nodeToCropInput1TransformMap;
    // Take equivalence nodes if provided.
    ComputationNodeBase* equivalenceNode1 = nullptr;
    ComputationNodeBase* equivalenceNode2 = nullptr;
    if (GetInputs().size() == 4)
    {
        equivalenceNode1 = GetInputs()[2].get();
        equivalenceNode2 = GetInputs()[3].get();
    }

    // Push first input to traversal stack to start exploring paths starting from there.
    traversalStack.push(GetInputs()[0].get());
    // Push first input transform as identity to enable composing transforms.
    nodeToCropInput0TransformMap.insert(make_pair(GetInputs()[0].get(), SpaceTransform::Identity(2)));
    // Start traversing graph starting from the first input.
    while (!traversalStack.empty())
    {
        ComputationNodeBase* currNode = traversalStack.top();
        traversalStack.pop();
        ProcessInputs(currNode, traversalStack, nodeToCropInput0TransformMap);
    }

    // Now traverse from second input.
    traversalStack.push(GetInputs()[1].get());
    // Push second input transform as identity to enable composing transforms.
    nodeToCropInput1TransformMap.insert(make_pair(GetInputs()[1].get(), SpaceTransform::Identity(2)));
    // Once we meet node that is in nodeToCropInput0TransformMap or equivalence node we will compute offsets.
    double xOffset = numeric_limits<double>::max();
    double yOffset = numeric_limits<double>::max();
    while (!traversalStack.empty())
    {
        ComputationNodeBase* currNode = traversalStack.top();
        traversalStack.pop();
        // Check if node is in the map corresponding to the first input (path connected over common ancestor).
        auto it = nodeToCropInput0TransformMap.find(currNode);
        const SpaceTransform* firstInputTransform = nullptr;
        if (it != nodeToCropInput0TransformMap.end())
        {
            // We have closed the path between nodes, save the first input transform.
            firstInputTransform = &it->second;
        }
        // Check if node is equivalent to one from the first subtree (path connected over equivalence nodes).
        else if (currNode == equivalenceNode2)
        {
            // We have closed the path between nodes using equivalence nodes, save the first equivalence node transform.
            firstInputTransform = &nodeToCropInput0TransformMap.find(equivalenceNode1)->second;
        }

        if (firstInputTransform)
        {
            // Calculate final transform.
            SpaceTransform finalTransform = nodeToCropInput1TransformMap.find(currNode)->second.Compose(firstInputTransform->Inverse());
            for (size_t ia = 0; ia < finalTransform.m_axisTransforms.size(); ia++)
            {
                // In crop node we expect no scaling.
                if (finalTransform.m_axisTransforms[ia].scale != 1.0f)
                    RuntimeError("Composite transform has non 1 scale in crop node.");
                if (finalTransform.m_axisTransforms[ia].translate > 0)
                    RuntimeError("Composite transform has positive translate (negative offset) in crop node.");
            }
            // Crop offsets are defined with transform translations.
            xOffset = -finalTransform.m_axisTransforms[0].translate;
            yOffset = -finalTransform.m_axisTransforms[1].translate;
            // Finished.
            break;
        }
        // No connected path, keep searching.
        ProcessInputs(currNode, traversalStack, nodeToCropInput1TransformMap);
    }
    if (xOffset == numeric_limits<double>::max() || yOffset == numeric_limits<double>::max())
        LogicError("Connected path between crop inputs not found. Unable to compute crop offsets.");

    // Save computed offsets.
    m_xOffset = xOffset;
    m_yOffset = yOffset;
}

template <class ElemType>
void CropNode<ElemType>::ComputeTransforms()
{
    if (m_transforms[0].m_axisTransforms.empty())
    {
        m_transforms[0].m_axisTransforms.resize(2);
        m_transforms[0].m_axisTransforms[0].scale = 1;
        m_transforms[0].m_axisTransforms[0].translate = -m_xOffset;
        m_transforms[0].m_axisTransforms[1].scale = 1;
        m_transforms[0].m_axisTransforms[1].translate = -m_yOffset;
    }
    // else: already computed.
}

template <class ElemType>
bool CropNode<ElemType>::SupportsTransformOnInput(size_t inputIndex)
{
    // We support transform on cropped input.
    return (inputIndex == 0);
}

template class CropNode<float>;
template class CropNode<double>;

}}}
