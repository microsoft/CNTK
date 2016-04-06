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

namespace Microsoft { namespace MSR { namespace CNTK {

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
    let& inMBLayout = Input(0)->GetMBLayout();
    let& input = Input(0)->Value();
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
        indexSequence.clear();
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
            if (input(0, inMBLayout->GetColumnIndex(seq, t))) // this is the condition check that this node performs; the meat
                indexSequence.push_back(t);
        // Note: The above accesses m_value directly on the CPU, putting it into BOTH state, possibly for other consumers as well.
    }
    // create a new MBLayout
    let& outMBLayout = GetMBLayout();
    outMBLayout->InitAsPackedSequences(SequenceLengthVector(sequences, indexSequences), /*temp*/m_placementBuffer, /*temp*/m_rowAllocationsBuffer);
    // copy to output
    vector<ElemType> buf(outMBLayout->GetNumCols(), numeric_limits<ElemType>::quiet_NaN()); // STL cannot easily avoid initializing, so we might as well init with NaN for gaps
    for (size_t i = 0, j = 0; i < sequences.size();)
    {
        if (sequences[i].seqId == GAP_SEQUENCE_ID) // gaps will keep the NaN
        {
            ++i;
            continue;
        }
        let& seq = outMBLayout->GetAllSequences()[j];
        if (seq.seqId == GAP_SEQUENCE_ID) // When would we see this?
        {
            ++j;
            continue;
        }
        let& indexSequence = indexSequences[i];
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
            buf[outMBLayout->GetColumnIndex(seq, t)] = (ElemType)indexSequence[t];
        ++i;
        ++j;
    }
    // the result will be kept in CPUDEVICE, since most likely we will access it again in PackedIndexNode
    Value().TransferToDeviceIfNotThere(CPUDEVICE, /*isBeingMoved=*/ true, /*emptyTransfer=*/ true, /*updatePreferredDevice=*/ true);
    Value().SetValue(1, outMBLayout->GetNumCols(), CPUDEVICE, buf.data(), MatrixFormat::matrixFormatColMajor);
}

template <class ElemType>
/*virtual*/ void WhereNode<ElemType>::BackpropToNonLooping(size_t /*inputIndex*/) /*override*/
{
    // we cannot backprop through a condition
    // Can we?
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
        m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
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
    let& sourceMBLayout = Input(SOURCEDATA)->GetMBLayout(); // only used for index conversion
    let& indexMBLayout  = Input(INDEXDATA)->GetMBLayout();
    let&  index  = Input(INDEXDATA)->Value(); // per-seq index values that are to be mapped
    auto& result =                   Value(); // packed index values as mapped to sourceData's layout
    // loop over sourceSequences
    // Input matrix contains time indices for each sequence that refer to frames inside that sequence.
    // We replace every per-sequence index by the resolved column index w.r.t. the same MBLayout.
    let& sourceSequences = sourceMBLayout->GetAllSequences();
    for (size_t i = 0; i < sourceSequences.size(); i++)
    {
        let& sourceSeq = sourceSequences[i];
        if (sourceSeq.seqId == GAP_SEQUENCE_ID)
            continue;
        let& indexSeq = indexMBLayout->FindSequence(sourceSeq.seqId);          // find corresponding entry in indexMBLayout
        for (size_t tIndex = 0; tIndex < indexSeq.GetNumTimeSteps(); tIndex++) // map all index values in index sequence
        {
            let jIndex  = indexMBLayout->GetColumnIndex(indexSeq, tIndex);    // map time index to actual location in the matrix storage object
            let tSource = (size_t)index(0, jIndex);                           // the new time location (relative to source sequence)
            let jSource = sourceMBLayout->GetColumnIndex(sourceSeq, tSource); // map new time index as well. This performs a range check.
            result(0, jIndex) = (ElemType)jSource;
        }
    }
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
    Input(INDEXDATA)->MaskMissingValueColumnsTo(FrameRange(Input(INDEXDATA)->GetMBLayout()), -1); // indicates an invalid column to Gather/Scatter
    let&  index  = Input(INDEXDATA)->Value();  // column indices to copy from
    let&  source = Input(SOURCEDATA)->Value(); // source data to copy
    auto& output =                    Value(); // output goes here
    output.DoGatherColumnsOf(/*beta=*/0, index, source, /*alpha=*/1);
}

template <class ElemType>
/*virtual*/ void GatherPackedNode<ElemType>::BackpropToNonLooping(size_t inputIndex) /*override*/
{
    if (inputIndex == SOURCEDATA)
    {
        let&  index          = Input(INDEXDATA)->Value();     // column indices to copy from
        auto& sourceGradient = Input(SOURCEDATA)->Gradient(); // source to propagate the gradient intpu
        auto& outputGradient =                    Gradient(); // output gradient to propagate
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
        LogicError("%ls requires first argument (index data) to have a time dimension.", this->NodeDescription().c_str());

    bool sourceHasTimeDimension = Input(SOURCEDATA)->HasMBLayout();

    if (isFinalValidationPass && Input(INDEXDATA)->GetSampleLayout().GetNumElements() != 1)
        InvalidArgument("%ls requires the first argument (index data) to be a scalar time sequence.", this->NodeDescription().c_str());

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
    if (*Input(INDEXDATA)->GetMBLayout() != *Input(SOURCEDATA)->GetMBLayout())
        InvalidArgument("%ls %ls operation requires the minibatch layout of index and source data to be the same.", NodeName().c_str(), OperationName().c_str());
    Input(INDEXDATA)->MaskMissingValueColumnsTo(FrameRange(Input(INDEXDATA)->GetMBLayout()), -1); // indicates an invalid column to Gather/Scatter
    let&  index = Input(INDEXDATA)->Value();  // column indices to copy from
    let&  source = Input(SOURCEDATA)->Value(); // source data to copy
    auto& output =                    Value(); // output goes here
    output.DoScatterColumnsOf(/*beta=*/0, index, source, /*alpha=*/1);
}

template <class ElemType>
/*virtual*/ void ScatterPackedNode<ElemType>::BackpropToNonLooping(size_t inputIndex) /*override*/
{
    if (inputIndex == SOURCEDATA)
    {
        let&  index          = Input(INDEXDATA)->Value();     // column indices to copy from
        auto& sourceGradient = Input(SOURCEDATA)->Gradient(); // source to propagate the gradient input
        auto& outputGradient =                    Gradient(); // output gradient to propagate
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

}}}
