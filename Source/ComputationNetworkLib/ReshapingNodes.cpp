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

// TODO: move to MBLayout as a static method
// packing algorithm
//  - width: maximum width of structure; set to maximum over sequence lengths
//  - inputSequences: vector of input SequenceInfo records (only seqId and GetNumTimeSteps() are used)
//  - [out] *pMBLayout: MBLayout that describes the created packed sequence set
//  - placement, rowAllocations: temp buffers (passed in to be able to optimize memory allocations)
template<typename SequenceInfoVector>
static void PackSequences(const SequenceInfoVector& inputSequences,
    /*ref->out*/MBLayoutPtr pMBLayout,
    /*temp buffer*/std::vector<std::pair<size_t, size_t>>& placement,
    /*temp buffer*/std::vector<size_t> rowAllocations)
{
    placement.resize(inputSequences.size()); // [sequence index] result goes here (entries are invalid for gaps)
    // determine width of MBLayout
    size_t width = 0;
    for (size_t i = 0; i < inputSequences.size(); i++)
        if (inputSequences[i].seqId == GAP_SEQUENCE_ID)
            continue;
        else if (width < inputSequences[i].GetNumTimeSteps())
            width = inputSequences[i].GetNumTimeSteps();
    // allocate
    rowAllocations.clear();             // [row] we build rows one by one
    for (size_t i = 0; i < inputSequences.size(); i++)
    {
        if (inputSequences[i].seqId == GAP_SEQUENCE_ID)
            continue;
        let len = inputSequences[i].GetNumTimeSteps();
        // first see if we find a row that has enough space
        size_t s;
        for (s = 0; s < rowAllocations.size(); s++)
            if (rowAllocations[s] + len <= width)
                break; // yep, it fits
        // we did not find a s that fit then create a new one
        if (s == rowAllocations.size())
            rowAllocations.push_back(0);
        // sequence goes to (s, rowAllocations[s])
        placement[i] = make_pair(s, rowAllocations[s]);
        // and allocate it
        rowAllocations[s] += len;
    }
    // create MBLayout
    pMBLayout->Init(rowAllocations.size(), width);
    for (size_t i = 0; i < inputSequences.size(); i++)
    {
        if (inputSequences[i].seqId == GAP_SEQUENCE_ID)
            continue;
        size_t s, tBegin; tie
        (s, tBegin) = placement[i];
        pMBLayout->AddSequence(inputSequences[i].seqId, s, (ptrdiff_t)tBegin, tBegin + inputSequences[i].GetNumTimeSteps());
    }
    // need to fill the gaps as well
    for (size_t s = 0; s < rowAllocations.size(); s++)
        pMBLayout->AddGap(s, (size_t)rowAllocations[s], width);
}

// wrapper class to pass MBLayout sequence vector to PackSequences()
struct SequenceLengthVector
{
    typedef vector<vector<size_t>> SequenceVector;
    typedef MBLayout::SequenceInfo SequenceInfo;
    const SequenceVector& sequenceVector;       // 
    const vector<SequenceInfo>& sequenceInfo;    // original sequence info (for seqId)
    SequenceLengthVector(const vector<SequenceInfo>& sequenceInfo, const SequenceVector& sequenceVector) : sequenceInfo(sequenceInfo), sequenceVector(sequenceVector) { }
    size_t size() const { return sequenceInfo.size(); }
    MBLayout::SequenceInfo operator[](size_t i) const // return a descriptor of the new sequence
    {
        SequenceInfo seq;
        seq.seqId = sequenceInfo[i].seqId;
        seq.s = i;
        seq.tBegin = 0;
        seq.tEnd = sequenceVector[i].size();
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
    PackSequences(SequenceLengthVector(sequences, indexSequences), outMBLayout, /*temp*/m_placementBuffer, /*temp*/m_rowAllocationsBuffer);
    // copy to output
    vector<ElemType> buf(outMBLayout->GetNumCols(), numeric_limits<ElemType>::quiet_NaN()); // STL cannot easily avoid initializing, so we might as well init with NaN for gaps
    for (size_t i = 0; i < sequences.size(); i++)
    {
        let& seq = outMBLayout->GetAllSequences()[i];
        if (seq.seqId == GAP_SEQUENCE_ID) // gaps will keep the NaN
            continue;
        let& indexSequence = indexSequences[i];
        for (size_t t = 0; t < seq.GetNumTimeSteps(); t++)
            buf[outMBLayout->GetColumnIndex(seq, t)] = (ElemType)indexSequence[t];
    }
    Value().SetValue(outMBLayout->GetNumParallelSequences(), outMBLayout->GetNumTimeSteps(), Input(0)->Value().GetDeviceId(), buf.data(), MatrixFormat::matrixFormatColMajor);
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

}}}
