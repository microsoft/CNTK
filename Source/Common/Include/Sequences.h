//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Sequences.h -- all about iterating over sequences, that is, MBLayout, FrameRange, and iterators
//

#pragma once

#include <vector>
#include <memory> // for shared_ptr
#include <mutex>
#include "Basics.h"
#include "Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Forward declarations
class FrameRange;

typedef size_t UniqueSequenceId;
#define GAP_SEQUENCE_ID SIZE_MAX       // indicates no data
#define NEW_SEQUENCE_ID (SIZE_MAX - 1) // let SetSequence() assign a unique id; for old readers. Don't mix with actual reader-assigned ids.

// -----------------------------------------------------------------------
// MBLayout -- layout information of minibatch
//
// Minibatches are collections of one or more sequences, laid out in a way to
// allow to process one time step for multiple sequences in parallel in shared CUDA launches.
//
// This is achieved by interleaving storage. If f(s,t) denotes a frame of sequence s at time t,
// the minibatch matrix would contain this:
//   f(0,0) f(1,0) ... f(0,1) f(1,1) ...
// Much of CNTK's efficiency comes from this.
// (Note that some communities, such as language processing, often call sets f(0..s-1,t) a
// "minibatch," where in our definition, a minibatch consists of multiple entire sequences.)
//
// In the special case of frame randomization, every frame is stored as a single-frame sequence.
//
// If we describe this in terms of tensors, a data matrix with sample layout (I,J,K) and
// MBLayout (S,T) can be interpreted as TensorShape(I,J,K,S,T).
//
// Sequences can also be concatenated to fill the space better. For this case,
// this object stores about every frame whether it is at the start or end of a sequence.
// Hence, we distinguish between "sequences" (logical units) and "parallel sequences"
// (where one "parallel sequence" may consist of multiple concatenated "sequences").
//
// When not all sequences have the same length, some parallel sequences have invalid frames (gaps).
// Gaps are identified by the MBLayouyt as well. Currently, these gaps only occur at the end.
//
// Gaps may also arise due to invalid input data (e.g. a speech utterance for which no alignment could be generated).
//
// An MBLayout provides the following functions:
//  - (building:) add a new sequence (or gap range) to the MBLayout
//  - inquire the set of sequences (sequence ids) that intersect with this minibatch
//  - inquire whether any time step t has a gap or boundary across all sequences
//  - inquire for gap or boundary at (s,t)
//
// Truncated BPTT support (partial sequences):
//  - in truncated BPTT, minibatches only contain partial sequences, e.g. a range of 20 time steps
//  - boundary information is stored for every sequence that intersects with this minibatch,
//    including boundaries that fall outside of the time range of the minibatch
//
// An MBLayout object stores:
//  - for every sequence in the minibatch the n-tuple (global sequence id, s, first t, last t)
//    (where first and last t may sometimes lie outside of the minibatch, e.g. in case of truncated BPTT)
//  - number of time steps and parallel sequences (their product is equal to the #columns in the minibatch matrix)
//  - lookup tables for looking up gap and boundary information
// -----------------------------------------------------------------------

// Contract between ComputationNode, ComputationNetwork, and MBLayout:
//  - if a node has no MBLayout, m_{value,gradient} are not samples (they are not activations or input data), but e.g. model parameters
//  - ComputationNode::GetNumCols() == MBLayout::GetNumTimeSteps() * MBLayout::GetNumParallelSequences()
//  - ComputationNetwork ensures that m_{value,gradient} are allocated correctly before calling ForwardProp() on a node

// Relationship between MBLayout and FrameRange:
//  - an MBLayout represents a time axis
//     - a nullptr means absence of a time axis, e.g. a weight matrix
//     - two MBLayouts with identical content may be used together as if they describe the same time axis
//  - a FrameRange describes an iterator over a time axis
//     - the iterator can be either an index "all," implying a non-sequenced 'map' operation
//       This is true for both time and parallel-sequence dimension, although not all code supports the striding needed to select a sequence dimension.
//       The iterator can also represent a sub-range, e.g. to one packed sequence with a given start and end time.
//     - each FrameRange is bound to a MBLayout for that reason
// Towards nested loops:  --TODO: implement this
//  - an object with multiple time dimensions (such as state of an attention model) is described by a linked list of MBLayouts
//  - a nested iterator is described by a linked list of FrameRanges

struct MBLayout
{
    typedef std::shared_ptr<MBLayout> MBLayoutPtr;

    // information stored about sequences
    struct SequenceInfo
    {
        UniqueSequenceId seqId; // unique sequence id (or GAP_SEQUENCE_ID--TODO: don't include gaps here)
        size_t s;               // index of parallel sequence
        ptrdiff_t tBegin;       // first time index in this minibatch. Note that this may be negative of the sequence started before this MB.
        size_t tEnd;            // end = first frame index after final frame. May be beyond the minibatch if reql sequence is longer than the MB.
        bool operator==(const SequenceInfo &other) const
        {
            return seqId == other.seqId && s == other.s && tBegin == other.tBegin && tEnd == other.tEnd;
        }
        size_t GetNumTimeSteps() const { return (size_t)(tEnd - tBegin); }
    };

    // -------------------------------------------------------------------
    // construction
    // -------------------------------------------------------------------

    MBLayout(size_t numParallelSequences, size_t numTimeSteps, const std::wstring &name)
        : m_distanceToStart(CPUDEVICE), m_distanceToEnd(CPUDEVICE), m_columnsValidityMask(CPUDEVICE)
    {
        Init(numParallelSequences, numTimeSteps);
        SetUniqueAxisName(name != L"" ? name : L"DynamicAxis");
    }
    MBLayout()
        : MBLayout(1, 0, L"")
    {
    }

    // copy the content of another MBLayoutPtr over
    // Use this instead of actual assignment to make it super-obvious that this is not copying the pointer but actual content. The pointer is kept fixed.
    // Use "keepName" if the "identity" of the target is to be preserved, e.g. 
    // while copying from reader space to network space.
    void CopyFrom(const MBLayoutPtr& other, bool keepName=false)
    {
        m_numTimeSteps = other->m_numTimeSteps;
        m_numParallelSequences = other->m_numParallelSequences;
        m_sequences = other->m_sequences;
        m_numFramesDeclared = other->m_numFramesDeclared;
        m_numGapFrames = other->m_numGapFrames;

        m_distanceToStart.SetValue(other->m_distanceToStart);
        m_distanceToEnd.SetValue(other->m_distanceToEnd);

        m_distanceToNearestStart = other->m_distanceToNearestStart;
        m_distanceToNearestEnd = other->m_distanceToNearestEnd;

        m_timeStepHasGap = other->m_timeStepHasGap;

        m_columnsValidityMask.SetValue(other->m_columnsValidityMask);
        m_writable = other->m_writable;

        if (!keepName)
            m_axisName = other->m_axisName;
    }

    // Destructive copy that steals ownership if the content, like std::move()
    // Note: For some reason the VC++ compiler does not generate the 
    // move assignment and we have to do this ourselves
    void MoveFrom(MBLayoutPtr other)
    {
        m_numTimeSteps = other->m_numTimeSteps;
        m_numParallelSequences = other->m_numParallelSequences;
        m_sequences = std::move(other->m_sequences);
        m_numFramesDeclared = other->m_numFramesDeclared;
        m_numGapFrames = other->m_numGapFrames;

        m_distanceToStart = std::move(other->m_distanceToStart);
        m_distanceToEnd = std::move(other->m_distanceToEnd);

        m_distanceToNearestStart = std::move(other->m_distanceToNearestStart);
        m_distanceToNearestEnd = std::move(other->m_distanceToNearestEnd);

        m_timeStepHasGap = std::move(other->m_timeStepHasGap);

        m_columnsValidityMask = std::move(other->m_columnsValidityMask);
        m_writable = other->m_writable;

        m_axisName = std::move(other->m_axisName);
    }

    MBLayout(const MBLayout&) = delete;
    MBLayout& operator=(const MBLayout&) = delete;

public:
    // resize and reset all frames to None (note: this is an invalid state and must be fixed by caller afterwards)
    void Init(size_t numParallelSequences, size_t numTimeSteps)
    {
        // remember the dimensions
        m_numParallelSequences = numParallelSequences;
        m_numTimeSteps = numTimeSteps;
        m_distanceToStart.Resize(m_numParallelSequences, m_numTimeSteps);
        m_distanceToEnd.Resize(m_numParallelSequences, m_numTimeSteps);
        m_distanceToNearestStart.assign(m_numTimeSteps, PTRDIFF_MAX);
        m_distanceToNearestEnd.assign(m_numTimeSteps, PTRDIFF_MAX);
        m_timeStepHasGap.assign(m_numTimeSteps, false);
        m_columnsValidityMask.Resize(0, 0); // invalidate
        // reset state
        m_numFramesDeclared = 0;
        m_numGapFrames = 0;
        m_sequences.clear();
        m_writable = true;
    }

    // packing algorithm
    //  - width: maximum width of structure; set to maximum over sequence lengths
    //  - inputSequences: vector of input SequenceInfo records (only seqId and GetNumTimeSteps() are used)
    //  - placement, rowAllocations: temp buffers (passed in to be able to optimize memory allocations)
    template<typename SequenceInfoVector>
    void InitAsPackedSequences(const SequenceInfoVector& inputSequences,
        /*temp buffer*/std::vector<std::pair<size_t, size_t>>& placement,
        /*temp buffer*/std::vector<size_t> rowAllocations)
    {
        placement.resize(inputSequences.size()); // [sequence index] result goes here (entries are invalid for gaps)
        // determine width of MBLayout
        size_t width = 0;
        for (size_t i = 0; i < inputSequences.size(); i++)
        {
            if (inputSequences[i].seqId == GAP_SEQUENCE_ID)
                continue;
            else if (width < inputSequences[i].GetNumTimeSteps())
                width = inputSequences[i].GetNumTimeSteps();
        }
        // allocate
        rowAllocations.clear();             // [row] we build rows one by one
        for (size_t i = 0; i < inputSequences.size(); i++)
        {
            if (inputSequences[i].seqId == GAP_SEQUENCE_ID)
                continue;
            let len = inputSequences[i].GetNumTimeSteps();
            // first see if we find a row that has enough space
            // TODO: Should we use a proper priority_queue?
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
        Init(rowAllocations.size(), width);
        for (size_t i = 0; i < inputSequences.size(); i++)
        {
            if (inputSequences[i].seqId == GAP_SEQUENCE_ID)
                continue;
            size_t s, tBegin; tie
            (s, tBegin) = placement[i];
            AddSequence(inputSequences[i].seqId, s, (ptrdiff_t)tBegin, tBegin + inputSequences[i].GetNumTimeSteps());
        }
        // need to fill the gaps as well
        for (size_t s = 0; s < rowAllocations.size(); s++)
            AddGap(s, (size_t)rowAllocations[s], width);
    }

    // -------------------------------------------------------------------
    // accessors
    // -------------------------------------------------------------------

    size_t GetNumTimeSteps() const { return m_numTimeSteps; }
    size_t GetNumParallelSequences() const { return m_numParallelSequences; }
    size_t GetNumSequences() const
    {
        return std::count_if(m_sequences.begin(), m_sequences.end(), [](const SequenceInfo& sequence) {
            return sequence.seqId != GAP_SEQUENCE_ID;
        });
    }

    // axis names are for now only a debugging aid
    // In the future, there will be a mechanism to denote that axes are meant to be the same.
    const wchar_t* GetAxisName() const { return m_axisName.c_str(); }
    void SetAxisName(const std::wstring& name) { m_axisName = name; }
    void SetUniqueAxisName(std::wstring name) // helper for constructing
    {
        // Unfortunatelly, initialization of local static variables is not thread-safe in VS2013.
        // As workaround, it is moved to the struct level. 
        // Todo: when upgraded to VS2013, change back to use the local static mutex, and remove also Sequences.cpp.
        // The mutex is need to make access to nameIndices be thread-safe.
        // static std::mutex nameIndiciesMutex;
        // static std::map<std::wstring, size_t> nameIndices;

        size_t index;

        // Use the block to make sure that nameIndiciesMutex is unlocked as soon as possible.
        {
            std::unique_lock<std::mutex> lock(s_nameIndiciesMutex);
            index = s_nameIndices[name]++;
        }

        if (index > 0)
            name += msra::strfun::wstrprintf(L"%d", (int)index);
        SetAxisName(name);
    }

    // how many columns the underlying MB matrix has
    size_t GetNumCols() const
    {
        return GetNumTimeSteps() * GetNumParallelSequences();
    }

    // Get the number of frames of the input sequence that belong to the MB, i.e. disregarding sequence elements that are outside of the MB boundaries
    // Input sequence is expected to belong to this MBLayout
    size_t GetNumSequenceFramesInCurrentMB(const SequenceInfo& sequenceInfo) const
    {
        return min(sequenceInfo.tEnd, GetNumTimeSteps()) - max(sequenceInfo.tBegin, (ptrdiff_t)0);
    }

    // return all sequences stored in this minibatch
    const vector<SequenceInfo>& GetAllSequences() const
    {
        return m_sequences;
    }

    // compute the number of actual samples in this layout (not counting gaps)
    // This is used by MeanNode and InvStdDevNode, and by statistics reporting.
    size_t GetActualNumSamples() const;

    const Matrix<char>& GetColumnsValidityMask(DEVICEID_TYPE deviceId) const;

    // compare whether two layouts are the same
    bool operator==(const MBLayout& other) const
    {
        if (this == &other)
            return true;
        bool res =
            m_numTimeSteps == other.m_numTimeSteps &&
            m_numParallelSequences == other.m_numParallelSequences &&
            m_sequences == other.m_sequences;
        return res;
    }
    bool operator!=(const MBLayout &other) const
    {
        return !(*this == other);
    } // duh

    operator std::string() const
    {
        std::stringstream s;
        s << "{numTimeSteps:" << m_numTimeSteps << ", numParallelSequences:" << m_numParallelSequences << ", sequences:[";

        bool first = true;
        for (const auto &seq : m_sequences)
        {
            if (!first)
                s << ", ";
            s << "{seqId:" << seq.seqId << ", s:" << seq.s <<", begin:" << seq.tBegin << ", end:" << seq.tEnd << "}";
            first = false;
        }
        s << "]}";
        return s.str();
    }

    // -------------------------------------------------------------------
    // building (adding sequences or gaps)
    // -------------------------------------------------------------------

    // mark a range of frames in a parallel sequence as one sentence
    // Note that endTime is the last frame +1. Like begin/end as used in STL containers.
    void AddSequence(UniqueSequenceId seqId, size_t s, ptrdiff_t beginTime, size_t endTime)
    {
        // old readers can just pass this to get an auto-assigned id (which is fine as long as we only have one MBLayout per minibatch)
        if (seqId == NEW_SEQUENCE_ID)
        {
            static UniqueSequenceId makeSeqIdCounter = 0;
            seqId = makeSeqIdCounter++;
            if (seqId == GAP_SEQUENCE_ID)
                LogicError("AddSequence: ran out of bits..."); // (will never happen anyway)
        }

        AddSequence(SequenceInfo{seqId, s, beginTime, endTime});
    }

    // version that passes a SequenceInfo record directly
    void AddSequence(const SequenceInfo &seqDesc)
    {
        const auto beginTime = seqDesc.tBegin;
        const auto endTime = seqDesc.tEnd;

        CheckWritable();
        if ((ptrdiff_t) endTime <= beginTime)
            LogicError("AddSequence: Sequences must be a least one frame long.");
        if (beginTime >= (ptrdiff_t) m_numTimeSteps) // no need to test endTime since it is always non-negative (size_t)
            LogicError("AddSequence: Sequence added to an MBLayout must overlap with minibatch.");

        // remember it
        m_sequences.push_back(seqDesc);

        // create all the cached fast-lookup information
        const auto seqId = seqDesc.seqId;
        const auto s = seqDesc.s;
        size_t b = (size_t)(max(beginTime, (ptrdiff_t) 0));
        size_t e = min(endTime, m_numTimeSteps);
        m_numFramesDeclared += (e - b);
        if (seqId == GAP_SEQUENCE_ID)
        {
            m_numGapFrames += (e - b);
            for (size_t t = b; t < e; t++)
            {
                m_timeStepHasGap[t] = true;
                m_distanceToStart(s, t) = -1; // start flags also encode gaps
            }
        }
        else
            for (size_t t = b; t < e; t++)
            {
                // update the nearest sentence boundaries, minimum over all parallel sequences
                // If 0, then we are on a boundary. If not 0, we can still test in presence of FrameRange.m_timeOffset.
                ptrdiff_t distanceToStart = (ptrdiff_t) t - beginTime;
                ptrdiff_t distanceToEnd = (ptrdiff_t)(endTime - 1 - t);
                m_distanceToStart(s, t) = (float) distanceToStart;
                m_distanceToEnd(s, t) = (float) distanceToEnd;
                // and the aggregate
                if (m_distanceToNearestStart[t] > distanceToStart)
                    m_distanceToNearestStart[t] = distanceToStart;
                if (m_distanceToNearestEnd[t] > distanceToEnd)
                    m_distanceToNearestEnd[t] = distanceToEnd;
            }
    }

    // short-hand to initialize an MBLayout for the common case of frame mode
    // In frame mode, there is one parallel "sequence" per sample, which is 1 frame long.
    // This function provides an efficient short-cut implementation of AddSequence(t, t, 0, 1) for every sample t.
    void InitAsFrameMode(size_t numSamples)
    {
        Init(numSamples, 1);

        // create sequences array
        SequenceInfo virginSeqInfo = {0, 0, 0, 1};
        m_sequences.resize(numSamples, virginSeqInfo); // pass it here since otherwise STL will initialize everything to 0 unnecessarily

        // update sequence indices
        for (size_t s = 0; s < numSamples; s++)
        {
            // remember it
            auto &seqDesc = m_sequences[s];
            seqDesc.seqId = s;
            seqDesc.s = s;
        }
        m_numFramesDeclared = numSamples;

        // create all the cached fast-lookup information
        m_distanceToStart.SetValue(0);
        m_distanceToEnd.SetValue(0);
        m_distanceToNearestStart[0] = 0;
        m_distanceToNearestEnd[0] = 0;

        Lock();
    }

    // mark a range of frames in a parallel sequence as invalid
    // I'd love to start with all-gaps, but that would require to set flags upfront, and then clearing them.
    void AddGap(size_t s, ptrdiff_t beginTime, size_t endTime)
    {
        if ((ptrdiff_t) endTime > beginTime)
            AddSequence(GAP_SEQUENCE_ID, s, beginTime, endTime);
    }

    // find a sequence by its id
    const SequenceInfo& FindSequence(UniqueSequenceId seqId) const
    {
        for (const auto &seqInfo : m_sequences)
            if (seqInfo.seqId == seqId)
                return seqInfo;
        LogicError("FindSequence: Requested sequence (id %u) not found.", (unsigned int) seqId);
    }

    // find a sequence by SequenceInfo array and position
    // Use this if sequences may be matching 1:1.
    const SequenceInfo& FindMatchingSequence(const vector<SequenceInfo>& querySequences, size_t i) const
    {
        // TODO: What are our sorted-ness guarantees?
        let seqId = querySequences[i].seqId; // the seq id we are looking for
        if (seqId == GAP_SEQUENCE_ID)
            LogicError("FindMatchingSequence: Cannot be applied go gaps.");
        if (seqId == m_sequences[i].seqId)   // if both sequence arrays match 1:1 then we found it
            return m_sequences[i];
        else
            return FindSequence(seqId);
    }

    // -------------------------------------------------------------------
    // inquire about gaps or boundaries
    // -------------------------------------------------------------------

    bool HasGaps() const;
    bool HasGaps(const FrameRange &fr) const;

    // test boundary flags for a specific condition
    bool IsBeyondStartOrEnd(const FrameRange& fr) const;
    bool IsGap(const FrameRange& fr) const;
    bool IsBeyondMinibatch(const FrameRange& fr) const;

    // test whether at least one sequence crosses the bounds of this minibatch
    bool HasSequenceBeyondBegin() const
    {
        for (const auto &seq : m_sequences)
            if (seq.tBegin < 0)
                return true;
        return false;
    }

    bool HasSequenceBeyondEnd() const
    {
        for (const auto &seq : m_sequences)
            if (seq.tEnd > m_numTimeSteps)
                return true;
        return false;
    }

    // -------------------------------------------------------------------
    // indexing
    // -------------------------------------------------------------------

    // get the matrix-column index for a given time step in a given sequence
    size_t GetColumnIndex(const SequenceInfo& seq, size_t t) const
    {
        if (t > seq.GetNumTimeSteps())
            LogicError("GetColumnIndex: t out of sequence bounds.");
        if (seq.s > GetNumParallelSequences())
            LogicError("GetColumnIndex: seq.s out of sequence bounds."); // can only happen if 'seq' does not come out of our own m_sequences array, which is verboten
        ptrdiff_t tIn = (ptrdiff_t)t + seq.tBegin;       // shifted time index
        if (tIn < 0 || (size_t)tIn >= GetNumTimeSteps())
            LogicError("GetColumnIndex: Attempted to access a time step that is accessing a portion of a sequence that is not included in current minibatch."); // we may encounter this for truncated BPTT
        size_t col = (size_t)tIn * GetNumParallelSequences() + seq.s;
        assert(col < GetNumCols());
        return col;
    }

    // get the matrix-column indices for a given sequence
    // sequence is expected to belong to this MB
    vector<size_t> GetColumnIndices(const SequenceInfo& seq) const
    {
        size_t numFrames = GetNumSequenceFramesInCurrentMB(seq);
        vector<size_t> res;
        res.reserve(numFrames);
        for (size_t i = 0; i < numFrames;++i)
            res.push_back(GetColumnIndex(seq,i));
        return res;
    }

private:
    // we are trying to access content--this verifies that the structure is consistent
    // All frames must now be declared.
    void CheckIsValid() const
    {
        if (m_numFramesDeclared != GetNumCols())
            LogicError("MBLayout: Attempting to read out flags, but only %d out of %d frames have been defined.",
                       (int) m_numFramesDeclared, (int) (m_numTimeSteps * m_numParallelSequences));
    }

    // Ensure that the MBLayout allows writes
    void CheckWritable() const
    {
        if (!m_writable)
            LogicError("Modification attempted on a MBLayout that is no longer writable.");
    }

    // Freeze the MBLayout disallowing further modifications through set operations
    void Lock() const
    {
        CheckIsValid();
        m_writable = false;
    }

private:
    // -------------------------------------------------------------------
    // data members: main information
    // -------------------------------------------------------------------

    // dimensions
    size_t m_numTimeSteps;
    size_t m_numParallelSequences;

    // all sequences that live inside this minibatch
    vector<SequenceInfo> m_sequences;

private:
    // -------------------------------------------------------------------
    // data members: cached information and inverse lookup tables
    // -------------------------------------------------------------------

    // counters on how much has been declared, for fast access (this can be recomputed from m_sequences as well)
    size_t m_numFramesDeclared;
    size_t m_numGapFrames;

    // Lookup tables for determining whether any sequence at time t is a boundary or gap.
    // An optional time delay can be given, then the test is whether going from t to (t + time delay) crosses a boundary.
    // The purpose is for knowing when to reset state of a recurrent node.
    //
    // For every (s,t), we store the distance to the corresponding sequence begin and end.
    // We also store for every [t] an aggregate to know the nearest boundary.
    // For example, two sentences used in parallel, one with 5 and one with 3 time steps, in one minibatch, both starting at step 0
    // Would be described by these [2 x 5]  matrices:
    // m_distanceToStart        = [ 0  1  2  3  4 ;
    //                              0  1  2 -1 -1 ]          // (last two time steps have no content)
    // m_distanceToEnd          = [ 4  3  2  1  0 ;
    //                              2  1  0  .  . ]          // (last two time steps undefined)
    // m_distanceToNearestStart = [ 0  1  2  3  4 ]
    // m_distanceToNearestEnd   = [ 2  1  0  1  0 ]
    Matrix<float> m_distanceToStart, m_distanceToEnd;                   // (s,t); value<0 stands for gap
    vector<ptrdiff_t> m_distanceToNearestStart, m_distanceToNearestEnd; // [t]    (does not store info about gaps; consult m_timeStepHasGap[] vector instead)

    vector<bool> m_timeStepHasGap; // [t] true if at least one gap in time step t

    // Cached mask indicating the validity of each column in the MBLayout
    // TODO: We actually just need a boolean matrix for this.
    // A value of 1 indicates that the column has valid content
    // and 0 indicates invalid (aka MinibatchPackingFlags::NoInput)
    mutable Matrix<char> m_columnsValidityMask;

    // A boolean flag indicating whether the MBLayout can be further modified
    // When it's value is false, no set operations are allowed on the MBLayout.
    // Meant to guard in lazy creation of m_columnsValidityMask.
    mutable bool m_writable;

    // The axis this MBLayout represents.
    // For now only a string meant for debugging.
    std::wstring m_axisName;

    // The mutex to searilize the access to nameIndices in SetUniqueAxisName().
    // Todo: after upgraded to VS2015, move both static variables into SetUnqiueAxisName() as local static variables there.
    static std::mutex s_nameIndiciesMutex;
    static std::map<std::wstring, size_t> s_nameIndices;

public:

    // special accessor for sequence training  --TODO: must be replaced by a different mechanism
    bool IsEnd(size_t s, size_t t) const
    {
        auto distanceToStart = (ptrdiff_t) m_distanceToStart(s, t);
#if 1 // I don't exactly know what this does, so try assert() first
        assert(distanceToStart != -1);
        distanceToStart;
#else
        if (distanceToStart == -1) // indicates a gap
            return false;
#endif
        auto distanceToEnd = (size_t) m_distanceToEnd(s, t);
        return distanceToEnd == 0;
    }
};
typedef MBLayout::MBLayoutPtr MBLayoutPtr;

// -----------------------------------------------------------------------
// FrameRange -- identifies a frame or a set of frames to apply computation to
//
// Operations can be applied all at once to all frames (PAR) or sequentially (SEQ).
//
// PAR is typically encountered in feed-forward DNNs, where all frames of a minibatch are independent.
// Thus, operations can be applied to all frames concurrently, using a single CUDA
// launche for all frames at once. In this case, the FrameRange identifies the
// entire minibatch.
//
// SEQ is needed for recurrent networks, where frames must be processed iteratively.
// However, we still process multiple parallel sequences concurrently. In this case, the
// FrameRange would identify frames of the same time step across all sequences.
//
// To access the subset of a minibatch matrix selected by FrameFange, use DataWithMBLayoutFor().
//
// TODO: This will in the future be able to hold sub-ranges for nested loops as well.
// -----------------------------------------------------------------------

class FrameRange
{
public:                       // TODO: make private (currently used from masking and DataFor) ; TODO: rename all members with m_ prefix
    size_t timeIdxInSeq;      // start frame; SIZE_MAX = all frames in MB
    ptrdiff_t m_timeOffset;   // this is added to timeIdxInSeq wherever it is used
    size_t m_timeRange;       // use this to describe a custom range > 1 frame
    size_t seqIndex;          // parallel-sequence index; SIZE_MAX = all sequences in MB (most common case)  --TODO: Bad name, 'sequence' and 'parallel sequence' are two different things
    MBLayoutPtr m_pMBLayout;  // layout associated with this
    bool m_broadcastAllowed;  // frame range may be broadcast from outer layout (e.g. a matrix with NULL layout and 1 column is acceptable to this frame range). Only applies when iterating over time; otherwise broadcasting is always OK.
    const FrameRange *parent; // or NULL: parent range, relative to which this FrameRange is interpreted  --TODO: not used yet

public:
    // can construct from a single size_t -> a single-frame range
    FrameRange(MBLayoutPtr pMBLayout, size_t timeIdxInSeq)
        : timeIdxInSeq(timeIdxInSeq), m_timeOffset(0), m_timeRange(1), seqIndex(SIZE_MAX), m_pMBLayout(pMBLayout), m_broadcastAllowed(false), parent(nullptr)
    {
    }

    // or without arguments -> entire minibatch / no frame-range
    FrameRange(MBLayoutPtr pMBLayout)
        : FrameRange(pMBLayout, SIZE_MAX)
    {
    }

    // no arguments--used if passed as an out parameter
    FrameRange()
        : FrameRange(MBLayoutPtr(), SIZE_MAX)
    {
    }

    // return a frame range with broadcast allowed
    // This is used, e.g., by PlusNode which can combine minibatch data and single-column vectors.
    FrameRange AllowBroadcast() const
    {
        FrameRange ret = *this;
        ret.m_broadcastAllowed = true;
        return ret;
    }

    // create a FrameRange that accesses a single sequence only
    // FrameRange(t).Sequence(seq)
    FrameRange Sequence(size_t s) const
    {
        FrameRange ret = *this;
        ret.seqIndex = s;
        return ret;
    }

    // create a FrameRange with its MBLayout replaced by another
    // You must check yourself whether this is correct.
    FrameRange WithLayout(MBLayoutPtr pMBLayout) const
    {
        FrameRange ret = *this;
        ret.m_pMBLayout = pMBLayout;
        return ret;
    }

    // create a FrameRange with a time offset
    // If IsAllFrames() then this will cause out-of-bounds slices.
    FrameRange WithTimeOffset(ptrdiff_t offset) const
    {
        FrameRange ret = *this;
        ret.m_timeOffset += offset;
        return ret;
    }

    // remove a time offset from a FrameRange
    FrameRange WithoutTimeOffset() const
    {
        FrameRange ret = *this;
        ret.m_timeOffset = 0;
        return ret;
    }

    // create a FrameRange with a time range > 1
    FrameRange WithTimeRange(size_t range) const
    {
        FrameRange ret = *this;
        if (!IsAllFrames())
            ret.m_timeRange = range;
        return ret;
    }

    // create a FrameRange from another with an updated time index
    FrameRange WithTimeStep(size_t begin) const
    {
        FrameRange ret = *this;
        ret.timeIdxInSeq = begin;
        return ret;
    }

    std::pair<size_t,size_t> GetSequenceRange() const
    {
        if (!m_pMBLayout) return
            make_pair(0, 1);
        else if (seqIndex == SIZE_MAX) return
            make_pair(0, m_pMBLayout->GetNumParallelSequences());
        else return
            make_pair(seqIndex, seqIndex + 1);
    }

    std::pair<size_t, size_t> GetTimeRange() const
    {
        if (!m_pMBLayout) return
            make_pair(0, 1);
        else if (IsAllFrames()) return
            make_pair(0, m_pMBLayout->GetNumTimeSteps());
        else return
            make_pair(timeIdxInSeq + m_timeOffset, timeIdxInSeq + m_timeOffset + m_timeRange);
    }

    bool IsOneColumnWrt(const shared_ptr<MBLayout> &pMBLayout) const
    {
        if (!pMBLayout) return
            true; // target has no layout: This would broadcast.
        else return
            (pMBLayout->GetNumTimeSteps()         == 1 || (!IsAllFrames() && m_timeRange == 1)) &&
            (pMBLayout->GetNumParallelSequences() == 1 || seqIndex != SIZE_MAX);
    }

    // code that can only handle single-frame ranges will call t() to get the time index, which will throw if numFrames != 1
    // Some functions need just the time index, e.g. for looking up stuff in m_boundaryInfo. That's where an unscaled index is needed.
    // Really only used in RecurrentNodes(), where it will be replaced by FrameRange::WithDelay() which allows to access delayed frames through the FrameRange object.
    size_t t() const
    {
        EnsureNotAllFrames();
        ptrdiff_t t = m_timeOffset + (ptrdiff_t) timeIdxInSeq;
        if (t < 0 || (size_t) t >= m_pMBLayout->GetNumTimeSteps())
            InvalidArgument("FrameRange::t(): Time offset caused time index to be out of range.");
        return (size_t) t;
    }

    bool IsAllFrames() const
    {
        return timeIdxInSeq == SIZE_MAX;
    } // if true then above functions may not be called; caller must use entire batch instead (PAR mode)

private:
    void EnsureNotAllFrames() const
    {
        if (IsAllFrames())
            LogicError("FrameRange::t() called when frame range refers to whole minibatch");
    }
};

// -----------------------------------------------------------------------
// MBLayout functions that require FrameRange
// -----------------------------------------------------------------------

inline bool MBLayout::HasGaps() const
{
    return m_numGapFrames > 0; /*HasGaps(FrameRange());*/
}

inline bool MBLayout::HasGaps(const FrameRange &fr) const
{
    CheckIsValid();
    if (fr.IsAllFrames())
        return m_numGapFrames > 0; // test entire minibatch
    if (fr.seqIndex == SIZE_MAX)
        return m_timeStepHasGap[fr.timeIdxInSeq]; // test all seq for one time step
    else
        return IsGap(fr); // test one sequence
}

// test whether a given frame is or contains a gap
inline bool MBLayout::IsGap(const FrameRange &fr) const
{
    CheckIsValid();

    if (fr.IsAllFrames())
        LogicError("MBLayout::Get() cannot be applied to FrameRange that specifies more than a single time step.");

    const auto t = fr.timeIdxInSeq; // we test off the frame without offset
    const auto s = fr.seqIndex;
    if (s == SIZE_MAX) // aggregate requested
        return m_timeStepHasGap[t];

    // determine flags from matrices
    return m_distanceToStart(s, t) < 0; // value is -1 for gaps, non-negative otherwise
}

// test whether frame is exceeding the bounds of the MB
inline bool MBLayout::IsBeyondMinibatch(const FrameRange& fr) const
{
    CheckIsValid();

    if (fr.IsAllFrames())
        LogicError("MBLayout::IsBeyondStartOrEnd() cannot be applied to FrameRange that specifies more than a single time step.");

    const auto beginTime = (ptrdiff_t)fr.timeIdxInSeq + fr.m_timeOffset; // we test off the frame with offset
    const auto endTime = beginTime + (ptrdiff_t)fr.m_timeRange;
    return beginTime < 0 || endTime > (ptrdiff_t)GetNumTimeSteps();
}

// test whether frame is exceeding the sentence boundaries
// In case of a gap, this returns false.
inline bool MBLayout::IsBeyondStartOrEnd(const FrameRange &fr) const
{
    CheckIsValid();

    if (fr.IsAllFrames())
        LogicError("MBLayout::IsBeyondStartOrEnd() cannot be applied to FrameRange that specifies more than a single time step.");

    const auto t = fr.timeIdxInSeq; // we test off the frame without offset
    const auto s = fr.seqIndex;
    if (s == SIZE_MAX) // aggregate requested
    {
        // determine flags from aggregate vectors
        // Note: We allow that all parallel sequences contain gaps (m_distanceToNearestStart[t] == PTRDIFF_MAX)
        // because that makes implementation of the reader easier for truncated BPTT (it knows too late that there are not that many frames left).
        auto distanceToStart = (ptrdiff_t) m_distanceToNearestStart[t];
        if (distanceToStart < -fr.m_timeOffset)
            return true;
        auto distanceToEnd = (ptrdiff_t) m_distanceToNearestEnd[t];
        if (distanceToEnd < fr.m_timeOffset)
            return true;
        return false;
    }

    // determine flags from matrices
    auto distanceToStart = (ptrdiff_t) m_distanceToStart(s, t);
    if (distanceToStart == -1) // indicates a gap
    {
        assert(m_timeStepHasGap[t]);
        return false; // a gap is not outside, so that we can allow collating
    }
    else
    {
        if (distanceToStart < -fr.m_timeOffset)
            return true;
        auto distanceToEnd = (ptrdiff_t) m_distanceToEnd(s, t);
        if (distanceToEnd < fr.m_timeOffset)
            return true;
    }
    return false;
}

// TODO: Remove this version (with sanity checks) after this has been tested. Then the function can be inlined above.
inline size_t MBLayout::GetActualNumSamples() const { return m_numFramesDeclared - m_numGapFrames; }

// return m_columnsValidityMask(,), which is lazily created here upon first call
// only called from MaskMissingColumnsTo()
// TODO: Can probably be faster by using the sequence array directly.
// TODO: Or should we just blast m_distanceToStart to GPU, and maks based on that? It is small compared to features.
inline const Matrix<char>& MBLayout::GetColumnsValidityMask(DEVICEID_TYPE deviceId) const
{
    CheckIsValid();
    // lazily compute the validity mask
    if (m_columnsValidityMask.IsEmpty())
    {
        assert(HasGaps()); // must only be called if there are gaps
        Lock();

        // Determine indices of all invalid columns in the minibatch
        // TODO: This can be done more efficiently by using m_sequences[].
        size_t nT = GetNumTimeSteps();
        size_t nS = GetNumParallelSequences();

        std::vector<char> columnsValidityMask(nT * nS, 1); // form the mask in a CPU-side STL vector first
        size_t gapsFound = 0;
        for (size_t t = 0; t < nT; t++)
        {
            FrameRange fr(nullptr, t);
            if (IsGap(fr))
            {
                for (size_t s = 0; s < nS; s++)
                {
                    if (IsGap(fr.Sequence(s)))
                    {
                        columnsValidityMask[(t * nS) + s] = 0;
                        gapsFound++;
                    }
                }
            }
        }
        assert(gapsFound == m_numGapFrames); // sanity check

        if (deviceId != m_columnsValidityMask.GetDeviceId())
            m_columnsValidityMask = Matrix<char>(deviceId);
        m_columnsValidityMask.SetValue(1, nS * nT, deviceId, columnsValidityMask.data());
    }
    return m_columnsValidityMask;
}

// class for defining an iteration over a sequence, forward and backward
// One day, we may also have nested structures. For those, FrameRangeIterations will be able to be instantiated from FrameRange objects to loop over their nested dimension.
class FrameRangeIteration
{
    MBLayoutPtr m_pMBLayout;
    int m_step;

public:
    // one-dimensional iteration (time sequences)
    // 'Step' specifies the stepping direction of the loop:
    //  - for left-to-right models -> pass step = +1
    //  - for right-to-left models -> pass step = -1
    FrameRangeIteration(MBLayoutPtr pMBLayout, int step)
        : m_pMBLayout(pMBLayout), m_step(step)
    {
    }
    // This class is returned by begin() and end().
    // It is a FrameRange with additions ++ and != operators needed in the for loop.
    class FrameRangeIterator : public FrameRange
    {
        ptrdiff_t m_step;

    public:
        FrameRangeIterator(const FrameRange &begin, ptrdiff_t step)
            : FrameRange(begin), m_step(step)
        {
        }
        bool operator!=(const FrameRangeIterator &other) const
        {
            return timeIdxInSeq != other.timeIdxInSeq;
        }
        void operator++(int)
        {
            timeIdxInSeq = (size_t)(m_step + (ptrdiff_t) timeIdxInSeq);
        } // going through (int) to avoid undefined behavior
    };
    // iterators for iterating forward
    FrameRangeIterator begin() const
    {
        if (m_step > 0)
            return FrameRangeIterator(FrameRange(m_pMBLayout, 0), +1);
        else
            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps() - 1), -1);
    }
    FrameRangeIterator end() const
    {
        if (m_step < 0)
            return FrameRangeIterator(FrameRange(m_pMBLayout, (size_t) -1), 0 /*dummy*/);
        else
            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps()), 0);
    }
    // iterators for iterating in reverse order (as needed for gradient update)
    FrameRangeIterator rbegin() const
    {
        if (m_step < 0)
            return FrameRangeIterator(FrameRange(m_pMBLayout, 0), +1);
        else
            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps() - 1), -1);
    }
    FrameRangeIterator rend() const
    {
        if (m_step > 0)
            return FrameRangeIterator(FrameRange(m_pMBLayout, (size_t) -1), 0);
        else
            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps()), 0);
    }
};

// -----------------------------------------------------------------------
// ColumnRangeWithMBLayoutFor() -- Return column range for a FrameRange of a Matrix with specified number of columns with a given MBLayout
// -----------------------------------------------------------------------

static inline std::pair<size_t, size_t> ColumnRangeWithMBLayoutFor(size_t numCols /*of data matrix to slice*/,
                                                                   const FrameRange &fr /*select frame or entire batch*/,
                                                                   const MBLayoutPtr &pMBLayout /*the MB layout of 'data'*/)
{
    if (!fr.m_pMBLayout && !fr.IsAllFrames())
        LogicError("ColumnRangeWithMBLayoutFor: FrameRange refers to a time slice while being outside of a loop.");

    // MBLayout of data and of FrameRange must be identical pointers,
    // or in case of broadcasting, respective parent pointers.
    // MBLayouts that are identical in content but not object identity (pointer) are admissible.
    // We rely on a runtime check. If this is inefficient, use a ReconcileDynamicAxis node.
    // (Note: Earlier versions of CNTK did not accept same-content MBLayouts.)
    if (fr.m_pMBLayout != pMBLayout)
    {
        // if broadcast allowed then it is allowed to broadcast from an outer-loop value
        // Currently, the only 'outer' loop we have is to have no layout.
        if (fr.m_broadcastAllowed && !pMBLayout && numCols == 1)
            return std::pair<size_t, size_t>(0, numCols);
        if (fr.m_pMBLayout && pMBLayout && *fr.m_pMBLayout == *pMBLayout)
            ; // layouts are compatible--you may proceed
        else
            LogicError("ColumnRangeWithMBLayoutFor: FrameRange's dynamic axis is inconsistent with matrix. They are compatible though--are you missing a ReconcileDynamicAxis operation?");
    }
    // if FrameRange refers to whole minibatch (map mode)
    // or if we don't even have a layout
    // then return the whole matrix
    // but as a reference (e.g. it cannot be resized)
    if (!pMBLayout || fr.IsAllFrames())
    {
        if (fr.m_timeOffset != 0)
            LogicError("ColumnRangeWithMBLayoutFor: Time offset must not be specified for FrameRanges that reference the entire minibatch."); // (note: the tensor version allows this)
        if (fr.seqIndex == SIZE_MAX)
            return std::pair<size_t, size_t>(0, numCols);
        else
        {
            if (!pMBLayout)
                LogicError("ColumnRangeWithMBLayoutFor: Attempting to retrieve a parallel sequence from data without layout.");
            else
                LogicError("ColumnRangeWithMBLayoutFor: Individual parallel sequences cannot be retrieved in Matrix representation. Use TensorView instead.");
        }
    }
    // FrameRange refers to a time slice -> return that
    else
    {
        size_t numParallelSequences = pMBLayout->GetNumParallelSequences();
        size_t startColumn = (fr.timeIdxInSeq + fr.m_timeOffset) * numParallelSequences;
        if (startColumn >= numCols)
            LogicError("ColumnRangeWithMBLayoutFor: FrameRange specifies a time index that is out of range.");
        if (fr.seqIndex == SIZE_MAX)
            return std::pair<size_t, size_t>(startColumn, numParallelSequences * fr.m_timeRange);
        else if (fr.m_timeRange != 1)
            LogicError("ColumnRangeWithMBLayoutFor: FrameRange only support per-sequence time ranges with tensor slices, not matrix slices.");
        else
            return std::pair<size_t, size_t>(startColumn + fr.seqIndex, 1);
    }
}

// -----------------------------------------------------------------------
// DataWithMBLayoutFor() -- create view for a FrameRange of a Matrix with a given MBLayout
// This function binds the above together.
// Any access by FrameRange should only be done through this function.
// -----------------------------------------------------------------------

template <class ElemType>
static inline Matrix<ElemType> DataWithMBLayoutFor(const Matrix<ElemType> &data,
                                                   const FrameRange &fr /*select frame or entire batch*/,
                                                   const MBLayoutPtr &pMBLayout /*the MB layout of 'data'*/)
{
    auto columnRange = ColumnRangeWithMBLayoutFor(data.GetNumCols(), fr, pMBLayout);
    return data.ColumnSlice(columnRange.first, columnRange.second);
}

// -----------------------------------------------------------------------
// TensorSliceWithMBLayoutFor() -- Return tensor slice for a FrameRange with a given MBLayout.
// This implements the logic of interpreting the FrameRange object.
// This function happily returns tensor bounds that are out of bounds, assuming caller will do the right thing.
// -----------------------------------------------------------------------

template <class DimensionVector> // e.g. std::vector<size_t> or SmallVector<size_t>
static inline std::pair<DimensionVector, DimensionVector> TensorSliceWithMBLayoutFor(const DimensionVector &shape /*actual tensor shape of 'data'*/,
                                                                                     const FrameRange &fr /*select frame or entire batch from 'data'*/,
                                                                                     const MBLayoutPtr &pMBLayout /*the MB layout of 'data'*/)
{
    if (!fr.m_pMBLayout && !fr.IsAllFrames())
        LogicError("TensorSliceWithMBLayoutFor: FrameRange refers to a time slice while being outside of a loop.");

    std::pair<DimensionVector, DimensionVector> result;
    typedef decltype(result.first[0]) ElemType;

    // this creates a slice for the entire matrix, which we will then narrow down
    result.first.resize(shape.size(), 0);
    result.second = shape;

    // get position of time and sequence index
    const size_t iterDim = shape.size() -1; // valid if data has MBLayout

    // MBLayout of data and of FrameRange must be identical pointers,
    // or in case of broadcasting, respective parent pointers.
    // MBLayouts that are identical in content but not object identity (pointer) are admissible.
    // We rely on a runtime check. If this is inefficient, use a ReconcileDynamicAxis node.
    // (Note: Earlier versions of CNTK did not accept same-content MBLayouts.)
    if (fr.m_pMBLayout != pMBLayout)
    {
        // if broadcast allowed then it is allowed to broadcast from an outer-loop value
        // Currently, the only 'outer' loop we have is to have no layout.
        if (fr.m_pMBLayout /*get data for a loop*/ && !pMBLayout /*'data' is not samples*/ && fr.m_broadcastAllowed /*we're OK with that*/)
            ; // the time dimension is broadcasting--leave it as is
        else if (fr.m_pMBLayout && pMBLayout && *fr.m_pMBLayout == *pMBLayout)
            ; // layouts are compatible--you may proceed
        else if (!fr.m_pMBLayout)
            LogicError("TensorSliceWithMBLayoutFor: FrameRange has no layout, incompatible with data's layout: %s",
                       static_cast<string>(*(pMBLayout)).c_str());
        else
            LogicError("TensorSliceWithMBLayoutFor: FrameRange's dynamic axis is inconsistent with data: %s vs. %s", 
                       static_cast<string>(*(fr.m_pMBLayout)).c_str(), static_cast<string>(*(pMBLayout)).c_str());
    }
    // if FrameRange refers to whole minibatch (map mode)
    // or if we don't even have a layout
    // then return the whole matrix
    // but as a reference (e.g. it cannot be resized)
    else if (!pMBLayout || fr.IsAllFrames())
    {
        if (fr.m_timeOffset != 0)
        {
            if (!pMBLayout)
                LogicError("TensorSliceWithMBLayoutFor: Time offset cannot be applied to tensors that have no time dimension.");
            result.first[iterDim] += (ElemType) fr.m_timeOffset; // Note: If we have an offset, this is guaranteed to yield a slice that is out of bounds.
            result.second[iterDim] += (ElemType) fr.m_timeOffset;
            if (result.first[iterDim] > result.second[iterDim])
                LogicError("TensorSliceWithMBLayoutFor: Numeric wraparound. You used a size_t vector where an int vector would be needed.");
        }
    }
    // FrameRange refers to a time slice -> return that
    else if (result.second[iterDim] > 1) // (if iter dim is broadcasting then always return that one independent of requested index)
    {
        assert(pMBLayout);
        size_t ts = fr.timeIdxInSeq + fr.m_timeOffset;
        size_t te = ts + fr.m_timeRange;
        result.first[iterDim] = (ElemType) ts;
        result.second[iterDim] = (ElemType) te;
    }

    // sequence index
    if (fr.seqIndex != SIZE_MAX)  // sequence requested?
    {
        if (pMBLayout) // (if no layout then broadcast to all sequences)
        {
            size_t sequenceDim = shape.size() - 2; // (only valid if pMBLayout)  --TODO: In case of multiple time dims, this must be adjusted.
            if (result.second[sequenceDim] > 1 /*>1 sequence (not broadcasting)*/)
            {
                size_t s = fr.seqIndex;
                if (s >= result.second[sequenceDim])
                    LogicError("TensorSliceWithMBLayoutFor: FrameRange specifies a parallel-sequence index that is out of range.");
                result.first[sequenceDim] = (ElemType)s;
                result.second[sequenceDim] = (ElemType)s + 1;
            }
        }
    }

    return result;
}

// -----------------------------------------------------------------------
// MaskMissingColumnsTo() -- function to set gaps to zero or NaN
// -----------------------------------------------------------------------

// This sets MB columns to 0 (or any 'val') that have the NoLabel or NoFeature flag set.
// Such situation happens when packing multiple sequences for parallel processing--there will be some gaps, which are flagged by these flags.
// Nodes that operate in 'map' style (input(j) -> output(j) independently) can ignore this; it will be garbage-in-garbage-out.
// However, nodes that 'reduce' minibatches (e.g. computing the sum of all frames across all sequences) must deal with the garbage.
// This function sets those to 0, assuming that now they can be reduced without affecting the result.
// This function can operate on the whole range or on a selected single frame and/or a single sequence.
// 'Reduce' style operations--the criterion nodes and gradient computation--call this.
// Warning: The layout used here must match the matrix. E.g. don't pass a child's matrix from a criterion node (use Input(x)->MaskMissing{Values,Gradient}ColumnsToZero() instead.
template <class ElemType>
static inline void MaskMissingColumnsTo(Matrix<ElemType>& matrixToMask, const MBLayoutPtr& pMBLayout, const FrameRange& fr, ElemType val)
{
    if (pMBLayout && pMBLayout->HasGaps(fr))
    {
        const auto& maskMatrix = pMBLayout->GetColumnsValidityMask(matrixToMask.GetDeviceId());

        maskMatrix.TransferToDeviceIfNotThere(matrixToMask.GetDeviceId(), /*ismoved=*/ false, /*emptyTransfer=*/ false, /*updatePreferredDevice=*/ false);
        auto maskSlice = DataWithMBLayoutFor(maskMatrix, fr, pMBLayout);

        auto matrixSliceToMask = DataWithMBLayoutFor(matrixToMask, fr, pMBLayout);
        if ((matrixSliceToMask.GetNumCols() % maskSlice.GetNumCols()) != 0)
            LogicError("MaskMissingColumnsTo: The number of columns of the matrix slice to be masked is not a multiple of the number of columns of the mask slice.");

        matrixSliceToMask.MaskColumnsValue(maskSlice, val, matrixSliceToMask.GetNumCols() / maskSlice.GetNumCols());
    }
}

}}}
