// Sequences.h -- all about iterating over sequences, that is, MBLayout, FrameRange, and iterators
//
// <copyright file="Sequences.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// TODO:
//  - fix RecurrentNode (remove shifted layout, use time offset in condition test)
//  - finally remove the old bit masks  --nearly there, only used for checking the new code
//  - split Is() into IsStart, IsEnd, IsGap; then eliminate MinibatchPackingFlags as well

#pragma once

#include "Basics.h"
#include "Matrix.h"
#include <vector>
#include <memory>   // for shared_ptr

enum class MinibatchPackingFlags : char     // (note: not using unsigned char because these go into a matrix, and we use Matrix<char>, since we use it as a data holder)
{
    None = 0,
    SequenceStart = 1 << 0,         // binary 0001  frame is first of an utterance
    SequenceEnd = 1 << 1,           // binary 0010  frame is last of an utterance
    NoFeature = 1 << 2,             // binary 0100  frame has no feature (e.g. a gap due to BPTT)
    NoLabel = 1 << 3,               // binary 1000  frame has no label

    NoInput = NoFeature | NoLabel,  // Note: Once we refactorized the reader, NoInput will no longer needed.
    SequenceStartOrNoFeature = SequenceStart | NoFeature,
    SequenceEndOrNoFeature = SequenceEnd | NoFeature,
    SequenceStartOrEndOrNoFeature = SequenceStart | SequenceEnd | NoFeature,
};

inline MinibatchPackingFlags operator| (MinibatchPackingFlags a, MinibatchPackingFlags b)
{
    return static_cast<MinibatchPackingFlags>(static_cast<unsigned char>(a) | static_cast<unsigned char>(b));
}

inline MinibatchPackingFlags& operator|= (MinibatchPackingFlags& a, MinibatchPackingFlags b)
{
    a = a | b;
    return a;
}

inline bool operator& (MinibatchPackingFlags a, MinibatchPackingFlags b)
{
    return (static_cast<unsigned char>(a) & static_cast<unsigned char>(b)) != 0;
}

namespace Microsoft { namespace MSR { namespace CNTK {

    // Forward declarations
    class FrameRange;

    typedef size_t UniqueSequenceId;
#define GAP_SEQUENCE_ID SIZE_MAX            // indicates no data
#define NEW_SEQUENCE_ID (SIZE_MAX-1)       // let SetSequence() assign a unique id; for old readers. Don't mix with actual reader-assigned ids.

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
    //  - inquire whether any time step t has any flags across all sequences
    //  - inquire the flags at (s,t)
    //
    // Truncated BPTT support (partial sequences):
    //  - in truncated BPTT, minibatches only contain partial sequences, e.g. a range of 20 time steps
    //  - the flags are stored for every sequence that intersects with this minibatch
    //  - that is also true for flags that fall outside the time-step range of the minibatch
    //
    // An MBLayout object stores:
    //  - for every sequence in the minibatch the n-tuple (global sequence id, s, first t, last t)
    //    (where first and last t may sometimes lie outside of the minibatch, e.g. in case of truncated BPTT)
    //  - number of time steps and parallel sequences (their product is equal to the #columns in the minibatch matrix)
    //  - MinibatchPackingFlags: information whether a frame (s,t) is
    //     - SequenceBegin (first frame of a sequence)
    //     - SequenceEnd (last frame of a sequence--in frame-randomization, each frame is both)
    //     - NoInput (a gap or missing input frame)
    //  - a column-wise OR of those flags for fast testing entire time steps at once
    // -----------------------------------------------------------------------

    // This object allocates its storage lazily, i.e. if there are no flags ever set, no memory is allocated. This is transparent to the caller.
    // Note: With truncated BPTT, it is possible to have sequential data, yet not a single flag set in a minibatch (if all frames are sequence-internal ranges).
    // Contract between ComputationNode, ComputationNetwork, and MBLayout:
    //  - if a node has no MBLayout, m_{function,gradient}Values are not samples (they are not activations or input data), but e.g. model parameters
    //  - ComputationNode::GetNumCols() == MBLayout::GetNumTimeSteps() * MBLayout::GetNumParallelSequences()
    //  - ComputationNetwork ensures that m_{function,gradient}Values are allocated correctly before calling ForwardProp() on a node
    // NOTE: Parts of this class represents the result of refactoring code, including a few irregular edge cases.
    //       Some code below represents the actual use cases I encountered. Not all are, I believe, needed to be as they are; this class could be simplified/streamlined much further.

    struct MBLayout
    {
        typedef std::shared_ptr<MBLayout> MBLayoutPtr;

        MBLayout(size_t numParallelSequences, size_t numTimeSteps) : m_sentenceBoundaryFlags(CPUDEVICE) { Init(numParallelSequences, numTimeSteps); }
        MBLayout() : MBLayout(1, 0) { }

        // copy the content of another MBLayoutPtr over
        // Use this instead of actual assignment to make it super-obvious that this is not copying the pointer but actual content. The pointer is kept fixed.
        void CopyFrom(const MBLayoutPtr & other) { *this = *other; }
        void MoveFrom(MBLayoutPtr other) { *this = move(*other); other->Init(0, 0); }    // destructive copy that steals ownership if the content, like std::move()
    private:
        MBLayout & operator=(const MBLayout &) = default;   // make this private --use CopyFrom() instead, which makes it very clear that it's copying content, not copying the reference
    public:

        // resize and reset all frames to None (note: this is an invalid state and must be fixed by caller afterwards)
        void Init(size_t numParallelSequences, size_t numTimeSteps)
        {
            // remember the dimensions
            m_numParallelSequences = numParallelSequences;
            m_numTimeSteps = numTimeSteps;
            // allocate lookup tables (note: except at the start, these don't really allocate new memory most of the time)
            m_sentenceBoundaryFlags.Resize(m_numParallelSequences, m_numTimeSteps);
            m_sentenceBoundaryFlags.SetValue((float)((int)MinibatchPackingFlags::None));
            m_minibatchPackingFlags.assign(m_sentenceBoundaryFlags.GetNumCols(), MinibatchPackingFlags::None);
            // PTRDIFF_MAX indicates not initialized (also in the matrix, which is stored as float).
            m_distanceToStart.Resize(m_numParallelSequences, m_numTimeSteps); m_distanceToStart.SetValue((float)PTRDIFF_MAX);
            m_distanceToEnd.Resize(m_numParallelSequences, m_numTimeSteps); m_distanceToEnd.SetValue((float)PTRDIFF_MAX);
            m_distanceToNearestStart.assign(m_numTimeSteps, PTRDIFF_MAX);
            m_distanceToNearestEnd.assign(m_numTimeSteps, PTRDIFF_MAX);
            m_timeStepHasGap.assign(m_numTimeSteps, false);
            // reset state
            m_numFramesDeclared = 0;
            m_numGapFrames = 0;
            m_sequences.clear();
            m_writable = true;
        }

        // short-hand to initialize an MBLayout for the special case of frame mode
        // In frame mode, there is one parallel "sequence" per sample, which is 1 frame long.
        void InitAsFrameMode(size_t numSamples)
        {
            Init(numSamples, 1);
            SequenceInfo seqInfo { 0, 0, 0, 1 };
            for (size_t s = 0; s < numSamples; s++)
            {
                seqInfo.seqId = seqInfo.s = s;
                AddSequence(seqInfo);
            }
            Lock();
        }
    private:
        // we are trying to access content--this verifies that the structure is consistent
        // All frames must now be declared.
        void CheckIsValid() const
        {
            if (m_numFramesDeclared != GetNumCols())
                LogicError("MBLayout: Attempting to read out flags, but only only %d out of %d frames have been defined.",
                           (int)m_numFramesDeclared, (int)( m_numTimeSteps * m_numParallelSequences));
        }
    public:

        size_t GetNumTimeSteps()         const { return m_numTimeSteps; }
        size_t GetNumParallelSequences() const { return m_numParallelSequences; }

        // how many columns the MB matrix has
        size_t GetNumCols()              const { return GetNumTimeSteps() * GetNumParallelSequences(); }

        // information stored about sequences
        struct SequenceInfo
        {
            UniqueSequenceId seqId; // unique sequence id (or GAP_SEQUENCE_ID--TODO: don't include gaps here)
            size_t s;               // index of parallel sequence
            ptrdiff_t tBegin;       // first time index in this minibatch. Note that this may be negative of the sequence started before this MB.
            size_t tEnd;            // end = first frame index after final frame. May be beyond the minibatch if reql sequence is longer than the MB.
            bool operator==(const SequenceInfo & other) const { return seqId == other.seqId && s == other.s && tBegin == other.tBegin && tEnd == other.tEnd; }
        };
        // return all sequences stored in this minibatch
        const vector<SequenceInfo> & GetAllSequences() const { return m_sequences; }

    public:

        // compare whether two layouts are the same
        bool operator==(const MBLayout & other) const
        {
            if (this == &other)
                return true;
            bool res =
                m_numTimeSteps == other.m_numTimeSteps &&
                m_numParallelSequences == other.m_numParallelSequences &&
                m_distanceToStart.IsEqualTo(other.m_distanceToStart) &&
                m_distanceToEnd  .IsEqualTo(other.m_distanceToEnd)   &&
                m_distanceToNearestStart == other.m_distanceToNearestStart &&
                m_distanceToNearestEnd   == other.m_distanceToNearestEnd   &&
                m_timeStepHasGap == other.m_timeStepHasGap &&
                m_sequences == other.m_sequences;
#if 1
            bool res1 =
                m_numTimeSteps == other.m_numTimeSteps &&
                m_numParallelSequences == other.m_numParallelSequences &&
                m_minibatchPackingFlags == other.m_minibatchPackingFlags &&
                m_sentenceBoundaryFlags.IsEqualTo(other.m_sentenceBoundaryFlags);
            assert(res == res1);
#endif
            return res;
        }
        bool operator!=(const MBLayout & other) const { return !(*this == other); } // duh

        // get boundary flags
    private:
        MinibatchPackingFlags Get(size_t t) const;
        MinibatchPackingFlags Get(const FrameRange & fr) const;
        MinibatchPackingFlags Get(size_t s, size_t t) const;

    private:
    public:     // naw, these are still used in RecurrentNodes. Will soon be replaced by a different mechanism.
        bool Is(size_t t, MinibatchPackingFlags f) const { return (Get(t) & f) != 0; }
        bool Is(size_t s, size_t t, MinibatchPackingFlags f) const { return (Get(s, t) & f) != 0; }

    public:
        // test boundary flags for a specific condition
        // TODO: Remove the direct-index versions in lieu of FrameRange version.
        //       Direct-index versions are currently used here:
        //        - LUSequenceReader.cpp: a sanity check
        //        - ClassBasedCrossEntropyWithSoftmaxNode (where the correct FrameRange object is already available)
        //        - RecurrentNode (which will be rewritten after MBLayout can handle tests outside its time range)
        // FrameRange version allows to test with time offset
        bool Is(const FrameRange & fr, MinibatchPackingFlags f) const { return (Get(fr) & f) != 0; }
        bool IsBeyondStartOrEnd(const FrameRange & fr) const;
        bool IsGap(size_t t) const { return Is(t, MinibatchPackingFlags::NoInput); }
        bool IsGap(size_t s, size_t t) const { return Is(s, t, MinibatchPackingFlags::NoInput); }

        // only used in sequence training, must be replaced by a different mechanism
        bool IsEnd(size_t s, size_t t) const { return Is(s, t, MinibatchPackingFlags::SequenceEnd); }

    private:
        // set a boundary flag (OR it on top of the existing layout)
        // Currently not yet updated/disabled:
        //  - RecurrentNode for m_timeStep > 1 (this will be fixed differently)
        // Currently marginally broken:
        //  - EvalReader.h
        void Set(size_t s, size_t t, MinibatchPackingFlags f)
        {
            CheckWritable();

            if (f == MinibatchPackingFlags::None)   // actually not setting anything: skip allocation
                return;
            m_sentenceBoundaryFlags.SetValue(s, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(s, t)) | f));
            m_minibatchPackingFlags[t] |= f;
        }

    public:

        // mark a range of frames in a parallel sequence as one sentence
        // Note that endTime is the last frame +1. Like begin/end as used in STL containers.
        void AddSequence(UniqueSequenceId seqId, size_t s, ptrdiff_t beginTime, size_t endTime)
        {
            // old readers can just pass this to get an auto-assigned id (which is fine as long as we only have one MBLayout per minibatch)
            if (seqId == NEW_SEQUENCE_ID)
            {
                static UniqueSequenceId makeSeqIdCounter = 0;
                seqId = makeSeqIdCounter++;
                if (seqId == GAP_SEQUENCE_ID) LogicError("AddSequence: ran out of bits...");    // (will never happen anyway)
            }

            AddSequence(SequenceInfo { seqId, s, beginTime, endTime });
        }

        // version that passes a SequenceInfo record directly
        void AddSequence(const SequenceInfo & seqDesc)
        {
            const auto beginTime = seqDesc.tBegin;
            const auto endTime = seqDesc.tEnd;

            CheckWritable();
            if ((ptrdiff_t)endTime <= beginTime)
                LogicError("AddSequence: Sequences must be a least one frame long.");
            if (beginTime >= (ptrdiff_t)m_numTimeSteps)         // no need to test endTime since it is always non-negative (size_t)
                LogicError("AddSequence: Sequence added to an MBLayout must overlap with minibatch.");

            // remember it
            m_sequences.push_back(seqDesc);

            // create all the cached fast-lookup information
            const auto seqId = seqDesc.seqId;
            const auto s = seqDesc.s;
            if (beginTime >= 0 && seqId != GAP_SEQUENCE_ID)
                Set(s, beginTime, MinibatchPackingFlags::SequenceStart);
            if (endTime <= m_numTimeSteps && seqId != GAP_SEQUENCE_ID)
                Set(s, endTime - 1, MinibatchPackingFlags::SequenceEnd);
            size_t b = (size_t)(max(beginTime, (ptrdiff_t)0));
            size_t e = min(endTime, m_numTimeSteps);
            m_numFramesDeclared += (e - b);
            if (seqId == GAP_SEQUENCE_ID)
            {
                m_numGapFrames += (e - b);
                for (size_t t = b; t < e; t++)
                {
                    Set(s, t, MinibatchPackingFlags::NoInput);
                    m_timeStepHasGap[t] = true;
                    m_distanceToStart(s, t) = -1;   // start flags also encode gaps
                }
            }
            else for (size_t t = b; t < e; t++)
            {
                // update the nearest sentence boundaries, minimum over all parallel sequences
                // -1 in distanceToStart(,) stands for a gap
                assert(m_distanceToStart(s, t) != -1);  // gaps not allowed to overlap
                // If 0, then we are on a boundary. If not 0, we can still test in presence of FrameRange.m_timeOffset.
                ptrdiff_t distanceToStart = t - beginTime;
                if (m_distanceToStart(s, t) > (float)distanceToStart)
                    m_distanceToStart(s, t) = (float)distanceToStart;
                if (m_distanceToNearestStart[t] > distanceToStart)
                    m_distanceToNearestStart[t] = distanceToStart;
                ptrdiff_t distanceToEnd = endTime - 1 - t;
                if (m_distanceToEnd(s, t) > (float) distanceToEnd)
                    m_distanceToEnd(s, t) = (float) distanceToEnd;
                if (m_distanceToNearestEnd[t] > distanceToEnd)
                    m_distanceToNearestEnd[t] = distanceToEnd;
                assert(t == (size_t)beginTime || t == endTime - 1 || m_sentenceBoundaryFlags(s, t) == 0);
            }
        }

        // mark a range of frames in a parallel sequence as invalid
        // I'd love to start with all-gaps, but that would require to set flags upfront, and then clearing them.
        void AddGap(size_t s, ptrdiff_t beginTime, size_t endTime) { if ((ptrdiff_t)endTime > beginTime) AddSequence(GAP_SEQUENCE_ID, s, beginTime, endTime); }

        // compute the number of actual samples in this layout (not counting gaps)
        // This is used by MeanNode and InvStdDevNode, and by statistics reporting.
        // TODO: rename Determine- to Get-, as it is a trivial operation now
        size_t DetermineActualNumSamples() const
        {
#if 1
            size_t n = GetNumCols();
            if (HasGaps())
            {
                for (size_t t = 0; t < GetNumTimeSteps(); t++)
                {
                    if (Is(t, MinibatchPackingFlags::NoInput))
                    {
                        for (size_t s = 0; s < GetNumParallelSequences(); s++)
                        {
                            if (Is(s, t, MinibatchPackingFlags::NoInput))
                                n--;
                        }
                    }
                }
            }
            if (m_numGapFrames != GetNumCols() - n)
                LogicError("DetermineActualNumSamples: Gap counting broken, measured %d vs. originally counted %d", (int)(GetNumCols() - n), (int)m_numGapFrames);
            assert(m_numFramesDeclared - m_numGapFrames == n);
#endif
            return m_numFramesDeclared - m_numGapFrames;
        }

        // function that must flatten gaps can call this to check whether there are any
        bool HasGaps() const;
        bool HasGaps(const FrameRange & fr) const;

        const Matrix<char> & GetColumnsValidityMask(DEVICEID_TYPE deviceId) const;

    private:

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

        // dimensions
        size_t m_numTimeSteps;
        size_t m_numParallelSequences;

        // counters on how much has been declared, for checks
        size_t m_numFramesDeclared;
        size_t m_numGapFrames;

        // all sequences that live inside this minibatch
        vector<SequenceInfo> m_sequences;

        // a matrix of S x T
        // S is the number of parallel sequences, T is the number of time steps (possibly of multiple concatenated sequences).
        // For example, two sentences used in parallel, one with 5 and one with 3 time steps, in one minibatch
        // would be described by this [2 x 5]  matrix:
        //   S . . . E
        //   S . E G G          // (last two time steps have no content)
        // where S, E, and G stand for bit-mask values of MinibatchPackingFlags::SequenceStart, MinibatchPackingFlags::SequenceEnd, and MinibatchPackingFlags::NoInput, respectively.
        Matrix<float> m_sentenceBoundaryFlags;  // (s,t)
        // TODO: we should change to a Matrix<char>.

        // a short-hand vector or-ing the above flags over all parallel sequences
        vector<MinibatchPackingFlags> m_minibatchPackingFlags;  // column-wise OR over m_sentenceBoundaryFlags for fast testing

        // a short-hand for determining whether any sequence at time t is a boundary or gap
        // TODO: Remove m_minibatchPackingFlags, and implement through these two. This will require to make Set() private, i.e. gotta change the readers.
        // PTRDIFF_MAX stands for 'not initialized'.
        Matrix<float> m_distanceToStart, m_distanceToEnd;                   // (s,t); -1 stands for gap
        vector<ptrdiff_t> m_distanceToNearestStart, m_distanceToNearestEnd; // [t]    (-1 does NOT stand for gap; consult m_timeStepHasGap[] vector instead)
        vector<bool> m_timeStepHasGap;                                      // [t]

        // A boolean flag indicating whether the MBLayout can be further modified
        // When it's value is false, no set operations are allowed on the MBLayout
        mutable bool m_writable;

        // Cached mask indicating the validity of each column in the MBLayout
        // TODO: We actually just need a boolean matrix for this.
        // A value of 1 indicates that the column has valid content 
        // and 0 indicates invalid (aka MinibatchPackingFlags::NoInput)
        // If the matrix is empty it means all columns are valid
        mutable Matrix<char> m_columnsValidityMask;

    public:
#if 1
        pair<Matrix<float>, MinibatchPackingFlags> GetFrame(size_t) const { NOT_IMPLEMENTED; }
        Matrix<float> & GetM() { NOT_IMPLEMENTED; }
#else
        // specialized functions to replicate old behavior that shouldn't be there but I cannot test
        // TODO: these should all go away one day

        // get info for one frame; used in DelayedValueNode
        // TODO: clean this up, we can do this more nicely. DelayedValueNode can just access individual elements, like everybody else.
        pair<Matrix<float>, MinibatchPackingFlags> GetFrame(size_t t) const
        {
            return make_pair(m_sentenceBoundaryFlags.ColumnSlice(t, 1), m_minibatchPackingFlags[t]);
        }
        // for LSTMNode ony, which is deprecated, only to make it compile easily:  also used in FindBestPathWithVariableLength() and FindBestPath() in a strange way
        Matrix<float> & GetM() { return m_sentenceBoundaryFlags; }
#endif

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

    // TODO: We should also have a FrameRange that selects all frames of a single sequence. Currently now possible since that would require Matrix::RowSlice()
    // TODO: Where this design currently breaks:  // <- BUGBUG: I think these are outdated
    //  - BatchModeNodes must access GetNumParallelSequences(), yet operate on the whole sequence
    //  - likewise, LSTMNode does its own iteration, hence needs access to GetNumParallelSequences() or NumCols() in the whole-batch iterator
    // BUGBUG: These nodes are currently broken and will need to be fixed:
    //  - CRFNode does not support > 1 parallel sequence
    class FrameRange
    {
    public: // TODO: make private (currently used from masking and DataFor) ; TODO: rename all members with m_ prefix
        size_t timeIdxInSeq;                // start frame; SIZE_MAX = all frames in MB
        ptrdiff_t m_timeOffset;             // this is added to timeIdxInSeq wherever it is used
        size_t seqIndex;                    // sequence index; SIZE_MAX = all sequences in MB (most common case)
        MBLayoutPtr m_pMBLayout;            // layout associated with this
        bool m_broadcastAllowed;            // frame range may be broadcast from outer layout (e.g. a matrix with NULL layout and 1 column is acceptable to this frame range)
        const FrameRange *parent;           // or NULL: parent range, relative to which this FrameRange is interpreted  --TODO: not used yet

    public:
        // can construct from a single size_t -> a single-frame range
        FrameRange(MBLayoutPtr pMBLayout, size_t timeIdxInSeq) : timeIdxInSeq(timeIdxInSeq), m_timeOffset(0), seqIndex(SIZE_MAX), m_pMBLayout(pMBLayout), m_broadcastAllowed(false), parent(nullptr) {}

        // or without arguments -> entire minibatch / no frame-range
        //FrameRange(MBLayoutPtr pMBLayout) : timeIdxInSeq(SIZE_MAX), seqIndex(SIZE_MAX), m_pMBLayout(pMBLayout), parent(nullptr) {}
        FrameRange(MBLayoutPtr pMBLayout) : FrameRange(pMBLayout, SIZE_MAX) {}

        // return a frame range with broadcast allowed
        // This is used, e.g., by PlusNode which can combine minibatch data and single-column vectors.
        FrameRange AllowBroadcast() const
        {
            if (seqIndex != SIZE_MAX)
                LogicError("FrameRange::AllowBroadcast() is incompatible with frame ranges that select a single sequence.");
            FrameRange ret = *this;
            ret.m_broadcastAllowed = true;
            return ret;
        }

        // create a FrameRange that accesses a single sequence only
        // FrameRange(t).Sequence(seq)
        FrameRange Sequence(size_t s) const
        {
            if (m_broadcastAllowed)
                LogicError("FrameRange::Sequence() is incompatible with frame ranges with m_broadcastAllowed.");
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
        // Note: This currently does not work in conjunction with IsAllFrames(). This would be a nice-to have, but tricky w.r.t. out-of-bounds accesses.
        FrameRange WithTimeOffset(ptrdiff_t offset) const
        {
            FrameRange ret = *this;
            ret.m_timeOffset += offset;
            return ret;
        }
        // check a FrameRange with time offset
        // Returns 0 if time index with offset is inside the layout; -1 if left of it, and +1 if right of it.
        int LocateTimeOffset() const
        {
            EnsureNotAllFrames();
            if (m_timeOffset == 0)      // no time offset: the time index itself must always be inside the actual MB
                return 0;
            if (!m_pMBLayout)
                InvalidArgument("FrameRange::LocateTimeOffset(): Time offset requires an MBLayout.");
            ptrdiff_t t = m_timeOffset + (ptrdiff_t)timeIdxInSeq;
            if (t < 0)
                return -1;
            else if ((size_t)t >= m_pMBLayout->GetNumTimeSteps())
                return +1;
            else
                return 0;
        }

        class IndexIteration    // range for range-based for over sequences
        {
            size_t m_beginIndex, m_endIndex;
        public:
            IndexIteration(size_t beginIndex, size_t endIndex) : m_beginIndex(beginIndex), m_endIndex(endIndex) { }
            size_t begin() const { return m_beginIndex; }
            size_t   end() const { return m_endIndex; }
        };
        IndexIteration GetSequenceRange(const shared_ptr<MBLayout> & pMBLayout) const { return IndexIteration(seqIndex == SIZE_MAX ? 0 : seqIndex, seqIndex == SIZE_MAX ? pMBLayout->GetNumParallelSequences() : seqIndex + 1); }

        // code that can only handle single-frame ranges will call t() to get the time index, which will throw if numFrames != 1
        // Some functions need just the time index, e.g. for looking up stuff in m_boundaryInfo. That's where an unscaled index is needed.
        // Really only used in RecurrentNodes(), where it will be replaced by FrameRange::WithDelay() which allows to access delayed frames through the FrameRange object.
        size_t t() const
        {
            EnsureNotAllFrames();
            ptrdiff_t t = m_timeOffset + (ptrdiff_t)timeIdxInSeq;
            if (LocateTimeOffset() != 0)
                InvalidArgument("FrameRange::t(): Time offset caused time index to be out of range.");
            return (size_t)t;
        }

        bool IsAllFrames() const { return timeIdxInSeq == SIZE_MAX; } // if true then above functions may not be called; caller must use entire batch instead (PAR mode)

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

    inline bool MBLayout::HasGaps() const { return m_numGapFrames > 0; /*HasGaps(FrameRange());*/ }
    inline bool MBLayout::HasGaps(const FrameRange & fr) const
    {
        CheckIsValid();
        if (fr.IsAllFrames())
            return m_numGapFrames > 0;      // test entire minibatch
        auto t = fr.t();
        if (fr.seqIndex == SIZE_MAX)
            return m_timeStepHasGap[t];     // test all seq for one time step
        else
            return Is(fr.seqIndex, t, MinibatchPackingFlags::NoInput);  // test one sequence
    }

    inline MinibatchPackingFlags MBLayout::Get(size_t t) const
    {
        return Get(FrameRange(nullptr/*shared_from_this(); stop-gap, not really correct but won't hurt, until this whole function goes away*/, t));
    }
    inline MinibatchPackingFlags MBLayout::Get(size_t s, size_t t) const
    {
        return Get(FrameRange(nullptr/*shared_from_this(); stop-gap, not really correct but won't hurt, until this whole function goes away*/, t).Sequence(s));
    }

    // get packing flags from a frame range
    // TODO: Can we always use this, and make the ones taking a time index private or absorb them here?
    // TODO: This generic function will soon no longer be used; instead, tests for gap and boundary will be separate calls.
    // BUGBUG: broken for time offsets, since off by one. Use IsBeyondStartOrEnd() instead.
    inline MinibatchPackingFlags MBLayout::Get(const FrameRange & fr) const
    {
        CheckIsValid();

        if (fr.IsAllFrames())
            LogicError("MBLayout::Get() cannot be applied to FrameRange that specifies more than a single time step.");

        MinibatchPackingFlags f = MinibatchPackingFlags::None;
        auto t = fr.timeIdxInSeq;        // we test off the frame without offset
        auto s = fr.seqIndex;
        if (s == SIZE_MAX)              // aggregate requested
        {
            // determine flags from aggregate vectors
            if (m_timeStepHasGap[t])      // indicates a gap
                f |= MinibatchPackingFlags::NoInput;
            auto distanceToStart = (ptrdiff_t)m_distanceToNearestStart[t];
            if (distanceToStart <= -fr.m_timeOffset)
                f |= MinibatchPackingFlags::SequenceStart;
            auto distanceToEnd = (ptrdiff_t)m_distanceToNearestEnd[t];
            if (distanceToEnd <= fr.m_timeOffset)
                f |= MinibatchPackingFlags::SequenceEnd;

            auto f1 = m_minibatchPackingFlags[t];
            assert(f1 == f); f1;

            return f;
        }

        // determine flags from matrices

        auto distanceToStart = (ptrdiff_t)m_distanceToStart(s, t);
        if (distanceToStart == -1)      // indicates a gap
        {
            assert(m_timeStepHasGap[t]);
            f |= MinibatchPackingFlags::NoInput;
        }
        else
        {
            if (distanceToStart <= -fr.m_timeOffset)
                f |= MinibatchPackingFlags::SequenceStart;
            auto distanceToEnd = (ptrdiff_t)m_distanceToEnd(s, t);
            if (distanceToEnd <= fr.m_timeOffset)
                f |= MinibatchPackingFlags::SequenceEnd;
        }

        auto f1 = (MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(s, t);
        assert(f1 == f); f1;

        return f;
    }

    // test whether frame is exceeding the sentence boundaries
    inline bool MBLayout::IsBeyondStartOrEnd(const FrameRange & fr) const
    {
        CheckIsValid();

        if (fr.IsAllFrames())
            LogicError("MBLayout::IsBeyondStartOrEnd() cannot be applied to FrameRange that specifies more than a single time step.");

        auto t = fr.timeIdxInSeq;        // we test off the frame without offset
        auto s = fr.seqIndex;
        if (s == SIZE_MAX)              // aggregate requested
        {
            // determine flags from aggregate vectors
            auto distanceToStart = (ptrdiff_t)m_distanceToNearestStart[t];
            if (distanceToStart < -fr.m_timeOffset)
                return true;
            auto distanceToEnd = (ptrdiff_t)m_distanceToNearestEnd[t];
            if (distanceToEnd < fr.m_timeOffset)
                return true;
            return false;
        }

        // determine flags from matrices
        auto distanceToStart = (ptrdiff_t)m_distanceToStart(s, t);
        if (distanceToStart == -1)      // indicates a gap
        {
            assert(m_timeStepHasGap[t]);
            return false;   // a gap is not outside, so that we can allow collating
        }
        else
        {
            if (distanceToStart < -fr.m_timeOffset)
                return true;
            auto distanceToEnd = (ptrdiff_t)m_distanceToEnd(s, t);
            if (distanceToEnd < fr.m_timeOffset)
                return true;
        }
        return false;
    }

    // return m_columnsValidityMask(,), which is lazily created here upon first call
    // only called from MaskMissingColumnsTo()
    // TODO: Can probably be faster by using the sequence array directly.
    inline const Matrix<char> & MBLayout::GetColumnsValidityMask(DEVICEID_TYPE deviceId) const
    {
        CheckIsValid();
        // lazily compute the validity mask
        if (m_columnsValidityMask.IsEmpty())
        {
            assert(HasGaps());  // must only be called if there are gaps
            Lock();

            // Determine indices of all invalid columns in the minibatch
            // TODO: This can be done more efficiently by using m_sequences[].
            size_t nT = GetNumTimeSteps();
            size_t nS = GetNumParallelSequences();

            std::vector<char> columnsValidityMask(nT * nS, 1);  // form the mask in a CPU-side STL vector first
            size_t gapsFound = 0;
            for (size_t t = 0; t < nT; t++)
            {
                if (IsGap(t))
                {
                    for (size_t s = 0; s < nS; s++)
                    {
                        if (IsGap(s, t))
                        {
                            columnsValidityMask[(t * nS) + s] = 0;
                            gapsFound++;
                        }
                    }
                }
            }
            assert(gapsFound == m_numGapFrames);    // sanity check

            if (deviceId != m_columnsValidityMask.GetDeviceId())
                m_columnsValidityMask = Matrix<char>(deviceId);
            m_columnsValidityMask.SetValue(1, nS * nT, deviceId, columnsValidityMask.data());
        }
        return m_columnsValidityMask;
    }

    // class for defining an iteration over a sequence
    // Currently supports time sequences, forward and backward.
    // TODO: It is meant to some day generalize to multi-dimensional iterations, e.g. across an image:
    //  - abstract delay direction to be multi-dimensional (let's call it FrameStep)
    //  - DelayedValueNode::direction gets replaced with a FrameStep
    //  - recInfo->m_steppingDirection will be replaced by a FrameStep
    //  - FrameRangeIterator derives from FrameStep, and operator++ adds tat to FrameRange
    // Longer-term, we will also have nested structures. For those, FrameRangeIterations will be able to be instantiated from FrameRange objects to loop over their nested dimension.
    class FrameRangeIteration
    {
        MBLayoutPtr m_pMBLayout;
        int m_step;
    public:
        // one-dimensional iteration (time sequences)
        // 'Step' specifies the stepping direction of the loop:
        //  - for left-to-right models -> pass step = +1
        //  - for right-to-left models -> pass step = -1
        FrameRangeIteration(MBLayoutPtr pMBLayout, int step) : m_pMBLayout(pMBLayout), m_step(step) { }
        // in the future we may consier multi-dimensional iterators such as iterators over images
        // This class is returned by begin() and end().
        // It is a FrameRange with additions ++ and != operators needed in the for loop.
        class FrameRangeIterator : public FrameRange
        {
            ptrdiff_t m_step;
        public:
            FrameRangeIterator(const FrameRange & begin, ptrdiff_t step) : FrameRange(begin), m_step(step) { }
            bool operator!=(const FrameRangeIterator & other) const { return timeIdxInSeq != other.timeIdxInSeq; }
            void operator++(int) { timeIdxInSeq = (size_t)(m_step + (ptrdiff_t)timeIdxInSeq); }    // going through (int) to avoid undefined behavior
        };
        // iterators for iterating forward
        FrameRangeIterator begin() const
        {
            if (m_step > 0) return FrameRangeIterator(FrameRange(m_pMBLayout, 0),                                  +1);
            else            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps() - 1), -1);
        }
        FrameRangeIterator end() const
        {
            if (m_step < 0) return FrameRangeIterator(FrameRange(m_pMBLayout, (size_t)-1),                     0/*dummy*/);
            else            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps()), 0);
        }
        // iterators for iterating in reverse order (as needed for gradient update)
        FrameRangeIterator rbegin() const
        {
            if (m_step < 0) return FrameRangeIterator(FrameRange(m_pMBLayout, 0),                                  +1);
            else            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps() - 1), -1);
        }
        FrameRangeIterator rend() const
        {
            if (m_step > 0) return FrameRangeIterator(FrameRange(m_pMBLayout, (size_t)-1),                     0);
            else            return FrameRangeIterator(FrameRange(m_pMBLayout, m_pMBLayout->GetNumTimeSteps()), 0);
        }
    };

    // -----------------------------------------------------------------------
    // ColumnRangeWithMBLayoutFor() -- Return column range for a FrameRange of a Matrix with specified number of columns with a given MBLayout
    // -----------------------------------------------------------------------
    static inline std::pair<size_t, size_t> ColumnRangeWithMBLayoutFor(size_t numCols, 
                                                                       const FrameRange & fr/*select frame or entire batch*/,
                                                                       const MBLayoutPtr & pMBLayout/*the MB layout of 'data'*/)
    {
        // MBLayout of data and of FrameRange must be identical pointers,
        // or in case of broadcasting, respective parent pointers.
        // MBLayouts that are identical in content but not object identity (pointer) are not admissible.
        // For those cases, use a ReconcileMBLayout node.
        if (fr.m_pMBLayout != pMBLayout)
        {
            // if broadcast allowed then it is allowed to broadcast from an outer-loop value
            // Currently, the only 'outer' loop we have is to have no layout.
            if (fr.m_broadcastAllowed && !pMBLayout && numCols == 1)
                return std::pair<size_t, size_t>(0, numCols);
            if (fr.m_pMBLayout && pMBLayout && *fr.m_pMBLayout == *pMBLayout)
                LogicError("DataFor: fr's MBLayout inconsistent with matrix. They are compatible though--are you missing a ReconcileMBLayout operation?");
            else
                LogicError("DataFor: fr's MBLayout inconsistent with matrix");
        }
        // if FrameRange refers to whole minibatch (map mode)
        // or if we don't even have a layout
        // then return the whole matrix
        // but as a reference (e.g. it cannot be resized)
        if (!pMBLayout || fr.IsAllFrames())
        {
            if (fr.seqIndex == SIZE_MAX)
                return std::pair<size_t, size_t>(0, numCols);
            else
            {
                if (!pMBLayout)
                    LogicError("DataFor: Attempting to retrieve a parallel sequence from data without layout.");
#if 1
                else
                    LogicError("DataFor: To retrieve a parallel sequence, implement Matrix::RowSlice() first!");
#else
                // get a reshaped view that stacks all sequences into T long vectors
                auto mat = data.ColumnSlice(0, data.GetNumCols());
                mat.Resize(data.GetNumRows() * pMBLayout->GetNumParallelSequences(), data.GetNumRows() / pMBLayout->GetNumParallelSequences());
                return mat;   // .RowSlice(fr.seqIndex * data.GetNumRows());
                // TODO: Why does RowSlice() not exist? Seems simple. Is there a hidden assumption of contiguous memory?#endif
#endif
            }
        }
        // FrameRange refers to a time slice -> return that
        else
        {
            size_t numParallelSequences = pMBLayout->GetNumParallelSequences();
            size_t startColumn = fr.t() * numParallelSequences;
            if (fr.seqIndex == SIZE_MAX)
                return std::pair<size_t, size_t>(startColumn, numParallelSequences);
            else
                return std::pair<size_t, size_t>(startColumn + fr.seqIndex, 1);
        }
    }

    // -----------------------------------------------------------------------
    // DataWithMBLayoutFor() -- create view for a FrameRange of a Matrix with a given MBLayout
    // This function binds the above together.
    // Any access by FrameRange should only be done through this function.
    // -----------------------------------------------------------------------

#if 0
    template<class ElemType>
    static inline Matrix<ElemType> DataWithMBLayoutFor(Matrix<ElemType> & data,
                                                       const FrameRange & fr/*select frame or entire batch*/,
                                                       const MBLayoutPtr & pMBLayout/*the MB layout of 'data'*/)
    {
        auto columnRange = ColumnRangeWithMBLayoutFor(data.GetNumCols(), fr, pMBLayout);
        //if ((columnRange.first == 0) && (columnRange.second == data.GetNumCols()))
        //    return data.AsReference();
        return data.ColumnSlice(columnRange.first, columnRange.second);
    }

    // const version (100% dup except the input type)
#endif
    template<class ElemType>
    static inline Matrix<ElemType> DataWithMBLayoutFor(const Matrix<ElemType> & data,
                                                       const FrameRange & fr/*select frame or entire batch*/,
                                                       const MBLayoutPtr & pMBLayout/*the MB layout of 'data'*/)
    {
        auto columnRange = ColumnRangeWithMBLayoutFor(data.GetNumCols(), fr, pMBLayout);
        //if ((columnRange.first == 0) && (columnRange.second == data.GetNumCols()))
        //    return data.AsReference();
        return data.ColumnSlice(columnRange.first, columnRange.second);
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
    template<class ElemType>
    static inline void MaskMissingColumnsTo(Matrix<ElemType>& matrixToMask, const MBLayoutPtr & pMBLayout, const FrameRange & fr, ElemType val)
    {
        if (pMBLayout && pMBLayout->HasGaps(fr))
        {
            const auto & maskMatrix = pMBLayout->GetColumnsValidityMask(matrixToMask.GetDeviceId());
            auto maskSlice          = DataWithMBLayoutFor(maskMatrix,   fr, pMBLayout);
            auto matrixSliceToMask  = DataWithMBLayoutFor(matrixToMask, fr, pMBLayout);
            matrixSliceToMask.MaskColumnsValue(maskSlice, val);
        }
    }

}}}
