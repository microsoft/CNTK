// Sequences.h -- all about iterating over sequences, that is, MBLayout, FrameRange, and iterators
//
// <copyright file="Sequences.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

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

    NoInput = NoFeature | NoLabel,  // when we refactorize reader, NoInput will no longer needed
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

    // -----------------------------------------------------------------------
    // MBLayout -- layout information of minibatch
    //
    // Minibatches are collections of one or more sequences, laid out in a way to
    // allow to process one time step for multiple sequences in parallel in shared CUDA launches.
    //
    // This is achieved by interleaving storage. If f(s,t) denotes a frame of sequence s at time t,
    // the minibatch matrix would contain this:
    //   f(0,0) f(1,0) ... f(0,1) f(1,1) ...
    // In the special case of frame randomization, every frame is stored as a single-frame sequence.
    // Much of CNTK's superior efficiency comes from this.
    //
    // Sequences can also be concatenated, to fill the space better. For this case,
    // this object stores about every frame whether it is at the start or end of a sequence.
    // Frames can also be invalid, due to gaps (not enough space to concatenate another sequence at the end)
    // and due to invalid input data (e.g. a speech utterance for which no alignment could be generated).
    //
    // An MBLayout object stores:
    //  - number of time steps and parallel sequences (their product is equal to the #columns in the minibatch matrix)
    //  - whether the data is sequential or not
    //  - information for (every time step, every parallel sequence) MinibatchPackingFlags for every (sequence, time step)
    //  - a column-wise OR of those flags for fast testing entire time steps at once
    // -----------------------------------------------------------------------

    // This object allocates its storage lazily, i.e. if there are no flags ever set, no memory is allocated. This is transparent to the caller.
    // Note: With truncated BPTT, it is possible to have sequential data, yet not a single flag set in a minibatch (if all frames are sequence-internal ranges).
    // Contract between ComputationNode, ComputationNetwork, and MBLayout:
    //  - if a node has no MBLayout, m_{function,gradient}Values are not samples (they are not activations or input data), but e.g. model parameters
    //  - ComputationNode::GetNumCols() == MBLayout::GetNumTimeSteps() * MBLayout::GetNumParallelSequences()
    //  - ComputationNetwork ensures that m_{function,gradient}Values are allocated correctly before calling EvaluateThisNode() on a node
    // TODO: move this to an appropriate place and name it properly. This class has no relationship with Matrix
    // NOTE: This class represents an ongoing abstraction of an originally distributed/code-duped way of defining and accessing the MB layout.
    //       Some code below represents the actual use cases I encountered. Not all are, I believe, needed to be as they are; this class could be simplified/streamlined much further.
    //       Some wackiness below is explained by this.

    struct MBLayout
    {
        typedef std::shared_ptr<MBLayout> MBLayoutPtr;

        MBLayout() : m_sentenceBoundaryFlags(CPUDEVICE) { Init(1, 0, false); }
        // TODO: ^^ use forwarding constructor to this guy vv, or default args
        MBLayout(size_t numParallelSequences, size_t numTimeSteps, bool dataIsSequential) : m_sentenceBoundaryFlags(CPUDEVICE) { Init(numParallelSequences, numTimeSteps, dataIsSequential); }

        // copy the content of another MBLayoutPtr over
        // Use this instead of actual assignment to make it super-obvious that this is not copying the pointer but actual content. The pointer is kept fixed.
        void CopyFrom(const MBLayoutPtr & other) { *this = *other; }
        void MoveFrom(MBLayoutPtr other) { *this = move(*other); other->Init(0, 0, false); }    // destructive copy that steals ownership if the content, like std::move()
    private:
        MBLayout & operator=(const MBLayout &) = default;   // make this private --use CopyFrom() instead, which makes it very clear that it's copying content, not copying the reference
    public:

        // resize and reset all frames to None (note: this is an invalid state and must be fixed by caller afterwards)
        void Init(size_t numParallelSequences, size_t numTimeSteps, bool /*dataIsSequentialDummy*/ = true/*no longer needed*/)
        {
            // remember the dimensions..
            m_numParallelSequences = numParallelSequences;
            m_numTimeSteps = numTimeSteps;
            //m_dataIsSequential = dataIsSequential;
            // ...but don't actually allocate anything
            m_sentenceBoundaryFlags.Resize(0, 0);
            m_minibatchPackingFlags.clear();
            m_sequences.clear();
            m_writable = true;
        }

        size_t GetNumTimeSteps()         const { return m_numTimeSteps; }
        size_t GetNumParallelSequences() const { return m_numParallelSequences; }   // note: if initialized as a dummy, m_numParallelSequences is set to 1

        // how many columns the MB should be allocated for
        size_t GetNumCols()              const { return GetNumTimeSteps() * GetNumParallelSequences(); }

    private:
        // test whether we have not allocated anything (will also return true if the minibatch is empty)
        bool IsEmpty() const { return m_minibatchPackingFlags.empty(); }
        // call this before ever writing anything--this will create the matrix/vector upon first use
        void LazyAlloc() const
        {
            if (!IsEmpty() || m_numTimeSteps == 0)
                return;
            // this is where the actual allocation happens
            m_sentenceBoundaryFlags.Resize(m_numParallelSequences, m_numTimeSteps);
            m_sentenceBoundaryFlags.SetValue((float)((int)MinibatchPackingFlags::None));
            m_minibatchPackingFlags.assign(m_sentenceBoundaryFlags.GetNumCols(), MinibatchPackingFlags::None);
        }
    public:

        // compare whether two layouts are the same
        bool operator==(const MBLayout & other) const
        {
            // for now just check the object identity
            if (this == &other)
                return true;
            return          m_numTimeSteps == other.m_numTimeSteps &&
                    m_numParallelSequences == other.m_numParallelSequences &&
                   m_minibatchPackingFlags == other.m_minibatchPackingFlags &&
            m_sentenceBoundaryFlags.IsEqualTo(other.m_sentenceBoundaryFlags);
        }
        bool operator!=(const MBLayout & other) const { return !(*this == other); } // duh

        // get boundary flags
        MinibatchPackingFlags Get(size_t t) const { return IsEmpty() ? MinibatchPackingFlags::None : m_minibatchPackingFlags[t]; }
        MinibatchPackingFlags Get(size_t id, size_t t) const { return IsEmpty() ? MinibatchPackingFlags::None : (MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t); }

        // test boundary flags for a specific condition
        bool Is(size_t t, MinibatchPackingFlags f) const { return (Get(t) & f) != 0; }
        bool Is(size_t id, size_t t, MinibatchPackingFlags f) const { return (Get(id, t) & f) != 0; }
        // TODO: swap id and t for all of these functions; t is the more important parameter

        // tests if Is() is false for every frame and sequence
        // If this returns true, it means that boundary information need not be considered, just process the whole thing in one go.
        // TODO: Can it ever happen that no flag is set, yet we have m_numParallelSequences != 1? Or does that simply not matter?
        // This is currently the case for frame randomization.
        bool IsAllNone() const { return IsEmpty(); }

        // set a boundary flag (OR it on top of the existing layout)
        void Set(size_t s, size_t t, MinibatchPackingFlags f)
        {
            CheckWritable();

            if (f == MinibatchPackingFlags::None)   // actually not setting anything: skip allocation
                return;
            //if ((f & (MinibatchPackingFlags::SequenceStart | MinibatchPackingFlags::SequenceEnd)) && !m_dataIsSequential)
            //    LogicError("MBLayout::Set: attempted to set SequenceStart or -End in a layout with !m_dataIsSequential");
            LazyAlloc();
            m_sentenceBoundaryFlags.SetValue(s, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(s, t)) | f));
            m_minibatchPackingFlags[t] |= f;
        }

        // mark a range of frames in a parallel sequence as one sentence
        // Note that endTime is the last frame +1. Think of begin/end as used in STL containers.
        void SetAsSentence(size_t s, size_t beginTime, size_t endTime)
        {
            Set(s, beginTime, MinibatchPackingFlags::SequenceStart);
            Set(s, endTime-1, MinibatchPackingFlags::SequenceEnd);
            // Note: It is assumed that this is being constructed after Init().
#ifdef _DEBUG
            for (size_t t = beginTime; t < endTime; t++)
            {
                assert(!Is(s, t, MinibatchPackingFlags::NoInput));
                assert(t == beginTime || !Is(s, t, MinibatchPackingFlags::SequenceStart));
                assert(t == endTime-1 || !Is(s, t, MinibatchPackingFlags::SequenceEnd));
            }
#endif
            AddSequence(beginTime, endTime, true);
        }

        // mark a range of frames in a parallel sequence as invalid
        void SetAsNoInput(size_t s, size_t beginTime, size_t endTime)
        {
            for (size_t t = beginTime; t < endTime; t++)
                Set(s, t, MinibatchPackingFlags::NoInput);
            AddSequence(beginTime, endTime, false);
        }

        // TODO: This can go away once frame mode returns multiple sequence sof one frame each; or we can test against cols==1
        // HAH! This function is only ever used for Decimate(). It can completely go away, as can methods of the same name in the readers!
        //bool RequireSentenceSeg() const { return m_dataIsSequential; }        // this is the name of a function on DataReader which really belongs here

        // compute the number of actual samples in this layout (not counting NoLabel ones)
        // This is used by MeanNode and InvStdDevNode.
        size_t DetermineActualNumSamples() const
        {
            size_t n = GetNumTimeSteps() * GetNumParallelSequences();
            if (!IsAllNone())
            {
                for (size_t t = 0; t < GetNumTimeSteps(); t++)
                {
                    if (Is(t, MinibatchPackingFlags::NoInput)) for (size_t s = 0; s < GetNumParallelSequences(); s++)
                    {
                        if (Is(s, t, MinibatchPackingFlags::NoInput))
                            n--;
                    }
                }
            }
            return n;
        }

        // test function for those pieces of the code that cannot handle gaps
        // TODO: Not efficient (linear scan). Use a global OR of all values.
        bool HasGaps() const
        {
            if (!IsAllNone())
                for (size_t t = 0; t < GetNumTimeSteps(); t++)
                    if (Is(t, MinibatchPackingFlags::NoInput))
                        return true;
            return false;
        }

    private:
        size_t m_numTimeSteps;
        size_t m_numParallelSequences;
        //bool m_dataIsSequential;
        // TODO: ^^ is m_dataIsSequential necessary? Who ues it?

        // TODO: rename the following two variables, or even implement it with a very different structure

        /// a matrix of n_stream x n_length
        /// n_stream is the number of streams
        /// n_length is the maximum lenght of each stream
        /// for example, two sentences used in parallel in one minibatch would be
        /// [2 x 5] if the max length of one of the sentences is 5
        /// the elements of the matrix is 0, 1, or -1, defined as ((int) MinibatchPackingFlags::SequenceStart), ((int) MinibatchPackingFlags::None), ((int) MinibatchPackingFlags::NoInput) in cbasetype.h 
        /// 0 1 1 0 1
        /// 1 0 1 0 0 
        /// for two parallel data streams. The first has two sentences, with 0 indicating begining of a sentence
        /// the second data stream has two sentences, with 0 indicating begining of sentences
        /// you may use 1 even if a sentence begins at that position, in this case, the trainer will carry over hidden states to the following
        /// frame. 
        mutable Matrix<float> m_sentenceBoundaryFlags;  // (t,stream)
        // ^^ float -> MinibatchPackingFlags, right? Or unsigned char; or change that to 'char' because Matrix<char> already exists
        // This matrix ^^ is always in CPU memory  --TODO: should rather be a matrix of some int
        /// conditionally point to either a pointer to that provided by network, or point to 
        /// an individual sentence boundary info, which happens if timeStep > 1 is required for PastValue node
        /// a matrix of 1 x n_length
        /// != 0 denotes the case that there exists sentence begin or no_labels case in this frame
        /// == 0 denotes such case is not in this frame
        mutable vector<MinibatchPackingFlags> m_minibatchPackingFlags;  // column-wise OR over m_sentenceBoundaryFlags for fast testing

        // A boolean flag indicating whether the MBLayout can be further modified
        // When it's value is false, no set operations are allowed on the MBLayout
        mutable bool m_writable;

        // Cached mask indicating the validity of each column in the MBLayout
        // TODO: We actually just need a boolean matrix for this.
        // A value of 1 indicates that the column has valid content 
        // and 0 indicates invalid (aka MinibatchPackingFlags::NoInput)
        // If the matrix is empty it means all columns are valid
        mutable std::shared_ptr<Matrix<char>> m_columnsValidityMask;

        // Ensure that the MBLayout allows writes
        void CheckWritable() const
        {
            if (!m_writable)
                LogicError("Modification attempted on a MBLayout that is no longer writable.");
        }

        // Freeze the MBLayout disallowing further modifications through set operations
        void Lock() const
        {
            m_writable = false;
        }

        // explicit list of sequences
        // Currently this is for diagnostics only, but in the future this will include utterance ids etc, meant for lining up inconsistent MB layouts.
        struct SequenceDesc
        {
            size_t tBegin, tEnd;
            bool hasData;           // false means it's a gap
        };
        mutable vector<SequenceDesc> m_sequences;
        void AddSequence(size_t b, size_t e, bool d)
        {
            m_sequences.push_back(SequenceDesc{ b, e, d });
        }

    public:
        // specialized functions to replicate old behavior that shouldn't be there but I cannot test
        // TODO: these should all go away one day

        // get info for one frame; used in DelayedValueNode
        // TODO: clean this up, we can do this more nicely. DelayedValueNode can just access individual elements, like everybody else.
        pair<Matrix<float>, MinibatchPackingFlags> GetFrame(size_t t) const
        {
            LazyAlloc();
            return make_pair(m_sentenceBoundaryFlags.ColumnSlice(t, 1), m_minibatchPackingFlags[t]);
        }

        // same as Set() but not ORing  --TODO: is this distinction needed?
        void SetWithoutOr(size_t id, size_t t, MinibatchPackingFlags f)
        {
            if (f == MinibatchPackingFlags::None)
                return;
            LazyAlloc();
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(int)f); // no OR
            m_minibatchPackingFlags[t] |= f;
        }
        // needed in DelayedValueNodeBase
        // TODO: this is wicked in that the matrix keeps only the NoLabel flag, while the vector keeps all (just gets ORed into)
        void Mask(size_t id, size_t t, MinibatchPackingFlags f)
        {
            if (IsEmpty())
                return;
            m_sentenceBoundaryFlags.SetValue(id, t, (float)(((MinibatchPackingFlags)(int)m_sentenceBoundaryFlags(id, t)) & f));
            //m_minibatchPackingFlags[t] &= f;
        }
        // for LSTMNode ony, which is deprecated, only to make it compile easily:  also used in FindBestPathWithVariableLength() and FindBestPath() in a strange way
        Matrix<float> & GetM() { LazyAlloc(); return m_sentenceBoundaryFlags; }

        // TODO: this function is only used in Kaldi2Reader for the moment, and
        //       we plan to remove it in the future. It copies the current
        //       MBLayout from an existing object but only copies <numTimeSteps>
        //       steps starting from <startTimeStep>.
        void CopyFromRange(const MBLayoutPtr & other, size_t startTimeStep, size_t numTimeSteps)
        {
            m_numParallelSequences = other->m_numParallelSequences;
            m_numTimeSteps = numTimeSteps;
            //m_dataIsSequential = other->m_dataIsSequential;
            m_sentenceBoundaryFlags.SetValue(other->m_sentenceBoundaryFlags.ColumnSlice(startTimeStep, numTimeSteps));
            m_minibatchPackingFlags.resize(numTimeSteps);
            m_minibatchPackingFlags.assign(
                other->m_minibatchPackingFlags.begin() + startTimeStep,
                other->m_minibatchPackingFlags.begin() + startTimeStep + numTimeSteps);
        }

        shared_ptr<Matrix<char>> GetColumnsValidityMask(const FrameRange& frameRange, DEVICEID_TYPE deviceId) const;
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
    // To access the subset of a minibatch matrix selected by FrameFange, use DataSliceWithMBLayout().
    //
    // TODO: This will in the future be able to hold sub-ranges for nested loops as well.
    // -----------------------------------------------------------------------

    // TODO: We should also have a FrameRange that selects all frames of a single sequence. Currently now possible since that would require Matrix::RowSlice()
    // TODO: Where this design currently breaks:  // <- BUGBUG: I think these are outdated
    //  - BatchModeNodes must access GetNumParallelSequences(), yet operate on the whole sequence
    //  - likewise, LSTMNode does its own iteration, hence needs access to GetNumParallelSequences() or NumCols() in the whole-batch iterator
    // BUGBUG: These are currently broken and will need to be fixed:
    //  - CRFNode does not support > 1 parallel sequence
    class FrameRange
    {
    public: // TODO: fix this (currently used from masking and DataSlice)
        size_t timeIdxInSeq;                // start frame; SIZE_MAX = all frames in MB
        size_t seqIndex;                    // sequence index; SIZE_MAX = all sequences in MB (most common case)
        MBLayoutPtr m_pMBLayout;            // layout associated with this
        bool m_broadcastAllowed;            // frame range may be broadcast from outer layout (e.g. a matrix with NULL layout and 1 column is acceptable to this frame range)
        const FrameRange *parent;           // or NULL: parent range, relative to which this FrameRange is interpreted  --TODO: not used yet

    public:
        // can construct from a single size_t -> a single-frame range
        FrameRange(MBLayoutPtr pMBLayout, size_t timeIdxInSeq) : timeIdxInSeq(timeIdxInSeq), seqIndex(SIZE_MAX), m_pMBLayout(pMBLayout), m_broadcastAllowed(false), parent(nullptr) {}

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
        // Some functions need just the time index, e.g. for looking up stuff in m_boundaryInfo. That's where an unscaled index is needed (as opposed to startColumn()).
        size_t t() const { EnsureNotAllFrames(); return timeIdxInSeq; }
        // multi-frame slice case: these two get startFrame and numFrames
        //size_t StartColumn() const { EnsureNotAllFrames(); return timeIdxInSeq * samplesInRecurrentStep; }
        //size_t NumCols() const { EnsureNotAllFrames(); return samplesInRecurrentStep; }
        // TODO: remove these ^^ two in favor of these vv
        size_t StartColumn(const shared_ptr<MBLayout> & pMBLayout) const { EnsureNotAllFrames(); return timeIdxInSeq * pMBLayout->GetNumParallelSequences(); }
        size_t NumCols(const shared_ptr<MBLayout> & pMBLayout) const { EnsureNotAllFrames(); return pMBLayout->GetNumParallelSequences(); }
        bool IsAllFrames() const { return timeIdxInSeq == SIZE_MAX; } // if true then above functions may not be called; caller must use entire batch instead (PAR mode)

        const FrameRange & Check(size_t expectedStartColumn, size_t expectedNumCols, const shared_ptr<MBLayout> & pMBLayout) const
        {
            if (!IsAllFrames() && (expectedStartColumn != StartColumn(pMBLayout) || expectedNumCols != NumCols(pMBLayout)))
                LogicError("FrameRange::Check: FrameRange object gives different range than original explicit code. Logic is borked.");
            return *this;
        }
        const FrameRange & Check_t(size_t expectedNumCols, const shared_ptr<MBLayout> & pMBLayout) const
        {
#if 1       // temporary workaround
            if (expectedNumCols == SIZE_MAX || !pMBLayout)
                return *this;
#endif
            if (!IsAllFrames())
                Check(t() * expectedNumCols, expectedNumCols, pMBLayout);
            return *this;
        }
    private:
        void EnsureNotAllFrames() const
        {
            if (IsAllFrames())
                LogicError("FrameRange::t() called when frame range refers to whole minibatch");
        }
    };

    inline shared_ptr<Matrix<char>> MBLayout::GetColumnsValidityMask(const FrameRange& frameRange, DEVICEID_TYPE deviceId) const
    {
        // lazily compute the validity mask
        if (m_columnsValidityMask == nullptr)
        {
            Lock();
            m_columnsValidityMask.reset(new Matrix<char>(deviceId));

            // Determine indices of all invalid columns in the specified frameRange
            if (!IsAllNone())       // TODO: use HasGaps() (but currently that would mean a second linear scan, which is not efficient)
            {
                size_t nT = GetNumTimeSteps();
                size_t nS = GetNumParallelSequences();

                std::vector<char> columnsValidityMask(nT * nS, 1);  // form the mask in a CPU-side STL vector first
                bool foundInvalidColumn = false;
                for (size_t t = 0; t < nT; t++)
                {
                    if (Is(t, MinibatchPackingFlags::NoInput))
                    {
                        for (size_t s = 0; s < nS; s++)
                        {
                            if (Is(s, t, MinibatchPackingFlags::NoInput))
                                columnsValidityMask[(t * nS) + s] = 0;
                        }

                        foundInvalidColumn = true;
                    }
                }

                if (foundInvalidColumn)                     // if any then blast it over to the GPU side
                    m_columnsValidityMask->SetValue(1, columnsValidityMask.size(), deviceId, columnsValidityMask.data());
            }
        }

        if (m_columnsValidityMask->IsEmpty())               // mask matrix was kept empty, which means no gaps detected
            return nullptr;

        // we have a validity mask: decide what to return
        if (frameRange.IsAllFrames())
            return m_columnsValidityMask;

        // Check if there are any invalid frames in the specified frameRange
        bool foundInvalidColumnsInRange = false;
        if (frameRange.seqIndex == SIZE_MAX)
        {
            foundInvalidColumnsInRange = Is(frameRange.t(), MinibatchPackingFlags::NoInput);
        }
        else
        {
            foundInvalidColumnsInRange = Is(frameRange.seqIndex, frameRange.t(), MinibatchPackingFlags::NoInput);
        }

        if (!foundInvalidColumnsInRange)
            return nullptr;

        // we get here if there is an actual validity mask and there are invalid frames in its range
        size_t startColumn = (frameRange.t() * GetNumParallelSequences()) + ((frameRange.seqIndex == SIZE_MAX) ? 0 : frameRange.seqIndex);
        size_t numColumns = (frameRange.seqIndex == SIZE_MAX) ? GetNumParallelSequences() : 1;

        // TODO: why use ColumnSlice() and not DataSlice()?
        return make_shared<Matrix<char>>(m_columnsValidityMask->ColumnSlice(startColumn, numColumns));
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
    // DataSliceWithMBLayout() -- create view for a FrameRange of a Matrix with a given MBLayout
    // This function binds the above together.
    // Any access by FrameRange should only be done through this function.
    // -----------------------------------------------------------------------

    template<class ElemType>
    static inline Matrix<ElemType> DataSliceWithMBLayout(Matrix<ElemType> & data,
                                                         const FrameRange & frameRange/*select frame or entire batch*/,
                                                         const MBLayoutPtr & pMBLayout/*the MB layout of 'data'*/)
    {
        // MBLayout of data and of FrameRange must be identical pointers,
        // or in case of broadcasting, respective parent pointers.
        // MBLayouts that are identical in content but not object identity (pointer) are not admissible.
        // For those cases, use a ReconcileMBLayout node.
        if (frameRange.m_pMBLayout != pMBLayout)
        {
            // if broadcast allowed then it is allowed to broadcast from an outer-loop value
            // Currently, the only 'outer' loop we have is to have no layout.
            if (frameRange.m_broadcastAllowed && !pMBLayout && data.GetNumCols() == 1)
                return data.AsReference();
            if (frameRange.m_pMBLayout && pMBLayout && *frameRange.m_pMBLayout == *pMBLayout)
                LogicError("DataSlice: frameRange's MBLayout inconsistent with matrix. They are compatible though--are you missing a ReconcileMBLayout operation?");
            else
                LogicError("DataSlice: frameRange's MBLayout inconsistent with matrix");
        }
        // if FrameRange refers to whole minibatch (map mode)
        // or if we don't even have a layout
        // then return the whole matrix
        // but as a reference (e.g. it cannot be resized)
        if (!pMBLayout || frameRange.IsAllFrames())
        {
            if (frameRange.seqIndex == SIZE_MAX)
                return data.AsReference();
            else
            {
                if (!pMBLayout)
                    LogicError("DataSlice: Attempting to retrieve a parallel sequence from data without layout.");
#if 1
                else
                    LogicError("DataSlice: To retrieve a parallel sequence, implement Matrix::RowSlice() first!");
#else
                // get a reshaped view that stacks all sequences into T long vectors
                auto mat = data.ColumnSlice(0, data.GetNumCols());
                mat.Resize(data.GetNumRows() * pMBLayout->GetNumParallelSequences(), data.GetNumRows() / pMBLayout->GetNumParallelSequences());
                return mat;   // .RowSlice(frameRange.seqIndex * data.GetNumRows());
                // TODO: Why does RowSlice() not exist? Seems simple. Is there a hidden assumption of contiguous memory?#endif
#endif
            }
        }
        // FrameRange refers to a time slice -> return that
        else
        {
            size_t numParallelSequences = pMBLayout->GetNumParallelSequences();
            size_t startColumn = frameRange.t() * numParallelSequences;
            if (frameRange.seqIndex == SIZE_MAX)
                return data.ColumnSlice(startColumn, numParallelSequences);
            else
                return data.ColumnSlice(startColumn + frameRange.seqIndex, 1);
        }
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
    // It is indirectly guarded by the m_maskMissingColumnsToZero flag, which, if false, will install a layout with IsAllNone() to be true. TODO: we better always install the same layout, and instead test m_maskMissingColumnsToZero here.
    // Note that existing 'reduce' style operations--the criterion nodes and gradient computation--already call this.  --BUGBUG: They can't, wrong layout!
    // Warning: The layout used here must match the matrix. E.g. don't pass a child's matrix from a criterion node (use Inputs(x)->MaskMissing{Values,Gradient}ColumnsToZero() instead.
    template<class ElemType>
    static inline bool MaskMissingColumnsTo(Matrix<ElemType>& matrixToBeMasked, const MBLayoutPtr & pMBLayout, const FrameRange & frameRange, ElemType val)
    {
        bool foundLabelOrFeatureMissing = false;    // return value: set to true if either nolabel or feature missing is processed

        if (pMBLayout && !pMBLayout->IsAllNone())   // TODO: This should check whether there are any gaps.
        {
            size_t nT = pMBLayout->GetNumTimeSteps();
            size_t nS = pMBLayout->GetNumParallelSequences();

            if (matrixToBeMasked.GetNumCols() != nT * nS)
                LogicError("MaskMissingColumnsToZero: pMBLayout->m_minibatchPackingFlags should have one element for each timestep of all streams. Check feature reader. ");

            shared_ptr<Matrix<char>> columnsValidityMask = pMBLayout->GetColumnsValidityMask(frameRange, matrixToBeMasked.GetDeviceId());
            if (columnsValidityMask != nullptr)
            {
                auto matrixSliceToMask = DataSliceWithMBLayout(matrixToBeMasked, frameRange, pMBLayout);
                foundLabelOrFeatureMissing = true;
                matrixSliceToMask.MaskColumnsValue(*columnsValidityMask, val);
            }
        }

        return foundLabelOrFeatureMissing;
    }

}}}
