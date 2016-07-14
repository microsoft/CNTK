//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// utterancesourcemulti.h -- implementation of utterancesource.h that supports multiple feature and label sets
//

#pragma once

#include "Basics.h"         // for attempt()
#include "htkfeatio.h"      // for htkmlfreader
#include "latticearchive.h" // for reading HTK phoneme lattices (MMI training)
#include "minibatchsourcehelpers.h"
#include "minibatchiterator.h"
#include "unordered_set"

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// minibatchutterancesource -- feature source to provide randomized utterances
// This also implements a frame-wise mode, which is layered on top of the utterance-wise mode
// and thus benefits from its goodies such as corpus-wide high-level randomization and chunk paging.
// ---------------------------------------------------------------------------
class minibatchutterancesourcemulti : public minibatchsource
{
    void operator=(const minibatchutterancesourcemulti &other); // non-assignable
    std::vector<size_t> vdim;                                   // feature dimension after augmenting neighhors
    std::vector<size_t> leftcontext;                            // number of frames to the left of the target frame in the context window
    std::vector<size_t> rightcontext;                           // number of frames to the right of the target frame in the context window
    std::vector<unsigned int> sampperiod;                       // (for reference and to check against model)
    std::vector<string> featkind;
    std::vector<size_t> featdim;
    std::vector<bool> expandToUtt;           // indicator of whether features should be applied to entire utterance, e.g. ivectors
    const bool framemode;                    // true -> actually return frame-level randomized frames (not possible in lattice mode)
    std::vector<std::vector<size_t>> counts; // [s] occurence count for all states (used for priors)
    int verbosity;
    // lattice reader
    // const std::vector<unique_ptr<latticesource>> &lattices;
    const latticesource &lattices;

    // std::vector<latticesource> lattices;
    // word-level transcripts (for MMI mode when adding best path to lattices)
    const map<wstring, msra::lattices::lattice::htkmlfwordsequence> &allwordtranscripts; // (used for getting word-level transcripts)
                                                                                         // std::vector<map<wstring,msra::lattices::lattice::htkmlfwordsequence>> allwordtranscripts;
    // data store (incl. paging in/out of features and lattices)
    struct utterancedesc // data descriptor for one utterance
    {
        msra::asr::htkfeatreader::parsedpath parsedpath; // archive filename and frame range in that file
        size_t classidsbegin;                            // index into allclassids[] array (first frame)

        utterancedesc(msra::asr::htkfeatreader::parsedpath &&ppath, size_t classidsbegin)
            : parsedpath(std::move(ppath)), classidsbegin(classidsbegin), framesToExpand(0), needsExpansion(false)
        {
        }
        bool needsExpansion; // ivector type of feature
        size_t framesToExpand; // expected number of frames (to expand ivectors) 
        wstring logicalpath() const
        {
            return parsedpath; /*type cast will return logical path*/
        }
        size_t numframes() const
        {
            if (needsExpansion)
                return framesToExpand;
            else
                return parsedpath.numframes();
        }
        wstring key() const // key used for looking up lattice (not stored to save space)
        {
#ifdef _MSC_VER
            static const wstring emptywstring;
            static const wregex deleteextensionre(L"\\.[^\\.\\\\/:]*$");
            return regex_replace(logicalpath(), deleteextensionre, emptywstring); // delete extension (or not if none)
#else
            return removeExtension(logicalpath());
#endif
        }
        void expandtoutterance(size_t requiredFrames)
        {
            needsExpansion = true;
            framesToExpand = requiredFrames;
        }
    };

    // Make sure type 'utterancedesc' has a move constructor
    static_assert(std::is_move_constructible<utterancedesc>::value, "Type 'utterancedesc' should be move constructible!");

    struct utterancechunkdata // data for a chunk of utterances
    {
        std::vector<utterancedesc> utteranceset; // utterances in this set
        size_t numutterances() const
        {
            return utteranceset.size();
        }

        std::vector<size_t> firstframes;                                            // [utteranceindex] first frame for given utterance
        mutable msra::dbn::matrix frames;                                           // stores all frames consecutively (mutable since this is a cache)
        size_t totalframes;                                                         // total #frames for all utterances in this chunk
        mutable std::vector<shared_ptr<const latticesource::latticepair>> lattices; // (may be empty if none)

        // construction
        utterancechunkdata()
            : totalframes(0)
        {
        }
        void push_back(utterancedesc && /*destructive*/ utt)
        {
            if (isinram())
                LogicError("utterancechunkdata: frames already paged into RAM--too late to add data");
            firstframes.push_back(totalframes);
            totalframes += utt.numframes();
            utteranceset.push_back(std::move(utt));
        }

        // accessors to an utterance's data
        size_t numframes(size_t i) const
        {
            return utteranceset[i].numframes();
        }
        size_t getclassidsbegin(size_t i) const
        {
            return utteranceset[i].classidsbegin;
        }
        msra::dbn::matrixstripe getutteranceframes(size_t i) const // return the frame set for a given utterance
        {
            if (!isinram())
                LogicError("getutteranceframes: called when data have not been paged in");
            const size_t ts = firstframes[i];
            const size_t n = numframes(i);
            return msra::dbn::matrixstripe(frames, ts, n);
        }
        shared_ptr<const latticesource::latticepair> getutterancelattice(size_t i) const // return the frame set for a given utterance
        {
            if (!isinram())
                LogicError("getutteranceframes: called when data have not been paged in");
            return lattices[i];
        }

        // paging
        // test if data is in memory at the moment
        bool isinram() const
        {
            return !frames.empty();
        }
        // page in data for this chunk
        // We pass in the feature info variables by ref which will be filled lazily upon first read
        void requiredata(string &featkind, size_t &featdim, unsigned int &sampperiod, const latticesource &latticesource, int verbosity = 0) const
        {
            if (numutterances() == 0)
                LogicError("requiredata: cannot page in virgin block");
            if (isinram())
                LogicError("requiredata: called when data is already in memory");
            try // this function supports retrying since we read from the unrealible network, i.e. do not return in a broken state
            {
                msra::asr::htkfeatreader reader; // feature reader (we reinstantiate it for each block, i.e. we reopen the file actually)
                std::unique_ptr<auto_timer> pageintimer = nullptr;
                if (verbosity > 2)
                    pageintimer.reset(new auto_timer());
                // if this is the first feature read ever, we explicitly open the first file to get the information such as feature dimension
                if (featdim == 0)
                {
                    reader.getinfo(utteranceset[0].parsedpath, featkind, featdim, sampperiod);
                    fprintf(stderr, "requiredata: determined feature kind as %d-dimensional '%s' with frame shift %.1f ms\n", (int) featdim, featkind.c_str(), sampperiod / 1e4);
                }
                // read all utterances; if they are in the same archive, htkfeatreader will be efficient in not closing the file
                frames.resize(featdim, totalframes);
                if (!latticesource.empty())
                    lattices.resize(utteranceset.size());
                foreach_index (i, utteranceset)
                {
                    // fprintf (stderr, ".");
                    // read features for this file
                    auto uttframes = getutteranceframes(i);                                                    // matrix stripe for this utterance (currently unfilled)
                    reader.read(utteranceset[i].parsedpath, (const string &)featkind, sampperiod, uttframes, utteranceset[i].needsExpansion);  // note: file info here used for checkuing only
                    // page in lattice data
                    if (!latticesource.empty())
                        latticesource.getlattices(utteranceset[i].key(), lattices[i], uttframes.cols());
                }
                if (verbosity)
                {
                    fprintf(stderr, "requiredata: %d utterances read\n", (int)utteranceset.size());
                    if (verbosity > 2)
                    {
                        if (pageintimer != nullptr)
                        {
                            double pageintime = (double)(*pageintimer);
#ifdef _MSC_VER
                            fprintf(stderr, "Chunk read statistics; Total time = %.8g, Num Frames read = %Iu, Num bytes per frame = %Iu, Avg I/O bandwidth = %.2g MB/sec).\n",
                              pageintime, totalframes, featdim * sizeof(float), (double)(featdim * sizeof(float) * totalframes / 1024 / 1024 / pageintime));
#else
                            fprintf(stderr, "Chunk read statistics; Total time = %.8g, Num Frames read = %zu, Num bytes per frame = %zu, Avg I/O bandwidth = %.2g MB/sec).\n",
                              pageintime, totalframes, featdim * sizeof(float), (double)(featdim * sizeof(float) * totalframes / 1024 / 1024 / pageintime));
#endif

                        }
                    }
                }
            }
            catch (...)
            {
                // Clean up in a non-throwable way in order not to hide the original exception.
                cleandata();
                throw;
            }
        }

        // page out data for this chunk
        void releasedata() const
        {
            if (numutterances() == 0)
                LogicError("releasedata: cannot page out virgin block");
            if (!isinram())
                LogicError("releasedata: called when data is not memory");
            cleandata();
        }

    private:
        void cleandata() const
        {
            // release frames
            frames.resize(0, 0);
            // release lattice data
            lattices.clear();
        }
    };
    std::vector<std::vector<utterancechunkdata>> allchunks;           // set of utterances organized in chunks, referred to by an iterator (not an index)
    std::vector<unique_ptr<biggrowablevector<CLASSIDTYPE>>> classids; // [classidsbegin+t] concatenation of all state sequences

    bool m_generatePhoneBoundaries;
    std::vector<unique_ptr<biggrowablevector<HMMIDTYPE>>> phoneboundaries;
    bool issupervised() const
    {
        return !classids.empty();
    }
    size_t numutterances; // total number of utterances
    size_t _totalframes;  // total frames (same as classids.size() if we have labels)
    double timegetbatch;  // [v-hansu] for time measurement
    // sequence in random order of actual use (randomized, where randomization is cached)
    const size_t randomizationrange; // parameter remembered; this is the full window (e.g. 48 hours), not the half window
    size_t currentsweep;             // randomization is currently cached for this sweep; if it changes, rebuild all below
    struct chunk                     // chunk as used in actual processing order (randomized sequence)
    {
        // the underlying chunk (as a non-indexed reference into the chunk set)
        std::vector<utterancechunkdata>::const_iterator uttchunkdata;
        const utterancechunkdata &getchunkdata() const
        {
            return *uttchunkdata;
        }
        size_t numutterances() const
        {
            return uttchunkdata->numutterances();
        }
        size_t numframes() const
        {
            return uttchunkdata->totalframes;
        }

        // position in utterance-position space
        size_t utteranceposbegin;
        size_t utteranceposend() const
        {
            return utteranceposbegin + numutterances();
        }

        // position on global time line
        size_t globalts; // start frame on global timeline (after randomization)
        size_t globalte() const
        {
            return globalts + numframes();
        }

        // randomization range limits
        size_t windowbegin; // randomizedchunk index of earliest chunk that utterances in here can be randomized with
        size_t windowend;   // and end index [windowbegin, windowend)
        chunk(std::vector<utterancechunkdata>::const_iterator uttchunkdata, size_t utteranceposbegin, size_t globalts)
            : uttchunkdata(uttchunkdata), utteranceposbegin(utteranceposbegin), globalts(globalts)
        {
        }
    };
    std::vector<std::vector<chunk>> randomizedchunks; // utterance chunks after being brought into random order (we randomize within a rolling window over them)
    size_t chunksinram;                               // (for diagnostics messages)
    struct utteranceref                               // describes the underlying random utterance associated with an utterance position
    {
        size_t chunkindex;     // lives in this chunk (index into randomizedchunks[])
    private:
        size_t m_utteranceindex; // utterance index in that chunk
    public:
        size_t utteranceindex() const
        {
            return m_utteranceindex;
        }

        size_t numframes;      // (cached since we cannot directly access the underlying data from here)
        size_t globalts;       // start frame in global space after randomization (for mapping frame index to utterance position)
        size_t globalte() const
        {
            return globalts + numframes;
        } // end frame
        utteranceref(size_t chunkindex, size_t utteranceindex)
            : chunkindex(chunkindex), m_utteranceindex(utteranceindex), globalts(SIZE_MAX), numframes(0)
        {
        }
        void swap(utteranceref &other) // used in randomization
        {
            ::swap(chunkindex, other.chunkindex);
            ::swap(m_utteranceindex, other.m_utteranceindex);
            assert(globalts == SIZE_MAX && other.globalts == SIZE_MAX && numframes == 0 && other.numframes == 0); // can only swap before assigning these
        }
    };
    std::vector<utteranceref> randomizedutterancerefs;            // [pos] randomized utterance ids
    std::unordered_map<size_t, size_t> randomizedutteranceposmap; // [globalts] -> pos lookup table
    struct positionchunkwindow                                    // chunk window required in memory when at a certain position, for controlling paging
    {
        std::vector<chunk>::iterator definingchunk; // the chunk in randomizedchunks[] that defined the utterance position of this utterance
        size_t windowbegin() const
        {
            return definingchunk->windowbegin;
        }
        size_t windowend() const
        {
            return definingchunk->windowend;
        }
        bool isvalidforthisposition(const utteranceref &utt) const
        {
            return utt.chunkindex >= windowbegin() && utt.chunkindex < windowend(); // check if 'utt' lives in is in allowed range for this position
        }
        positionchunkwindow(std::vector<chunk>::iterator definingchunk)
            : definingchunk(definingchunk)
        {
        }
    };
    std::vector<positionchunkwindow> positionchunkwindows; // [utterance position] -> [windowbegin, windowend) for controlling paging

    // frame-level randomization layered on top of utterance chunking (randomized, where randomization is cached)
    #pragma pack(push)
    #pragma pack(1)
    struct frameref
    {
        unsigned short chunkindex;     // lives in this chunk (index into randomizedchunks[])

    private:
        // The utterance index and frame index are stored in a byte array
        // with the first NUM_UTTERANCE_INDEX_BITS bits being the utterance index 
        // and the remaining being the frame index
        static const unsigned int NUM_STORAGE_BYTES = 4;
        static const unsigned int NUM_UTTERANCE_INDEX_BITS = 16;
        static const unsigned int NUM_FRAME_INDEX_BITS = (NUM_STORAGE_BYTES * 8) - NUM_UTTERANCE_INDEX_BITS;

        unsigned char m_buffer[NUM_STORAGE_BYTES];
        static_assert(NUM_STORAGE_BYTES <= sizeof(unsigned int), "Number of storage bytes in a frameref should not exceed sizeof(unsigned int)");
        unsigned int fullvalue() const
        {
            unsigned int fullValue = m_buffer[0];
            for (int i = 1; i < NUM_STORAGE_BYTES; ++i)
            {
                fullValue = (fullValue << 8) + m_buffer[i];
            }

            return fullValue;
        }
    public:
        static const size_t maxutterancesperchunk = (1 << NUM_UTTERANCE_INDEX_BITS) - 1;
        static const size_t maxframesperutterance = (1 << NUM_FRAME_INDEX_BITS) - 1;

        // utterance index in that chunk
        unsigned short utteranceindex() const
        {
            return (unsigned short)(fullvalue() >> NUM_FRAME_INDEX_BITS);
        }

        // frame index within the utterance
        unsigned short frameindex() const
        {
            return (unsigned short)(fullvalue() & ((1U << NUM_FRAME_INDEX_BITS) - 1));
        }

        frameref(size_t ci, size_t ui, size_t fi)
            : chunkindex((unsigned short) ci)
        {
            if ((ui <= maxutterancesperchunk) && (fi <= maxframesperutterance))
            {
                unsigned int fullValue = (unsigned int)ui;
                fullValue = (fullValue << NUM_FRAME_INDEX_BITS) + (unsigned int)fi;
                for (int i = NUM_STORAGE_BYTES - 1; i >=0; --i)
                {
                    m_buffer[i] = fullValue & ((1U << 8) - 1);
                    fullValue = fullValue >> 8;
                }
            }
            else
                LogicError("frameref: bit fields too small");
        }

        frameref()
            : chunkindex(0)
        {
            for (int i = 0; i < NUM_STORAGE_BYTES; ++i)
                m_buffer[i] = 0;
        }

        void print() const
        {
            fprintf(stderr, "chunkindex=%d, utteranceindex=%d, frameindex=%d\n", (int)chunkindex, (int)utteranceindex(), (int)frameindex());
        }
    };
    #pragma pack(pop)

    class framerandomizer
    {
    private:
        const std::vector<std::vector<chunk>>& m_randomizedChunks;

        // When true we use a rolling window of randomized framerefs to minimize memory
        // footprint, instead of using a large vector listing all frames in the training corpus
        // Functionally, the 2 methods are identical.
        const bool m_minimizeMemoryFootprint;

        // [globalt-sweepts] -> (chunk, utt, frame) lookup table for randomized frames  --this can be REALLY big!
        // Only used when m_minimizeMemoryFootprint is false
        biggrowablevector<frameref> m_randomizedframerefs;

        // A rolling windows of chunks of framerefs used for randomization in frame mode
        // Along with each frameref, we also store the chunk index of the original frame
        // at that index before randomization, to be used for determining the chunk range
        // to be used for randomization of that frame's position
        // Only used when m_minimizeMemoryFootprint is true
        std::deque<std::vector<std::pair<unsigned short, frameref>>> m_randomizedframerefsWindow;
        size_t m_currentRangeBeginChunkIdx;
        size_t m_currentRangeEndChunkIdx;
        size_t m_nextFramePosNotYetRandomized;

    public:
        framerandomizer(const std::vector<std::vector<chunk>>& randomizedChunks, bool minimizeMemoryFootprint)
            : m_randomizedChunks(randomizedChunks), m_minimizeMemoryFootprint(minimizeMemoryFootprint), m_currentRangeBeginChunkIdx(0), m_currentRangeEndChunkIdx(0), m_nextFramePosNotYetRandomized(0)
        {
        }

        void randomizeFrameRange(size_t globalts, size_t globalte)
        {
            if (m_nextFramePosNotYetRandomized == m_randomizedChunks[0].back().globalte())
                return;

            assert(m_nextFramePosNotYetRandomized >= m_randomizedChunks[0][0].globalts);

            size_t firstFramePosToRandomize = m_nextFramePosNotYetRandomized;

            // Find the smallest chunk index whose windowbegin exceeds the chunk index
            // of the frame position (globalte - 1). We will randomize up to this chunk
            // as the final position of (globalte - 1) is guaranteed to have been determined 
            // when all frames up to that chunk have been randomized
            size_t lastFramePosChunkIdx = chunkIdx(globalte - 1);
            size_t endChunkIdxToRandomize = lastFramePosChunkIdx;
            while ((endChunkIdxToRandomize < m_randomizedChunks[0].size()) &&
                   (m_randomizedChunks[0][endChunkIdxToRandomize].windowbegin <= lastFramePosChunkIdx))
            {
                endChunkIdxToRandomize++;
            }

            size_t endFramePosToRandomize = m_randomizedChunks[0][endChunkIdxToRandomize - 1].globalte();

            // Determine the range of chunks that need to be in m_randomizedframerefsWindow for us
            // to perform the necessary randomization
            size_t startChunkIdx = std::min(chunkIdx(globalts), m_randomizedChunks[0][chunkIdx(firstFramePosToRandomize)].windowbegin);
            size_t endChunkIdx = m_randomizedChunks[0][chunkIdx(endFramePosToRandomize - 1)].windowend;

            // Lets drop everything that is outside the new range [startChunkIdx, endChunkIdx)
            for (size_t i = m_currentRangeBeginChunkIdx; i < startChunkIdx; ++i)
            {
                m_randomizedframerefsWindow.pop_front();
                m_currentRangeBeginChunkIdx++;
            }

            // Lets page in everything from m_currentRangeEndChunkIdx to endChunkIdx
            for (size_t i = m_currentRangeEndChunkIdx; i < endChunkIdx; ++i)
                addRandomizedFramesForChunk(i);

            assert(m_currentRangeEndChunkIdx == endChunkIdx);

            // now randomize them --we use the nested loop again to avoid storing a backpointer
            // The condition is that a randomized frame may not be moved out of its associated chunk window.
            // The actual range we randomize is up to the last frame that position (globalte - 1) could
            // potentially swap with
            for (size_t t = firstFramePosToRandomize; t < endFramePosToRandomize; ++t)
            {
                size_t currentChunkIdx = chunkIdx(t);

                size_t chunkWindowBegin = m_randomizedChunks[0][currentChunkIdx].windowbegin;
                size_t chunkWindowEnd = m_randomizedChunks[0][currentChunkIdx].windowend;

                // Chunk implies that if we are at position 't', we are guaranteed to have chunks [chunkWindowBegin, chunkWindowEnd) in RAM.
                // These chunks are associated with a range of frame positions.
                // It is implied that if we are at position 't', the frames covered by chunks [chunkWindowBegin, chunkWindowEnd) are in RAM.
                const size_t postbegin = m_randomizedChunks[0][chunkWindowBegin].globalts;
                const size_t postend = m_randomizedChunks[0][chunkWindowEnd - 1].globalte();
                // The position that this frame gets randomized to must be guaranteed to belong to a chunk within [postbegin, postend).

                for (;;) // (randomization retry loop)
                {
                    size_t tswap = Microsoft::MSR::CNTK::rand(postbegin, postend); // random frame position within allowed range
                    // We want to swap 't' to 'tswap' and 'tswap' to 't'.
                    //  - Both may have been swapped before.
                    //  - Both must stay within the randomization window of their respective position.
                    // check admissibility of where the element at 'tswap' gets swapped to 't' (range = [windowbegin,windowend))
                    size_t tswapchunkindex = randomizedframeref(tswap).chunkindex;
                    if (tswapchunkindex < chunkWindowBegin || tswapchunkindex >= chunkWindowEnd)
                        continue;
                    // check admissibility of where the element at t gets swapped to (which is frame position 'tswap')
                    const size_t sourcechunkindex = randomizedframeref(t).chunkindex;
                    size_t targetchunkindex = ttochunk(tswap); // chunk associated with this frame position defines value range
                    const auto &targetchunk = m_randomizedChunks[0][targetchunkindex];
                    const size_t targetwindowbegin = targetchunk.windowbegin;
                    const size_t targetwindowend = targetchunk.windowend;
                    if (sourcechunkindex < targetwindowbegin || sourcechunkindex >= targetwindowend)
                        continue;
                    // admissible--swap the two
                    ::swap(randomizedframeref(t), randomizedframeref(tswap));

                    // do a post-check if we got it right  --we seem not to
                    if (isframepositionvalid(t) && isframepositionvalid(tswap))
                        break;
                    // not valid: swap them back and try again  --we actually discovered a bug in the code above
                    ::swap(randomizedframeref(t), randomizedframeref(tswap));
                    fprintf(stderr, "randomizeFrameRange: BUGBUG --invalid swapping condition detected\n");
                }
            }

            m_nextFramePosNotYetRandomized = endFramePosToRandomize;

            // Verify no frameref has violated its range constraints
            for (size_t t = globalts; t < globalte; ++t)
            {
                size_t chunkIdx = ttochunk(t);
                const auto &chunk = m_randomizedChunks[0][chunkIdx]; // for window and chunkdata
                const size_t poswindowbegin = chunk.windowbegin;
                const size_t poswindowend = chunk.windowend;

                const size_t randomizedchunkindex = randomizedframeref(t).chunkindex;
                if (randomizedchunkindex < poswindowbegin || randomizedchunkindex >= poswindowend)
                    LogicError("randomizeFrameRange: nope, you got frame randomization wrong, dude");
            }
        }

        void reset(unsigned int randSeed)
        {
            srand(randSeed);
            size_t sweepts = m_randomizedChunks[0][0].globalts;
            size_t totalFrames = m_randomizedChunks[0].back().globalte() - sweepts;
            if (m_minimizeMemoryFootprint)
            {
                m_randomizedframerefsWindow.clear();
                m_currentRangeBeginChunkIdx = m_randomizedChunks[0][0].windowbegin;
                m_currentRangeEndChunkIdx = m_currentRangeBeginChunkIdx;
                m_nextFramePosNotYetRandomized = sweepts;
            }
            else
            {
                if (m_randomizedframerefs.size() != totalFrames)
                    m_randomizedframerefs.resize(totalFrames);

                // An original timeline is established by the randomized chunks, denoted by 't'.
                // Returned frames are indexed by frame position j = (globalt - sweept), which have an associated underlying 't'.
                // It is guaranteed that uttterance frame position j maps to an underlying frame within the corresponding chunk window.
                biggrowablevector<unsigned short> ttochunk; // randomized chunk index associated with frame position
                ttochunk.resize(totalFrames);
                size_t t = 0;
                // enumerate chunks in their randomized order and assign frame indices in that order -> randomizedframerefs[t]
                // At this point, chunks are in randomized order, but utterances and frames within utterances are not randomized.
                // Later we will randomize those as well.
                foreach_index(i, m_randomizedChunks[0])
                {
                    const auto &chunk = m_randomizedChunks[0][i];
                    const auto &chunkdata = chunk.getchunkdata();
                    const size_t numutt = chunkdata.numutterances();
                    for (size_t k = 0; k < numutt; k++)
                    {
                        const size_t n = chunkdata.numframes(k);
                        for (size_t m = 0; m < n; m++)
                        {
                            m_randomizedframerefs[t] = { (size_t)i, k, m }; // hopefully this is a memory copy, not a bit-wise assignment! If not, then code it explicitly
                            ttochunk[t] = (unsigned short)i;
                            checkoverflow(ttochunk[t], i, "ttochunk[]");
                            t++;
                        }
                    }
                }
                assert(t == totalFrames);

                // now randomize them --we use the nested loop again to avoid storing a backpointer
                // The condition is that a randomized frame may not be moved out of its associated chunk window.
                foreach_index(t, m_randomizedframerefs)
                {
                    const size_t positionchunkindex = ttochunk[t];               // position 't' lies within this chunk (relationship is monotonous, not random)
                    const auto &chunk = m_randomizedChunks[0][positionchunkindex]; // for window

                    // get in-RAM chunk range for this frame position (shared across all frame positions within the same chunk)
                    const size_t poswindowbegin = chunk.windowbegin; // rolling window over chunks (which under the hood have been randomized)
                    const size_t poswindowend = chunk.windowend;
                    // Chunk implies that if we are at position 't', we are guaranteed to have chunks [poswindowbegin, poswindowend) in RAM.
                    // These chunks are associated with a range of frame positions.
                    // It is implied that if we are at position 't', the frames covered by chunks [poswindowbegin, poswindowend) are in RAM.
                    const size_t postbegin = m_randomizedChunks[0][poswindowbegin].globalts - sweepts;
                    const size_t postend = m_randomizedChunks[0][poswindowend - 1].globalte() - sweepts;
                    // The position that this frame gets randomized to must be guaranteed to belong to a chunk within [postbegin, postend).

                    for (;;) // (randomization retry loop)
                    {
                        size_t tswap = Microsoft::MSR::CNTK::rand(postbegin, postend); // random frame position within allowed range
                        // We want to swap 't' to 'tswap' and 'tswap' to 't'.
                        //  - Both may have been swapped before.
                        //  - Both must stay within the randomization window of their respective position.
                        // check admissibility of where the element at 'tswap' gets swapped to 't' (range = [windowbegin,windowend))
                        size_t tswapchunkindex = m_randomizedframerefs[tswap].chunkindex;
                        if (tswapchunkindex < poswindowbegin || tswapchunkindex >= poswindowend)
                            continue;
                        // check admissibility of where the element at t gets swapped to (which is frame position 'tswap')
                        const size_t sourcechunkindex = m_randomizedframerefs[t].chunkindex;
                        size_t targetchunkindex = ttochunk[tswap]; // chunk associated with this frame position defines value range
                        const auto &targetchunk = m_randomizedChunks[0][targetchunkindex];
                        const size_t targetwindowbegin = targetchunk.windowbegin;
                        const size_t targetwindowend = targetchunk.windowend;
                        if (sourcechunkindex < targetwindowbegin || sourcechunkindex >= targetwindowend)
                            continue;
                        // admissible--swap the two
                        ::swap(m_randomizedframerefs[t], m_randomizedframerefs[tswap]);
                        // do a post-check if we got it right  --we seem not to
                        if (isframepositionvalid(t, ttochunk) && isframepositionvalid(tswap, ttochunk))
                            break;
                        // not valid: swap them back and try again  --we actually discovered a bug in the code above
                        ::swap(m_randomizedframerefs[t], m_randomizedframerefs[tswap]);
                        fprintf(stderr, "lazyrandomization: BUGBUG --invalid swapping condition detected\n");
                    }
                }

                // check it --my head spins
                t = 0;
                foreach_index(i, m_randomizedChunks[0])
                {
                    const auto &chunk = m_randomizedChunks[0][i]; // for window and chunkdata
                    const size_t poswindowbegin = chunk.windowbegin;
                    const size_t poswindowend = chunk.windowend;

                    const auto &chunkdata = chunk.getchunkdata(); // for numutterances/numframes
                    const size_t numutt = chunkdata.numutterances();
                    for (size_t k = 0; k < numutt; k++)
                    {
                        const size_t n = chunkdata.numframes(k);
                        for (size_t m = 0; m < n; m++)
                        {
                            const size_t randomizedchunkindex = m_randomizedframerefs[t].chunkindex;
                            if (randomizedchunkindex < poswindowbegin || randomizedchunkindex >= poswindowend)
                                LogicError("lazyrandomization: nope, you got frame randomization wrong, dude");
                            t++;
                        }
                    }
                }
                assert(t == totalFrames);

                m_nextFramePosNotYetRandomized = m_randomizedChunks[0].back().globalte();
                m_currentRangeBeginChunkIdx = 0;
                m_currentRangeEndChunkIdx = m_randomizedChunks[0].size();
            }
        }

        frameref& randomizedframeref(size_t globalts)
        {
            if (m_minimizeMemoryFootprint)
                return randomizedframeentry(globalts).second;
            else
            {
                size_t sweepts = m_randomizedChunks[0][0].globalts;
                return m_randomizedframerefs[globalts - sweepts];
            }
        }

    private:
        void addRandomizedFramesForChunk(size_t chunkIdx)
        {
            assert(m_minimizeMemoryFootprint);
            assert(chunkIdx == m_currentRangeEndChunkIdx);

            const auto &chunk = m_randomizedChunks[0][chunkIdx];
            const auto &chunkdata = chunk.getchunkdata();
            const size_t numutt = chunkdata.numutterances();
            std::vector<std::pair<unsigned short, frameref>> chunkFrames(chunkdata.totalframes);
            size_t t = 0;
            for (size_t k = 0; k < numutt; k++)
            {
                const size_t n = chunkdata.numframes(k);
                for (size_t m = 0; m < n; m++)
                {
                    chunkFrames[t] = { (unsigned short)chunkIdx, { chunkIdx, k, m } };
                    checkoverflow(chunkFrames[t].first, chunkIdx, "ttochunk[]");
                    t++;
                }
            }

            m_randomizedframerefsWindow.push_back(std::move(chunkFrames));

            m_currentRangeEndChunkIdx++;
        }

        size_t chunkIdx(size_t t)
        {
            assert(t >= m_randomizedChunks[0][m_currentRangeBeginChunkIdx].globalts);

            // TODO: Use std::lower_bound
            size_t low = m_currentRangeBeginChunkIdx;
            size_t high = m_randomizedChunks[0].size() - 1;
            while (high > low)
            {
                size_t mid = (high + low) / 2;
                if (t >= m_randomizedChunks[0][mid].globalte())
                {
                    low = mid + 1;
                }
                else if (t < m_randomizedChunks[0][mid].globalts)
                {
                    assert(mid > 0);
                    high = mid - 1;
                }
                else
                {
                    return mid;
                }
            }

            assert((high == low) && ((t >= m_randomizedChunks[0][low].globalts) && (t < m_randomizedChunks[0][low].globalte())));
            return low;
        }

        // helper for testing whether a swapped frame position is valid (w.r.t. beign in RAM when being at position 't')
        bool isframepositionvalid(const size_t t)
        {
            assert(m_minimizeMemoryFootprint);

            // look up valid range for time position
            const size_t positionchunkindex = ttochunk(t); // position 't' lies within this original chunk (relationship is monotonous, not random)
            const auto &chunk = m_randomizedChunks[0][positionchunkindex];
            // get in-RAM chunk range for this frame position (shared across all frame positions within the same chunk)
            const size_t poswindowbegin = chunk.windowbegin; // rolling window over chunks (which under the hood have been randomized)
            const size_t poswindowend = chunk.windowend;
            // Chunk implies that if we are at position 't', we are guaranteed to have chunks [poswindowbegin, poswindowend) in RAM.

            // now see if the randomized location is within that window
            const size_t actualchunkindexforpos = randomizedframeref(t).chunkindex; // where this frame pos has been mapped to
            return actualchunkindexforpos >= poswindowbegin && actualchunkindexforpos < poswindowend;
            // We only need to test the chunk index. Utterance and frame can be randomized within a chunk as we want, as long it is in RAM.
        }

        // helper for testing whether a swapped frame position is valid (w.r.t. beign in RAM when being at position 't')
        bool isframepositionvalid(const size_t t, const biggrowablevector<unsigned short> &ttochunk) const
        {
            assert(!m_minimizeMemoryFootprint);

            // look up valid range for time position
            const size_t positionchunkindex = ttochunk[t]; // position 't' lies within this original chunk (relationship is monotonous, not random)
            const auto &chunk = m_randomizedChunks[0][positionchunkindex];
            // get in-RAM chunk range for this frame position (shared across all frame positions within the same chunk)
            const size_t poswindowbegin = chunk.windowbegin; // rolling window over chunks (which under the hood have been randomized)
            const size_t poswindowend = chunk.windowend;
            // Chunk implies that if we are at position 't', we are guaranteed to have chunks [poswindowbegin, poswindowend) in RAM.

            // now see if the randomized location is within that window
            const size_t actualchunkindexforpos = m_randomizedframerefs[t].chunkindex; // where this frame pos has been mapped to
            return actualchunkindexforpos >= poswindowbegin && actualchunkindexforpos < poswindowend;
            // We only need to test the chunk index. Utterance and frame can be randomized within a chunk as we want, as long it is in RAM.
        }

        unsigned short& ttochunk(size_t globalts)
        {
            assert(m_minimizeMemoryFootprint);
            return randomizedframeentry(globalts).first;
        }

        std::pair<unsigned short, frameref>& randomizedframeentry(size_t globalts)
        {
            assert(m_minimizeMemoryFootprint);
            size_t globaltsChunkIdx = chunkIdx(globalts);
            assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);
            return m_randomizedframerefsWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx][globalts - m_randomizedChunks[0][globaltsChunkIdx].globalts];
        }

        framerandomizer& operator=(const framerandomizer&) = delete;
    };

    framerandomizer m_frameRandomizer;

    // TODO: this may go away if we store classids directly in the utterance data
    template <class VECTOR>
    class shiftedvector // accessing a vector with a non-0 starting index
    {
        void operator=(const shiftedvector &);
        VECTOR &v;
        size_t first;
        size_t n;
        void check(size_t i) const
        {
            if (i >= n)
                LogicError("shiftedvector: index out of bounds");
        }

    public:
        shiftedvector(VECTOR &v, size_t first, size_t n)
            : v(v), first(first), n(n)
        {
        }
        // TODO: the following is not templated--do it if needed; also should return a const reference then
        size_t operator[](size_t i) const
        {
            check(i);
            return v[first + i];
        }
    };
    template <class UTTREF>
    std::vector<shiftedvector<biggrowablevector<CLASSIDTYPE>>> getclassids(const UTTREF &uttref) // return sub-vector of classids[] for a given utterance
    {
        std::vector<shiftedvector<biggrowablevector<CLASSIDTYPE>>> allclassids;

        if (!issupervised())
        {
            foreach_index (i, classids)
                allclassids.push_back(std::move(shiftedvector<biggrowablevector<CLASSIDTYPE>>((*classids[i]), 0, 0)));
            return allclassids; // nothing to return
        }
        const auto &chunk = randomizedchunks[0][uttref.chunkindex];
        const auto &chunkdata = chunk.getchunkdata();
        const size_t classidsbegin = chunkdata.getclassidsbegin(uttref.utteranceindex()); // index of first state label in global concatenated classids[] array
        const size_t n = chunkdata.numframes(uttref.utteranceindex());
        foreach_index (i, classids)
        {
            if ((*classids[i])[classidsbegin + n] != (CLASSIDTYPE) -1)
                LogicError("getclassids: expected boundary marker not found, internal data structure screwed up");
            allclassids.push_back(std::move(shiftedvector<biggrowablevector<CLASSIDTYPE>>((*classids[i]), classidsbegin, n)));
        }
        return allclassids; // nothing to return
    }
    template <class UTTREF>
    std::vector<shiftedvector<biggrowablevector<HMMIDTYPE>>> getphonebound(const UTTREF &uttref) // return sub-vector of classids[] for a given utterance
    {
        if (!m_generatePhoneBoundaries)
            LogicError("getphonebound: generation of phone boundaries is not enabled for this utterance source!");

        std::vector<shiftedvector<biggrowablevector<HMMIDTYPE>>> allphoneboundaries;
        allphoneboundaries.empty();

        if (!issupervised())
        {
            foreach_index (i, classids)
                allphoneboundaries.push_back(std::move(shiftedvector<biggrowablevector<HMMIDTYPE>>((*phoneboundaries[i]), 0, 0)));
            return allphoneboundaries; // nothing to return
        }
        const auto &chunk = randomizedchunks[0][uttref.chunkindex];
        const auto &chunkdata = chunk.getchunkdata();
        const size_t classidsbegin = chunkdata.getclassidsbegin(uttref.utteranceindex()); // index of first state label in global concatenated classids[] array
        const size_t n = chunkdata.numframes(uttref.utteranceindex());
        foreach_index (i, phoneboundaries)
        {
            if ((*phoneboundaries[i])[classidsbegin + n] != (HMMIDTYPE) -1)
                LogicError("getclassids: expected boundary marker not found, internal data structure screwed up");
            allphoneboundaries.push_back(std::move(shiftedvector<biggrowablevector<HMMIDTYPE>>((*phoneboundaries[i]), classidsbegin, n)));
        }
        return allphoneboundaries; // nothing to return
    }

public:
    // constructor
    // Pass empty labels to denote unsupervised training (so getbatch() will not return uids).
    // This mode requires utterances with time stamps.
    minibatchutterancesourcemulti(const std::vector<std::vector<wstring>> &infiles, const std::vector<map<wstring, std::vector<msra::asr::htkmlfentry>>> &labels,
                                  std::vector<size_t> vdim, std::vector<size_t> udim, std::vector<size_t> leftcontext, std::vector<size_t> rightcontext, size_t randomizationrange,
                                  const latticesource &lattices, const map<wstring, msra::lattices::lattice::htkmlfwordsequence> &allwordtranscripts, const bool framemode, bool minimizeMemoryFootprint, std::vector<bool> expandToUtt)
                                  : vdim(vdim), leftcontext(leftcontext), rightcontext(rightcontext), sampperiod(0), featdim(0), randomizationrange(randomizationrange), currentsweep(SIZE_MAX), lattices(lattices), allwordtranscripts(allwordtranscripts), framemode(framemode), chunksinram(0), timegetbatch(0), verbosity(2), m_generatePhoneBoundaries(!lattices.empty()), m_frameRandomizer(randomizedchunks, minimizeMemoryFootprint), expandToUtt(expandToUtt)
    // [v-hansu] change framemode (lattices.empty()) into framemode (false) to run utterance mode without lattice
    // you also need to change another line, search : [v-hansu] comment out to run utterance mode without lattice
    {
        // process infiles to know dimensions of things (but not loading features)
        size_t nomlf = 0;               // number of entries missing in MLF (diagnostics)
        size_t nolat = 0;               // number of entries missing in lattice archive (diagnostics)
        std::vector<size_t> numclasses; // number of output classes as found in the label file (diagnostics)
        _totalframes = 0;
        wstring key;
        size_t numutts = 0;

        std::vector<bool> uttisvalid;    // boolean flag to check that utterance is valid. valid means number of
                                         // frames is consistent across all feature and label streams
        std::vector<size_t> uttduration; // track utterance durations to determine utterance validity

        std::vector<size_t> classidsbegin;

        allchunks = std::vector<std::vector<utterancechunkdata>>(infiles.size(), std::vector<utterancechunkdata>());
        featdim = std::vector<size_t>(infiles.size(), 0);
        sampperiod = std::vector<unsigned int>(infiles.size(), 0);
        featkind = std::vector<string>(infiles.size(), "");

        numclasses = std::vector<size_t>(labels.size(), 0);
        counts = std::vector<std::vector<size_t>>(labels.size(), std::vector<size_t>());

        foreach_index (i, labels)
        {
            classids.push_back(unique_ptr<biggrowablevector<CLASSIDTYPE>>(new biggrowablevector<CLASSIDTYPE>()));
            if (m_generatePhoneBoundaries)
                phoneboundaries.push_back(unique_ptr<biggrowablevector<HMMIDTYPE>>(new biggrowablevector<HMMIDTYPE>()));

            // std::pair<std::vector<wstring>,std::vector<wstring>> latticetocs;
            // std::unordered_map<std::string,size_t> modelsymmap;
            // lattices.push_back(shared_ptr<latticesource>(new latticesource(latticetocs, modelsymmap)));
        }

        // first check consistency across feature streams
        // We'll go through the SCP files for each stream to make sure the duration is consistent
        // If not, we'll plan to ignore the utterance, and inform the user
        // m indexes the feature stream
        // i indexes the files within a stream, i.e. in the SCP file)
        foreach_index (m, infiles)
        {
            if (m == 0)
            {
                numutts = infiles[m].size();
                uttisvalid = std::vector<bool>(numutts, true);
                uttduration = std::vector<size_t>(numutts, 0);
                if (expandToUtt[m])
                    RuntimeError("minibatchutterancesourcemulti: the first feature stream must be frame-based not utterance based");
            }
            else if (infiles[m].size() != numutts)
                RuntimeError("minibatchutterancesourcemulti: all feature files must have same number of utterances");

            foreach_index (i, infiles[m])
            {
                utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(infiles[m][i]), 0); // mseltzer - is this foolproof for multiio? is classids always non-empty?
                const size_t uttframes = utterance.numframes();                                  // will throw if frame bounds not given --required to be given in this mode
                if (expandToUtt[m] && uttframes != 1)
                    RuntimeError("minibatchutterancesource: utterance-based features must be 1 frame in duration");
                // we need at least 2 frames for boundary markers to work
                else if (!expandToUtt[m] && uttframes < 2)
                    RuntimeError("minibatchutterancesource: utterances < 2 frames not supported");
                if (uttframes > frameref::maxframesperutterance)
                {
                    fprintf(stderr, "minibatchutterancesource: skipping %d-th file (%d frames) because it exceeds max. frames (%d) for frameref bit field: %ls\n", i, (int) uttframes, (int) frameref::maxframesperutterance, key.c_str());
                    uttduration[i] = 0;
                    uttisvalid[i] = false;
                }
                else
                {
                    if (m == 0)
                    {
                        uttduration[i] = uttframes;
                        uttisvalid[i] = true;
                    }
                    else if (uttduration[i] != uttframes && !expandToUtt[m])
                    {
                        fprintf(stderr, "minibatchutterancesource: skipping %d-th file due to inconsistency in duration in different feature streams (%d vs %d frames)\n", i, (int) uttduration[i], (int) uttframes);
                        uttduration[i] = 0;
                        uttisvalid[i] = false;
                    }
                }
            }
        }
        size_t invalidutts = 0;
        foreach_index (i, uttisvalid)
        {
            if (!uttisvalid[i])
                invalidutts++;
        }
        if (invalidutts > uttisvalid.size() / 2)
            RuntimeError("minibatchutterancesource: too many files with inconsistent durations, assuming broken configuration\n");
        else if (invalidutts > 0)
            fprintf(stderr, "Found inconsistent durations across feature streams in %d out of %d files\n", (int) invalidutts, (int) uttisvalid.size());

        // now process the features and labels
        size_t utterancesetsize = 0;
        foreach_index (m, infiles)
        {
            std::vector<utterancedesc> utteranceset; // read all utterances to here first; at the end, distribute to chunks
            utteranceset.reserve(infiles[m].size());
            // if (m==0)
            //    numutts = infiles[m].size();
            // else
            //    if (infiles[m].size()!=numutts)
            //        RuntimeError("minibatchutterancesourcemulti: all feature files must have same number of utterances\n");
            if (m == 0)
                classidsbegin.clear();

            foreach_index (i, infiles[m])
            {
                if (i % (infiles[m].size() / 100 + 1) == 0)
                {
                    fprintf(stderr, ".");
                    fflush(stderr);
                }
                // build utterance descriptor
                if (m == 0 && !labels.empty())
                    classidsbegin.push_back(classids[0]->size());

                if (uttisvalid[i])
                {
                    utterancedesc utterance(msra::asr::htkfeatreader::parsedpath(infiles[m][i]), labels.empty() ? 0 : classidsbegin[i]); // mseltzer - is this foolproof for multiio? is classids always non-empty?
                    const size_t uttframes = utterance.numframes();                                                                      // will throw if frame bounds not given --required to be given in this mode
                    if (expandToUtt[m])
                    {
                        assert(uttframes == 1);
                        utterance.expandtoutterance(uttduration[i]);
                    }
                    else
                    {
                        assert(uttframes == uttduration[i]);                                                                                 // ensure nothing funky happened
                    }
                    // already performed these checks above
                    // we need at least 2 frames for boundary markers to work
                    // if (uttframes < 2)
                    //    RuntimeError("minibatchutterancesource: utterances < 2 frames not supported");
                    // if (uttframes > frameref::maxframesperutterance)
                    // {
                    //    fprintf (stderr, "minibatchutterancesource: skipping %d-th file (%d frames) because it exceeds max. frames (%d) for frameref bit field: %ls", i, uttframes, frameref::maxframesperutterance, key.c_str());
                    //    continue;
                    // }

                    // check whether we have the ref transcript
                    // auto labelsiter = labels[0].end();
                    bool lacksmlf = true;
                    if (!labels.empty()) // empty means unsupervised mode (don't load any)
                    {
                        key = utterance.key();
                        // check if labels are available (if not, it normally means that no path was found in realignment)
                        auto labelsiter = labels[0].find(key);
                        // const bool lacksmlf = (labelsiter == labels[0].end());
                        lacksmlf = (labelsiter == labels[0].end());
                        if (lacksmlf)
                            if (nomlf++ < 5)
                                fprintf(stderr, " [no labels for  %ls]", key.c_str());
                        // check if lattice is available (when in lattice mode)
                        // TODO: also check the #frames here; requires a design change of the TOC format & a rerun
                        const bool lackslat = !lattices.empty() && !lattices.haslattice(key); // ('true' if we have no lattices)
                        if (lackslat)
                            if (nolat++ < 5)
                                fprintf(stderr, " [no lattice for %ls]", key.c_str());
                        // skip if either one is missing
                        if (lacksmlf || lackslat)
                        {
                            uttisvalid[i] = false;
                            continue; // skip this utterance at all
                        }
                    }
                    // push the label sequence into classids[], since we already looked it up
                    // TODO: we can store labels more efficiently now since we don't do frame-wise random access anymore.

                    // OK, utterance has all we need --remember it

                    if (m == 0)
                    {
                        if (!labels.empty() && !lacksmlf)
                        // if (!labels.empty() && labelsiter != labels[0].end())
                        {
                            // first verify that all the label files have the proper duration
                            foreach_index (j, labels)
                            {
                                const auto &labseq = labels[j].find(key)->second;
                                // check if durations match; skip if not
                                size_t labframes = labseq.empty() ? 0 : (labseq[labseq.size() - 1].firstframe + labseq[labseq.size() - 1].numframes);
                                if (labframes != uttframes)
                                {
                                    fprintf(stderr, " [duration mismatch (%d in label vs. %d in feat file), skipping %ls]", (int) labframes, (int) uttframes, key.c_str());
                                    nomlf++;
                                    uttisvalid[i] = false;
                                    // continue;   // skip this utterance at all
                                    break;
                                }
                            }
                            if (uttisvalid[i])
                            {
                                utteranceset.push_back(std::move(utterance));
                                _totalframes += uttframes;
                                // then parse each mlf if the durations are consistent
                                foreach_index (j, labels)
                                {
                                    const auto &labseq = labels[j].find(key)->second;
                                    // expand classid sequence into flat array
                                    foreach_index (i, labseq)
                                    {
                                        const auto &e = labseq[i];
                                        if ((i > 0 && labseq[i - 1].firstframe + labseq[i - 1].numframes != e.firstframe) || (i == 0 && e.firstframe != 0))
                                        {
                                            RuntimeError("minibatchutterancesource: labels not in consecutive order MLF in label set: %ls", key.c_str());
                                        }
                                        if (e.classid >= udim[j])
                                        {
                                            RuntimeError("minibatchutterancesource: class id %d exceeds model output dimension %d in file %ls", (int) e.classid, (int) udim[j], key.c_str());
                                        }
                                        if (e.classid != (CLASSIDTYPE) e.classid)
                                            RuntimeError("CLASSIDTYPE has too few bits");
                                        for (size_t t = e.firstframe; t < e.firstframe + e.numframes; t++)
                                        {
                                            classids[j]->push_back(e.classid);
                                            if (m_generatePhoneBoundaries)
                                            {
                                                if (e.phonestart != 0 && t == e.firstframe)
                                                    phoneboundaries[j]->push_back((HMMIDTYPE)e.phonestart);
                                                else
                                                    phoneboundaries[j]->push_back((HMMIDTYPE)0);
                                            }
                                        }
                                        numclasses[j] = max(numclasses[j], (size_t)(1u + e.classid));
                                        counts[j].resize(numclasses[j], 0);
                                        counts[j][e.classid] += e.numframes;
                                    }

                                    classids[j]->push_back((CLASSIDTYPE) -1);      // append a boundary marker marker for checking
                                    if (m_generatePhoneBoundaries)
                                        phoneboundaries[j]->push_back((HMMIDTYPE) -1); // append a boundary marker marker for checking

                                    if (!labels[j].empty() && classids[j]->size() != _totalframes + utteranceset.size())
                                        LogicError("minibatchutterancesource: label duration inconsistent with feature file in MLF label set: %ls", key.c_str());
                                    assert(labels[j].empty() || classids[j]->size() == _totalframes + utteranceset.size());
                                }
                            }
                        }
                        else
                        {
                            assert(classids.empty() && labels.empty());
                            utteranceset.push_back(std::move(utterance));
                            _totalframes += uttframes;
                        }
                    }
                    else
                    {
                        utteranceset.push_back(std::move(utterance));
                    }
                }
            }
            if (m == 0)
                utterancesetsize = utteranceset.size();
            else
                assert(utteranceset.size() == utterancesetsize);

            fprintf(stderr, "feature set %d: %d frames in %d out of %d utterances\n", m, (int) _totalframes, (int) utteranceset.size(), (int) infiles[m].size());

            if (!labels.empty())
            {
                foreach_index (j, labels)
                {
                    biggrowablevector<CLASSIDTYPE> &cid = *classids[j];
                    foreach_index (i, utteranceset)
                    {
                        // if ((*classids[j])[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (CLASSIDTYPE) -1)
                        // printf("index = %d\n",utteranceset[i].classidsbegin + utteranceset[i].numframes());
                        // printf("cid[index] = %d\n",cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()]);
                        // printf("CLASSIDTYPE(-1) = %d\n",(CLASSIDTYPE) -1);
                        if (cid[utteranceset[i].classidsbegin + utteranceset[i].numframes()] != (CLASSIDTYPE) -1)
                            LogicError("minibatchutterancesource: classids[] out of sync");
                    }
                }
            }
            if (nomlf + nolat > 0)
            {
                fprintf(stderr, "minibatchutterancesource: out of %d files, %d files not found in label set and %d have no lattice\n", (int) infiles[0].size(), (int) nomlf, (int) nolat);
                if (nomlf + nolat > infiles[m].size() / 2)
                    RuntimeError("minibatchutterancesource: too many files not found in label set--assuming broken configuration\n");
            }
            if (m == 0)
            {
                foreach_index (j, numclasses)
                {
                    fprintf(stderr, "label set %d: %d classes\n", j, (int) numclasses[j]);
                }
            }
            // distribute them over chunks
            // We simply count off frames until we reach the chunk size.
            // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
            const size_t framespersec = 100;                   // we just assume this; our efficiency calculation is based on this
            const size_t chunkframes = 15 * 60 * framespersec; // number of frames to target for each chunk
            // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
            // When paging chunk by chunk, chunk size ~14 MB.
            std::vector<utterancechunkdata> &thisallchunks = allchunks[m];

            thisallchunks.resize(0);
            thisallchunks.reserve(_totalframes / chunkframes);

            foreach_index (i, utteranceset)
            {
                // if exceeding current entry--create a new one
                // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
                if (thisallchunks.empty() || thisallchunks.back().totalframes > chunkframes || thisallchunks.back().numutterances() >= frameref::maxutterancesperchunk)
                    thisallchunks.push_back(utterancechunkdata());
                // append utterance to last chunk
                utterancechunkdata &currentchunk = thisallchunks.back();
                currentchunk.push_back(std::move(utteranceset[i])); // move it out from our temp array into the chunk
                // TODO: above push_back does not actually 'move' because the internal push_back does not accept that
            }
            numutterances = utteranceset.size();
            fprintf(stderr, "minibatchutterancesource: %d utterances grouped into %d chunks, av. chunk size: %.1f utterances, %.1f frames\n",
                    (int) numutterances, (int) thisallchunks.size(), numutterances / (double) thisallchunks.size(), _totalframes / (double) thisallchunks.size());
            // Now utterances are stored exclusively in allchunks[]. They are never referred to by a sequential utterance id at this point, only by chunk/within-chunk index.
        }
    }

private:
    // shuffle a vector into random order by randomly swapping elements

    template <typename VECTOR>
    static void randomshuffle(VECTOR &v, size_t randomseed)
    {
        if (v.size() > RAND_MAX * (size_t) RAND_MAX)
            RuntimeError("randomshuffle: too large set: need to change to different random generator!");
        srand((unsigned int) randomseed);
        foreach_index (i, v)
        {
            // pick a random location
            const size_t irand = Microsoft::MSR::CNTK::rand(0, v.size());

            // swap element i with it
            if (irand == (size_t) i)
                continue;
            ::swap(v[i], v[irand]);
        }
    }
    static void checkoverflow(size_t fieldval, size_t targetval, const char *fieldname)
    {
        if (fieldval != targetval)
            RuntimeError("checkoverflow: bit field %s too small for value 0x%x (cut from 0x%x)", fieldname, (int) targetval, (int) fieldval);
    }

    // big long helper to update all cached randomization information
    // This is a rather complex process since we randomize on two levels:
    //  - chunks of consecutive data in the feature archive
    //  - within a range of chunks that is paged into RAM
    //     - utterances (in utt mode), or
    //     - frames (in frame mode)
    // The 'globalts' parameter is the start time that triggered the rerandomization; it is NOT the base time of the randomized area.
    size_t lazyrandomization(const size_t globalts)
    {
        const size_t sweep = globalts / _totalframes; // which sweep (this determines randomization)
        if (sweep == currentsweep)                    // already got this one--nothing to do
            return sweep;

        currentsweep = sweep;
        if (verbosity > 0)
            fprintf(stderr, "lazyrandomization: re-randomizing for sweep %d in %s mode\n", (int) currentsweep, framemode ? "frame" : "utterance");

        const size_t sweepts = sweep * _totalframes; // first global frame index for this sweep

        // first randomize chunks
        std::vector<std::vector<std::vector<utterancechunkdata>::const_iterator>> randomizedchunkrefs;
        foreach_index (i, allchunks)
            randomizedchunkrefs.push_back(std::vector<std::vector<utterancechunkdata>::const_iterator>());

        foreach_index (i, allchunks)
            randomizedchunkrefs[i].reserve(allchunks[i].size());

        foreach_index (i, allchunks) // TODO: this cries for iterating using the iterator!
        {
            foreach_index (j, allchunks[i])
                randomizedchunkrefs[i].push_back(allchunks[i].begin() + j);
            assert(randomizedchunkrefs[i].size() == allchunks[i].size());

            // note that sincew randomshuffle() uses sweep as seed, this will keep the randomization common across all feature streams
            randomshuffle(randomizedchunkrefs[i], sweep); // bring into random order (with random seed depending on sweep)
        }

        // place them onto the global timeline -> randomizedchunks[]
        // We are processing with randomization within a rolling window over this chunk sequence.
        // Paging will happen on a chunk-by-chunk basis.
        // The global time stamp is needed to determine the paging window.
        randomizedchunks.clear(); // data chunks after being brought into random order (we randomize within a rolling window over them)

        foreach_index (i, allchunks)
            randomizedchunks.push_back(std::vector<chunk>());

        foreach_index (i, allchunks)
        {
            randomizedchunks[i].reserve(randomizedchunkrefs[i].size());
            foreach_index (k, randomizedchunkrefs[i])
                randomizedchunks[i].push_back(chunk(randomizedchunkrefs[i][k], randomizedchunks[i].empty() ? 0 : randomizedchunks[i].back().utteranceposend(), randomizedchunks[i].empty() ? sweepts : randomizedchunks[i].back().globalte()));
            assert(randomizedchunks[i].size() == allchunks[i].size());
            assert(randomizedchunks[i].empty() || (randomizedchunks[i].back().utteranceposend() == numutterances && randomizedchunks[i].back().globalte() == sweepts + _totalframes));
        }
        // for each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
        foreach_index (i, randomizedchunks)
        {
            foreach_index (k, randomizedchunks[i])
            {
                chunk &chunk = randomizedchunks[i][k];
                // start with the range of left neighbor
                if (k == 0)
                {
                    chunk.windowbegin = 0;
                    chunk.windowend = 1;
                }
                else
                {
                    chunk.windowbegin = randomizedchunks[i][k - 1].windowbegin; // might be too early
                    chunk.windowend = randomizedchunks[i][k - 1].windowend;     // might have more space
                }
                while (chunk.globalts - randomizedchunks[i][chunk.windowbegin].globalts > randomizationrange / 2)
                    chunk.windowbegin++; // too early
                while (chunk.windowend < randomizedchunks[i].size() && randomizedchunks[i][chunk.windowend].globalte() - chunk.globalts < randomizationrange / 2)
                    chunk.windowend++; // got more space
            }
        }
        if (!framemode) // utterance mode
        {
            // This sets up the following members:
            //  - positionchunkwindows
            //  - randomizedutterancerefs
            //  - randomizedutteranceposmap

            // We will now introduce the concept of utterance *position*.
            // During processing, utterances will be indexed by position (which is in turn derived from a frame index in getbatch()),
            // and it is assumed (required) that positions are requested consecutively.
            // Each utterance position has an underlying associated utterance, which is represented as (chunkid, within-chunk index) and randomly assigned.
            // Each utterance position also has an associated range of chunks that are kept in memory,
            // and the associated underlying utterance is guaranteed to be found within that associated range of chunks.
            // That allows to page out/in data when processing utterance positions in a consecutive manner.

            // compute chunk windows for every utterance position -> positionchunkwindows[]
            // Utterance positions can only reference underlying utterance data within the chunk window.
            // Utterance positions are defined by the randomized chunk sequence (i.e. their underlying 'defining' chunk differs from sweep to sweep).
            positionchunkwindows.clear(); // [utterance position] -> [windowbegin, windowend) for controlling paging
            positionchunkwindows.reserve(numutterances);

            // positionchunkwindows should be consistent for all inputs (distinct feature streams), so just build based on feature[0]
            // contains pointer to chunk elements but only to compute index
            foreach_index (k, randomizedchunks[0]) // TODO: this really cries for iterating using iterators!
            {
                chunk &chunk = randomizedchunks[0][k];
                for (size_t i = chunk.utteranceposbegin; i < chunk.utteranceposend(); i++) // loop over utterances in this chunk
                    positionchunkwindows.push_back(randomizedchunks[0].begin() + k);
                // to look up the chunk range in memory for a position, look up the defining chunk and its range
            }
            assert(positionchunkwindows.size() == numutterances);

            // build the randomized utterances array -> randomizedutterancerefs[]
            // start by assigning all utterance positions to utterances in non-random consecutive manner
            randomizedutterancerefs.clear(); // [pos] randomized utterance ids
            randomizedutterancerefs.reserve(numutterances);
            foreach_index (k, randomizedchunks[0])
            {
                chunk &chunk = randomizedchunks[0][k];
                for (size_t i = 0; i < chunk.numutterances(); i++) // loop over utterances in this chunk
                    randomizedutterancerefs.push_back(utteranceref(k, i));
            }
            assert(randomizedutterancerefs.size() == numutterances);
            foreach_index (i, randomizedutterancerefs)
            {
                auto &uttref = randomizedutterancerefs[i];
                assert(positionchunkwindows[i].isvalidforthisposition(uttref));
                uttref;
            }

            // check we got those setup right

            // we now randomly shuffle randomizedutterancerefs[pos], while considering the constraints of what chunk range needs to be in memory
            srand((unsigned int) sweep + 1);
            for (size_t i = 0; i < randomizedutterancerefs.size(); i++)
            {
                // get valid randomization range, expressed in chunks
                const size_t windowbegin = positionchunkwindows[i].windowbegin();
                const size_t windowend = positionchunkwindows[i].windowend();

                // get valid randomization range, expressed in utterance positions
                // Remember, utterance positions are defined by chunks.
                const size_t posbegin = randomizedchunks[0][windowbegin].utteranceposbegin;
                const size_t posend = randomizedchunks[0][windowend - 1].utteranceposend();

                // randomization range for this utterance position is [posbegin, posend)
                for (;;)
                {
                    // pick a random location
                    const size_t j = Microsoft::MSR::CNTK::rand(posbegin, posend); // a random number within the window
                    if (i == j)
                        break; // the random gods say "this one points to its original position"... nothing wrong about that, but better not try to swap

                    // We want to swap utterances at i and j, but need to make sure they remain in their allowed range.
                    // This is guaranteed for a so-far untouched utterance, but both i and j may have been touched by a previous swap.

                    // We want to use the utterance previously referenced at utterance position j at position i. Is that allowed?
                    if (!positionchunkwindows[i].isvalidforthisposition(randomizedutterancerefs[j]))
                        continue; // nope --try another

                    // Likewise may we use the utterance previously referenced at utterance position i at position j?
                    if (!positionchunkwindows[j].isvalidforthisposition(randomizedutterancerefs[i]))
                        continue; // nope --try another

                    // yep--swap them
                    randomizedutterancerefs[i].swap(randomizedutterancerefs[j]);
                    break;
                }
            }

            // place the randomized utterances on the global timeline so we can find them by globalts
            size_t t = sweepts;
            foreach_index (i, randomizedutterancerefs)
            {
                auto &uttref = randomizedutterancerefs[i];
                uttref.globalts = t;
                uttref.numframes = randomizedchunks[0][uttref.chunkindex].getchunkdata().numframes(uttref.utteranceindex());
                t = uttref.globalte();
            }
            assert(t == sweepts + _totalframes);

            // verify that we got it right (I got a knot in my head!)
            foreach_index (i, randomizedutterancerefs)
            {
                // get utterance referenced at this position
                const auto &uttref = randomizedutterancerefs[i];
                // check if it is valid for this position
                if (uttref.chunkindex < positionchunkwindows[i].windowbegin() || uttref.chunkindex >= positionchunkwindows[i].windowend())
                    LogicError("lazyrandomization: randomization logic mangled!");
            }

            // create lookup table for (globalts values -> pos) -> randomizedutteranceposmap[]
            randomizedutteranceposmap.clear(); // [globalts] -> pos lookup table
            foreach_index (pos, randomizedutterancerefs)
            {
                auto &uttref = randomizedutterancerefs[pos];
                randomizedutteranceposmap[uttref.globalts] = (size_t) pos;
            }
        }
        else // frame mode
        {
            m_frameRandomizer.reset((unsigned int)sweep + 1);
        }

        return sweep;
    }

    // helper to page out a chunk with log message
    void releaserandomizedchunk(size_t k)
    {
        size_t numreleased = 0;
        foreach_index (m, randomizedchunks)
        {
            auto &chunkdata = randomizedchunks[m][k].getchunkdata();
            if (chunkdata.isinram())
            {
                if (verbosity)
                    fprintf(stderr, "releaserandomizedchunk: paging out randomized chunk %d (frame range [%d..%d]), %d resident in RAM\n",
                            (int) k, (int) randomizedchunks[m][k].globalts, (int) (randomizedchunks[m][k].globalte() - 1), (int) (chunksinram - 1));
                chunkdata.releasedata();
                numreleased++;
            }
        }
        if (numreleased > 0 && numreleased < randomizedchunks.size())
        {
            LogicError("releaserandomizedchunk: inconsistency detected - some inputs have chunks in ram, some not");
        }
        else if (numreleased == randomizedchunks.size())
        {
            chunksinram--;
        }
        return;
    }

    // helper to page in a chunk for a given utterance
    // (window range passed in for checking only)
    // Returns true if we actually did read something.
    bool requirerandomizedchunk(const size_t chunkindex, const size_t windowbegin, const size_t windowend)
    {
        size_t numinram = 0;

        if (chunkindex < windowbegin || chunkindex >= windowend)
            LogicError("requirerandomizedchunk: requested utterance outside in-memory chunk range");

        foreach_index (m, randomizedchunks)
        {
            auto &chunk = randomizedchunks[m][chunkindex];
            auto &chunkdata = chunk.getchunkdata();
            if (chunkdata.isinram())
                numinram++;
        }
        if (numinram == randomizedchunks.size())
            return false;
        else if (numinram == 0)
        {
            foreach_index (m, randomizedchunks)
            {
                auto &chunk = randomizedchunks[m][chunkindex];
                auto &chunkdata = chunk.getchunkdata();
                if (verbosity)
                    fprintf(stderr, "feature set %d: requirerandomizedchunk: paging in randomized chunk %d (frame range [%d..%d]), %d resident in RAM\n", m, (int) chunkindex, (int) chunk.globalts, (int) (chunk.globalte() - 1), (int) (chunksinram + 1));
                msra::util::attempt(5, [&]() // (reading from network)
                                    {
                                        chunkdata.requiredata(featkind[m], featdim[m], sampperiod[m], this->lattices, verbosity);
                                    });
            }
            chunksinram++;
            return true;
        }
        else
        {
            LogicError("requirerandomizedchunk: inconsistency detected - some inputs need chunks paged in, some not");
        }
    }

    class matrixasvectorofvectors // wrapper around a matrix that views it as a vector of column vectors
    {
        void operator=(const matrixasvectorofvectors &); // non-assignable
        msra::dbn::matrixbase &m;

    public:
        matrixasvectorofvectors(msra::dbn::matrixbase &m)
            : m(m)
        {
        }
        size_t size() const
        {
            return m.cols();
        }
        const_array_ref<float> operator[](size_t j) const
        {
            return array_ref<float>(&m(0, j), m.rows());
        }
    };

    size_t chunkforframepos(const size_t t) const // find chunk for a given frame position
    {
        // inspect chunk of first feature stream only
        auto iter = std::lower_bound(randomizedchunks[0].begin(), randomizedchunks[0].end(), t, [&](const chunk &chunk, size_t t)
                                     {
                                         return chunk.globalte() <= t;
                                     });
        const size_t chunkindex = iter - randomizedchunks[0].begin();
        if (t < randomizedchunks[0][chunkindex].globalts || t >= randomizedchunks[0][chunkindex].globalte())
            LogicError("chunkforframepos: dude, learn STL!");
        return chunkindex;
    }

public:
    void setverbosity(int newverbosity)
    {
        verbosity = newverbosity;
    }

    // get the next minibatch
    // A minibatch is made up of one or more utterances.
    // We will return less than 'framesrequested' unless the first utterance is too long.
    // Note that this may return frames that are beyond the epoch end, but the first frame is always within the epoch.
    // We specify the utterance by its global start time (in a space of a infinitely repeated training set).
    // This is efficient since getbatch() is called with sequential 'globalts' except at epoch start.
    // Note that the start of an epoch does not necessarily fall onto an utterance boundary. The caller must use firstvalidglobalts() to find the first valid globalts at or after a given time.
    // Support for data parallelism:  If mpinodes > 1 then we will
    //  - load only a subset of blocks from the disk
    //  - skip frames/utterances in not-loaded blocks in the returned data
    //  - 'framesadvanced' will still return the logical #frames; that is, by how much the global time index is advanced
    bool getbatch(const size_t globalts, const size_t framesrequested,
                  const size_t subsetnum, const size_t numsubsets, size_t &framesadvanced,
                  std::vector<msra::dbn::matrix> &feat, std::vector<std::vector<size_t>> &uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> &transcripts,
                  std::vector<shared_ptr<const latticesource::latticepair>> &latticepairs, std::vector<std::vector<size_t>> &sentendmark,
                  std::vector<std::vector<size_t>> &phoneboundaries) override
    {
        bool readfromdisk = false; // return value: shall be 'true' if we paged in anything

        auto_timer timergetbatch;
        assert(_totalframes > 0);

        // update randomization if a new sweep is entered  --this is a complex operation that updates many of the data members used below
        const size_t sweep = lazyrandomization(globalts);

        size_t mbframes = 0;
        const std::vector<char> noboundaryflags; // dummy
        if (!framemode)                          // regular utterance mode
        {
            // find utterance position for globalts
            // There must be a precise match; it is not possible to specify frames that are not on boundaries.
            auto positer = randomizedutteranceposmap.find(globalts);
            if (positer == randomizedutteranceposmap.end())
                LogicError("getbatch: invalid 'globalts' parameter; must match an existing utterance boundary");
            const size_t spos = positer->second;

            // determine how many utterances will fit into the requested minibatch size
            mbframes = randomizedutterancerefs[spos].numframes; // at least one utterance, even if too long
            size_t epos;
            for (epos = spos + 1; epos < numutterances && ((mbframes + randomizedutterancerefs[epos].numframes) < framesrequested); epos++) // add more utterances as long as they fit within requested minibatch size
                mbframes += randomizedutterancerefs[epos].numframes;

            // do some paging housekeeping
            // This will also set the feature-kind information if it's the first time.
            // Free all chunks left of the range.
            // Page-in all chunks right of the range.
            // We are a little more blunt for now: Free all outside the range, and page in only what is touched. We could save some loop iterations.
            const size_t windowbegin = positionchunkwindows[spos].windowbegin();
            const size_t windowend = positionchunkwindows[epos - 1].windowend();
            for (size_t k = 0; k < windowbegin; k++)
                releaserandomizedchunk(k);
            for (size_t k = windowend; k < randomizedchunks[0].size(); k++)
                releaserandomizedchunk(k);
            for (size_t pos = spos; pos < epos; pos++)
                if ((randomizedutterancerefs[pos].chunkindex % numsubsets) == subsetnum)
                    readfromdisk |= requirerandomizedchunk(randomizedutterancerefs[pos].chunkindex, windowbegin, windowend); // (window range passed in for checking only)

            // Note that the above loop loops over all chunks incl. those that we already should have.
            // This has an effect, e.g., if 'numsubsets' has changed (we will fill gaps).

            // determine the true #frames we return, for allocation--it is less than mbframes in the case of MPI/data-parallel sub-set mode
            size_t tspos = 0;
            for (size_t pos = spos; pos < epos; pos++)
            {
                const auto &uttref = randomizedutterancerefs[pos];
                if ((uttref.chunkindex % numsubsets) != subsetnum) // chunk not to be returned for this MPI node
                    continue;

                tspos += uttref.numframes;
            }

            // resize feat and uids
            feat.resize(vdim.size());
            uids.resize(classids.size());
            if (m_generatePhoneBoundaries)
                phoneboundaries.resize(classids.size());
            sentendmark.resize(vdim.size());
            assert(feat.size() == vdim.size());
            assert(feat.size() == randomizedchunks.size());
            foreach_index (i, feat)
            {
                feat[i].resize(vdim[i], tspos);

                if (i == 0)
                {
                    foreach_index (j, uids)
                    {
                        if (issupervised()) // empty means unsupervised training -> return empty uids
                        {
                            uids[j].resize(tspos);
                            if (m_generatePhoneBoundaries)
                                phoneboundaries[j].resize(tspos);
                        }
                        else
                        {
                            uids[i].clear();
                            if (m_generatePhoneBoundaries)
                                phoneboundaries[i].clear();
                        }
                        latticepairs.clear(); // will push_back() below
                        transcripts.clear();
                    }
                    foreach_index (j, sentendmark)
                    {
                        sentendmark[j].clear();
                    }
                }
            }
            // return these utterances
            if (verbosity > 0)
                fprintf(stderr, "getbatch: getting utterances %d..%d (%d subset of %d frames out of %d requested) in sweep %d\n", (int) spos, (int) (epos - 1), (int) tspos, (int) mbframes, (int) framesrequested, (int) sweep);
            tspos = 0; // relative start of utterance 'pos' within the returned minibatch
            for (size_t pos = spos; pos < epos; pos++)
            {
                const auto &uttref = randomizedutterancerefs[pos];
                if ((uttref.chunkindex % numsubsets) != subsetnum) // chunk not to be returned for this MPI node
                    continue;

                size_t n = 0;
                foreach_index (i, randomizedchunks)
                {
                    const auto &chunk = randomizedchunks[i][uttref.chunkindex];
                    const auto &chunkdata = chunk.getchunkdata();
                    assert((numsubsets > 1) || (uttref.globalts == globalts + tspos));
                    auto uttframes = chunkdata.getutteranceframes(uttref.utteranceindex());
                    matrixasvectorofvectors uttframevectors(uttframes); // (wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors())
                    n = uttframevectors.size();
                    sentendmark[i].push_back(n + tspos);
                    assert(n == uttframes.cols() && uttref.numframes == n && chunkdata.numframes(uttref.utteranceindex()) == n);

                    // copy the frames and class labels
                    for (size_t t = 0; t < n; t++) // t = time index into source utterance
                    {
                        size_t leftextent, rightextent;
                        // page in the needed range of frames
                        if (leftcontext[i] == 0 && rightcontext[i] == 0)
                        {
                            leftextent = rightextent = augmentationextent(uttframevectors[t].size(), vdim[i]);
                        }
                        else
                        {
                            leftextent = leftcontext[i];
                            rightextent = rightcontext[i];
                        }
                        augmentneighbors(uttframevectors, noboundaryflags, t, leftextent, rightextent, feat[i], t + tspos);
                        // augmentneighbors(uttframevectors, noboundaryflags, t, feat[i], t + tspos);
                    }

                    // copy the frames and class labels
                    if (i == 0)
                    {
                        auto uttclassids = getclassids(uttref);
                        std::vector<shiftedvector<biggrowablevector<HMMIDTYPE>>> uttphoneboudaries;
                        if (m_generatePhoneBoundaries)
                            uttphoneboudaries = getphonebound(uttref);
                        foreach_index (j, uttclassids)
                        {
                            for (size_t t = 0; t < n; t++) // t = time index into source utterance
                            {
                                if (issupervised())
                                {
                                    uids[j][t + tspos] = uttclassids[j][t];
                                    if (m_generatePhoneBoundaries)
                                        phoneboundaries[j][t + tspos] = uttphoneboudaries[j][t];
                                }
                            }

                            if (!this->lattices.empty())
                            {
                                auto latticepair = chunkdata.getutterancelattice(uttref.utteranceindex());
                                latticepairs.push_back(latticepair);
                                // look up reference
                                const auto &key = latticepair->getkey();
                                if (!allwordtranscripts.empty())
                                {
                                    const auto &transcript = allwordtranscripts.find(key)->second;
                                    transcripts.push_back(transcript.words);
                                }
                            }
                        }
                    }
                }
                tspos += n;
            }

            foreach_index (i, feat)
            {
                assert(tspos == feat[i].cols());
            }
        }
        else
        {
            const size_t sweepts = sweep * _totalframes;                      // first global frame index for this sweep
            const size_t sweepte = sweepts + _totalframes;                    // and its end
            const size_t globalte = min(globalts + framesrequested, sweepte); // we return as much as requested, but not exceeding sweep end
            mbframes = globalte - globalts;                                   // that's our mb size

            // Perform randomization of the desired frame range
            m_frameRandomizer.randomizeFrameRange(globalts, globalte);

            // determine window range
            // We enumerate all frames--can this be done more efficiently?
            const size_t firstchunk = chunkforframepos(globalts);
            const size_t lastchunk = chunkforframepos(globalte - 1);
            const size_t windowbegin = randomizedchunks[0][firstchunk].windowbegin;
            const size_t windowend = randomizedchunks[0][lastchunk].windowend;
            if (verbosity > 0)
                fprintf(stderr, "getbatch: getting randomized frames [%d..%d] (%d frames out of %d requested) in sweep %d; chunks [%d..%d] -> chunk window [%d..%d)\n",
                        (int) globalts, (int) globalte, (int) mbframes, (int) framesrequested, (int) sweep, (int) firstchunk, (int) lastchunk, (int) windowbegin, (int) windowend);
            // release all data outside, and page in all data inside
            for (size_t k = 0; k < windowbegin; k++)
                releaserandomizedchunk(k);
            for (size_t k = windowbegin; k < windowend; k++)
                if ((k % numsubsets) == subsetnum)                                     // in MPI mode, we skip chunks this way
                    readfromdisk |= requirerandomizedchunk(k, windowbegin, windowend); // (window range passed in for checking only, redundant here)
            for (size_t k = windowend; k < randomizedchunks[0].size(); k++)
                releaserandomizedchunk(k);

            // determine the true #frames we return--it is less than mbframes in the case of MPI/data-parallel sub-set mode
            // First determine it for all nodes, then pick the min over all nodes, as to give all the same #frames for better load balancing.
            // TODO: No, return all; and leave it to caller to redistribute them [Zhijie Yan]
            std::vector<size_t> subsetsizes(numsubsets, 0);
            for (size_t i = 0; i < mbframes; i++) // i is input frame index; j < i in case of MPI/data-parallel sub-set mode
            {
                const frameref &frameref = m_frameRandomizer.randomizedframeref(globalts + i);
                subsetsizes[frameref.chunkindex % numsubsets]++;
            }
            size_t j = subsetsizes[subsetnum];                                           // return what we have  --TODO: we can remove the above full computation again now
            const size_t allocframes = max(j, (mbframes + numsubsets - 1) / numsubsets); // we leave space for the desired #frames, assuming caller will try to pad them later

            // resize feat and uids
            feat.resize(vdim.size());
            uids.resize(classids.size());
            assert(feat.size() == vdim.size());
            assert(feat.size() == randomizedchunks.size());
            foreach_index (i, feat)
            {
                feat[i].resize(vdim[i], allocframes);
                feat[i].shrink(vdim[i], j);

                if (i == 0)
                {
                    foreach_index (k, uids)
                    {
                        if (issupervised()) // empty means unsupervised training -> return empty uids
                            uids[k].resize(j);
                        else
                            uids[k].clear();
                        latticepairs.clear(); // will push_back() below
                        transcripts.clear();
                    }
                }
            }

            // return randomized frames for the time range of those utterances
            size_t currmpinodeframecount = 0;
            for (size_t j = 0; j < mbframes; j++)
            {
                if (currmpinodeframecount >= feat[0].cols()) // MPI/data-parallel mode: all nodes return the same #frames, which is how feat(,) is allocated
                    break;

                // map to time index inside arrays
                const frameref &frameref = m_frameRandomizer.randomizedframeref(globalts + j);

                // in MPI/data-parallel mode, skip frames that are not in chunks loaded for this MPI node
                if ((frameref.chunkindex % numsubsets) != subsetnum)
                    continue;

                // random utterance
                readfromdisk |= requirerandomizedchunk(frameref.chunkindex, windowbegin, windowend); // (this is just a check; should not actually page in anything)

                foreach_index (i, randomizedchunks)
                {
                    const auto &chunk = randomizedchunks[i][frameref.chunkindex];
                    const auto &chunkdata = chunk.getchunkdata();
                    auto uttframes = chunkdata.getutteranceframes(frameref.utteranceindex());
                    matrixasvectorofvectors uttframevectors(uttframes); // (wrapper that allows m[.].size() and m[.][.] as required by augmentneighbors())
                    const size_t n = uttframevectors.size();
                    assert(n == uttframes.cols() && chunkdata.numframes(frameref.utteranceindex()) == n);
                    n;

                    // copy frame and class labels
                    const size_t t = frameref.frameindex();

                    size_t leftextent, rightextent;
                    // page in the needed range of frames
                    if (leftcontext[i] == 0 && rightcontext[i] == 0)
                    {
                        leftextent = rightextent = augmentationextent(uttframevectors[t].size(), vdim[i]);
                    }
                    else
                    {
                        leftextent = leftcontext[i];
                        rightextent = rightcontext[i];
                    }
                    augmentneighbors(uttframevectors, noboundaryflags, t, leftextent, rightextent, feat[i], currmpinodeframecount);

                    if (issupervised() && i == 0)
                    {
                        auto frameclassids = getclassids(frameref);
                        foreach_index (k, uids)
                            uids[k][currmpinodeframecount] = frameclassids[k][t];
                    }
                }

                currmpinodeframecount++;
            }
        }
        timegetbatch = timergetbatch;

        // this is the number of frames we actually moved ahead in time
        framesadvanced = mbframes;

        return readfromdisk;
    }

    bool supportsbatchsubsetting() const override
    {
        return true;
    }

    bool getbatch(const size_t globalts,
                  const size_t framesrequested, std::vector<msra::dbn::matrix> &feat, std::vector<std::vector<size_t>> &uids,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> &transcripts,
                  std::vector<shared_ptr<const latticesource::latticepair>> &lattices, std::vector<std::vector<size_t>> &sentendmark,
                  std::vector<std::vector<size_t>> &phoneboundaries)
    {
        size_t dummy;
        return getbatch(globalts, framesrequested, 0, 1, dummy, feat, uids, transcripts, lattices, sentendmark, phoneboundaries);
    }

    double gettimegetbatch()
    {
        return timegetbatch;
    }

    // alternate (updated) definition for multiple inputs/outputs - read as a vector of feature matrixes or a vector of label strings
    bool getbatch(const size_t /*globalts*/,
                  const size_t /*framesrequested*/, msra::dbn::matrix & /*feat*/, std::vector<size_t> & /*uids*/,
                  std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> & /*transcripts*/,
                  std::vector<shared_ptr<const latticesource::latticepair>> & /*latticepairs*/)
    {
        // should never get here
        RuntimeError("minibatchframesourcemulti: getbatch() being called for single input feature and single output feature, should use minibatchutterancesource instead\n");

        // for single input/output set size to be 1 and run old getbatch
        // feat.resize(1);
        // uids.resize(1);
        // return getbatch(globalts, framesrequested, feat[0], uids[0], transcripts, latticepairs);
    }

    size_t totalframes() const
    {
        return _totalframes;
    }

    // return first valid globalts to ask getbatch() for
    // In utterance mode, the epoch start may fall in the middle of an utterance.
    // We return the end time of that utterance (which, in pathological cases, may in turn be outside the epoch; handle that).
    /*implement*/ size_t firstvalidglobalts(const size_t globalts)
    {
        // update randomization if a new sweep is entered  --this is a complex operation that updates many of the data members used below
        const size_t sweep = lazyrandomization(globalts);
        // frame mode: start at sweep boundary directly
        if (framemode)
            return globalts;
        // utterance mode
        assert(globalts >= sweep * _totalframes && globalts < (sweep + 1) * _totalframes);
        sweep;
        foreach_index (pos, randomizedutterancerefs)
            if (randomizedutterancerefs[pos].globalts >= globalts)
                return randomizedutterancerefs[pos].globalts; // exact or inexact match
        return randomizedutterancerefs.back().globalte();     // boundary case: requested time falls within the last utterance
    }

    const std::vector<size_t> &unitcounts() const
    {
        return counts[0];
    }
    const std::vector<size_t> &unitcounts(size_t index) const
    {
        return counts[index];
    }
};
};
};
