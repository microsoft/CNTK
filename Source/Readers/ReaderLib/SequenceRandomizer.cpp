//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "SequenceRandomizer.h"
#include <algorithm>
#include <utility>
#include <deque>
#include "RandomOrdering.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    SequenceRandomizer::SequenceRandomizer(
        int verbosity,
        IDataDeserializerPtr deserializer,
        ChunkRandomizerPtr chunkRandomizer)
        : m_verbosity(verbosity),
        m_randomizedChunks(chunkRandomizer->GetRandomizedChunks()),
        m_chunkWindowBegin(0),
        m_randomizedWindowEnd(0),
        m_randomizationCursor(0),
        m_chunkWindowEnd(0),
        m_currentSequenceCursor(0),
        m_currentChunkCursor(0),
        m_currentSampleCursor(0),
        m_deserializer(deserializer)
    {
        size_t max = 0;
        for (const auto& c : m_randomizedChunks)
        {
            if (max < c.m_original->m_numberOfSequences)
            {
                max = c.m_original->m_numberOfSequences;
            }
        }

        m_bufferOriginalSequences.reserve(max);
    }

    // Resets the current sweep according to the randomization seed provided.
    void SequenceRandomizer::Reset(size_t randSeed)
    {
        m_rng.seed((unsigned long)randSeed);

        m_sequenceWindow.clear();
        m_randomizedChunkInfo.clear();

        m_chunkWindowBegin = 0;
        m_randomizedWindowEnd = 0;
        m_randomizationCursor = 0;
        m_chunkWindowEnd = 0;

        m_currentChunkCursor = 0;
        m_currentSequenceCursor = 0;
        m_currentSampleCursor = 0;

        // Prepare the chunk for reading
        RandomizeNextChunkIfNeeded();
    }

    // Gets the next randomized sequence descriptions not exceeding the global and local sample count,
    // when atLeastOneSequenceNeeded is false. Otherwise (when atLeastOneSequenceNeeded is true), 
    // returns at least one sequence description even when its length is greater than the required sample counts.
    // Whether a sequence is considered local is defined by the isLocalSequence predicate.
    // Returns a pair whose first element indicates the number of global samples read,
    // and second -- the number of local samples read (== sum of number of sample over all elements in the 
    // 'sequences' vector).
    std::pair<size_t, size_t> SequenceRandomizer::GetNextSequenceDescriptions(
        size_t globalSampleCount,
        size_t localSampleCount,
        const std::function<bool(const RandomizedSequenceDescription*)>& isLocalSequence,
        ClosedOpenChunkInterval& requiredChunks,
        std::vector<RandomizedSequenceDescription>& sequences,
        bool atLeastOneSequenceNeeded)
    {
        assert(globalSampleCount != 0);
        assert(localSampleCount != 0);

        if (globalSampleCount > std::numeric_limits<int>::max() &&
            localSampleCount > std::numeric_limits<int>::max())
            RuntimeError("Global and local size of the minibatch cannot exceed max int.");

        // Initialize the range to the current chunk.
        requiredChunks.m_begin = (ChunkIdType)std::min(m_currentChunkCursor, m_randomizedChunks.size() - 1);
        requiredChunks.m_end = requiredChunks.m_begin + 1;

        sequences.reserve(localSampleCount);
        sequences.clear();

        size_t globalSamplesRead = 0, localSamplesRead = 0;
        while (m_currentChunkCursor < m_randomizedChunks.size() &&
               localSamplesRead < localSampleCount &&
               globalSamplesRead < globalSampleCount)
        {
            size_t sequenceOffsetInsideChunk = m_currentSequenceCursor - m_randomizedChunks[m_currentChunkCursor].m_sequencePositionStart;
            const RandomizedSequenceDescription* sequence = &m_sequenceWindow[m_currentChunkCursor - m_chunkWindowBegin][sequenceOffsetInsideChunk];
            int sequenceLength = (int)sequence->m_numberOfSamples;
            bool isLocal = isLocalSequence(sequence);
            bool enoughData = !sequences.empty() || !atLeastOneSequenceNeeded;

            // Let's check whether we need to break because we exceeded global counter.
            if (enoughData && globalSamplesRead + sequenceLength > globalSampleCount)
                break;

            // Let's check whether we need to break because we exceeded local counter.
            if (enoughData && isLocal && localSamplesRead + sequenceLength > localSampleCount)
                break;

            if (isLocal) // Ok good to add it to the result.
            {
                sequences.push_back(*sequence);
                localSamplesRead += sequenceLength;
            }

            // even when the next sequence is not local, somebody else would return it, so
            // we need to ivalidate the 'atLeastOneSequenceNeeded' flag.
            atLeastOneSequenceNeeded = false; 

            globalSamplesRead += sequenceLength;

            // Update the required chunk window.
            requiredChunks.m_begin = std::min(m_randomizedChunks[m_currentChunkCursor].m_randomizationWindow.m_begin, requiredChunks.m_begin);
            requiredChunks.m_end = std::max(m_randomizedChunks[m_currentChunkCursor].m_randomizationWindow.m_end, requiredChunks.m_end);

            // Update current cursor to the next sequence.
            m_currentSequenceCursor++;
            m_currentSampleCursor += sequenceLength;
            if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkCursor].m_original->m_numberOfSequences)
            {
                // Moving to the next chunk,
                // Be careful, this invalidates the sequence from above.
                MoveChunkCursor();
            }
        }

        return { globalSamplesRead, localSamplesRead };
    }

    // Move the chunk cursor to the next chunk, randomizing more sequences if necessary.
    void SequenceRandomizer::MoveChunkCursor()
    {
        m_currentChunkCursor++;
        RandomizeNextChunkIfNeeded();

        // Release chunks that are not needed anymore.
        ReleaseChunks();
    }

    // Release chunks from the chunk window that are not needed anymore.
    void SequenceRandomizer::ReleaseChunks()
    {
        // We should drop chunks, but firstly make sure that they are not used any more.
        // That means the sequence description that we have got from the previous call can still be in the BlockRandomizer.
        size_t currentChunk = std::min(m_currentChunkCursor, m_randomizedChunks.size() - 1);
        size_t candidateToUnload = m_chunkWindowBegin;
        size_t releasedChunks = 0;
        while (candidateToUnload < m_randomizedChunks.size() &&
               candidateToUnload < m_randomizedChunks[currentChunk].m_randomizationWindow.m_begin &&
               m_randomizedChunks[candidateToUnload].m_randomizationWindow.m_end <= m_currentChunkCursor)
        {
            m_sequenceWindow.pop_front();
            m_randomizedChunkInfo.pop_front();
            m_chunkWindowBegin++;
            candidateToUnload++;
            releasedChunks++;
        }

        if (m_verbosity && 0 < releasedChunks)
            fprintf(stderr,
                "SequenceRandomizer::ReleaseChunks(): "
                "released %" PRIu64 " chunks, now "
                "chunk window [%" PRIu64 "..%u), cursor %" PRIu64 ", "
                "randomized window [%" PRIu64 "..%" PRIu64 "), randomization cursor %" PRIu64 "\n",
                releasedChunks,
                m_chunkWindowBegin, m_chunkWindowEnd,
                m_currentChunkCursor,
                m_chunkWindowBegin, m_randomizedWindowEnd,
                m_randomizationCursor);
    }

    // Randomize one more chunk if needed after the chunk cursor has been incremented.
    void SequenceRandomizer::RandomizeNextChunkIfNeeded()
    {
        if (m_currentChunkCursor < m_randomizedWindowEnd)
        {
            assert(m_currentChunkCursor >= m_chunkWindowBegin);
            return;
        }
        assert(m_randomizedWindowEnd == m_currentChunkCursor);

        if (m_randomizedWindowEnd == m_randomizedChunks.size())
        {
            return;
        }

        // Chunk not yet randomized.
        // of the sample position we have to randomized (current + sampleCount).
        // We will randomize up to this chunk as the final position of windows end is guaranteed to have been determined
        // when all sequences up to that chunk have been randomized
        size_t nextRandomizationCursor = m_randomizedChunks[m_randomizedWindowEnd].m_randomizationWindow.m_end;
        while (nextRandomizationCursor < m_randomizedChunks.size() &&
               m_randomizedChunks[nextRandomizationCursor].m_randomizationWindow.m_begin <= m_randomizedWindowEnd)
        {
            nextRandomizationCursor++;
        }

        // Determine the end chunk that we need to load into memory.
        ChunkIdType nextChunkWindowEnd = m_randomizedChunks[nextRandomizationCursor - 1].m_randomizationWindow.m_end;

        // Lets page in everything from m_currentRangeEndChunkIndex to endChunkIdx
        for (ChunkIdType i = m_chunkWindowEnd; i < nextChunkWindowEnd; ++i)
        {
            AddRandomizedSequencesForChunk(i);
        }

        size_t firstSequencePositionToRandomize =
            m_randomizationCursor == 0 ? 0 : m_randomizedChunks[m_randomizationCursor - 1].SequenceEndPosition();

        size_t endSequencePosToRandomize = m_randomizedChunks[nextRandomizationCursor - 1].SequenceEndPosition();
        for (size_t t = firstSequencePositionToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // Get valid randomization range, expressed in chunks
            // TODO: This can be done more efficiently, we know the range of chunks already.
            const ChunkIdType currentChunkIdx = GetChunkIndexForSequencePosition(t);

            size_t chunkWindowBegin = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_begin;
            size_t chunkWindowEnd = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_end;

            // Get valid randomization range, expressed in sequence positions.
            size_t posBegin = m_randomizedChunks[chunkWindowBegin].m_sequencePositionStart;
            size_t posEnd = m_randomizedChunks[chunkWindowEnd - 1].SequenceEndPosition();

            ChunkIdType tChunkIndex = GetChunkIndexForSequencePosition(t);
            auto& tSequence = GetRandomizedSequenceDescriptionByPosition(tChunkIndex, t);

            for (;;)
            {
                // Pick a sequence position from [posBegin, posEnd)
                const size_t j = RandMT(posBegin, posEnd, m_rng);

                // Pick up j sequence.
                ChunkIdType jChunkIndex = GetChunkIndexForSequencePosition(j);
                auto& jSequence = GetRandomizedSequenceDescriptionByPosition(jChunkIndex, j);

                // Try again if the sequence currently at j cannot be placed at position i.
                if (!IsValidForPosition(tChunkIndex, jSequence))
                    continue;

                // Try again if the sequence currently at i cannot be placed at position j.
                if (!IsValidForPosition(jChunkIndex, tSequence))
                    continue;

                // Swap and break out.
                std::swap(tSequence, jSequence);
                break;
            }
        }

        // Verify that we got it right
        for (size_t t = firstSequencePositionToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // TODO assert only
            ChunkIdType tChunkIndex = GetChunkIndexForSequencePosition(t);
            if (!IsValidForPosition(tChunkIndex, GetRandomizedSequenceDescriptionByPosition(tChunkIndex, t)))
            {
                LogicError("SequenceRandomizer::RandomizeNextSequenceDescriptions: randomization logic mangled!");
            }
        }

        // Let's recalculate number of samples in the randomized chunks for efficient indexing in seek.
        size_t sampleCount = 0;
        size_t randomizedChunk = m_randomizedWindowEnd - m_chunkWindowBegin;
        for (size_t index = 0; index < m_sequenceWindow[randomizedChunk].size(); index++)
        {
            sampleCount += m_sequenceWindow[randomizedChunk][index].m_numberOfSamples;
        }

        // Save the sample information.
        ChunkInfo info;
        info.numberOfSamples = sampleCount;
        info.start = m_randomizedChunkInfo.empty() ? 0 : m_randomizedChunkInfo.back().start + m_randomizedChunkInfo.back().numberOfSamples;
        m_randomizedChunkInfo.push_back(info);

        // Update the cursors.
        m_randomizedWindowEnd++;
        m_randomizationCursor = nextRandomizationCursor;
        m_chunkWindowEnd = nextChunkWindowEnd;

        if (m_verbosity)
            fprintf(stderr,
                "SequenceRandomizer::RandomizeNextChunkIfNeeded(): "
                "chunk window [%" PRIu64 "..%u), cursor %" PRIu64 ", "
                "randomized window [%" PRIu64 "..%" PRIu64 "), randomization cursor %" PRIu64 "\n",
                m_chunkWindowBegin, m_chunkWindowEnd,
                m_currentChunkCursor,
                m_chunkWindowBegin, m_randomizedWindowEnd,
                m_randomizationCursor);
    }

    // Sets current cursor to the given sample offset.
    // If offset is in the middle of the sequence, the next sequence is picked up.
    // If there is no sequence, an offset outside the sweep is returned.
    size_t SequenceRandomizer::Seek(size_t sweepSampleOffset, size_t sweep)
    {
        // Determine sample range that is randomized within the chunk window.
        size_t randomizeWindowBeginInSamples = 0;
        size_t randomizedWindowEndInSamples = 0;
        if (!m_randomizedChunkInfo.empty())
        {
            randomizeWindowBeginInSamples = m_randomizedChunkInfo.front().start;
            randomizedWindowEndInSamples = m_randomizedChunkInfo.back().start + m_randomizedChunkInfo.back().numberOfSamples;
        }

        if (m_verbosity)
            fprintf(stderr, "SequenceRandomizer::Seek(): seeking offset %" PRIu64 " in sweep %" PRIu64 "\n",
                sweepSampleOffset,
                sweep);

        if (sweepSampleOffset < randomizeWindowBeginInSamples)
        {
            // The requested offset is before the earliest randomized sequences we still have.
            // Need to start over.
            if (m_verbosity)
                fprintf(stderr, "SequenceRandomizer::Seek(): starting over \n");

            Reset(sweep);
        }
        else if (sweepSampleOffset < randomizedWindowEndInSamples)
        {
            // The requested offset is within the randomized window.
            // We change the current chunk cursor to contain the requested offset.
            if (m_verbosity)
                fprintf(stderr, "SequenceRandomizer::Seek(): offset is within randomized window\n");
            size_t index;
            for (index = 0; index < m_randomizedChunkInfo.size(); index++)
            {
                if (m_randomizedChunkInfo[index].start <= sweepSampleOffset &&
                    sweepSampleOffset < (m_randomizedChunkInfo[index].start + m_randomizedChunkInfo[index].numberOfSamples))
                {
                    break;
                }
            }
            assert(index != m_randomizedChunkInfo.size());

            m_currentChunkCursor = m_chunkWindowBegin + index;
            m_currentSequenceCursor = m_randomizedChunks[m_currentChunkCursor].m_sequencePositionStart;
            m_currentSampleCursor = m_randomizedChunkInfo[index].start;

            // TODO most of the time, we can advance to the right sequence here
            // (unless we need to go past the randomized chunk window)
        }

        // Advance sequence by sequence until the desire offset is reached.
        if (m_verbosity)
            fprintf(stderr, "SequenceRandomizer::Seek(): advancing cursor from %" PRIu64 " to %" PRIu64 "\n",
                m_currentSampleCursor,
                sweepSampleOffset);

        // TODO perhaps optimize this
        ClosedOpenChunkInterval window;
        vector<RandomizedSequenceDescription> tmp;
        while (m_currentSampleCursor < sweepSampleOffset)
        {
            tmp.clear();
            GetNextSequenceDescriptions(1, 1, [](const RandomizedSequenceDescription*) { return true; }, window, tmp);
        }

        return m_currentSampleCursor;
    }

    // Checks if the randomized sequence is valid for a target chunk.
    bool SequenceRandomizer::IsValidForPosition(ChunkIdType chunkIndex, const RandomizedSequenceDescription& seqDesc) const
    {
        const auto& chunk = m_randomizedChunks[chunkIndex];
        return chunk.m_randomizationWindow.m_begin <= seqDesc.m_chunk->m_chunkId && seqDesc.m_chunk->m_chunkId < chunk.m_randomizationWindow.m_end;
    }

    // Gets randomized chunk index using a sequence position in the sweep.
    ChunkIdType SequenceRandomizer::GetChunkIndexForSequencePosition(size_t sequencePosition) const
    {
        auto result = std::upper_bound(
            m_randomizedChunks.begin(),
            m_randomizedChunks.end(),
            sequencePosition,
            [](size_t sp, const RandomizedChunk& c) { return sp < c.m_sequencePositionStart; });
        return (ChunkIdType)(result - 1 - m_randomizedChunks.begin());
    }

    // Add randomizes sequences for the chunk with a given index.
    void SequenceRandomizer::AddRandomizedSequencesForChunk(ChunkIdType chunkIdx)
    {
        assert(chunkIdx == m_chunkWindowEnd);

        const RandomizedChunk& chunk = m_randomizedChunks[chunkIdx];
        std::vector<RandomizedSequenceDescription> chunkSequences;

        m_bufferOriginalSequences.clear();
        m_deserializer->GetSequencesForChunk(chunk.m_original->m_id, m_bufferOriginalSequences);
        chunkSequences.reserve(m_bufferOriginalSequences.size());
        for (size_t k = 0; k < m_bufferOriginalSequences.size(); k++)
        {
            RandomizedSequenceDescription s;
            s.m_indexInOriginalChunk = m_bufferOriginalSequences[k].m_indexInChunk;
            s.m_numberOfSamples = m_bufferOriginalSequences[k].m_numberOfSamples;
            s.m_chunk = &chunk;
            chunkSequences.push_back(s);
        }

        m_sequenceWindow.push_back(std::move(chunkSequences));
        m_chunkWindowEnd++;
    }

    // Gets randomized sequence by the sequence position in the sweep and randomized chunk index.
    RandomizedSequenceDescription& SequenceRandomizer::GetRandomizedSequenceDescriptionByPosition(ChunkIdType chunkIndex, size_t sequenceSweepPosition)
    {
        size_t sequenceOffsetInsideChunk = sequenceSweepPosition - m_randomizedChunks[chunkIndex].m_sequencePositionStart;
        return m_sequenceWindow[chunkIndex - m_chunkWindowBegin][sequenceOffsetInsideChunk];
    }
}}}
