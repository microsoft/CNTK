//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "SequenceRandomizer.h"
#include <algorithm>
#include <utility>
#include <deque>

namespace Microsoft { namespace MSR { namespace CNTK {

    // NOTE: This is an old code, used for legacy randomization to make sure we preserve the same behavior for the tests.
    // TODO: Deprecate when the new randomizer is in place.
    static inline size_t rand(const size_t begin, const size_t end)
    {
        // still only covers 32-bit range
        const size_t randomNumber = ::rand() * RAND_MAX + ::rand();
        return begin + randomNumber % (end - begin);
    }

    SequenceRandomizer::SequenceRandomizer(
        IDataDeserializerPtr deserializer,
        ChunkRandomizerPtr chunkRandomizer)
        : m_randomizedChunks(chunkRandomizer->GetRandomizedChunks()),
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
        srand((unsigned int)randSeed);

        m_sequenceWindow.clear();
        m_chunkWindow.clear();
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

    // Gets next randomized sequence descriptions not exceeding the sample count.
    std::vector<RandomizedSequenceDescription> SequenceRandomizer::GetNextSequenceDescriptions(size_t sampleCount)
    {
        int samples = (int)sampleCount;

        std::vector<RandomizedSequenceDescription> result;
        result.reserve(sampleCount);

        size_t sequenceOffsetInsideChunk = m_currentSequenceCursor - m_randomizedChunks[m_currentChunkCursor].m_sequencePositionStart;
        RandomizedSequenceDescription* sequence = &m_sequenceWindow[m_currentChunkCursor - m_chunkWindowBegin][sequenceOffsetInsideChunk];

        result.push_back(*sequence);
        samples -= (int)sequence->m_numberOfSamples;
        m_currentSequenceCursor++;
        m_currentSampleCursor += (int)sequence->m_numberOfSamples;

        if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkCursor].m_original->m_numberOfSequences)
        {
            // Moving to the next chunk.
            MoveChunkCursor();
        }

        while (samples > 0 && m_currentChunkCursor < m_randomizedChunks.size())
        {
            sequenceOffsetInsideChunk = m_currentSequenceCursor - m_randomizedChunks[m_currentChunkCursor].m_sequencePositionStart;
            sequence = &m_sequenceWindow[m_currentChunkCursor - m_chunkWindowBegin][sequenceOffsetInsideChunk];
            if (samples - sequence->m_numberOfSamples >= 0)
            {
                result.push_back(*sequence);
                m_currentSequenceCursor++;
                samples -= (int)sequence->m_numberOfSamples;
                m_currentSampleCursor += (int)sequence->m_numberOfSamples;

                if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkCursor].m_original->m_numberOfSequences)
                {
                    // Moving to the next chunk.
                    MoveChunkCursor();
                }
            }
        }

        return result;
    }

    // Move the chunk cursor to the next chunk, randomizing more sequences if necessary.
    void SequenceRandomizer::MoveChunkCursor()
    {
        m_currentChunkCursor++;
        RandomizeNextChunkIfNeeded();
    }

    // Release chunks from the chunk window that are not needed anymore.
    void SequenceRandomizer::ReleaseChunks()
    {
        // We should drop chunks, but firstly make sure that they are not used any more.
        // That means the sequence description that we have got from the previous call can still be in the BlockRandomizer.
        size_t currentChunk = std::min(m_currentChunkCursor, m_randomizedChunks.size() - 1);
        size_t candidateToUnload = m_chunkWindowBegin;
        while (candidateToUnload < m_randomizedChunks.size() &&
               candidateToUnload < m_randomizedChunks[currentChunk].m_randomizationWindow.m_begin &&
               m_randomizedChunks[candidateToUnload].m_randomizationWindow.m_end <= m_currentChunkCursor)
        {
            m_sequenceWindow.pop_front();
            m_chunkWindow.pop_front();
            m_randomizedChunkInfo.pop_front();
            m_chunkWindowBegin++;
            candidateToUnload++;
        }
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
        size_t nextChunkWindowEnd = m_randomizedChunks[nextRandomizationCursor - 1].m_randomizationWindow.m_end;

        // Lets page in everything from m_currentRangeEndChunkIndex to endChunkIdx
        for (size_t i = m_chunkWindowEnd; i < nextChunkWindowEnd; ++i)
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
            const size_t currentChunkIdx = GetChunkIndexForSequencePosition(t);

            size_t chunkWindowBegin = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_begin;
            size_t chunkWindowEnd = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_end;

            // Get valid randomization range, expressed in sequence positions.
            size_t posBegin = m_randomizedChunks[chunkWindowBegin].m_sequencePositionStart;
            size_t posEnd = m_randomizedChunks[chunkWindowEnd - 1].SequenceEndPosition();

            for (;;)
            {
                // Pick a sequence position from [posBegin, posEnd)
                const size_t j = rand(posBegin, posEnd);

                // Try again if the sequence currently at j cannot be placed at position i.
                if (!IsValidForPosition(t, GetRandomizedSequenceDescriptionBySequenceId(j)))
                    continue;

                // Try again if the sequence currently at i cannot be placed at position j.
                if (!IsValidForPosition(j, GetRandomizedSequenceDescriptionBySequenceId(t)))
                    continue;

                // Swap and break out.
                std::swap(GetRandomizedSequenceDescriptionBySequenceId(t), GetRandomizedSequenceDescriptionBySequenceId(j)); // TODO old swap was perhaps more efficient
                break;
            }
        }

        // Verify that we got it right
        for (size_t t = firstSequencePositionToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // TODO assert only
            if (!IsValidForPosition(t, GetRandomizedSequenceDescriptionBySequenceId(t)))
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

        if (sweepSampleOffset < randomizeWindowBeginInSamples)
        {
            // The requested offset is before the earliest randomized sequences we still have.
            // Need to start over.
            Reset(sweep + 1);
        }
        else if (sweepSampleOffset < randomizedWindowEndInSamples)
        {
            // The requested offset is within the randomized window.
            // We change the current chunk cursor to contain the requested offset.
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
        // TODO perhaps optimize this
        while (m_currentSampleCursor < sweepSampleOffset)
        {
            GetNextSequenceDescriptions(1);
        }

        return m_currentSampleCursor;
    }

    // Checks if the randomized sequence is valid for a target position using its chunk randomization window.
    bool SequenceRandomizer::IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const
    {
        const auto& chunk = m_randomizedChunks[GetChunkIndexForSequencePosition(targetPosition)];
        return chunk.m_randomizationWindow.m_begin <= seqDesc.m_chunk->m_chunkId && seqDesc.m_chunk->m_chunkId < chunk.m_randomizationWindow.m_end;
    }

    // Gets randomized chunk index using a sequence position in the sweep.
    size_t SequenceRandomizer::GetChunkIndexForSequencePosition(size_t sequencePosition) const
    {
        auto result = std::upper_bound(
            m_randomizedChunks.begin(),
            m_randomizedChunks.end(),
            sequencePosition,
            [](size_t sp, const RandomizedChunk& c) { return sp < c.m_sequencePositionStart; });
        return result - 1 - m_randomizedChunks.begin();
    }

    // Add randomizes sequences for the chunk with a given index.
    void SequenceRandomizer::AddRandomizedSequencesForChunk(size_t chunkIdx)
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
            s.m_id = m_bufferOriginalSequences[k].m_id;
            s.m_numberOfSamples = m_bufferOriginalSequences[k].m_numberOfSamples;
            s.m_chunk = &chunk;
            chunkSequences.push_back(s);
        }

        m_sequenceWindow.push_back(std::move(chunkSequences));
        m_chunkWindow.push_back(chunk);
        m_chunkWindowEnd++;
    }

    // Gets randomized sequence by the sequence id.
    RandomizedSequenceDescription& SequenceRandomizer::GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId)
    {
        size_t globalChunkIdx = GetChunkIndexForSequencePosition(sequenceId);
        size_t sequenceOffsetInsideChunk = sequenceId - m_randomizedChunks[globalChunkIdx].m_sequencePositionStart;
        return m_sequenceWindow[globalChunkIdx - m_chunkWindowBegin][sequenceOffsetInsideChunk];
    }
}}}
