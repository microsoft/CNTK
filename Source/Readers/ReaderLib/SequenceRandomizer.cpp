//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "SequenceRandomizer.h"
#include <algorithm>
#include <utility>
#include <deque>

#include "DataReader.h"
#include <random>
#include <set>

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
        m_currentRangeBeginChunkIndex(0),
        m_currentRangeEndChunkIndex(0),
        m_nextSamplePositionNotYetRandomized(0),
        m_nextSequencePositionNotYetRandomized(0),
        m_currentSequencePosition(0),
        m_currentChunkPosition(0),
        m_currentSamplePosition(0),
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

    // Gets next randomized sequence descriptions not exceeding the count.
    std::vector<RandomizedSequenceDescription> SequenceRandomizer::GetNextSequenceDescriptions(size_t sampleCount)
    {
        RandomizeNextSequenceDescriptions(sampleCount);

        int samples = (int)sampleCount;

        std::vector<RandomizedSequenceDescription> result;
        result.reserve(sampleCount);

        assert(IsChunkInWindow(m_currentChunkPosition));

        size_t sequenceOffsetInsideChunk = m_currentSequencePosition - m_randomizedChunks[m_currentChunkPosition].m_sequencePositionStart;
        RandomizedSequenceDescription* sequence = &m_sequenceWindow[m_currentChunkPosition - m_currentRangeBeginChunkIndex][sequenceOffsetInsideChunk];

        result.push_back(*sequence);
        samples -= (int)sequence->m_numberOfSamples;
        m_currentSequencePosition++;
        m_currentSamplePosition += sequence->m_numberOfSamples;

        if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkPosition].m_original->m_numberOfSequences)
        {
            // Moving to the next chunk.
            m_currentChunkPosition++;
        }

        while (samples > 0 && m_currentChunkPosition < m_randomizedChunks.size())
        {
            sequenceOffsetInsideChunk = m_currentSequencePosition - m_randomizedChunks[m_currentChunkPosition].m_sequencePositionStart;
            sequence = &m_sequenceWindow[m_currentChunkPosition - m_currentRangeBeginChunkIndex][sequenceOffsetInsideChunk];
            if (samples - sequence->m_numberOfSamples >= 0)
            {
                result.push_back(*sequence);
                m_currentSequencePosition++;
                samples -= (int)sequence->m_numberOfSamples;
                m_currentSamplePosition += sequence->m_numberOfSamples;

                if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkPosition].m_original->m_numberOfSequences)
                {
                    // Moving to the next chunk.
                    m_currentChunkPosition++;
                }
            }
            else
            {
                break;
            }
        }

        return result;
    }

    void SequenceRandomizer::RandomizeNextSequenceDescriptions(size_t sampleCount)
    {
        assert(m_currentSamplePosition <= m_nextSamplePositionNotYetRandomized);
        if (m_currentSamplePosition + sampleCount <= m_nextSamplePositionNotYetRandomized)
        {
            return;
        }

        if (m_nextSamplePositionNotYetRandomized == m_randomizedChunks.back().SampleEndPosition())
        {
            return;
        }

        if (m_nextSequencePositionNotYetRandomized == m_randomizedChunks.back().SequenceEndPosition())
        {
            assert(false);
            return;
        }

        assert(m_nextSamplePositionNotYetRandomized >= m_randomizedChunks[0].m_samplePositionStart);

        size_t firstSamplePositionToRandomize = m_nextSamplePositionNotYetRandomized;
        size_t firstSequencePositionToRandomize = m_nextSequencePositionNotYetRandomized;

        // Find the smallest chunk index whose windows begin exceeds the chunk index
        // of the sample position we have to randomize (current + sampleCount).
        // We will randomize up to this chunk as the final position of windows end is guaranteed to have been determined
        // when all sequences up to that chunk have been randomized
        size_t lastSamplePositionChunkIdx = GetChunkIndexOf(m_currentSamplePosition + sampleCount - 1);
        size_t endChunkIdxToRandomize = lastSamplePositionChunkIdx;
        while (endChunkIdxToRandomize < m_randomizedChunks.size() &&
            m_randomizedChunks[endChunkIdxToRandomize].m_randomizationWindow.m_begin <= lastSamplePositionChunkIdx)
        {
            endChunkIdxToRandomize++;
        }

        size_t endFramePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].SampleEndPosition();
        size_t endSequencePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].SequenceEndPosition();
        assert(GetChunkIndexOf(endFramePosToRandomize - 1) == endChunkIdxToRandomize - 1);

        // Determine the range of chunks that need to be in m_sequenceWindows for us
        // to perform the necessary randomization
        size_t startChunkIdx = std::min(GetChunkIndexOf(m_currentSamplePosition), m_randomizedChunks[GetChunkIndexOf(firstSamplePositionToRandomize)].m_randomizationWindow.m_begin);
        size_t endChunkIdx = m_randomizedChunks[GetChunkIndexOf(endFramePosToRandomize - 1)].m_randomizationWindow.m_end;
        assert(endChunkIdxToRandomize <= endChunkIdx);

        // Let's drop everything that is outside the new range [startChunkIdx, endChunkIdx)
        for (size_t i = m_currentRangeBeginChunkIndex; i < startChunkIdx; ++i)
        {
            m_sequenceWindow.pop_front();
            m_chunkWindow.pop_front();
            m_currentRangeBeginChunkIndex++;
        }

        // Let's page in everything from m_currentRangeEndChunkIndex to endChunkIdx
        for (size_t i = m_currentRangeEndChunkIndex; i < endChunkIdx; ++i)
        {
            AddRandomizedSequencesForChunk(i);
        }

        for (size_t t = firstSequencePositionToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // Get valid randomization range, expressed in chunks
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

        m_nextSamplePositionNotYetRandomized = endFramePosToRandomize;
        m_nextSequencePositionNotYetRandomized = endSequencePosToRandomize;
    }

    // Resets the current sweep according to the randomization seed provided.
    void SequenceRandomizer::Reset(size_t randSeed)
    {
        srand((unsigned int)randSeed);
        size_t sweepts = m_randomizedChunks[0].m_samplePositionStart;

        m_sequenceWindow.clear();
        m_chunkWindow.clear();
        m_currentRangeBeginChunkIndex = m_randomizedChunks[0].m_randomizationWindow.m_begin;
        m_currentRangeEndChunkIndex = m_currentRangeBeginChunkIndex;
        m_nextSamplePositionNotYetRandomized = sweepts;
        m_nextSequencePositionNotYetRandomized = 0;

        m_currentSequencePosition = 0;
        m_currentChunkPosition = 0;
        m_currentSamplePosition = 0;
    }

    // Sets current sequence position to the sample offset.
    // If offset is in the middle of the sequence, the next sequence is picked up.
    size_t SequenceRandomizer::Seek(size_t offset, size_t sweep)
    {
        size_t chunkIdx = GetChunkIndexOf(offset);
        if (!IsChunkInWindow(chunkIdx))
        {
            Reset(sweep + 1);
            size_t count = offset;
            // We need to randomize at least a single sequence (expectation of RandomizeNextSequenceDescriptions),
            // so we increase count by one if it is zero.
            if (count == 0)
            {
                count++;
            }

            RandomizeNextSequenceDescriptions(count);
        }

        assert(chunkIdx >= m_currentRangeBeginChunkIndex);
        assert(chunkIdx < m_currentRangeEndChunkIndex);

        size_t sampleOffsetInsideChunk = offset - m_randomizedChunks[chunkIdx].m_samplePositionStart;
        auto& sequences = m_sequenceWindow[chunkIdx - m_currentRangeBeginChunkIndex];

        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size() && numberOfSamples < sampleOffsetInsideChunk; ++i)
        {
            numberOfSamples += sequences[i].m_numberOfSamples;
            sequenceId++;
        }

        m_currentSequencePosition = sequenceId + m_randomizedChunks[chunkIdx].m_sequencePositionStart;
        return m_randomizedChunks[chunkIdx].m_samplePositionStart + numberOfSamples;
    }

    // Checks if the randomized sequence is valid for a target position using its chunk randomization window.
    bool SequenceRandomizer::IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const
    {
        const auto& chunk = m_randomizedChunks[GetChunkIndexForSequencePosition(targetPosition)];
        return chunk.m_randomizationWindow.m_begin <= seqDesc.m_chunk->m_chunkId && seqDesc.m_chunk->m_chunkId < chunk.m_randomizationWindow.m_end;
    }

    // Gets chunk index using a sequence position in the sweep.
    // TODO: upper bound should be used instead.
    size_t SequenceRandomizer::GetChunkIndexForSequencePosition(size_t sequencePosition) const
    {
        struct PositionConverter
        {
            size_t m_position;
            PositionConverter(const RandomizedChunk & chunk) : m_position(chunk.m_sequencePositionStart) {};
            PositionConverter(size_t sequencePosition) : m_position(sequencePosition) {};
        };

        auto result = std::lower_bound(m_randomizedChunks.begin(), m_randomizedChunks.end(), sequencePosition,
            [](const PositionConverter& a, const PositionConverter& b)
        {
            return a.m_position <= b.m_position;
        });

        return result - 1 - m_randomizedChunks.begin();
    }

    // Gets chunk index using a sample position in the sweep.
    // TODO: upper bound should be used instead.
    size_t SequenceRandomizer::GetChunkIndexOf(size_t sampleOffsetInSweep)
    {
        size_t low = 0; // TODO: m_currentRangeBeginChunkIdx; can be done more efficient?
        size_t high = m_randomizedChunks.size() - 1;
        while (high > low)
        {
            size_t mid = (high + low) / 2;
            if (sampleOffsetInSweep >= m_randomizedChunks[mid].SampleEndPosition())
            {
                low = mid + 1;
            }
            else if (sampleOffsetInSweep < m_randomizedChunks[mid].m_samplePositionStart)
            {
                assert(mid > 0);
                high = mid - 1;
            }
            else
            {
                return mid;
            }
        }

        assert((high == low) && ((sampleOffsetInSweep >= m_randomizedChunks[low].m_samplePositionStart) && (sampleOffsetInSweep < m_randomizedChunks[low].SampleEndPosition())));
        return low;
    }

    // Checks if chunk index is in the current window.
    bool SequenceRandomizer::IsChunkInWindow(size_t chunkIdx) const
    {
        return chunkIdx >= m_currentRangeBeginChunkIndex && chunkIdx < m_currentRangeEndChunkIndex;
    }

    // Add randomizes sequences for the chunk with a given index.
    void SequenceRandomizer::AddRandomizedSequencesForChunk(size_t chunkIdx)
    {
        assert(chunkIdx == m_currentRangeEndChunkIndex);

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
        m_currentRangeEndChunkIndex++;
    }

    // Gets randomized sequence by the sequence id.
    RandomizedSequenceDescription& SequenceRandomizer::GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId)
    {
        size_t globalChunkIdx = GetChunkIndexForSequencePosition(sequenceId);
        size_t sequenceOffsetInsideChunk = sequenceId - m_randomizedChunks[globalChunkIdx].m_sequencePositionStart;
        return m_sequenceWindow[globalChunkIdx - m_currentRangeBeginChunkIndex][sequenceOffsetInsideChunk];
    }
}}}
