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
        m_currentRangeBeginChunkIdx(0),
        m_currentRangeEndChunkIdx(0),
        m_nextFramePosNotYetRandomized(0),
        m_nextSequencePosNotYetRandomized(0),
        m_currentSequencePosition(0),
        m_currentChunkPosition(0),
        m_currentFramePosition(0),
        m_deserializer(deserializer)
    {
        size_t max = 0;
        for (const auto& c : m_randomizedChunks)
        {
            if (max < c.m_original->numberOfSequences)
            {
                max = c.m_original->numberOfSequences;
            }
        }

        m_bufferOriginalSequences.reserve(max);
    }

    std::vector<RandomizedSequenceDescription> SequenceRandomizer::GetSequencesForRange(size_t sampleCount)
    {
        int samples = (int)sampleCount;

        std::vector<RandomizedSequenceDescription> result;
        result.reserve(sampleCount);

        assert(IsChunkInRange(m_currentChunkPosition));

        size_t sequenceOffsetInsideChunk = m_currentSequencePosition - m_randomizedChunks[m_currentChunkPosition].m_sequencePositionStart;
        RandomizedSequenceDescription* sequence = &m_randomizedSequenceWindow[m_currentChunkPosition - m_currentRangeBeginChunkIdx][sequenceOffsetInsideChunk].second;

        result.push_back(*sequence);
        samples -= (int)sequence->m_numberOfSamples;
        m_currentSequencePosition++;
        m_currentFramePosition += sequence->m_numberOfSamples;

        if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkPosition].m_original->numberOfSequences)
        {
            // Moving to the next chunk.
            m_currentChunkPosition++;
        }

        while (samples > 0 && m_currentChunkPosition < m_randomizedChunks.size())
        {
            sequenceOffsetInsideChunk = m_currentSequencePosition - m_randomizedChunks[m_currentChunkPosition].m_sequencePositionStart;
            sequence = &m_randomizedSequenceWindow[m_currentChunkPosition - m_currentRangeBeginChunkIdx][sequenceOffsetInsideChunk].second;
            if (samples - sequence->m_numberOfSamples >= 0)
            {
                result.push_back(*sequence);
                m_currentSequencePosition++;
                samples -= (int)sequence->m_numberOfSamples;
                m_currentFramePosition += sequence->m_numberOfSamples;

                if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkPosition].m_original->numberOfSequences)
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

    void SequenceRandomizer::RandomizeSequenceForRange(size_t sampleCount)
    {
        assert(m_currentFramePosition <= m_nextFramePosNotYetRandomized);// is this true between sweeps?
        if (m_nextFramePosNotYetRandomized == m_randomizedChunks.back().SampleEndPosition())
        {
            return;
        }

        if (m_currentFramePosition + sampleCount < m_nextFramePosNotYetRandomized)
        {
            return;
        }

        if (m_nextSequencePosNotYetRandomized == m_randomizedChunks.back().SequenceEndPosition())
        {
            assert(false);
            return;
        }

        assert(m_nextFramePosNotYetRandomized >= m_randomizedChunks[0].m_samplePositionStart);

        size_t firstFramePosToRandomize = m_nextFramePosNotYetRandomized;
        size_t firstSequencePosToRandomize = m_nextSequencePosNotYetRandomized;

        // Find the smallest chunk index whose windowbegin exceeds the chunk index
        // of the frame position (globalte - 1). We will randomize up to this chunk
        // as the final position of (globalte - 1) is guaranteed to have been determined
        // when all frames up to that chunk have been randomized

        size_t lastFramePosChunkIdx = GetChunkIndexOf(m_currentFramePosition + sampleCount - 1);
        size_t endChunkIdxToRandomize = lastFramePosChunkIdx;
        while (endChunkIdxToRandomize < m_randomizedChunks.size() &&
            m_randomizedChunks[endChunkIdxToRandomize].m_randomizationWindow.m_begin <= lastFramePosChunkIdx)
        {
            endChunkIdxToRandomize++;
        }

        size_t endFramePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].SampleEndPosition();
        size_t endSequencePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].SequenceEndPosition();

        // Determine the range of chunks that need to be in m_randomizedframerefsWindow for us
        // to perform the necessary randomization
        size_t startChunkIdx = std::min(GetChunkIndexOf(m_currentFramePosition), m_randomizedChunks[GetChunkIndexOf(firstFramePosToRandomize)].m_randomizationWindow.m_begin);
        size_t endChunkIdx = m_randomizedChunks[GetChunkIndexOf(endFramePosToRandomize - 1)].m_randomizationWindow.m_end;

        // Lets drop everything that is outside the new range [startChunkIdx, endChunkIdx)
        for (size_t i = m_currentRangeBeginChunkIdx; i < startChunkIdx; ++i)
        {
            m_randomizedSequenceWindow.pop_front();
            m_randomizedChunkWindow.pop_front();
            m_currentRangeBeginChunkIdx++;
        }

        // Lets page in everything from m_currentRangeEndChunkIdx to endChunkIdx
        for (size_t i = m_currentRangeEndChunkIdx; i < endChunkIdx; ++i)
        {
            AddRandomizedFramesForChunk(i);
        }

        for (size_t t = firstSequencePosToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // Get valid randomization range, expressed in chunks
            const size_t currentChunkIdx = GetChunkIndexForSequencePosition(t);
            //size_t currentChunkIdx = GetChunkIndexOf(t);

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
        for (size_t t = firstSequencePosToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // TODO assert only
            if (!IsValidForPosition(t, GetRandomizedSequenceDescriptionBySequenceId(t)))
            {
                LogicError("BlockRandomizer::Randomize: randomization logic mangled!");
            }
        }

        m_nextFramePosNotYetRandomized = endFramePosToRandomize;
        m_nextSequencePosNotYetRandomized = endSequencePosToRandomize;
    }

    void SequenceRandomizer::Reset(size_t randSeed)
    {
        srand((unsigned int)randSeed);
        size_t sweepts = m_randomizedChunks[0].m_samplePositionStart;

        m_randomizedSequenceWindow.clear();
        m_randomizedChunkWindow.clear();
        m_currentRangeBeginChunkIdx = m_randomizedChunks[0].m_randomizationWindow.m_begin;
        m_currentRangeEndChunkIdx = m_currentRangeBeginChunkIdx;
        m_nextFramePosNotYetRandomized = sweepts;
        m_nextSequencePosNotYetRandomized = 0;

        m_currentSequencePosition = 0;
        m_currentChunkPosition = 0;
    }

    void SequenceRandomizer::SetSequencePositionTo(size_t offset, size_t sweep)
    {
        size_t globaltsChunkIdx = GetChunkIndexOf(offset);
        if (!this->IsChunkInRange(globaltsChunkIdx))
        {
            Reset(sweep + 1);
            size_t count = offset;
            if (count == 0)
            {
                count++;
            }

            // TODO: should not really require the data here, this can lead to many chunks in memory.
            RandomizeSequenceForRange(count);
        }

        assert(globaltsChunkIdx >= m_currentRangeBeginChunkIdx);
        assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);

        size_t sampleOffsetInsideChunk = offset - m_randomizedChunks[globaltsChunkIdx].m_samplePositionStart;
        auto& sequences = m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx];

        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size(); ++i)
        {
            size_t sequenceSize = sequences[i].second.m_numberOfSamples;
            if (sequenceSize + numberOfSamples > sampleOffsetInsideChunk)
            {
                break;
            }

            numberOfSamples += sequenceSize;
            sequenceId++;
        }

        m_currentSequencePosition = sequenceId + m_randomizedChunks[globaltsChunkIdx].m_sequencePositionStart;
    }


    bool SequenceRandomizer::IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const
    {
        const auto& chunk = m_randomizedChunks[GetChunkIndexForSequencePosition(targetPosition)];
        return chunk.m_randomizationWindow.m_begin <= seqDesc.m_chunk->m_chunkId && seqDesc.m_chunk->m_chunkId < chunk.m_randomizationWindow.m_end;
    }


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

    RandomizedSequenceDescription& SequenceRandomizer::GetRandomizedSequenceDescription(size_t globalts)
    {
        return RandomizedSequenceByGlobalSample(globalts).second;
    }

    RandomizedSequenceDescription& SequenceRandomizer::GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId)
    {
        return GetRandomizedSequenceBySequenceId(sequenceId).second;
    }

    size_t SequenceRandomizer::GetChunkIndexOf(size_t t)
    {
        //assert(t >= m_randomizedChunks[m_currentRangeBeginChunkIdx].m_samplePositionStart);
        size_t low = 0; // m_currentRangeBeginChunkIdx; can be done more efficient?
        size_t high = m_randomizedChunks.size() - 1;
        while (high > low)
        {
            size_t mid = (high + low) / 2;
            if (t >= m_randomizedChunks[mid].SampleEndPosition())
            {
                low = mid + 1;
            }
            else if (t < m_randomizedChunks[mid].m_samplePositionStart)
            {
                assert(mid > 0);
                high = mid - 1;
            }
            else
            {
                return mid;
            }
        }

        assert((high == low) && ((t >= m_randomizedChunks[low].m_samplePositionStart) && (t < m_randomizedChunks[low].SampleEndPosition())));
        return low;
    }

    bool SequenceRandomizer::IsChunkInRange(size_t chunkIdx) const
    {
        return chunkIdx >= m_currentRangeBeginChunkIdx && chunkIdx < m_currentRangeEndChunkIdx;
    }

    void SequenceRandomizer::AddRandomizedFramesForChunk(size_t chunkIdx)
    {
        assert(chunkIdx == m_currentRangeEndChunkIdx);

        const RandomizedChunk& chunk = m_randomizedChunks[chunkIdx];
        std::vector<std::pair<unsigned short, RandomizedSequenceDescription>> chunkSequences;

        m_bufferOriginalSequences.clear();
        m_deserializer->GetSequencesForChunk(chunk.m_original->id, m_bufferOriginalSequences);
        chunkSequences.reserve(m_bufferOriginalSequences.size());
        for (size_t k = 0; k < m_bufferOriginalSequences.size(); k++)
        {
            RandomizedSequenceDescription s;
            s.m_id = m_bufferOriginalSequences[k].m_id;
            s.m_numberOfSamples = m_bufferOriginalSequences[k].m_numberOfSamples;
            s.m_chunk = &chunk;
            chunkSequences.push_back(std::make_pair((unsigned short)chunkIdx, s));
        }

        m_randomizedSequenceWindow.push_back(std::move(chunkSequences));
        m_randomizedChunkWindow.push_back(chunk);
        m_currentRangeEndChunkIdx++;
    }

    std::pair<unsigned short, RandomizedSequenceDescription>& SequenceRandomizer::GetRandomizedSequenceBySequenceId(size_t sequenceId)
    {
        size_t globalChunkIdx = GetChunkIndexForSequencePosition(sequenceId);
        size_t sequenceOffsetInsideChunk = sequenceId - m_randomizedChunks[globalChunkIdx].m_sequencePositionStart;
        return m_randomizedSequenceWindow[globalChunkIdx - m_currentRangeBeginChunkIdx][sequenceOffsetInsideChunk];
    }

    std::pair<unsigned short, RandomizedSequenceDescription>& SequenceRandomizer::RandomizedSequenceByGlobalSample(size_t globalts)
    {
        size_t globaltsChunkIdx = GetChunkIndexOf(globalts);
        assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);

        size_t sampleOffsetInsideChunk = globalts - m_randomizedChunks[globaltsChunkIdx].m_samplePositionStart;
        auto& sequences = m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx];

        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size(); ++i)
        {
            size_t sequenceSize = sequences[i].second.m_numberOfSamples;
            if (sequenceSize + numberOfSamples > sampleOffsetInsideChunk)
            {
                break;
            }

            numberOfSamples += sequenceSize;
            sequenceId++;
        }

        assert(sequenceId < sequences.size());
        return m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx][sequenceId];
    }

    size_t SequenceRandomizer::GetRandomizedSequenceIdByGlobalSample(size_t globalts)
    {
        size_t globaltsChunkIdx = GetChunkIndexOf(globalts);
        assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);

        size_t sampleOffsetInsideChunk = globalts - m_randomizedChunks[globaltsChunkIdx].m_samplePositionStart;
        auto& sequences = m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx];
        size_t currentSequenceId = m_randomizedChunks[globaltsChunkIdx].m_sequencePositionStart;
        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size(); ++i)
        {
            size_t sequenceSize = sequences[i].second.m_numberOfSamples;
            if (sequenceSize + numberOfSamples > sampleOffsetInsideChunk)
            {
                break;
            }

            numberOfSamples += sequenceSize;
            sequenceId++;
            currentSequenceId++;
        }

        assert(sequenceId < sequences.size());
        return currentSequenceId;
    }

}}}
