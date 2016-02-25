//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "PartialBlockRandomizer.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <deque>

#include "DataReader.h"
#include <random>
#include "../HTKMLFReader/biggrowablevectors.h"
#include <set>
#include <unordered_set>

namespace Microsoft { namespace MSR { namespace CNTK {

// NOTE: This is an old code, used for legacy randomization to make sure we preserve the same behavior for the tests.
// TODO: Deprecate when the new randomizer is in place.
static inline size_t rand(const size_t begin, const size_t end)
{
    // still only covers 32-bit range
    const size_t randomNumber = ::rand() * RAND_MAX + ::rand();
    return begin + randomNumber % (end - begin);
}

// NOTE: This is an old code, used for legacy randomization to make sure we preserve the same behavior for the tests.
// TODO: Deprecate when the new randomizer is in place.
template <typename TVector>
void RandomShuffle(TVector& v, size_t randomSeed)
{
    if (v.size() > RAND_MAX * static_cast<size_t>(RAND_MAX))
    {
        RuntimeError("RandomShuffle: too large set: need to change to different random generator!");
    }

    srand(static_cast<unsigned int>(randomSeed));
    foreach_index(currentLocation, v)
    {
        // Pick a random location a location and swap with current
        const size_t randomLocation = rand(0, v.size());
        std::swap(v[currentLocation], v[randomLocation]);
    }
}

class SequenceRandomizer
{
    const std::vector<RandomizedChunk>& m_randomizedChunks;

    // A rolling windows of chunks of framerefs used for randomization in frame mode
    // Along with each frameref, we also store the chunk index of the original frame
    // at that index before randomization, to be used for determining the chunk range
    // to be used for randomization of that frame's position
    std::deque<std::vector<std::pair<unsigned short, RandomizedSequenceDescription>>> m_randomizedSequenceWindow;

    size_t m_currentRangeBeginChunkIdx;
    size_t m_currentRangeEndChunkIdx;

    size_t m_nextFramePosNotYetRandomized;
    ICorpusPtr m_corpus;
    PartialBlockRandomizer& m_parent;

public:
    SequenceRandomizer(
        PartialBlockRandomizer& parent,
        ICorpusPtr corpus,
        const std::vector<RandomizedChunk>& randomizedChunks)
        : m_randomizedChunks(randomizedChunks),
        m_currentRangeBeginChunkIdx(0),
        m_currentRangeEndChunkIdx(0),
        m_nextFramePosNotYetRandomized(0),
        m_corpus(corpus),
        m_parent(parent)
    {
    }

    std::vector<RandomizedSequenceDescription> GetSequencesForRange(size_t globalts, size_t globalte)
    {
        std::vector<RandomizedSequenceDescription> result;
        RandomizedSequenceDescription sequence = GetRandomizedSequenceDescription(globalts);

        result.push_back(sequence);
        globalts += sequence.m_original->m_numberOfSamples;

        while (globalts < globalte)
        {
            sequence = GetRandomizedSequenceDescription(globalts);
            if (sequence.m_original->m_numberOfSamples + globalts < globalte)
            {
                result.push_back(sequence);
            }

            globalts += sequence.m_original->m_numberOfSamples;
        }

        return result;
    }

    void RandomizeSequenceForRange(size_t globalts, size_t globalte)
    {
        assert(globalts <= m_nextFramePosNotYetRandomized);// is this true between sweeps?
        if (m_nextFramePosNotYetRandomized == m_randomizedChunks.back().globalte())
        {
            // not clear why. there is some expectation from the calling side it seems. last one?
            return;
        }

        assert(m_nextFramePosNotYetRandomized >= m_randomizedChunks[0].m_samplePositionStart);

        size_t firstFramePosToRandomize = m_nextFramePosNotYetRandomized;

        // Find the smallest chunk index whose windowbegin exceeds the chunk index
        // of the frame position (globalte - 1). We will randomize up to this chunk
        // as the final position of (globalte - 1) is guaranteed to have been determined 
        // when all frames up to that chunk have been randomized

        size_t lastFramePosChunkIdx = GetChunkIndexOf(globalte - 1);
        size_t endChunkIdxToRandomize = lastFramePosChunkIdx;
        while (endChunkIdxToRandomize < m_randomizedChunks.size() &&
               m_randomizedChunks[endChunkIdxToRandomize].m_randomizationWindow.m_begin <= lastFramePosChunkIdx)
        {
            endChunkIdxToRandomize++;
        }

        size_t endFramePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].globalte();

        // Determine the range of chunks that need to be in m_randomizedframerefsWindow for us
        // to perform the necessary randomization
        size_t startChunkIdx = std::min(GetChunkIndexOf(globalts), m_randomizedChunks[GetChunkIndexOf(firstFramePosToRandomize)].m_randomizationWindow.m_begin);
        size_t endChunkIdx = m_randomizedChunks[GetChunkIndexOf(endFramePosToRandomize - 1)].m_randomizationWindow.m_end;

        // Lets drop everything that is outside the new range [startChunkIdx, endChunkIdx)
        for (size_t i = m_currentRangeBeginChunkIdx; i < startChunkIdx; ++i)
        {
            m_randomizedSequenceWindow.pop_front();
            m_currentRangeBeginChunkIdx++;
        }

        // Lets page in everything from m_currentRangeEndChunkIdx to endChunkIdx
        for (size_t i = m_currentRangeEndChunkIdx; i < endChunkIdx; ++i)
        {
            AddRandomizedFramesForChunk(i);
        }

        // now randomize them --we use the nested loop again to avoid storing a backpointer
        // The condition is that a randomized frame may not be moved out of its associated chunk window.
        // The catual range we randomize is up to the last frame that position (globalte - 1) could
        // potentially swap with
        for (size_t t = firstFramePosToRandomize; t < endFramePosToRandomize; ++t)
        {
            size_t currentChunkIdx = GetChunkIndexOf(t);

            size_t chunkWindowBegin = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_begin;
            size_t chunkWindowEnd = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_end;

            // Chunk implies that if we are at position 't', we are guaranteed to have chunks [chunkWindowBegin, chunkWindowEnd) in RAM.
            // These chunks are associated with a range of frame positions.
            // It is implied that if we are at position 't', the frames covered by chunks [chunkWindowBegin, chunkWindowEnd) are in RAM.
            const size_t postbegin = m_randomizedChunks[chunkWindowBegin].m_samplePositionStart;
            const size_t postend = m_randomizedChunks[chunkWindowEnd - 1].globalte();
            // The position that this frame gets randomized to must be guaranteed to belong to a chunk within [postbegin, postend).

            for (;;) // (randomization retry loop)
            {
                size_t tswap = rand(postbegin, postend); // random frame position within allowed range
                // We want to swap 't' to 'tswap' and 'tswap' to 't'.
                //  - Both may have been swapped before.
                //  - Both must stay within the randomization window of their respective position.
                // check admissibility of where the element at 'tswap' gets swapped to 't' (range = [windowbegin,windowend))
                size_t tswapchunkindex = GetRandomizedSequenceDescription(tswap).m_chunk->m_chunkId;
                if (tswapchunkindex < chunkWindowBegin || tswapchunkindex >= chunkWindowEnd)
                    continue;

                // check admissibility of where the element at t gets swapped to (which is frame position 'tswap')
                const size_t sourcechunkindex = GetRandomizedSequenceDescription(t).m_chunk->m_chunkId;
                size_t targetchunkindex = TimestampToRandomizedChunkIndex(tswap); // chunk associated with this frame position defines value range
                const auto &targetchunk = m_randomizedChunks[targetchunkindex];
                const size_t targetwindowbegin = targetchunk.m_randomizationWindow.m_begin;
                const size_t targetwindowend = targetchunk.m_randomizationWindow.m_end;
                if (sourcechunkindex < targetwindowbegin || sourcechunkindex >= targetwindowend)
                    continue;
                // admissible--swap the two
                ::swap(GetRandomizedSequenceDescription(t), GetRandomizedSequenceDescription(tswap));

                // do a post-check if we got it right  --we seem not to
                if (IsFramePositionValid(t) && IsFramePositionValid(tswap))
                    break;
                // not valid: swap them back and try again  --we actually discovered a bug in the code above
                ::swap(GetRandomizedSequenceDescription(t), GetRandomizedSequenceDescription(tswap));
                fprintf(stderr, "lazyrandomization: BUGBUG --invalid swapping condition detected\n");
            }
        }

        m_nextFramePosNotYetRandomized = endFramePosToRandomize;

        // Verify no frameref has violated its range constraints
        for (size_t t = globalts; t < globalte; ++t)
        {
            size_t chunkIdx = TimestampToRandomizedChunkIndex(t);
            const auto &chunk = m_randomizedChunks[chunkIdx]; // for window and chunkdata
            const size_t poswindowbegin = chunk.m_randomizationWindow.m_begin;
            const size_t poswindowend = chunk.m_randomizationWindow.m_end;

            const size_t randomizedchunkindex = GetRandomizedSequenceDescription(t).m_chunk->m_chunkId;
            if (randomizedchunkindex < poswindowbegin || randomizedchunkindex >= poswindowend)
                LogicError("lazyrandomization: nope, you got frame randomization wrong, dude");
        }
    }

    void Reset(size_t randSeed)
    {
        srand((unsigned int)randSeed);
        size_t sweepts = m_randomizedChunks[0].m_samplePositionStart;

        m_randomizedSequenceWindow.clear();
        m_currentRangeBeginChunkIdx = m_randomizedChunks[0].m_randomizationWindow.m_begin;
        m_currentRangeEndChunkIdx = m_currentRangeBeginChunkIdx;
        m_nextFramePosNotYetRandomized = sweepts;
    }

    RandomizedSequenceDescription& GetRandomizedSequenceDescription(size_t globalts)
    {
        return RandomizedSequenceByGlobalSample(globalts).second;
    }

private:
    void AddRandomizedFramesForChunk(size_t chunkIdx)
    {
        assert(chunkIdx == m_currentRangeEndChunkIdx);

        const RandomizedChunk& chunk = m_randomizedChunks[chunkIdx];
        std::vector<std::pair<unsigned short, RandomizedSequenceDescription>> chunkSequences;
        chunkSequences.reserve(chunk.m_original->numberOfSequences);

        SequenceDescriptions originalSequences = m_parent.m_corpus->GetSequencesForChunk(chunk.m_original->id);
        for (size_t k = 0; k < originalSequences.size(); k++)
        {
            RandomizedSequenceDescription s;
            s.m_original = originalSequences[k];
            s.m_chunk = &chunk;
            chunkSequences.push_back(std::make_pair((unsigned short)chunkIdx, s));
        }

        m_randomizedSequenceWindow.push_back(std::move(chunkSequences));
        m_currentRangeEndChunkIdx++;
    }

    size_t GetChunkIndexOf(size_t t)
    {
        assert(t >= m_randomizedChunks[m_currentRangeBeginChunkIdx].m_samplePositionStart);
        size_t low = m_currentRangeBeginChunkIdx;
        size_t high = m_randomizedChunks.size() - 1;
        while (high > low)
        {
            size_t mid = (high + low) / 2;
            if (t >= m_randomizedChunks[mid].globalte())
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

        assert((high == low) && ((t >= m_randomizedChunks[low].m_samplePositionStart) && (t < m_randomizedChunks[low].globalte())));
        return low;
    }

    // helper for testing whether a swapped frame position is valid (w.r.t. beign in RAM when being at position 't')
    bool IsFramePositionValid(const size_t t)
    {
        // look up valid range for time position
        const size_t positionchunkindex = TimestampToRandomizedChunkIndex(t); // position 't' lies within this original chunk (relationship is monotonous, not random)
        const auto &chunk = m_randomizedChunks[positionchunkindex];
        // get in-RAM chunk range for this frame position (shared across all frame positions within the same chunk)
        const size_t poswindowbegin = chunk.m_randomizationWindow.m_begin; // rolling window over chunks (which under the hood have been randomized)
        const size_t poswindowend = chunk.m_randomizationWindow.m_begin;
        // Chunk implies that if we are at position 't', we are guaranteed to have chunks [poswindowbegin, poswindowend) in RAM.

        // now see if the randomized location is within that window
        const size_t actualchunkindexforpos = GetRandomizedSequenceDescription(t).m_chunk->m_chunkId; // where this frame pos has been mapped to
        return actualchunkindexforpos >= poswindowbegin && actualchunkindexforpos < poswindowend;
        // We only need to test the chunk index. Utterance and frame can be randomized within a chunk as we want, as long it is in RAM.
    }

    unsigned short& TimestampToRandomizedChunkIndex(size_t globalts)
    {
        return RandomizedSequenceByGlobalSample(globalts).first;
    }

    std::pair<unsigned short, RandomizedSequenceDescription>& RandomizedSequenceByGlobalSample(size_t globalts)
    {
        size_t globaltsChunkIdx = GetChunkIndexOf(globalts);
        assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);

        size_t sampleOffsetInsideChunk = globalts - m_randomizedChunks[globaltsChunkIdx].m_samplePositionStart;
        auto sequences = m_corpus->GetSequencesForChunk(m_randomizedChunks[globaltsChunkIdx].m_original->id);
        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size(); ++i)
        {
            if (sequences[i]->m_numberOfSamples + numberOfSamples > sampleOffsetInsideChunk)
            {
                break;
            }
            numberOfSamples += sequences[i]->m_numberOfSamples;
            sequenceId++;
        }

        assert(sequenceId < sequences.size());
        return m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx][sequenceId];
    }

    DISABLE_COPY_AND_MOVE(SequenceRandomizer);
};

PartialBlockRandomizer::PartialBlockRandomizer(
    int verbosity,
    size_t randomizationRangeInSamples,
    IDataDeserializerPtr deserializer,
    DistributionMode distributionMode,
    bool useLegacyRandomization,
    ICorpusPtr corpus)
    : m_verbosity(verbosity),
      m_randomizationRangeInSamples(randomizationRangeInSamples),
      m_deserializer(deserializer),
      m_distributionMode(distributionMode),
      m_useLegacyRandomization(useLegacyRandomization),
      m_sweep(SIZE_MAX),
      m_samplePositionInEpoch(SIZE_MAX),
      m_epochSize(SIZE_MAX),
      m_corpus(corpus),
      m_globalSamplePosition(SIZE_MAX)
{
    assert(deserializer != nullptr);

    m_originalChunks = m_corpus->GetChunkDescriptions();
    m_streams = m_deserializer->GetStreamDescriptions();
    m_sequenceRandomizer = std::make_unique<SequenceRandomizer>(*this, m_corpus, m_randomizedChunks);
}

void PartialBlockRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_config = config;
    if (config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_epochSize = m_corpus->TotalNumberOfSamples();
    }
    else
    {
        m_epochSize = config.m_totalEpochSizeInSamples;
    }

    m_globalSamplePosition = m_epochSize * config.m_epochIndex;

    RandomizeForGlobalSamplePosition(m_globalSamplePosition);
}

void PartialBlockRandomizer::RandomizeForGlobalSamplePosition(size_t samplePosition)
{
    size_t sweep = samplePosition / m_corpus->TotalNumberOfSamples();
    if (m_sweep != sweep)
    {
        m_sweep = sweep;
        m_sweepStartInSamples = sweep * m_corpus->TotalNumberOfSamples();
        RandomizeChunks();
        m_sequenceRandomizer->RandomizeSequenceForRange(m_sweepStartInSamples, samplePosition);
    }
}

void PartialBlockRandomizer::RandomizeChunks()
{
    std::vector<size_t> randomizedChunkIndices;
    randomizedChunkIndices.reserve(m_originalChunks.size());
    for (size_t i = 0; i < m_originalChunks.size(); i++)
    {
        randomizedChunkIndices.push_back(i);
    }

    if (m_useLegacyRandomization)
    {
        RandomShuffle(randomizedChunkIndices, m_sweep);
    }
    else
    {
        std::mt19937 m_rng(static_cast<int>(m_sweep));
        std::shuffle(randomizedChunkIndices.begin(), randomizedChunkIndices.end(), m_rng);
    }

    // Place randomized chunks on global timeline
    m_randomizedChunks.clear();
    m_randomizedChunks.reserve(m_originalChunks.size());
    size_t samplePosition = m_sweepStartInSamples;
    for (size_t chunkIndex = 0; chunkIndex < m_originalChunks.size(); chunkIndex++)
    {
        const size_t originalChunkIndex = randomizedChunkIndices[chunkIndex];
        const size_t numberOfSamples = m_originalChunks[originalChunkIndex].numberOfSamples;

        RandomizedChunk randomizedChunk;
        randomizedChunk.m_original = &m_originalChunks[originalChunkIndex];
        randomizedChunk.m_samplePositionStart = samplePosition;
        m_randomizedChunks.push_back(randomizedChunk);
        samplePosition += numberOfSamples;
    }

    // Add sentinel
    RandomizedChunk sentinel;
    sentinel.m_original = nullptr;
    sentinel.m_randomizationWindow.m_begin = SIZE_MAX;
    sentinel.m_randomizationWindow.m_end = SIZE_MAX;
    m_randomizedChunks.push_back(sentinel);
    assert(m_originalChunks.size() + 1 == m_randomizedChunks.size());

    // For each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
    size_t halfWindowRange = m_randomizationRangeInSamples / 2;
    for (size_t chunkId = 0; chunkId < m_originalChunks.size(); chunkId++)
    {
        auto& chunk = m_randomizedChunks[chunkId];

        // start with the range of left neighbor
        if (chunkId == 0)
        {
            chunk.m_randomizationWindow.m_begin = 0;
            chunk.m_randomizationWindow.m_end = 1;
        }
        else
        {
            chunk.m_randomizationWindow.m_begin = m_randomizedChunks[chunkId - 1].m_randomizationWindow.m_begin; // might be too early
            chunk.m_randomizationWindow.m_end = m_randomizedChunks[chunkId - 1].m_randomizationWindow.m_end; // might have more space
        }

        // Need to adapt now.
        while (chunk.m_samplePositionStart - m_randomizedChunks[chunk.m_randomizationWindow.m_begin].m_samplePositionStart
               > halfWindowRange)
        {
            // too early, need to increase
            chunk.m_randomizationWindow.m_begin++;
        }

        while (chunk.m_randomizationWindow.m_end < m_originalChunks.size() &&
               m_randomizedChunks[chunk.m_randomizationWindow.m_end + 1].m_samplePositionStart - chunk.m_samplePositionStart < halfWindowRange)
        {
            // got more space, move window to the right.
            chunk.m_randomizationWindow.m_end++;
        }
    }
}
/*
bool PartialBlockRandomizer::RandomizeIfNewSweepIsEntered()
{
    // Check that StartEpoch() was called
    assert(m_sequencePositionInSweep != SIZE_MAX);

    if (m_sequencePositionInSweep >= m_numSequences)
    {
        if (m_verbosity > 0)
            std::cerr << __FUNCTION__ << ": re-randomizing for sweep " << m_sweep
            << " in " << (m_frameMode ? "frame" : "utterance") << " mode" << endl;
        m_sweep++;
        m_sweepStartInSamples += m_numSamples;
        Randomize();
        m_sequenceRandomizer->Reset(m_sweep);
        m_sequencePositionInSweep -= m_numSequences;
        assert(m_sequencePositionInSweep < m_numSequences); // cannot jump ahead more than a sweep
        return true;
    };

    return false;
}*/

Sequences PartialBlockRandomizer::GetNextSequences(size_t sampleCount)
{
    assert(m_samplePositionInEpoch != SIZE_MAX); // SetEpochConfiguration() must be called first

    std::vector<size_t> originalIds;
    std::unordered_set<size_t> originalChunks;

    Sequences result;
    std::vector<RandomizedSequenceDescription> sequences;
    result.m_endOfEpoch = GetNextSequenceDescriptions(sampleCount, sequences);

    if (sequences.size() == 0)
    {
        return result;
    }

    std::map<size_t, ChunkPtr> chunks;
    std::set<size_t> tracked;
    for (auto s: sequences)
    {
        size_t chunkIndex = s.m_original->m_chunkId;
        if (tracked.find(chunkIndex) != tracked.end())
        {
            continue;
        }

        auto chunk = m_chunks.find(chunkIndex);
        if (chunk != m_chunks.end())
        {
            chunks[chunkIndex] = chunk->second;
            tracked.insert(chunkIndex);
        }
        else
        {
            chunks[chunkIndex] = m_deserializer->GetChunk(chunkIndex);
        }
    }
    std::swap(chunks, m_chunks);

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(sequences.size()));

    // TODO: This will be changed, when we move transformers under the randomizer.
    // TODO: Randomizer won't should not deal with multithreading.
//#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < sequences.size(); ++i)
    {
        const auto& sequenceDescription = sequences[i].m_original;
        auto sequence = m_chunks[sequenceDescription->m_chunkId]->GetSequence(originalIds[i]);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    }

    return result;
}

bool PartialBlockRandomizer::GetNextSequenceDescriptions(size_t sampleCount, std::vector<RandomizedSequenceDescription>& result)
{
    RandomizeForGlobalSamplePosition(m_globalSamplePosition);

    // Check epoch.
    if (m_globalSamplePosition - m_config.m_epochIndex * m_epochSize + sampleCount >= m_epochSize)
    {
        sampleCount = m_epochSize - m_globalSamplePosition + m_config.m_epochIndex * m_epochSize;
    }

    if (sampleCount <= 0)
    {
        return false;
    }

    // Check sweep if rerandomization is needed.
    size_t sweepPosition = m_globalSamplePosition % m_corpus->TotalNumberOfSamples();
    if (sweepPosition + sampleCount >= m_corpus->TotalNumberOfSamples())
    {
        sampleCount = m_corpus->TotalNumberOfSamples() - sweepPosition;
    }

    m_sequenceRandomizer->RandomizeSequenceForRange(m_globalSamplePosition, m_globalSamplePosition + sampleCount);
    std::vector<RandomizedSequenceDescription> sequences = m_sequenceRandomizer->GetSequencesForRange(m_globalSamplePosition, m_globalSamplePosition + sampleCount);

    if (m_distributionMode == DistributionMode::chunk)
    {
        for (const auto& sequence : sequences)
        {
            if (sequence.m_chunk->m_chunkId % m_config.m_numberOfWorkers == m_config.m_workerRank)
            {
                result.push_back(sequence);
            }
        }
    }
    else if (m_distributionMode == DistributionMode::sequence)
    {
        size_t strideBegin = sampleCount * m_config.m_workerRank / m_config.m_numberOfWorkers;
        size_t strideEnd = sampleCount * (m_config.m_workerRank + 1) / m_config.m_numberOfWorkers;
        for (size_t i = strideBegin; i < strideEnd; ++i)
        {
            result.assign(sequences.begin() + strideBegin, sequences.begin() + strideEnd);
        }
    }
    else
    {
        LogicError("Not supporeted mode.");
    }

    return true;
}

/*
size_t PartialBlockRandomizer::GetChunkIndexForSequencePosition(size_t sequencePosition) const
{
assert(sequencePosition <= m_numSamples);

struct PositionConverter
{
size_t m_position;
PositionConverter(const RandomizedChunk & chunk) : m_position(chunk.m_info.m_sequencePositionStart) {};
PositionConverter(size_t sequencePosition) : m_position(sequencePosition) {};
};

auto result = std::lower_bound(m_randomizedChunks.begin(), m_randomizedChunks.end(), sequencePosition,
[](const PositionConverter& a, const PositionConverter& b)
{
return a.m_position <= b.m_position;
});

return result - m_randomizedChunks.begin() - 1;
}

bool PartialBlockRandomizer::IsValidForPosition(size_t targetPosition, const SequenceDescription& seqDesc) const
{
const auto& chunk = m_randomizedChunks[GetChunkIndexForSequencePosition(targetPosition)];
return chunk.m_windowBegin <= seqDesc.m_chunkId && seqDesc.m_chunkId < chunk.m_windowEnd;
}
*/

}
}
}
