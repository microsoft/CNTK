//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "PartialBlockRandomizer.h"
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
    size_t m_nextSequencePosNotYetRandomized;
    IMetaDataPtr m_metaData;
    PartialBlockRandomizer& m_parent;
    size_t m_currentSequencePosition;

public:
    std::map<size_t, ChunkPtr> m_randomizedSequenceWindowChunks;

    SequenceRandomizer(
        PartialBlockRandomizer& parent,
        IMetaDataPtr metaData,
        const std::vector<RandomizedChunk>& randomizedChunks)
        : m_randomizedChunks(randomizedChunks),
        m_currentRangeBeginChunkIdx(0),
        m_currentRangeEndChunkIdx(0),
        m_nextFramePosNotYetRandomized(0),
        m_metaData(metaData),
        m_parent(parent),
        m_nextSequencePosNotYetRandomized(0),
        m_currentSequencePosition(0)
    {
    }

    std::vector<RandomizedSequenceDescription> GetSequencesForRange(size_t globalts, size_t globalte) // TODO should be simple count i suppose?
    {
        assert(globalts < globalte);
        std::vector<RandomizedSequenceDescription> result;
        result.reserve(globalte - globalts);

        RandomizedSequenceDescription* sequence = &GetRandomizedSequenceDescriptionBySequenceId(m_currentSequencePosition);
        result.push_back(*sequence);

        int samples = (int)(globalte - globalts);
        samples -= (int)sequence->m_original->m_numberOfSamples;
        m_currentSequencePosition++;

        while (samples > 0)
        {
            sequence = &GetRandomizedSequenceDescriptionBySequenceId(m_currentSequencePosition);
            if (samples - sequence->m_original->m_numberOfSamples >= 0)
            {
                result.push_back(*sequence);
                m_currentSequencePosition++;
                samples -= (int)sequence->m_original->m_numberOfSamples;
            }
            else
            {
                break;
            }
        }

        return result;
    }

    void RandomizeSequenceForRange(size_t globalts, size_t globalte)
    {
        assert(globalts < globalte);
        assert(globalts <= m_nextFramePosNotYetRandomized);// is this true between sweeps?
        if (m_nextFramePosNotYetRandomized == m_randomizedChunks.back().globalte())
        {
            return;
        }

        if (globalte < m_nextFramePosNotYetRandomized)
        {
            return;
        }

        if (m_nextSequencePosNotYetRandomized == m_randomizedChunks.back().PositionEnd())
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

        size_t lastFramePosChunkIdx = GetChunkIndexOf(globalte - 1);
        size_t endChunkIdxToRandomize = lastFramePosChunkIdx;
        while (endChunkIdxToRandomize < m_randomizedChunks.size() &&
               m_randomizedChunks[endChunkIdxToRandomize].m_randomizationWindow.m_begin <= lastFramePosChunkIdx)
        {
            endChunkIdxToRandomize++;
        }

        size_t endFramePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].globalte();
        size_t endSequencePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].PositionEnd();

        // Determine the range of chunks that need to be in m_randomizedframerefsWindow for us
        // to perform the necessary randomization
        size_t startChunkIdx = std::min(GetChunkIndexOf(globalts), m_randomizedChunks[GetChunkIndexOf(firstFramePosToRandomize)].m_randomizationWindow.m_begin);
        size_t endChunkIdx = m_randomizedChunks[GetChunkIndexOf(endFramePosToRandomize - 1)].m_randomizationWindow.m_end;

        // Lets drop everything that is outside the new range [startChunkIdx, endChunkIdx)
        for (size_t i = m_currentRangeBeginChunkIdx; i < startChunkIdx; ++i)
        {
            size_t chunkId = m_randomizedSequenceWindow.front()[0].first;
            m_randomizedSequenceWindow.pop_front();
            m_randomizedSequenceWindowChunks.erase(chunkId);

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
            size_t posEnd = m_randomizedChunks[chunkWindowEnd - 1].PositionEnd();

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

    bool IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const
    {
        const auto& chunk = m_randomizedChunks[GetChunkIndexForSequencePosition(targetPosition)];
        return chunk.m_randomizationWindow.m_begin <= seqDesc.m_chunk->m_chunkId && seqDesc.m_chunk->m_chunkId < chunk.m_randomizationWindow.m_end;
    }


    size_t GetChunkIndexForSequencePosition(size_t sequencePosition) const
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

    void Reset(size_t randSeed)
    {
        srand((unsigned int)randSeed);
        size_t sweepts = m_randomizedChunks[0].m_samplePositionStart;

        m_randomizedSequenceWindow.clear();
        m_currentRangeBeginChunkIdx = m_randomizedChunks[0].m_randomizationWindow.m_begin;
        m_currentRangeEndChunkIdx = m_currentRangeBeginChunkIdx;
        m_nextFramePosNotYetRandomized = sweepts;
        m_nextSequencePosNotYetRandomized = 0;
        m_currentSequencePosition = 0;
    }

    RandomizedSequenceDescription& GetRandomizedSequenceDescription(size_t globalts)
    {
        return RandomizedSequenceByGlobalSample(globalts).second;
    }

    RandomizedSequenceDescription& GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId)
    {
        return GetRandomizedSequenceBySequenceId(sequenceId).second;
    }

    void SetSequencePositionTo(size_t globalSample)
    {
        size_t globaltsChunkIdx = GetChunkIndexOf(globalSample);
        assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);

        size_t sampleOffsetInsideChunk = globalSample - m_randomizedChunks[globaltsChunkIdx].m_samplePositionStart;
        auto& sequences = m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx];

        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size(); ++i)
        {
            size_t sequenceSize = sequences[i].second.m_original->m_numberOfSamples;
            if (sequenceSize + numberOfSamples > sampleOffsetInsideChunk)
            {
                break;
            }

            numberOfSamples += sequenceSize;
            sequenceId++;
        }

        m_currentSequencePosition = sequenceId + m_randomizedChunks[globaltsChunkIdx].m_sequencePositionStart;
    }

private:
    void AddRandomizedFramesForChunk(size_t chunkIdx)
    {
        assert(chunkIdx == m_currentRangeEndChunkIdx);

        const RandomizedChunk& chunk = m_randomizedChunks[chunkIdx];
        std::vector<std::pair<unsigned short, RandomizedSequenceDescription>> chunkSequences;
        chunkSequences.reserve(chunk.m_original->numberOfSequences);

        std::vector<SequenceDescriptionPtr> originalSequences = m_parent.m_metaData->GetSequencesForChunk(chunk.m_original->id);
        for (size_t k = 0; k < originalSequences.size(); k++)
        {
            RandomizedSequenceDescription s;
            s.m_original = originalSequences[k];
            s.m_chunk = &chunk;
            chunkSequences.push_back(std::make_pair((unsigned short)chunkIdx, s));
        }

        m_randomizedSequenceWindow.push_back(std::move(chunkSequences));
        m_randomizedSequenceWindowChunks[chunkIdx] = m_parent.m_deserializer->GetChunk(chunk.m_original->id);
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

    unsigned short& TimestampToRandomizedChunkIndex(size_t globalts)
    {
        return RandomizedSequenceByGlobalSample(globalts).first;
    }

    std::pair<unsigned short, RandomizedSequenceDescription>& GetRandomizedSequenceBySequenceId(size_t sequenceId)
    {
        size_t globalChunkIdx = GetChunkIndexForSequencePosition(sequenceId);
        size_t sequenceOffsetInsideChunk = sequenceId - m_randomizedChunks[globalChunkIdx].m_sequencePositionStart;
        return m_randomizedSequenceWindow[globalChunkIdx - m_currentRangeBeginChunkIdx][sequenceOffsetInsideChunk];
    }

    std::pair<unsigned short, RandomizedSequenceDescription>& RandomizedSequenceByGlobalSample(size_t globalts)
    {
        size_t globaltsChunkIdx = GetChunkIndexOf(globalts);
        assert(globaltsChunkIdx < m_currentRangeEndChunkIdx);

        size_t sampleOffsetInsideChunk = globalts - m_randomizedChunks[globaltsChunkIdx].m_samplePositionStart;
        auto& sequences = m_randomizedSequenceWindow[globaltsChunkIdx - m_currentRangeBeginChunkIdx];

        size_t numberOfSamples = 0;
        size_t sequenceId = 0;
        for (size_t i = 0; i < sequences.size(); ++i)
        {
            size_t sequenceSize = sequences[i].second.m_original->m_numberOfSamples;
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

    size_t GetRandomizedSequenceIdByGlobalSample(size_t globalts)
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
            size_t sequenceSize = sequences[i].second.m_original->m_numberOfSamples;
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

    DISABLE_COPY_AND_MOVE(SequenceRandomizer);
};

PartialBlockRandomizer::PartialBlockRandomizer(
    int verbosity,
    size_t randomizationRangeInSamples,
    IDataDeserializerPtr deserializer,
    DistributionMode distributionMode,
    bool useLegacyRandomization,
    IMetaDataPtr metadata)
    : m_verbosity(verbosity),
      m_randomizationRangeInSamples(randomizationRangeInSamples),
      m_deserializer(deserializer),
      m_distributionMode(distributionMode),
      m_useLegacyRandomization(useLegacyRandomization),
      m_sweep(SIZE_MAX),
      m_samplePositionInEpoch(SIZE_MAX),
      m_epochSize(SIZE_MAX),
      m_metaData(metadata),
      m_globalSamplePosition(SIZE_MAX),
      m_sweepTotalNumberOfSamples(0)
{
    assert(deserializer != nullptr);

    m_originalChunks = m_metaData->GetChunkDescriptions();
    m_streams = m_deserializer->GetStreamDescriptions();
    m_sequenceRandomizer = std::make_shared<SequenceRandomizer>(*this, m_metaData, m_randomizedChunks);
}

void PartialBlockRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_sweepTotalNumberOfSamples = m_metaData->GetTotalNumberOfSamples();
    m_config = config;
    if (config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_epochSize = m_sweepTotalNumberOfSamples;
    }
    else
    {
        m_epochSize = config.m_totalEpochSizeInSamples;
    }

    m_globalSamplePosition = m_epochSize * config.m_epochIndex;
    RandomizeForGlobalSamplePosition(m_globalSamplePosition);
    m_sequenceRandomizer->SetSequencePositionTo(m_globalSamplePosition);
}

void PartialBlockRandomizer::RandomizeForGlobalSamplePosition(size_t samplePosition)
{
    size_t sweep = samplePosition / m_sweepTotalNumberOfSamples;
    if (m_sweep != sweep)
    {
        m_sweep = sweep;
        m_sweepStartInSamples = sweep * m_sweepTotalNumberOfSamples;
        RandomizeChunks();
        m_sequenceRandomizer->Reset(m_sweep + 1);

        size_t start = m_sweepStartInSamples;
        size_t end = samplePosition;
        if (start == end)
        {
            // need at least to randomize the first sequences.
            end += 1;
        }

        m_sequenceRandomizer->RandomizeSequenceForRange(start, end);
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
    size_t sequencePosition = 0;
    for (size_t chunkIndex = 0; chunkIndex < m_originalChunks.size(); chunkIndex++)
    {
        const size_t originalChunkIndex = randomizedChunkIndices[chunkIndex];
        const size_t numberOfSamples = m_originalChunks[originalChunkIndex]->numberOfSamples;
        const size_t numberOfSequences = m_originalChunks[originalChunkIndex]->numberOfSequences;

        RandomizedChunk randomizedChunk;
        randomizedChunk.m_chunkId = chunkIndex;
        randomizedChunk.m_original = m_originalChunks[originalChunkIndex].get();
        randomizedChunk.m_samplePositionStart = samplePosition;
        randomizedChunk.m_sequencePositionStart = sequencePosition;
        m_randomizedChunks.push_back(randomizedChunk);
        samplePosition += numberOfSamples;
        sequencePosition += numberOfSequences;
    }

    // Add sentinel
  /*  RandomizedChunk sentinel;
    sentinel.m_original = nullptr;
    sentinel.m_randomizationWindow.m_begin = SIZE_MAX;
    sentinel.m_randomizationWindow.m_end = SIZE_MAX;
    m_randomizedChunks.push_back(sentinel);
    assert(m_originalChunks.size() + 1 == m_randomizedChunks.size());*/

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
               m_randomizedChunks[chunk.m_randomizationWindow.m_end].globalte() - chunk.m_samplePositionStart < halfWindowRange)
        {
            // got more space, move window to the right.
            chunk.m_randomizationWindow.m_end++;
        }
    }

    //m_randomizedChunks.resize(m_randomizedChunks.size() - 1); // remove sentinel.
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
    Sequences result;
    std::vector<RandomizedSequenceDescription> sequences;
    result.m_endOfEpoch = GetNextSequenceDescriptions(sampleCount, sequences);

    if (sequences.size() == 0)
    {
        return result;
    }
    /*
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
    std::swap(chunks, m_chunks);*/

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(sequences.size()));

    // TODO: This will be changed, when we move transformers under the randomizer.
    // TODO: Randomizer won't should not deal with multithreading.
//#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < sequences.size(); ++i)
    {
        const auto& sequenceDescription = sequences[i].m_original;
        auto sequence = m_sequenceRandomizer->m_randomizedSequenceWindowChunks[sequences[i].m_chunk->m_chunkId]->GetSequence(sequenceDescription->m_id);
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
        return true;
    }

    // Check sweep if rerandomization is needed.
    size_t sweepPosition = m_globalSamplePosition % m_sweepTotalNumberOfSamples;
    if (sweepPosition + sampleCount >= m_sweepTotalNumberOfSamples)
    {
        sampleCount = m_sweepTotalNumberOfSamples - sweepPosition;
    }
    assert(sampleCount != 0);

    m_sequenceRandomizer->RandomizeSequenceForRange(m_globalSamplePosition, m_globalSamplePosition + sampleCount);
    std::vector<RandomizedSequenceDescription> sequences = m_sequenceRandomizer->GetSequencesForRange(m_globalSamplePosition, m_globalSamplePosition + sampleCount);

    for (const auto& s : sequences)
    {
        m_globalSamplePosition += s.m_original->m_numberOfSamples;
    }

    result.reserve(sequences.size());
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
        result.assign(sequences.begin() + strideBegin, sequences.begin() + strideEnd);
    }
    else
    {
        LogicError("Not supporeted mode.");
    }

    return false;
}

}
}
}
