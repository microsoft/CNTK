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

PartialBlockRandomizer::PartialBlockRandomizer(
    int verbosity,
    size_t randomizationRangeInSamples,
    IDataDeserializerPtr deserializer,
    DistributionMode distributionMode,
    bool useLegacyRandomization)
    : m_verbosity(verbosity),
      m_deserializer(deserializer),
      m_distributionMode(distributionMode),
      m_sweep(SIZE_MAX),
      m_epochSize(SIZE_MAX),
      m_globalSamplePosition(SIZE_MAX),
      m_sweepTotalNumberOfSamples(0),
      m_lastSeenChunk(SIZE_MAX),
      m_chunkRandomizer(std::make_shared<ChunkRandomizer>(deserializer, useLegacyRandomization, randomizationRangeInSamples))
{
    assert(deserializer != nullptr);

    m_streams = m_deserializer->GetStreamDescriptions();
    m_sequenceRandomizer = std::make_shared<SequenceRandomizer>(m_deserializer, m_chunkRandomizer);

    m_sweepTotalNumberOfSamples = 0;
    for (auto const & chunk : m_deserializer->GetChunkDescriptions())
    {
        m_sweepTotalNumberOfSamples += chunk->numberOfSamples;
    }
}

void PartialBlockRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_config = config;
    if (config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_epochSize = m_sweepTotalNumberOfSamples;
    }
    else
    {
        m_epochSize = config.m_totalEpochSizeInSamples;
    }

    m_sequenceRandomizer->SetWorker(config.m_workerRank, config.m_numberOfWorkers);
    m_globalSamplePosition = m_epochSize * config.m_epochIndex;
    PrepareNewSweepIfNeeded(m_globalSamplePosition);
    m_sequenceRandomizer->SetSequencePositionTo(m_globalSamplePosition % m_sweepTotalNumberOfSamples, m_sweep);
}

void PartialBlockRandomizer::PrepareNewSweepIfNeeded(size_t samplePosition)
{
    size_t sweep = samplePosition / m_sweepTotalNumberOfSamples;
    if (m_sweep != sweep)
    {
        m_sweep = sweep;
        m_sweepStartInSamples = sweep * m_sweepTotalNumberOfSamples;
        m_chunkRandomizer->Randomize((unsigned int)m_sweep);
        m_sequenceRandomizer->Reset(m_sweep + 1);
        m_chunks.clear();
        m_lastSeenChunk = SIZE_MAX;
    }
}

Sequences PartialBlockRandomizer::GetNextSequences(size_t sampleCount)
{
    Sequences result;
    std::vector<RandomizedSequenceDescription> sequences;
    result.m_endOfEpoch = GetNextSequenceDescriptions(sampleCount, sequences);

    if (sequences.size() == 0)
    {
        return result;
    }

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(sequences.size()));

    // TODO: This will be changed, when we move transformers under the randomizer.
    // TODO: Randomizer won't should not deal with multithreading.
#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < sequences.size(); ++i)
    {
        const auto& description = sequences[i];
        std::vector<SequenceDataPtr> sequence;
        auto it = m_chunks.find(description.m_chunk->m_chunkId);
        if (it == m_chunks.end())
        {
            LogicError("Invalid chunk requested.");
        }
        it->second->GetSequence(description.m_id, sequence);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    }

    return result;
}

bool PartialBlockRandomizer::GetNextSequenceDescriptions(size_t sampleCount, std::vector<RandomizedSequenceDescription>& result)
{
    PrepareNewSweepIfNeeded(m_globalSamplePosition);

    // Check epoch.
    if (m_globalSamplePosition - m_config.m_epochIndex * m_epochSize + sampleCount >= m_epochSize)
    {
        sampleCount = m_epochSize - m_globalSamplePosition + m_config.m_epochIndex * m_epochSize;
    }

    if (sampleCount <= 0)
    {
        return true;
    }

    // Check that we do not go over the sweep.
    size_t sweepPosition = m_globalSamplePosition % m_sweepTotalNumberOfSamples;
    if (sweepPosition + sampleCount >= m_sweepTotalNumberOfSamples)
    {
        sampleCount = m_sweepTotalNumberOfSamples - sweepPosition;
    }
    assert(sampleCount != 0);

    m_sequenceRandomizer->RandomizeSequenceForRange(sampleCount);
    std::vector<RandomizedSequenceDescription> sequences = m_sequenceRandomizer->GetSequencesForRange(sampleCount);

    // Swap remove all old chunks and add new ones.
    // Require all data in chunks.
    RetrieveDataChunks();

    for (const auto& s : sequences)
    {
        m_globalSamplePosition += s.m_numberOfSamples;
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
        LogicError("Not supported mode.");
    }

    return false;
}

void PartialBlockRandomizer::RetrieveDataChunks()
{
    const auto& window = m_sequenceRandomizer->GetChunkWindow();
    if (window.back().m_chunkId == m_lastSeenChunk)
    {
        return; // nothing to retrieve.
    }

    m_lastSeenChunk = window.back().m_chunkId;
    std::map<size_t, ChunkPtr> chunks;
    for (auto const& chunk : window)
    {
        if (m_distributionMode == DistributionMode::chunk && chunk.m_chunkId % m_config.m_numberOfWorkers != m_config.m_workerRank)
        {
            continue;
        }

        auto it = m_chunks.find(chunk.m_original->id);
        if (it != m_chunks.end())
        {
            chunks[chunk.m_chunkId] = it->second;
        }
        else
        {
            chunks[chunk.m_chunkId] = m_deserializer->GetChunk(chunk.m_original->id);
        }
    }

    m_chunks.swap(chunks);
}

}}}
