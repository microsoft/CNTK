//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "NoRandomizer.h"
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

NoRandomizer::NoRandomizer(IDataDeserializerPtr deserializer)
    : m_deserializer(deserializer),
      m_samplePositionInEpoch(0),
      m_sequencePosition(0)
{
    assert(deserializer != nullptr);

    m_timeline = m_deserializer->GetSequenceDescriptions();
    for (const auto& sequence : m_timeline)
    {
        if (sequence->m_numberOfSamples != 1)
        {
            RuntimeError("Currently, no randomizer supports only frame mode. Received a sequence with %d number of samples.",
                static_cast<int>(sequence->m_numberOfSamples));
        }
    }

    m_streams = m_deserializer->GetStreamDescriptions();
}

void NoRandomizer::Initialize(TransformerPtr, const ConfigParameters&)
{
}

void NoRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_config = config;

    if (m_config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_config.m_totalEpochSizeInSamples = m_timeline.size();
    }

    m_samplePositionInEpoch = 0;
    size_t globalSamplePosition = m_config.m_totalEpochSizeInSamples * config.m_epochIndex;
    m_sequencePosition = globalSamplePosition % m_timeline.size();
};

Sequences NoRandomizer::GetNextSequences(size_t sampleCount)
{
    Sequences result;
    if (m_config.m_totalEpochSizeInSamples <= m_samplePositionInEpoch)
    {
        result.m_endOfEpoch = true;
        return result;
    }

    size_t maxSampleCount = std::min(sampleCount, m_config.m_totalEpochSizeInSamples - m_samplePositionInEpoch);
    size_t start = maxSampleCount * m_config.m_workerRank / m_config.m_numberOfWorkers;
    size_t end = maxSampleCount * (m_config.m_workerRank + 1) / m_config.m_numberOfWorkers;
    size_t subsetSize = end - start;

    std::vector<size_t> chunkIds;
    SequenceDescriptions sequences;
    sequences.reserve(subsetSize);
    size_t previousChunk = SIZE_MAX;
    for (size_t i = start; i < end; ++i)
    {
        const auto& sequence = m_timeline[(m_sequencePosition + i) % m_timeline.size()];
        assert(sequence->m_numberOfSamples == 1);
        sequences.push_back(sequence);

        if (previousChunk != sequence->m_chunkId)
        {
            chunkIds.push_back(sequence->m_chunkId);
            previousChunk = sequence->m_chunkId;
        }
    }

    m_samplePositionInEpoch += maxSampleCount;
    m_sequencePosition = (m_sequencePosition + maxSampleCount) % m_timeline.size();

    if (sequences.size() == 0)
    {
        return result;
    }

    // TODO: Currently we preserve chunks not for the complete window, only for minibatch
    // Should be changed
    std::map<size_t, ChunkPtr> chunks;
    for (size_t id : chunkIds)
    {
        auto chunk = m_chunks.find(id);
        if (chunk == m_chunks.end())
        {
            chunks[id] = m_deserializer->GetChunk(id);
        }
        else
        {
            chunks[id] = chunk->second;
        }
    }

    m_chunks.swap(chunks);

    // TODO: Not clear whether batching will make sense for this.
    // We have to re-assemble the exposed result from sequences from different chunks.
    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(sequences.size()));

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < sequences.size(); ++i)
    {
        auto sequence = m_chunks[sequences[i]->m_chunkId]->GetSequence(sequences[i]->m_id);

        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    }
    return result;
}

} } }
