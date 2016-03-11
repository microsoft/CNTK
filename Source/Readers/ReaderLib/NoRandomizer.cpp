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
      m_currentChunkPosition(SIZE_MAX),
      m_globalSamplePosition(0),
      m_totalNumberOfSamples(0),
      m_currentSequencePositionInChunk(0),
      m_chunkStartPosition(0),
      m_chunkEndPosition(0)
{
    assert(deserializer != nullptr);
    m_streams = m_deserializer->GetStreamDescriptions();
    m_chunkDescriptions = m_deserializer->GetChunkDescriptions();

    size_t sampleCount = 0;
    for (const auto& chunk : m_chunkDescriptions)
    {
        m_chunkSampleOffset.push_back(sampleCount);
        sampleCount += chunk->numberOfSamples;
    }

    m_totalNumberOfSamples = sampleCount;
    m_chunks.resize(m_chunkDescriptions.size());
}

void NoRandomizer::Initialize(TransformerPtr, const ConfigParameters&)
{
}

size_t NoRandomizer::GetChunkIndexOf(size_t samplePosition)
{
    size_t low = 0;
    size_t high = m_chunkDescriptions.size() - 1;
    while (high > low)
    {
        size_t mid = (high + low) / 2;
        if (samplePosition >= m_chunkSampleOffset[mid] + m_chunkDescriptions[mid]->numberOfSamples)
        {
            low = mid + 1;
        }
        else if (samplePosition < m_chunkSampleOffset[mid])
        {
            assert(mid > 0);
            high = mid - 1;
        }
        else
        {
            return mid;
        }
    }

    assert((high == low) && ((samplePosition >= m_chunkSampleOffset[low]) && (samplePosition < m_chunkSampleOffset[low] + m_chunkDescriptions[low]->numberOfSamples)));
    return low;
}

void NoRandomizer::StartEpoch(const EpochConfiguration& config)
{
    m_config = config;

    if (m_config.m_totalEpochSizeInSamples == requestDataSize)
    {
        m_config.m_totalEpochSizeInSamples = m_totalNumberOfSamples;
    }

    m_samplePositionInEpoch = 0;
    m_globalSamplePosition = m_config.m_totalEpochSizeInSamples * config.m_epochIndex;
    size_t localSamplePosition = m_globalSamplePosition % m_totalNumberOfSamples;

    size_t chunkIndex = GetChunkIndexOf(localSamplePosition);
    if (chunkIndex != m_currentChunkPosition)
    {
        m_currentChunkPosition = chunkIndex;
        m_sequenceWindow.clear();

        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
        m_currentSequencePositionInChunk = 0;

        // Dropping all chunk data from previous epoch
        while (m_chunkStartPosition < m_chunkEndPosition)
        {
            m_chunks[m_chunkStartPosition] = nullptr;
            m_chunkStartPosition = (m_chunkStartPosition + 1) % m_chunkDescriptions.size();
        }

        m_chunkStartPosition = 0;
        m_chunkEndPosition = 0;
    }

    size_t sampleOffsetInsideChunk = localSamplePosition - m_chunkSampleOffset[m_currentChunkPosition];
    size_t numberOfSamples = 0;
    size_t sequenceId = 0;

    // Currently linear, happens only at the border of epochs.
    for (size_t i = 0; i < m_sequenceWindow.size(); ++i)
    {
        size_t sequenceSize = m_sequenceWindow[i].m_numberOfSamples;
        if (sequenceSize + numberOfSamples > sampleOffsetInsideChunk)
        {
            break;
        }

        numberOfSamples += sequenceSize;
        sequenceId++;
    }

    m_currentSequencePositionInChunk = sequenceId;
};

std::vector<SequenceDescription> NoRandomizer::GetNextSequenceDescriptions(size_t sampleCount)
{
    assert(m_sequenceWindow.size() != 0);

    // Decimation is done based only on sequence ids
    int samples = (int)sampleCount;

    std::vector<SequenceDescription> descriptions;
    descriptions.reserve(sampleCount);

    size_t sequenceOffsetInsideChunk = m_currentSequencePositionInChunk;
    SequenceDescription sequence = m_sequenceWindow[m_currentSequencePositionInChunk];

    descriptions.push_back(sequence);

    samples -= (int)sequence.m_numberOfSamples;
    m_currentSequencePositionInChunk++;
    m_samplePositionInEpoch += sequence.m_numberOfSamples;
    m_globalSamplePosition += sequence.m_numberOfSamples;

    if (sequenceOffsetInsideChunk + 1 >= m_chunkDescriptions[m_currentChunkPosition]->numberOfSequences)
    {
        // Moving to the next chunk.
        m_currentChunkPosition = (m_currentChunkPosition + 1) % m_chunkDescriptions.size();
        m_currentSequencePositionInChunk = 0;
        m_sequenceWindow.clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
    }

    while (samples > 0)
    {
        sequenceOffsetInsideChunk = m_currentSequencePositionInChunk;
        sequence = m_sequenceWindow[sequenceOffsetInsideChunk];
        if (samples - (int)sequence.m_numberOfSamples >= 0)
        {
            descriptions.push_back(sequence);
            m_currentSequencePositionInChunk++;
            samples -= (int)sequence.m_numberOfSamples;
            m_samplePositionInEpoch += sequence.m_numberOfSamples;
            m_globalSamplePosition += sequence.m_numberOfSamples;

            if (sequenceOffsetInsideChunk + 1 >= m_chunkDescriptions[m_currentChunkPosition]->numberOfSequences)
            {
                // Moving to the next chunk.
                m_currentChunkPosition = (m_currentChunkPosition + 1) % m_chunkDescriptions.size();
                m_sequenceWindow.clear();
                m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
                m_currentSequencePositionInChunk = 0;
            }
        }
        else
        {
            break;
        }
    }

    return descriptions;
}

Sequences NoRandomizer::GetNextSequences(size_t sampleCount)
{
    Sequences result;
    if (m_config.m_totalEpochSizeInSamples <= m_samplePositionInEpoch)
    {
        result.m_endOfEpoch = true;
        return result;
    }

    // Check that we do not go over the sweep.
    size_t sweepPosition = m_globalSamplePosition % m_totalNumberOfSamples;
    if (sweepPosition + sampleCount >= m_totalNumberOfSamples)
    {
        sampleCount = m_totalNumberOfSamples - sweepPosition;
    }
    assert(sampleCount != 0);

    std::vector<SequenceDescription> descriptions = GetNextSequenceDescriptions(sampleCount);

    size_t start = descriptions.size() * m_config.m_workerRank / m_config.m_numberOfWorkers;
    size_t end = descriptions.size() * (m_config.m_workerRank + 1) / m_config.m_numberOfWorkers;
    size_t subsetSize = end - start;
    if (subsetSize == 0)
    {
        return result;
    }

    while (m_chunkStartPosition < m_chunkEndPosition && m_chunkStartPosition != descriptions[start].m_chunkId)
    {
        m_chunks[m_chunkStartPosition] = nullptr;
        m_chunkStartPosition = (m_chunkStartPosition + 1) % m_chunkDescriptions.size();
    }

    for (size_t i = 0; i < subsetSize; ++i)
    {
        size_t chunkId = (m_chunkEndPosition - 1) % m_chunkDescriptions.size();
        if (m_chunkStartPosition == m_chunkEndPosition || descriptions[start + i].m_chunkId != chunkId)
        {
            m_chunks[m_chunkEndPosition] = m_deserializer->GetChunk(descriptions[start + i].m_chunkId);
            m_chunkEndPosition = (m_chunkEndPosition + 1) % m_chunkDescriptions.size();
        }
    }

    // TODO: Not clear whether batching will make sense for this.
    // We have to re-assemble the exposed result from sequences from different chunks.
    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(subsetSize));
#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < subsetSize; ++i)
    {
        std::vector<SequenceDataPtr> sequence;
        m_chunks[descriptions[start + i].m_chunkId]->GetSequence(descriptions[start + i].m_id, sequence);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    }
    return result;

}

} } }
