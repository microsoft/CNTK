//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "NoRandomizer.h"
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

NoRandomizer::NoRandomizer(IDataDeserializerPtr deserializer, bool multithreadedGetNextSequences)
    : m_deserializer(deserializer),
      m_samplePositionInEpoch(0),
      m_currentChunkPosition(CHUNKID_MAX),
      m_globalSamplePosition(0),
      m_totalNumberOfSamples(0),
      m_currentSequencePositionInChunk(0),
      m_multithreadedGetNextSequences(multithreadedGetNextSequences)
{
    assert(deserializer != nullptr);
    m_streams = m_deserializer->GetStreamDescriptions();
    m_chunkDescriptions = m_deserializer->GetChunkDescriptions();

    size_t sampleCount = 0;
    for (const auto& chunk : m_chunkDescriptions)
    {
        // Check that position corresponds to chunk id.
        assert(m_chunkSampleOffset.size() == chunk->m_id);

        m_chunkSampleOffset.push_back(sampleCount);
        sampleCount += chunk->m_numberOfSamples;
    }

    if (sampleCount == 0)
    {
        RuntimeError("NoRandomizer: Expected input to contain samples, but the number of successfully read samples was 0.");
    }

    m_totalNumberOfSamples = sampleCount;
}

ChunkIdType NoRandomizer::GetChunkIndexOf(size_t samplePosition)
{
    auto result = std::upper_bound(m_chunkSampleOffset.begin(), m_chunkSampleOffset.end(), samplePosition);
    return (ChunkIdType) (result - 1 - m_chunkSampleOffset.begin());
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
    size_t sweepSamplePosition = m_globalSamplePosition % m_totalNumberOfSamples;

    ChunkIdType chunkIndex = GetChunkIndexOf(sweepSamplePosition);
    if (chunkIndex != m_currentChunkPosition)
    {
        // unloading everything.
        m_currentChunkId = CHUNKID_MAX;
        m_currentChunk = nullptr;

        // Need to load descriptions for the new current chunk.
        m_currentChunkPosition = chunkIndex;
        m_currentSequencePositionInChunk = 0;
        m_sequenceWindow.clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
    }

    // Moving current sequence inside the chunk to match the sample offset.
    size_t sampleOffsetInsideChunk = sweepSamplePosition - m_chunkSampleOffset[m_currentChunkPosition];
    size_t numberOfSamples = 0;
    size_t sequenceId = 0;

    // Currently linear, happens only at the border of epochs.
    for (size_t i = 0; i < m_sequenceWindow.size(); ++i)
    {
        size_t sequenceSize = m_sequenceWindow[i].m_numberOfSamples;
        if (sequenceSize + numberOfSamples > sampleOffsetInsideChunk)
        {
            // We have found our sequence.
            break;
        }

        numberOfSamples += sequenceSize;
        sequenceId++;
    }

    m_currentSequencePositionInChunk = sequenceId;
    assert(m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences > m_currentSequencePositionInChunk);
};

// Moving the cursor to the next sequence. Possibly updating the chunk information if needed.
void NoRandomizer::MoveToNextSequence()
{
    SequenceDescription& sequence = m_sequenceWindow[m_currentSequencePositionInChunk];
    m_samplePositionInEpoch += sequence.m_numberOfSamples;
    m_globalSamplePosition += sequence.m_numberOfSamples;

    if (m_currentSequencePositionInChunk + 1 >= m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences)
    {
        // Moving to the next chunk.
        m_currentChunkPosition = (m_currentChunkPosition + 1) % m_chunkDescriptions.size();
        m_currentSequencePositionInChunk = 0;
        m_sequenceWindow.clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
    }
    else
    {
        m_currentSequencePositionInChunk++;
    }
}

// Gets next sequence descriptions with total size less than sampleCount.
std::vector<SequenceDescription> NoRandomizer::GetNextSequenceDescriptions(size_t sampleCount)
{
    assert(m_sequenceWindow.size() != 0);
    assert(m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences > m_currentSequencePositionInChunk);

    int samples = (int)sampleCount;

    std::vector<SequenceDescription> result;

    do
    {
        const SequenceDescription& sequence = m_sequenceWindow[m_currentSequencePositionInChunk];
        result.push_back(sequence);
        samples -= (int)sequence.m_numberOfSamples;

        MoveToNextSequence();
    }
    // Check whether the next sequence fits into the sample count, if not, exit.
    while (samples - (int)m_sequenceWindow[m_currentSequencePositionInChunk].m_numberOfSamples >= 0);
    return result;
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
    // TODO: This preserves the old behavior. Could be done differently in the future.
    size_t sweepPosition = m_globalSamplePosition % m_totalNumberOfSamples;
    sampleCount = std::min(sampleCount, m_totalNumberOfSamples - sweepPosition);
    assert(sampleCount != 0);

    std::vector<SequenceDescription> descriptions = GetNextSequenceDescriptions(sampleCount);

    // Retrieve only sequences that are required by this worker.
    size_t start = descriptions.size() * m_config.m_workerRank / m_config.m_numberOfWorkers;
    size_t end = descriptions.size() * (m_config.m_workerRank + 1) / m_config.m_numberOfWorkers;
    size_t subsetSize = end - start;
    if (subsetSize == 0)
    {
        return result;
    }

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(subsetSize));

    // Collect all the chunks that we need
    std::map<ChunkIdType, ChunkPtr> chunks;

    if (m_currentChunk != nullptr)
    {
        chunks[m_currentChunkId] = m_currentChunk;
    }

    for (int i = 0; i < subsetSize; ++i)
    {
        const auto& sequenceDescription = descriptions[start + i];
        auto it = chunks.find(sequenceDescription.m_chunkId);
        if (it == chunks.end())
        {
            chunks[sequenceDescription.m_chunkId] = m_deserializer->GetChunk(sequenceDescription.m_chunkId);
        }
    }

    auto process = [&](int i) -> void {
        std::vector<SequenceDataPtr> sequence;
        const auto& sequenceDescription = descriptions[start + i];

        auto it = chunks.find(sequenceDescription.m_chunkId);
        if (it == chunks.end())
        {
            LogicError("Invalid chunk requested.");
        }

        it->second->GetSequence(sequenceDescription.m_id, sequence);
        for (int j = 0; j < m_streams.size(); ++j)
        {
            result.m_data[j][i] = sequence[j];
        }
    };

    // TODO: This will be changed, when we move transformers under the (no-) randomizer, should not deal with multithreading here.
    if (m_multithreadedGetNextSequences)
    {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < subsetSize; ++i)
            process(i);
    }
    else
    {
        for (int i = 0; i < subsetSize; ++i)
            process(i);
    }

    // Keep the last chunk for next time
    m_currentChunkId = descriptions[start + subsetSize - 1].m_chunkId;
    auto it = chunks.find(m_currentChunkId);
    assert(it != chunks.end());
    m_currentChunk = it->second;

    return result;
}

} } }
