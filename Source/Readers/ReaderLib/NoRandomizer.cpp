//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>

#include "NoRandomizer.h"
#include "DataReader.h"
#include "ExceptionCapture.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    NoRandomizer::NoRandomizer(IDataDeserializerPtr deserializer, bool multithreadedGetNextSequences, size_t maxNumberOfInvalidSequences)
    : m_deserializer(deserializer),
      m_currentChunkPosition(CHUNKID_MAX),
      m_globalSamplePosition(0),
      m_globalSequencePosition(0),
      m_sweepSizeInSamples(0),
      m_currentSequencePositionInChunk(0),
      m_multithreadedGetNextSequences(multithreadedGetNextSequences),
      m_cleaner(maxNumberOfInvalidSequences)
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

    m_sweepSizeInSamples = sampleCount;
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
        m_config.m_totalEpochSizeInSamples = m_sweepSizeInSamples;

    SetCurrentSamplePosition(m_config.m_totalEpochSizeInSamples * config.m_epochIndex);
}

// Moving the cursor to the next sequence. Possibly updating the chunk information if needed.
void NoRandomizer::MoveToNextSequence()
{
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

// Gets next sequences not exceeding local and global samples.
void NoRandomizer::GetNextSequenceDescriptions(size_t globalSampleCount, size_t localSampleCount, std::vector<SequenceDescription>& result)
{
    assert(globalSampleCount != 0);
    assert(localSampleCount != 0);

    if (globalSampleCount > std::numeric_limits<int>::max() &&
        localSampleCount > std::numeric_limits<int>::max())
        RuntimeError("Global and local size of the minibatch cannot exceed max int.");

    assert(m_sequenceWindow.size() != 0);
    assert(m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences > m_currentSequencePositionInChunk);

    size_t numGlobalSamplesLoaded = 0, numLocalSamplesLoaded = 0;

    result.reserve(localSampleCount);
    result.clear();

    while (globalSampleCount > numGlobalSamplesLoaded  && localSampleCount > numLocalSamplesLoaded)
    {
        const SequenceDescription& sequence = m_sequenceWindow[m_currentSequencePositionInChunk];
        auto sequenceLength = sequence.m_numberOfSamples;

        // Let's check whether we need to return this sequence or skip it.
        bool isLocal = m_globalSequencePosition % m_config.m_numberOfWorkers == m_config.m_workerRank;
        bool enoughData = !result.empty();

        // Let's check whether we need to break because we exceeded global counter.
        if (enoughData && globalSampleCount - numGlobalSamplesLoaded < sequenceLength)
            break;

        // Let's check whether we need to break because we exceeded local counter.
        if (enoughData && isLocal && localSampleCount - numLocalSamplesLoaded < sequenceLength)
            break;

        if (isLocal) // Ok good to add it to the result.
        {
            result.push_back(sequence);
            numLocalSamplesLoaded += sequence.m_numberOfSamples;
        }

        numGlobalSamplesLoaded += sequence.m_numberOfSamples;
        m_globalSamplePosition += sequence.m_numberOfSamples;
        m_globalSequencePosition++;

        MoveToNextSequence();
    }
}

size_t NoRandomizer::GetCurrentSamplePosition()
{
    return m_globalSamplePosition;
}

Sequences NoRandomizer::GetNextSequences(size_t globalSampleCount, size_t localSampleCount)
{
    if (globalSampleCount == 0)
        LogicError("Global sample count must not be zero.");

    if (localSampleCount == 0)
        LogicError("Local sample count must not be zero.");

    Sequences result;
    size_t endOfEpochPosition = m_config.m_totalEpochSizeInSamples * (m_config.m_epochIndex + 1);
    if (m_globalSamplePosition >= endOfEpochPosition)
    {
        result.m_endOfEpoch = true;
        result.m_endOfSweep = (m_globalSamplePosition >= m_sweepSizeInSamples) &&
            (m_globalSamplePosition % m_sweepSizeInSamples == 0);
        return result;
    }

    // Check we do not go over epoch.
    globalSampleCount = std::min(globalSampleCount, endOfEpochPosition - m_globalSamplePosition);

    if (!m_config.m_allowMinibatchesToCrossSweepBoundaries)
    {
        // Cut down the required sample count if we're not allowed to go over the
        // sweep boundary
        size_t sweepPosition = m_globalSamplePosition % m_sweepSizeInSamples;
        globalSampleCount = std::min(globalSampleCount, m_sweepSizeInSamples - sweepPosition);
    }

    if (globalSampleCount == 0)
        LogicError("Global sample count must not result in zero.");

    auto sweepIndex = m_globalSamplePosition / m_sweepSizeInSamples;

    m_sequenceBuffer.clear();
    GetNextSequenceDescriptions(globalSampleCount, localSampleCount, m_sequenceBuffer);

    // m_globalSamplePosition is already shifted in GetNextSequenceDescriptions() by the current minibatch size.
    // Set the end-of-epoch flag (true when the current batch is last in an epoch).
    result.m_endOfEpoch = (m_globalSamplePosition >= endOfEpochPosition);
    result.m_endOfSweep = sweepIndex != m_globalSamplePosition / m_sweepSizeInSamples;

    if (m_sequenceBuffer.size() == 0)
    {
        return result;
    }

    result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(m_sequenceBuffer.size()));

    // Collect all the chunks that we need
    std::map<ChunkIdType, ChunkPtr> chunks;
    for (const auto& s : m_sequenceBuffer)
    {
        auto it = chunks.find(s.m_chunkId);
        if (it == chunks.end())
        {
            auto old = m_chunks.find(s.m_chunkId);
            if (old != m_chunks.end())
            {
                chunks.insert(std::make_pair(s.m_chunkId, old->second));
            }
            else
            {
                chunks[s.m_chunkId] = m_deserializer->GetChunk(s.m_chunkId);
            }
        }
    }

    // swap current chunks with new ones:
    m_chunks.swap(chunks);

    auto process = [&](int i) -> void {
        std::vector<SequenceDataPtr> sequence;
        const auto& sequenceDescription = m_sequenceBuffer[i];

        auto it = m_chunks.find(sequenceDescription.m_chunkId);
        if (it == m_chunks.end())
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
        ExceptionCapture capture;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < m_sequenceBuffer.size(); ++i)
            capture.SafeRun(process, i);
        capture.RethrowIfHappened();
    }
    else
    {
        for (int i = 0; i < m_sequenceBuffer.size(); ++i)
            process(i);
    }

    m_cleaner.Clean(result);
    return result;
}

void NoRandomizer::SetCurrentSamplePosition(size_t samplePosition)
{
    m_currentSequencePositionInChunk = 0;
    m_globalSamplePosition = samplePosition;
    size_t sweepSamplePosition = m_globalSamplePosition % m_sweepSizeInSamples;

    ChunkIdType chunkIndex = GetChunkIndexOf(sweepSamplePosition);
    if (chunkIndex != m_currentChunkPosition)
    {
        // Need to load descriptions for the new current chunk.
        m_currentChunkPosition = chunkIndex;
        m_currentSequencePositionInChunk = 0;
        m_sequenceWindow.clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
    }

    // Moving current sequence inside the chunk to match the sample offset.
    // Currently linear, happens only at the border of epochs.
    size_t sampleOffsetInsideChunk = sweepSamplePosition - m_chunkSampleOffset[m_currentChunkPosition];
    size_t numberOfSamples = 0;
    while (m_currentSequencePositionInChunk < m_sequenceWindow.size() &&
        numberOfSamples < sampleOffsetInsideChunk)
    {
        numberOfSamples += m_sequenceWindow[m_currentSequencePositionInChunk].m_numberOfSamples;
        MoveToNextSequence();
    }

    // Updating the global position
    m_globalSamplePosition = m_globalSamplePosition - sampleOffsetInsideChunk + numberOfSamples;
    assert(m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences > m_currentSequencePositionInChunk);

    m_globalSequencePosition = 0;
    for (size_t i = 0; i < m_currentChunkPosition; ++i)
    {
        m_globalSequencePosition += m_chunkDescriptions[i]->m_numberOfSequences;
    }
    m_globalSequencePosition += m_currentSequencePositionInChunk;
}

void NoRandomizer::SetConfiguration(const ReaderConfiguration& config)
{
    *((ReaderConfiguration*)&m_config) = config;
}

} } }
