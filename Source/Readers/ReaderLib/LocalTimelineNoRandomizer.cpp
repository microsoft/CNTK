//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "LocalTimelineNoRandomizer.h"

namespace CNTK {

LocalTimelineNoRandomizer::LocalTimelineNoRandomizer(DataDeserializerPtr deserializer, bool multithreadedGetNextSequences, size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_currentChunkPosition(0),
  m_currentSequencePosition(0)
{
}

void LocalTimelineNoRandomizer::Prefetch()
{
    size_t capturedPosition = m_currentChunkPosition;
    m_prefetch = std::async(std::launch::async, [=]()
    {
        auto chunkId = m_originalChunkDescriptions[capturedPosition].m_id;
        std::get<0>(m_prefetchedChunk) = m_originalChunkDescriptions[capturedPosition];
        std::get<1>(m_prefetchedChunk) = m_deserializer->GetChunk(chunkId);

        std::get<2>(m_prefetchedChunk).clear();
        m_deserializer->GetSequencesForChunk(m_currentChunkPosition, std::get<2>(m_prefetchedChunk));
    });
}

void LocalTimelineNoRandomizer::RefillSequenceWindow()
{
    m_window.m_sequences.assign(std::get<2>(m_prefetchedChunk).begin(), std::get<2>(m_prefetchedChunk).end());
    m_window.m_dataChunks.clear();
    m_window.m_dataChunks[std::get<0>(m_prefetchedChunk).m_id] = std::get<1>(m_prefetchedChunk);

    if (m_config.m_numberOfWorkers > 1)
    {
        // Decimate according to the position.
        size_t currentSequencePosition = m_currentSequencePosition;
        size_t currentInputIndex = 0;
        for (size_t i = 0; i < m_window.m_sequences.size(); ++i, ++currentSequencePosition)
        {
            if (currentSequencePosition % m_config.m_numberOfWorkers == m_config.m_workerRank)
                std::swap(m_window.m_sequences[currentInputIndex++], m_window.m_sequences[i]);
        }

        m_currentSequencePosition += m_window.m_sequences.size();
        m_window.m_sequences.erase(m_window.m_sequences.begin() + currentInputIndex);
    }

    // If last chunk, add marker.
    if (m_currentChunkPosition == m_originalChunkDescriptions.size() - 1)
    {
        m_window.m_sequences.push_back(s_endOfSweep);
        m_currentSequencePosition = 0;
    }

    // Moving to the next chunk.
    m_currentChunkPosition = (m_currentChunkPosition + 1) % m_originalChunkDescriptions.size();
}

Dictionary LocalTimelineNoRandomizer::GetInnerState()
{
    Dictionary state;
    state[L"currentChunkPosition"] = (size_t)m_currentChunkPosition;
    state[L"currentSequencePosition"] = m_currentSequencePosition;
    return state;
}

void LocalTimelineNoRandomizer::SetInnerState(const Dictionary& state)
{
    m_currentChunkPosition = (ChunkIdType)ValueOf(state, L"currentChunkPosition");
    m_currentSequencePosition = ValueOf(state, L"currentSequencePosition");
}

}
