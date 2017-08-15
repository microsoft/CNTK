//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "LTNoRandomizer.h"

namespace CNTK {

LTNoRandomizer::LTNoRandomizer(DataDeserializerPtr deserializer, bool multithreadedGetNextSequences, size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_currentChunkPosition(0),
  m_currentSequencePosition(0)
{
}

void LTNoRandomizer::Prefetch() const
{
    auto chunkId = m_originalChunkDescriptions[m_currentChunkPosition].m_id;
    m_prefetchedChunk.m_info = m_originalChunkDescriptions[m_currentChunkPosition];
    m_prefetchedChunk.m_data = m_deserializer->GetChunk(chunkId);
    m_prefetchedChunk.m_sequenceInfos.clear();
    m_prefetchedChunk.m_data->SequenceInfos(m_prefetchedChunk.m_sequenceInfos);
}

void LTNoRandomizer::RefillSequenceWindow(SequenceWindow& window)
{
    window.m_sequences.assign(m_prefetchedChunk.m_sequenceInfos.begin(), m_prefetchedChunk.m_sequenceInfos.end());
    window.m_dataChunks.clear();
    window.m_dataChunks[m_prefetchedChunk.m_info.m_id] = m_prefetchedChunk.m_data;

    auto numberOfWorkers = Config().m_numberOfWorkers;
    if (numberOfWorkers > 1)
    {
        // Decimate according to the position.
        size_t workerSequencePosition = 0;
        for (size_t i = 0; i < window.m_sequences.size(); ++i, ++m_currentSequencePosition)
        {
            if (m_currentSequencePosition % numberOfWorkers == Config().m_workerRank)
                std::swap(window.m_sequences[workerSequencePosition++], window.m_sequences[i]);
        }

        window.m_sequences.erase(window.m_sequences.begin() + workerSequencePosition);
    }

    // If last chunk, add the sweep marker.
    if (m_currentChunkPosition == m_originalChunkDescriptions.size() - 1)
    {
        window.m_sequences.push_back(s_endOfSweep);
        m_currentSequencePosition = 0;
    }

    // Moving to the next chunk.
    m_currentChunkPosition = (m_currentChunkPosition + 1) % m_originalChunkDescriptions.size();
}

// Properties used in the checkpoint.
const static std::wstring s_currentChunkPositionProperty = L"currentChunkPosition";
const static std::wstring s_currentSequencePositionProperty = L"currentSequencePosition";

std::map<std::wstring, size_t> LTNoRandomizer::GetInnerState()
{
    std::map<std::wstring, size_t> state;
    state[s_currentChunkPositionProperty] = m_currentChunkPosition;
    state[s_currentSequencePositionProperty] = m_currentSequencePosition;
    return state;
}

void LTNoRandomizer::SetInnerState(const std::map<std::wstring, size_t>& state)
{
    m_currentChunkPosition = (ChunkIdType)ValueFrom(state, s_currentChunkPositionProperty);
    m_currentSequencePosition = ValueFrom(state, s_currentSequencePositionProperty);
}

}
