//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <random>

#include "LTTumblingWindowRandomizer.h"
#include "RandomOrdering.h"
#include <tuple>

namespace CNTK {

using Microsoft::MSR::CNTK::RandomShuffleMT;

LTTumblingWindowRandomizer::LTTumblingWindowRandomizer(
    DataDeserializerPtr deserializer,
    bool sampleBasedRandomizationWindow,
    size_t randomizationRange,
    size_t seedOffset,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
: Base(deserializer, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_randomizationRange(randomizationRange),
  m_seedOffset(seedOffset),
  m_chunkPosition(0),
  m_sampleBasedRandomizationWindow(sampleBasedRandomizationWindow),
  m_sweepCount(0)
{
    RandomizeChunks(m_sweepCount);
}

void LTTumblingWindowRandomizer::RandomizeWindow(size_t sweepCount, size_t chunkPositionOfWindow, size_t sequencePositionInWindow) const
{
    m_rng.seed((unsigned long)(chunkPositionOfWindow + sweepCount + m_seedOffset));
    RandomShuffleMT(m_prefetchedSequences, sequencePositionInWindow, m_prefetchedSequences.size(), m_rng);
}

void LTTumblingWindowRandomizer::RandomizeChunks(size_t sweepCount) const
{
    m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
    m_rng.seed((unsigned long)sweepCount + m_seedOffset);
    RandomShuffleMT(m_prefetchedChunkDescriptions, m_rng);
}

void LTTumblingWindowRandomizer::Prefetch() const
{
    size_t position = m_chunkPosition;
    size_t sweepIndex = m_sweepCount;

    // Prefetch does not change any state that cannot be recalculated,
    // only prefetches data.
    int64_t range = m_randomizationRange;
    m_prefetchedChunks.clear();
    m_prefetchedSequences.clear();

    size_t lastSequencePositionInWindow = 0;
    size_t lastWindowPosition = m_chunkPosition;
    while (range > 0)
    {
        auto desc = m_prefetchedChunkDescriptions[position];
        if (position % Config().m_numberOfWorkers == Config().m_workerRank) // Need to add to the window
        {
            size_t oldSize = m_prefetchedSequences.size();

            // Query deserializer.
            ChunkPtr data = m_deserializer->GetChunk(desc.m_id);
            data->SequenceInfos(m_prefetchedSequences);
            m_prefetchedChunks.push_back(std::make_tuple(desc, data));

            if (!m_sampleBasedRandomizationWindow)
                --range;
            else
                for (size_t i = oldSize; i < m_prefetchedSequences.size(); ++i)
                    range -= m_prefetchedSequences[i].m_numberOfSamples;
        }
        else
        {
            // Empty, we do not need data , only for tracking the current chunk.
            m_prefetchedChunks.push_back(std::make_tuple(ChunkInfo{}, nullptr));
        }

        if (position == m_originalChunkDescriptions.size() - 1)
        {
            // Sweep boundary, randomize all sequences in the window from the previous sweep.
            RandomizeWindow(sweepIndex, lastWindowPosition, lastSequencePositionInWindow);

            // Switch to next sweep, randomize chunks.
            sweepIndex++;
            RandomizeChunks(sweepIndex);

            // Put a marker and reset window position to the beginning of the sweep.
            m_prefetchedSequences.push_back(s_endOfSweep);
            lastWindowPosition = 0;
            lastSequencePositionInWindow = m_prefetchedSequences.size();
        }

        position = (position + 1) % m_originalChunkDescriptions.size();
    }

    // Rerandomize the last part of the sequences.
    RandomizeWindow(sweepIndex, lastWindowPosition, lastSequencePositionInWindow);
}

void LTTumblingWindowRandomizer::RefillSequenceWindow(SequenceWindow& window)
{
    window.m_dataChunks.clear();
    window.m_sequences = m_prefetchedSequences;
    for (const auto& s : window.m_sequences)
        if (IsEndOfSweep(s))
            m_sweepCount++;

    for (const auto& c : m_prefetchedChunks)
        window.m_dataChunks.insert(std::make_pair(std::get<0>(c).m_id, std::get<1>(c)));

    m_chunkPosition = (ChunkIdType)(m_chunkPosition + m_prefetchedChunks.size()) % m_originalChunkDescriptions.size();
}

// Properties used in the checkpoint.
const static std::wstring s_chunkPositionProperty = L"chunkPosition";
const static std::wstring s_sweepIndexProperty = L"sweepIndex";

std::map<std::wstring, size_t> LTTumblingWindowRandomizer::GetInnerState()
{
    std::map<std::wstring, size_t> state;
    state[s_chunkPositionProperty] = m_chunkPosition;
    state[s_sweepIndexProperty] = m_sweepCount;
    return state;
}

void LTTumblingWindowRandomizer::SetInnerState(const std::map<std::wstring, size_t>& state)
{
    m_sweepCount = ValueFrom(state, s_sweepIndexProperty);
    RandomizeChunks(m_sweepCount);
    m_chunkPosition = (ChunkIdType)ValueFrom(state, s_chunkPositionProperty);
}

}
