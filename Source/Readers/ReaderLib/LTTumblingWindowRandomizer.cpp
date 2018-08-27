//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <random>
#include <mutex>

#include "LTTumblingWindowRandomizer.h"
#include "RandomOrdering.h"
#include <tuple>

namespace CNTK {

using Microsoft::MSR::CNTK::RandomShuffleMT;

// Properties used in the checkpoint.
const static std::wstring s_chunkPositionProperty = L"chunkPosition";
const static std::wstring s_sweepIndexProperty = L"sweepIndex";

LTTumblingWindowRandomizer::LTTumblingWindowRandomizer(
    DataDeserializerPtr deserializer,
    bool sampleBasedRandomizationWindow,
    size_t randomizationRange,
    size_t seedOffset,
    bool multithreadedGetNextSequences,
    size_t maxNumberOfInvalidSequences)
    : Base(deserializer, { { s_chunkPositionProperty, 0}, { s_sweepIndexProperty, 0} }, multithreadedGetNextSequences, maxNumberOfInvalidSequences),
  m_randomizationRange(randomizationRange),
  m_seedOffset(seedOffset),
  m_chunkPosition(0),
  m_sampleBasedRandomizationWindow(sampleBasedRandomizationWindow),
  m_sweepCount(0)
{
    std::lock_guard<std::mutex> lock(m_prefetchStateMutex);
    RandomizeChunks(m_prefetchState, m_sweepCount);
}

void LTTumblingWindowRandomizer::RandomizeWindow(PrefetchState& prefetchState, size_t sweepCount, size_t chunkPositionOfWindow, size_t sequencePositionInWindow) const
{
    prefetchState.m_rng.seed((unsigned long)(chunkPositionOfWindow + sweepCount + m_seedOffset));
    RandomShuffleMT(prefetchState.m_prefetchedSequences, sequencePositionInWindow, prefetchState.m_prefetchedSequences.size(), prefetchState.m_rng);
}

void LTTumblingWindowRandomizer::RandomizeChunks(PrefetchState& prefetchState, size_t sweepCount) const
{
    prefetchState.m_prefetchedChunkDescriptions = m_originalChunkDescriptions;
    prefetchState.m_rng.seed((unsigned long)sweepCount + m_seedOffset);
    RandomShuffleMT(prefetchState.m_prefetchedChunkDescriptions, prefetchState.m_rng);
    assert(m_originalChunkDescriptions.size() == prefetchState.m_prefetchedChunkDescriptions.size());
}

void LTTumblingWindowRandomizer::Prefetch() const
{
    std::lock_guard<std::mutex> lock(m_prefetchStateMutex);
    assert(m_prefetchState.m_prefetchedChunkDescriptions.size() > 0);
    size_t numChunks = m_prefetchState.m_prefetchedChunkDescriptions.size();
    size_t position = m_chunkPosition;
    size_t sweepIndex = m_sweepCount;

    // Prefetch does not change any state that cannot be recalculated,
    // only prefetches data.
    int64_t range = m_randomizationRange;
    m_prefetchState.m_prefetchedChunks.clear();
    m_prefetchState.m_prefetchedSequences.clear();

    size_t lastSequencePositionInWindow = 0;
    size_t lastWindowPosition = m_chunkPosition;
    while (range > 0)
    {
        assert(position < numChunks);
        ChunkInfo desc = m_prefetchState.m_prefetchedChunkDescriptions.at(position);
        if (position % Config().m_numberOfWorkers == Config().m_workerRank) // Need to add to the window
        {   //modify m_prefetchedChunks, m_prefetchedSequences, m_prefetchedChunkDescriptions[position] in this if
            size_t oldSize = m_prefetchState.m_prefetchedSequences.size();

            // Query deserializer.
            ChunkPtr data = m_deserializer->GetChunk(desc.m_id);
            data->SequenceInfos(m_prefetchState.m_prefetchedSequences);
            //m_prefetchedChunkDescriptions[m_currentChunkPosition] (as it is copied from m_OriginalChunkDescriptions)
            //might not have the counts of sequencs and samples
            //especiallly for the deserializers without indexer, such as UserDeserializer
            if (!desc.HasCountsInitiated())
                UpdateChunkInfo(desc, m_prefetchState.m_prefetchedSequences);
            m_prefetchState.m_prefetchedChunks.push_back(std::make_tuple(desc, data));

            if (!m_sampleBasedRandomizationWindow)
                --range;
            else
                for (size_t i = oldSize; i < m_prefetchState.m_prefetchedSequences.size(); ++i)
                    range -= m_prefetchState.m_prefetchedSequences[i].m_numberOfSamples;
        }
        else
        {
            // Empty, we do not need data , only for tracking the current chunk.
            m_prefetchState.m_prefetchedChunks.push_back(std::make_tuple(ChunkInfo{}, nullptr));
        }

        if (position == numChunks - 1)
        {
            // Sweep boundary, randomize all sequences in the window from the previous sweep.
            RandomizeWindow(m_prefetchState, sweepIndex, lastWindowPosition, lastSequencePositionInWindow);

            // Switch to next sweep, randomize chunks.
            sweepIndex++;
            RandomizeChunks(m_prefetchState, sweepIndex);

            // Put a marker and reset window position to the beginning of the sweep.
            m_prefetchState.m_prefetchedSequences.push_back(s_endOfSweep);
            lastWindowPosition = 0;
            lastSequencePositionInWindow = m_prefetchState.m_prefetchedSequences.size();
        }

        position = (position + 1) % numChunks;
    }

    // Rerandomize the last part of the sequences.
    RandomizeWindow(m_prefetchState, sweepIndex, lastWindowPosition, lastSequencePositionInWindow);
}

void LTTumblingWindowRandomizer::RefillSequenceWindow(SequenceWindow& window)
{
    std::lock_guard<std::mutex> lock(m_prefetchStateMutex);

    window.m_dataChunks.clear();
    window.m_sequences = m_prefetchState.m_prefetchedSequences;
    for (const auto& s : window.m_sequences)
        if (IsEndOfSweep(s))
            m_sweepCount++;

    for (const auto& c : m_prefetchState.m_prefetchedChunks)
        window.m_dataChunks.insert(std::make_pair(std::get<0>(c).m_id, std::get<1>(c)));

    m_chunkPosition = (ChunkIdType)(m_chunkPosition + m_prefetchState.m_prefetchedChunks.size()) % m_originalChunkDescriptions.size();
}

std::map<std::wstring, size_t> LTTumblingWindowRandomizer::GetInnerState()
{
    std::map<std::wstring, size_t> state;
    state[s_chunkPositionProperty] = m_chunkPosition;
    state[s_sweepIndexProperty] = m_sweepCount;
    return state;
}

void LTTumblingWindowRandomizer::SetInnerState(const std::map<std::wstring, size_t>& state)
{
    std::lock_guard<std::mutex> lock(m_prefetchStateMutex);

    m_sweepCount = ValueFrom(state, s_sweepIndexProperty);
    RandomizeChunks(m_prefetchState, m_sweepCount);
    m_chunkPosition = (ChunkIdType)ValueFrom(state, s_chunkPositionProperty);
}

}
