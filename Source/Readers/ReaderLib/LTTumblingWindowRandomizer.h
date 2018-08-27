//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "LocalTimelineRandomizerBase.h"

namespace CNTK {

// LT - LocalTimeline
// A randomizer that firstly randomizes chunks and then sequences inside a tumbling window of chunks.
class LTTumblingWindowRandomizer : public LocalTimelineRandomizerBase
{
    typedef LocalTimelineRandomizerBase Base;

public:
    LTTumblingWindowRandomizer(
        DataDeserializerPtr deserializer,
        bool sampleBasedRandomizationWindow,
        size_t randomizationRange,
        size_t seedOffset = 0,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences= 0); // per worker

    std::map<std::wstring, size_t> GetInnerState() override;
    void SetInnerState(const std::map<std::wstring, size_t>& state) override;
    void RefillSequenceWindow(SequenceWindow& window) override;
    void Prefetch() const override;

private:
    struct PrefetchState
    {
        std::mt19937_64 m_rng;
        std::vector<ChunkInfo> m_prefetchedChunkDescriptions;
        std::vector<SequenceInfo> m_prefetchedSequences;
        std::vector<std::tuple<ChunkInfo, ChunkPtr>> m_prefetchedChunks;
    };
    void RandomizeWindow(PrefetchState& prefetchState, size_t sweepCount, size_t chunkPositionOfWindow, size_t sequencePositionInWindow) const;
    void RandomizeChunks(PrefetchState& prefetchState, size_t sweepCount) const;

    const size_t m_randomizationRange;
    const size_t m_seedOffset;
    const bool m_sampleBasedRandomizationWindow;

    // Current chunk position that the randomizer works with.
    ChunkIdType m_chunkPosition;
    // Current sweep count, incremented when the next window
    // is fetched.
    size_t m_sweepCount;


    mutable PrefetchState m_prefetchState;
    mutable std::mutex m_prefetchStateMutex;
};

}
