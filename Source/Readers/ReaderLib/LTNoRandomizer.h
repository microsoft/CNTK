//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "LocalTimelineRandomizerBase.h"

namespace CNTK {

// LT - LocalTimeline
// A randomizer that does not randomize input (identity function over the original timeline).
class LTNoRandomizer : public LocalTimelineRandomizerBase
{
    typedef LocalTimelineRandomizerBase Base;

public:
    LTNoRandomizer(
        DataDeserializerPtr deserializer,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences = 0); // per worker

    std::map<std::wstring, size_t> GetInnerState() override;
    void SetInnerState(const std::map<std::wstring, size_t>& state) override;

    void RefillSequenceWindow(SequenceWindow& window) override;
    void Prefetch() const override;

private:
    // Current chunk position.
    ChunkIdType m_currentChunkPosition;

    // Current sequence position.
    size_t m_currentSequencePosition;

    // Prefetched chunk, expandable - no need to include in a checkpoint.
    // Can be recomputed after restore.
    struct PrefetchedChunk
    {
        ChunkInfo m_info;
        ChunkPtr m_data;
        std::vector<SequenceInfo> m_sequenceInfos;
    };

    mutable PrefetchedChunk m_prefetchedChunk;
};

}
