//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"
#include "ReaderUtil.h"
#include "LocalTimelineRandomizerBase.h"

namespace CNTK {

// The class represents a randomizer that does not randomize input (identity function over the original timeline).
// Used training where the training data has already been pre - randomized.
class LocalTimelineNoRandomizer : public LocalTimelineRandomizerBase
{
    typedef LocalTimelineRandomizerBase Base;

public:
    LocalTimelineNoRandomizer(
        DataDeserializerPtr deserializer,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences = 0); // per worker

    Dictionary GetInnerState() override;
    void SetInnerState(const Dictionary& state) override;
    void RefillSequenceWindow() override;

    ~LocalTimelineNoRandomizer()
    {
        if (m_prefetch.valid())
            m_prefetch.wait_for(std::chrono::seconds(60));
    }

private:
    void Prefetch();

    // Current chunk position.
    ChunkIdType m_currentChunkPosition;

    // Current sequence position
    size_t m_currentSequencePosition;
    std::tuple<ChunkDescription, ChunkPtr, std::vector<SequenceDescription>> m_prefetchedChunk;
};

}
