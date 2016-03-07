//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    struct ClosedOpenInterval
    {
        size_t m_begin;
        size_t m_end;
    };

    struct RandomizedChunk
    {
        size_t m_chunkId;
        const ChunkDescription* m_original;
        size_t m_samplePositionStart;
        size_t m_sequencePositionStart;
        ClosedOpenInterval m_randomizationWindow;

        size_t SampleEndPosition() const
        {
            return m_original->numberOfSamples + m_samplePositionStart;
        }

        size_t SequenceEndPosition() const
        {
            return m_original->numberOfSequences + m_sequencePositionStart;
        }
    };

    class ChunkRandomizer
    {
        IDataDeserializerPtr m_deserializer;
        std::vector<RandomizedChunk> m_randomizedChunks;
        std::vector<ChunkDescriptionPtr> m_originalChunks;
        bool m_legacy;
        size_t m_randomizationRangeInSamples;

    public:
        ChunkRandomizer(IDataDeserializerPtr deserializer, bool legacy, size_t randomizationRangeInSamples);
        const std::vector<RandomizedChunk>& GetRandomizedChunks() const;
        void Randomize(unsigned int seed);
    };

    typedef std::shared_ptr<ChunkRandomizer> ChunkRandomizerPtr;
}}}