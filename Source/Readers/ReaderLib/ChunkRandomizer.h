//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Represents an interval of chunks closed on the left and opened on the right.
    struct ClosedOpenChunkInterval
    {
        ChunkIdType m_begin;
        ChunkIdType m_end;
    };

    // Information about randomized chunk.
    struct RandomizedChunk
    {
        // Chunk id.
        ChunkIdType m_chunkId;
        // Pointer to the original chunk.
        const ChunkDescription* m_original;
        // Position of the first sample of the chunk in the input.
        size_t m_samplePositionStart;
        // Position of the first sequence of the chunk in the input.
        size_t m_sequencePositionStart;
        // Randomization window for this chunk.
        ClosedOpenChunkInterval m_randomizationWindow;

        // Position of the last sample of the chunk in the input.
        size_t SampleEndPosition() const
        {
            return m_original->m_numberOfSamples + m_samplePositionStart;
        }

        // Position of the last sequence of the chunk in the input.
        size_t SequenceEndPosition() const
        {
            return m_original->m_numberOfSequences + m_sequencePositionStart;
        }
    };

    // Randomizes a set of chunks and calculates their possible randomization windows.
    // TODO: Currently, we have to preserve the same behavior for randomization in order to make all tests pass.
    // TODO: Randomization can be made simpler if we randomize only forwards.
    class ChunkRandomizer
    {
    public:
        ChunkRandomizer(IDataDeserializerPtr deserializer, size_t randomizationRangeInSamples, bool legacy = false);

        // Gets randomized chunks.
        const std::vector<RandomizedChunk>& GetRandomizedChunks() const;

        // Randomizes chunks based on the seed.
        void Randomize(unsigned int seed);

    private:
        IDataDeserializerPtr m_deserializer;
        // Randomized chunks.
        std::vector<RandomizedChunk> m_randomizedChunks;
        // Original chunks.
        std::vector<ChunkDescriptionPtr> m_originalChunks;
        // Whether to use legacy mode for randomization.
        bool m_legacy;
        // Randomization range in samples.
        size_t m_randomizationRangeInSamples;
    };

    typedef std::shared_ptr<ChunkRandomizer> ChunkRandomizerPtr;
}}}
