//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "DataDeserializer.h"
#include <random>

namespace CNTK {

    // Represents an interval of chunks closed on the left and opened on the right.
    struct ClosedOpenChunkInterval
    {
        ClosedOpenChunkInterval() : m_begin{ 0 }, m_end{ 0 } {}

        friend bool operator== (const ClosedOpenChunkInterval &a, const ClosedOpenChunkInterval &b);
        friend bool operator!= (const ClosedOpenChunkInterval &a, const ClosedOpenChunkInterval &b);

        size_t Size() const
        {
            return m_end - m_begin;
        }

        ChunkIdType m_begin;
        ChunkIdType m_end;
    };

    inline bool operator== (const ClosedOpenChunkInterval &a, const ClosedOpenChunkInterval &b)
    {
        return a.m_begin == b.m_begin && a.m_end == b.m_end;
    }

    inline bool operator!= (const ClosedOpenChunkInterval &a, const ClosedOpenChunkInterval &b)
    {
        return !(a == b);
    }

    // Information about randomized chunk.
    struct RandomizedChunk
    {
        // Chunk id.
        ChunkIdType m_chunkId;
        // Pointer to the original chunk.
        const ChunkInfo* m_original;
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
        ChunkRandomizer(DataDeserializerPtr deserializer, size_t randomizationRange, bool sampleBasedRandomizationWindow = true);

        // Gets randomized chunks.
        const std::vector<RandomizedChunk>& GetRandomizedChunks() const;

        // Randomizes chunks based on the seed.
        void Randomize(size_t seed);

        // Randomize by spraying original sequences over a window of "m_randomizationRange" samples.
        void RandomizeUsingWindowInSamples();

        // Randomize by spraying original sequences over a window of "m_randomizationRange" of chunks.
        void RandomizeUsingWindowInChunks();

    private:
        DataDeserializerPtr m_deserializer;
        // Randomized chunks.
        std::vector<RandomizedChunk> m_randomizedChunks;
        // Original chunks.
        std::vector<ChunkInfo> m_originalChunks;

        // Randomization range in samples.
        size_t m_randomizationRange;

        // Specifies how to interpret randomization range,
        // if true randomization range == number of samples, else 
        // randomization range = number of chunks.
        bool m_sampleBasedRandomizationWindow;

        std::mt19937_64 m_rng;
    };

    typedef std::shared_ptr<ChunkRandomizer> ChunkRandomizerPtr;
}
