//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"
#include "ChunkRandomizer.h"
#include <deque>

namespace Microsoft { namespace MSR { namespace CNTK {

// Randomized sequence description.
struct RandomizedSequenceDescription
{
    // Sequence id.
    size_t m_id;
    // Number of samples in sequence.
    size_t m_numberOfSamples;
    // Randomized chunk this sequence belongs to.
    const RandomizedChunk* m_chunk;
};

// Class that given randomized chunks, randomizes sequence descriptions in a window of chunks.
// TODO: This code is still based on the old behavior, so that all current tests pass.
// TODO: Can be simplified if we only randomized sequences forward.
class SequenceRandomizer
{
public:
    SequenceRandomizer(
        IDataDeserializerPtr deserializer,
        ChunkRandomizerPtr chunkRandomizer);

    // Resets current sequence sweep according to the seed.
    void Reset(size_t seed);

    // Sets current sequence cursor given the sample offset in a sweep.
    // If the sample offset point in the middle of sequence, the cursor is moved to the sequence end,
    // and a new sample offset is returned that points to the end of the sequence.
    size_t Seek(size_t sweepSampleOffset, size_t sweep);

    // Gets next sequence descriptions.
    std::vector<RandomizedSequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);

    // Gets current randomized chunk window.
    const std::deque<RandomizedChunk>& GetChunkWindow() const
    {
        return m_chunkWindow;
    }

private:
    DISABLE_COPY_AND_MOVE(SequenceRandomizer);

    // Randomizes next sequence descriptions not exceeding sample count.
    void RandomizeNextSequenceDescriptions(size_t sampleCount);

    // Validates if sequence description is valid for the current position.
    bool IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const;

    // Gets randomized chunk index by the sequence position inside the sweep.
    size_t GetChunkIndexForSequencePosition(size_t sequencePosition) const;

    // Gets randomized sequence description by the sample offset in the sweep.
    RandomizedSequenceDescription& GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId);

    // Gets chunk index given a sample offset in the sweep.
    size_t GetChunkIndexOf(size_t sampleOffsetInSweep);

    // Checks if chunk index is in the randomized window.
    bool IsChunkInWindow(size_t chunkIndex) const;

    // Adds randomized sequences to the window.
    void AddRandomizedSequencesForChunk(size_t chunkIndex);

private:
    // Randomized chunks.
    const std::vector<RandomizedChunk>& m_randomizedChunks;

    // A rolling windows of randomized chunks.
    // Which chunk to load is decided by the BlockRandomizer (i.e. decimation based on chunk).
    std::deque<RandomizedChunk> m_chunkWindow;

    // A rolling window of randomized sequences for the chunks.
    std::deque<std::vector<RandomizedSequenceDescription>> m_sequenceWindow;

    // Index (< m_randomizedChunks.size) of the first chunk in the window(m_chunkWindow).
    size_t m_currentRangeBeginChunkIndex;

    // Index (< m_randomizedChunks.size) of the last chunk in the window(m_chunkWindow).
    size_t m_currentRangeEndChunkIndex;

    // Next sample position not yet randomized.
    size_t m_nextSamplePositionNotYetRandomized;

    // Next sequence position not yet randomized.
    size_t m_nextSequencePositionNotYetRandomized;

    IDataDeserializerPtr m_deserializer;

    // Current sequence position.
    size_t m_currentSequencePosition;
    // Current chunk position.
    size_t m_currentChunkPosition;
    // Current sample position.
    size_t m_currentSamplePosition;

    // Used only as a buffer to get sequence descriptions without memory reallocation.
    std::vector<SequenceDescription> m_bufferOriginalSequences;
};

typedef std::shared_ptr<SequenceRandomizer> SequenceRandomizerPtr;
}}}
