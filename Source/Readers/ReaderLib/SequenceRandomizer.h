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
    // Sequnce id.
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
    // Randomized chunks.
    const std::vector<RandomizedChunk>& m_randomizedChunks;

    // A rolling windows of chunks of used for randomization.
    // Along with each sequence description, we store the chunk index of the original sequence
    // at that index before randomization, to be used for determining the chunk range
    // to be used for randomization of that frame's position
    std::deque<std::vector<std::pair<unsigned short, RandomizedSequenceDescription>>> m_randomizedSequenceWindow;
    std::deque<RandomizedChunk> m_randomizedChunkWindow;

    size_t m_currentRangeBeginChunkIdx;
    size_t m_currentRangeEndChunkIdx;

    size_t m_nextFramePosNotYetRandomized;
    size_t m_nextSequencePosNotYetRandomized;
    IDataDeserializerPtr m_deserializer;

    size_t m_currentSequencePosition;
    size_t m_currentChunkPosition;
    size_t m_currentFramePosition;

    std::vector<SequenceDescription> m_bufferOriginalSequences;

public:
    SequenceRandomizer(
        IDataDeserializerPtr deserializer,
        ChunkRandomizerPtr chunkRandomizer);

    void Reset(size_t seed);
    void SetSequencePositionTo(size_t globalSample, size_t sweep);

    std::vector<RandomizedSequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);
    void RandomizeSequenceForRange(size_t sampleCount);

    const std::deque<RandomizedChunk>& GetChunkWindow() const
    {
        return m_randomizedChunkWindow;
    }

private:
    bool IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const;
    size_t GetChunkIndexForSequencePosition(size_t sequencePosition) const;
    RandomizedSequenceDescription& GetRandomizedSequenceDescription(size_t globalts);
    RandomizedSequenceDescription& GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId);
    size_t GetChunkIndexOf(size_t t);
    bool IsChunkInRange(size_t chunkIdx) const;
    void AddRandomizedFramesForChunk(size_t chunkIdx);
    std::pair<unsigned short, RandomizedSequenceDescription>& GetRandomizedSequenceBySequenceId(size_t sequenceId);
    std::pair<unsigned short, RandomizedSequenceDescription>& RandomizedSequenceByGlobalSample(size_t globalts);
    size_t GetRandomizedSequenceIdByGlobalSample(size_t globalts);

    DISABLE_COPY_AND_MOVE(SequenceRandomizer);
};

typedef std::shared_ptr<SequenceRandomizer> SequenceRandomizerPtr;
}}}
