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

    // Resets the current sweep according to the randomization seed provided.
    void Reset(size_t seed);

    // Sets the current cursor to the given sample offset.
    // If the offset is in the middle of a sequence, the next sequence is picked up.
    // If the offset points in the middle of last sequence, the end of the sweep is returned.
    size_t Seek(size_t sweepSampleOffset, size_t sweep);

    // Gets the next randomized sequence descriptions not exceeding the sample count.
    std::vector<RandomizedSequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);

    // Gets the current randomized chunk window.
    const std::deque<RandomizedChunk>& GetChunkWindow() const
    {
        return m_chunkWindow;
    }

    // Release chunks from the chunk window that are not needed anymore.
    void ReleaseChunks();

private:
    DISABLE_COPY_AND_MOVE(SequenceRandomizer);

    // Randomize one more chunk if needed after the chunk cursor has been incremented.
    void RandomizeNextChunkIfNeeded();

    // Checks if the randomized sequence is valid for a target position using its chunk randomization window.
    bool IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const;

    // Gets randomized chunk index using a sequence position in the sweep.
    size_t GetChunkIndexForSequencePosition(size_t sequencePosition) const;

    // Gets randomized sequence by the sequence id.
    RandomizedSequenceDescription& GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId);

    // Add randomizes sequences for the chunk with a given index.
    void AddRandomizedSequencesForChunk(size_t chunkIndex);

    // Move the chunk cursor to the next chunk, randomizing more sequences if necessary.
    void MoveChunkCursor();

private:

    IDataDeserializerPtr m_deserializer;

    // Used only as a buffer to get sequence descriptions without memory reallocation.
    std::vector<SequenceDescription> m_bufferOriginalSequences;

    // Randomized chunks.
    const std::vector<RandomizedChunk>& m_randomizedChunks;

    //
    // We randomize sequences in a rolling window over the randomized chunks.
    // During randomization, sequences will be moved between different chunks in the window, but the total number
    // of sequences in any chunk stays the same.
    // The number of samples in each randomized chunk, however, may vary due to sequences being changed.
    //
    // NOTE: We do this in order to support the same randomization as used on all regression tests.
    // The rolling window is divided into three parts. The first part is fully randomized, and
    // has sequences at their final position (wrt. the randomization for the sweep). Only sequences
    // from this part are returned to the caller (GetNextSequenceDescriptions).
    // The second and third part correspond to sequences that are being randomized, i.e., within
    // which sequences may still change their position. The randomization cursor, which is located
    // at the boundary between part 2 and 3, indicates where to continue randomization by
    // swapping sequences forward or backward depending on the randomization window of a particular chunk.
    //
    //                              all chunks:
    //                          m_randomizedChunks[]
    //  ----------+------------+---------------+---------------------+-------------
    //            |               loaded chunks:                     |
    //            |      m_chunkWindow[], m_sequenceWindow[]         |
    //   unloaded +------------+------------------+------------------+ chunks to be
    //    chunks  | randomized | in randomization | in randomization |   loaded
    //            |            | (back window)    | (forward window) |
    //  ----------+------------+------------------+------------------+-------------
    //            |     ^      |                  |                  |
    //            |     |      |                  |                  | m_chunkWindowEnd
    //            |     |      |                  |
    //            |     |      |                  | m_randomizationCursor
    //            |     |      |
    //            |     |      | m_randomizedWindowEnd
    //            |     |
    //            |     | m_currentChunkCursor
    //            |
    //            | m_chunkWindowBegin
    //
    //

    // A rolling windows of randomized chunks.
    // Which chunk to load is decided by the BlockRandomizer (i.e. decimation based on chunk).
    std::deque<RandomizedChunk> m_chunkWindow;

    // A rolling window of randomized sequences for the chunks.
    // Contains randomized sequences from m_chunkWindow chunks.
    std::deque<std::vector<RandomizedSequenceDescription>> m_sequenceWindow;

    struct ChunkInfo
    {
        size_t start;
        size_t numberOfSamples;
    };

    // A rolling window of sample start positions and length for chunks that had their
    // sequenced randomized.
    std::deque<ChunkInfo> m_randomizedChunkInfo;

    // Index of the first chunk in the window (inclusive).
    size_t m_chunkWindowBegin;

    // Indices of chunk, sequence, and sample from which to return data to caller.
    size_t m_currentChunkCursor;
    size_t m_currentSequenceCursor;
    size_t m_currentSampleCursor;

    // Index of the last fully randomized chunk in the window (exclusive).
    size_t m_randomizedWindowEnd;

    // Index of the chunk in the window where to continue randomizing sequences.
    size_t m_randomizationCursor;

    // Index of the last chunk in the window (exclusive).
    size_t m_chunkWindowEnd;
};

typedef std::shared_ptr<SequenceRandomizer> SequenceRandomizerPtr;
}}}
