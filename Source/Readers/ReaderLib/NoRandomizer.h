//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"
#include "ReaderUtil.h"

namespace CNTK {

// The class represents a randomizer that does not randomize input (identity function over the original timeline).
// Used training where the training data has already been pre - randomized.
// TODO: currently this code moved from the old block randomizer.
// TODO: The class will be further refactored and common based will be extracted with BlockRandomizer.
class NoRandomizer : public SequenceEnumerator
{
public:
    NoRandomizer(
        DataDeserializerPtr deserializer, 
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences = 0); // per worker

    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t globalSampleCount, size_t localSampleCount) override;
    virtual std::vector<StreamInformation> GetStreamDescriptions() const override
    {
        return m_deserializer->StreamInfos();
    }

    std::map<std::wstring, size_t> GetState() override;
    void SetState(const std::map<std::wstring, size_t>& state) override;

    void SetConfiguration(const ReaderConfiguration& config) override;

private:
    // Gets next sequences not exceeding localSampleCount for this worker and globalSampleCount across workers.
    void GetNextSequenceDescriptions(size_t globalSampleCount, size_t localSampleCount, Sequences& result);

    // Get chunk index for the sample offset from the beginning of the sweep.
    ChunkIdType GetChunkIndexOf(size_t samplePosition);

    // Moves the cursor to the sequence possibly updating the chunk.
    void MoveToNextSequence();

    inline size_t GetEndOfEpochPosition() 
    {
        return m_config.m_totalEpochSizeInSamples * (m_config.m_epochIndex + 1);
    }

    DataDeserializerPtr m_deserializer;

    // Whether to get sequences using multiple thread.
    // Useful in case deserializer performs CPU intensive deserialization (e.g. decompression)
    bool m_multithreadedGetNextSequences;

    // Stream descriptions
    std::vector<StreamInformation> m_streams;

    // Epoch configuration
    EpochConfiguration m_config;

    // Chunk descriptions.
    std::vector<ChunkInfo> m_chunkDescriptions;

    // m_chunkDescription defines the complete sweep of samples: [0 .. N]
    // m_chunkSampleOffset for each chunk contains the sample offset in the sweep where the chunk begins.
    std::vector<size_t> m_chunkSampleOffset;

    // Current chunk data.
    std::map<ChunkIdType, ChunkPtr> m_chunks;

    // Current chunk data id.
    ChunkIdType m_currentChunkId;

    // Current window of sequence descriptions.
    std::vector<SequenceInfo> m_sequenceWindow;

    // Current sequence position the randomizer works with.
    size_t m_currentSequencePositionInChunk;

    // Current chunk position that the randomizer works with.
    // An index inside the m_chunkDescriptions.
    ChunkIdType m_currentChunkPosition;

    // Global sample position on the timeline.
    size_t m_globalSamplePosition;

    // Used for decimation.
    size_t m_globalSequencePosition;

    // Total number of samples in the sweep.
    size_t m_sweepSizeInSamples;

    // Temp buffer to avoid allocations.
    std::vector<SequenceInfo> m_sequenceBuffer;

    // Helper class for removing invalid sequences.
    SequenceCleaner m_cleaner;
};

}
