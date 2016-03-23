//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <map>
#include "Transformer.h"
#include "DataDeserializer.h"
#include "SequenceRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The class represents a randomizer that does not randomize input (identity function over the original timeline).
// Used training where the training data has already been pre - randomized.
// TODO: currently this code moved from the old block randomizer.
// TODO: The class will be further refactored and common based will be extracted with BlockRandomizer.
// TODO: This layering will be changed, when we move transformers under the randomizer, it won't be a transformer anymore.
class NoRandomizer : public Transformer
{
public:
    NoRandomizer(IDataDeserializerPtr deserializer);

    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;
    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t sampleCount) override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

private:
    // Gets next sequence descriptions with total size less than sampleCount.
    std::vector<SequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);

    // Get chunk index for the sample offset from the beginning of the sweep.
    size_t GetChunkIndexOf(size_t samplePosition);

    // Moves the cursor to the sequence possibly updating the chunk.
    void MoveToNextSequence();

    IDataDeserializerPtr m_deserializer;

    // Stream descriptions
    std::vector<StreamDescriptionPtr> m_streams;

    // Epoch configuration
    EpochConfiguration m_config;

    // Chunk descriptions.
    ChunkDescriptions m_chunkDescriptions;

    // m_chunkDescription defines the complete sweep of samples: [0 .. N]
    // m_chunkSampleOffset for each chunk contains the sample offset in the sweep where the chunk begins.
    std::vector<size_t> m_chunkSampleOffset;

    // Current chunk data.
    ChunkPtr m_currentChunk;
    // Current chunk data id.
    size_t m_currentChunkId;

    // Current window of sequence descriptions.
    std::vector<SequenceDescription> m_sequenceWindow;

    // Current sequence position the randomizer works with.
    size_t m_currentSequencePositionInChunk;

    // Current chunk position that the randomizer works with.
    // An index inside the m_chunkDescriptions.
    size_t m_currentChunkPosition;

    // Global sample position on the timeline.
    // TODO: possible recalculate it base on samplePositionInEpoch.
    size_t m_globalSamplePosition;

    // Current sample position in the epoch.
    size_t m_samplePositionInEpoch;

    // Total number of samples in the sweep.
    size_t m_totalNumberOfSamples;
};

}}}
