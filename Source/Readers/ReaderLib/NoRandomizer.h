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
// This class is used for inference and for training where the training data has already been pre - randomized.
// TODO: currently this code moved from the old block randomizer.
// TODO: The class will be further refactored and common based will be extracted with BlockRandomizer.
// TODO: Currently works only for frame mode (numberOfSample in sequence == 1) and without chunking
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
    std::vector<SequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);
    size_t GetChunkIndexOf(size_t t);
    void PrepareNewSweepIfNeeded(size_t samplePosition);

    // Deserializer and information on the original timeline
    IDataDeserializerPtr m_deserializer;

    // Stream descriptions
    std::vector<StreamDescriptionPtr> m_streams;

    // Epoch configuration
    EpochConfiguration m_config;

    ChunkDescriptions m_chunkDescriptions;
    std::vector<size_t> m_chunkSampleOffset;
    std::vector<size_t> m_chunkSequenceOffset;

    std::vector<SequenceDescription> m_sequenceWindow;
    std::map<size_t, ChunkPtr> m_chunks;

    size_t m_globalSamplePosition;
    size_t m_samplePositionInEpoch;
    size_t m_totalNumberOfSamples;
    size_t m_currentSequencePositionInChunk;
    size_t m_currentChunkPosition;
    size_t m_sweep;
};

}}}
