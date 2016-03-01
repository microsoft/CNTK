//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
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

    size_t globalte() const
    {
        return m_original->numberOfSamples + m_samplePositionStart;
    }

    size_t PositionEnd() const
    {
        return m_original->numberOfSequences + m_sequencePositionStart;
    }
};

struct RandomizedSequenceDescription
{
    SequenceDescriptionPtr m_original;
    const RandomizedChunk* m_chunk;
};

// The class represents a randomizer that uses a partial timeline for randomization.
class PartialBlockRandomizer : public Transformer
{
public:
    enum class DistributionMode 
    {
        chunk,
        sequence
    };

    PartialBlockRandomizer(
        int verbosity,
        size_t randomizationRangeInSamples,
        IDataDeserializerPtr deserializer,
        DistributionMode distributionMode,
        bool useLegacyRandomization,
        IMetaDataPtr metadata);

    virtual ~PartialBlockRandomizer()
    {
    }

    virtual void Initialize(TransformerPtr, const ConfigParameters&) override {};
    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t sampleCount) override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

private:
    size_t m_globalSamplePosition;
    EpochConfiguration m_config;
    ChunkDescriptions m_originalChunks;

    size_t m_sweep;
    size_t m_sweepStartInSamples;

    friend class SequenceRandomizer;
    std::shared_ptr<SequenceRandomizer> m_sequenceRandomizer;
    std::map<size_t, ChunkPtr> m_chunks;
    IDataDeserializerPtr m_deserializer;
    std::vector<RandomizedChunk> m_randomizedChunks;    // (includes a sentinel)
    std::vector<StreamDescriptionPtr> m_streams;
    IMetaDataPtr m_metaData;

    // General configuration
    bool m_useLegacyRandomization;
    int m_verbosity;
    size_t m_randomizationRangeInSamples; // full window
    DistributionMode m_distributionMode;


    bool GetNextSequenceDescriptions(size_t sampleCount, std::vector<RandomizedSequenceDescription>& result);

    // Per-epoch configuration
    size_t m_epochSize;
    size_t m_samplePositionInEpoch;

    // Per-randomization-sweep information
    size_t m_sequencePositionInSweep;

    void RandomizeChunks();

    void RandomizeForGlobalSamplePosition(size_t samplePosition);
};

}}}
