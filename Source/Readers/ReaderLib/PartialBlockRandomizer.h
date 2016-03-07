//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"
#include "ChunkRandomizer.h"
#include "SequenceRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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
        bool useLegacyRandomization);

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
    void RetrieveDataChunks();
    bool GetNextSequenceDescriptions(size_t sampleCount, std::vector<RandomizedSequenceDescription>& result);
    void PrepareNewSweepIfNeeded(size_t samplePosition);

    size_t m_globalSamplePosition;
    EpochConfiguration m_config;
    size_t m_epochSize;

    size_t m_sweep;
    size_t m_sweepStartInSamples;

    IDataDeserializerPtr m_deserializer;
    ChunkRandomizerPtr m_chunkRandomizer;
    SequenceRandomizerPtr m_sequenceRandomizer;
    std::vector<StreamDescriptionPtr> m_streams;

    std::map<size_t, ChunkPtr> m_chunks;
    size_t m_lastSeenChunk;

    // General configuration
    int m_verbosity;
    DistributionMode m_distributionMode;
    size_t m_sweepTotalNumberOfSamples;
};

}}}
