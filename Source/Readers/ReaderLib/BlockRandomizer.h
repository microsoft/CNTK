//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The class represents a randomizer that does randomization based on chunks/sequences inside a set of chunk.
// TODO: currently this code moved from the old block randomizer.
// The class will be further refactored and common based will be extracted with NoRandomizer.
// Currently works only for frame mode (numberOfSample in sequence == 1)
class BlockRandomizer : public Transformer
{
public:
    BlockRandomizer(int verbosity, size_t randomizationRangeInSamples, DataDeserializerPtr deserializer);
    virtual ~BlockRandomizer()
    {
    }

    virtual void Initialize(TransformerPtr next, const ConfigParameters& readerConfig) override;
    virtual void StartEpoch(const EpochConfiguration& config) override;
    virtual Sequences GetNextSequences(size_t sampleCount) override;
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

private:
    enum class DistributionMode {
        // TODO better names, description
        chunk_modulus,
        sequences_strides
    };

    // Structure for per-chunk information
    struct ChunkInformation
    {
        size_t m_sequencePositionStart;
        size_t m_samplePositionStart;
    };

    // Structure that will be maintained for each randomized chunk
    struct RandomizedChunk
    {
        struct ChunkInformation m_info; // sample positions are global // TODO could drop 'global' requirement?

        size_t m_originalChunkIndex;

        // Randomization range (in randomized chunk positions; right-side open)
        size_t m_windowBegin;
        size_t m_windowEnd;
    };

    // General configuration
    int m_verbosity;
    size_t m_randomizationRangeInSamples; // full window
    DistributionMode m_distributionMode;

    // Deserializer and information on the original timeline
    DataDeserializerPtr m_deserializer;
    size_t m_numSequences;
    size_t m_numChunks;
    size_t m_numSamples;
    bool m_frameMode;                                 // true iff only single-sample sequences
    std::vector<ChunkInformation> m_chunkInformation; // (includes a sentinel)

    // Per-epoch configuration
    size_t m_workerRank;
    size_t m_numberOfWorkers;
    size_t m_epochSize;
    size_t m_samplePositionInEpoch;

    // Per-randomization-sweep information
    size_t m_sweep;
    size_t m_sweepStartInSamples; // TODO do we need it?
    size_t m_sequencePositionInSweep;
    std::vector<RandomizedChunk> m_randomizedChunks;    // (includes a sentinel)
    std::vector<size_t> m_sequencePositionToChunkIndex; // TODO find on m_randomizedChunks instead?
    std::vector<SequenceDescription> m_randomTimeline;

    // Check that timeline has only valid sequences of non-zero length
    // with incrementing IDs and non-decreasing chunk identifiers.
    bool TimelineIsValidForRandomization(const SequenceDescriptions& timeline) const;

    void RandomizeChunks();

    bool IsValidForPosition(size_t targetPosition, const SequenceDescription& seqDesc) const;

    void Randomize();

    void RandomizeForGlobalSamplePosition(const size_t samplePosition);

    void RandomizeIfNewSweepIsEntered();

    bool GetNextSequenceIds(size_t sampleCount, std::vector<size_t>& ids);
};
} } }
