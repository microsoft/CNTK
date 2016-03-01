//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <unordered_set>

#include "Transformer.h"
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The class represents a randomizer that does randomization based on chunks/sequences inside a set of chunk.
// TODO: currently this code moved from the old block randomizer.
// TODO: The class will be further refactored and common based will be extracted with NoRandomizer.
// TODO: Currently works only for frame mode (numberOfSample in sequence == 1)
// TODO: This layering will be changed, when we move transformers under the randomizer, it won't be a transformer anymore.
class BlockRandomizer : public Transformer
{
public:
    enum class DistributionMode {
        chunk_modulus,
        sequences_strides
    };

    BlockRandomizer(int verbosity,
                    size_t randomizationRangeInSamples,
                    IDataDeserializerPtr deserializer,
                    DistributionMode distributionMode = DistributionMode::sequences_strides,
                    bool useLegacyRandomization = false);

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
    bool m_useLegacyRandomization;
    int m_verbosity;
    size_t m_randomizationRangeInSamples; // full window
    DistributionMode m_distributionMode;

    // Deserializer and information on the original timeline
    IDataDeserializerPtr m_deserializer;
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
    // TODO optimize footprint:
    //      (do not require full timeline, i.e., Amit's change in original HTKMLFReader)
    //      (instead of SequenceDescription, use something smaller)
    std::vector<SequenceDescription> m_randomTimeline;
    std::vector<StreamDescriptionPtr> m_streams;

    // Chunks that we currently hold a pointer to
    std::map<size_t, ChunkPtr> m_chunks; // TODO vector? or unordered_map

    // Check that timeline has only valid sequences of non-zero length
    // with incrementing IDs and non-decreasing chunk identifiers.
    bool TimelineIsValidForRandomization(const SequenceDescriptions& timeline) const;

    void RandomizeChunks();

    size_t GetChunkIndexForSequencePosition(size_t sequencePosition) const;

    bool IsValidForPosition(size_t targetPosition, const SequenceDescription& seqDesc) const;

    void Randomize();

    void RandomizeForGlobalSamplePosition(const size_t samplePosition);

    bool RandomizeIfNewSweepIsEntered();

    bool GetNextSequenceIds(size_t sampleCount, std::vector<size_t>& originalIds, std::unordered_set<size_t>& originalChunks);
};

}}}
