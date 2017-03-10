//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Config.h"
#include "CorpusDescriptor.h"
#include "UtteranceDescription.h"
#include "HTKChunkDescription.h"
#include "ConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an HTK deserializer.
// Provides a set of chunks/sequences to the upper layers.
class HTKDataDeserializer : public DataDeserializerBase
{
public:
    // Expects new configuration.
    HTKDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

    // TODO: Should be removed, when legacy config goes away, expects configuration in a legacy mode.
    HTKDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& featureConfig, const std::wstring& featureName, bool primary);

    // Get information about chunks.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Get information about particular chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result) override;

    // Retrieves data for a chunk.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Gets sequence description by the primary one.
    virtual bool GetSequenceDescription(const SequenceDescription& primary, SequenceDescription&) override;

private:
    class HTKChunk;
    DISABLE_COPY_AND_MOVE(HTKDataDeserializer);

    // Initialization functions.
    void InitializeChunkDescriptions(const std::vector<std::string>& paths);
    void InitializeStreams(const std::wstring& featureName);
    void InitializeFeatureInformation();
    void InitializeAugmentationWindow(const std::pair<size_t, size_t>& augmentationWindow);

    // Gets sequence by its chunk id and id inside the chunk.
    void GetSequenceById(ChunkIdType chunkId, size_t id, std::vector<SequenceDataPtr>&);

    // Dimension of features.
    size_t m_dimension;

    // Type of the features.
    ElementType m_elementType;

    // Chunk descriptions.
    std::vector<HTKChunkDescription> m_chunks;

    // Augmentation window.
    std::pair<size_t, size_t> m_augmentationWindow;

    CorpusDescriptorPtr m_corpus;

    // General configuration
    int m_verbosity;

    // Total number of frames.
    size_t m_totalNumberOfFrames = 0;

    // Flag that indicates whether a single speech frames should be exposed as a sequence.
    bool m_frameMode;

    // Used to correlate a sequence key with the sequence inside the chunk when deserializer is running not in primary mode.
    // Key -> <chunkid, offset inside chunk>
    std::map<size_t, std::pair<size_t, size_t>> m_keyToChunkLocation;

    // Auxiliary data for checking against the data in the feature file.
    unsigned int m_samplePeriod = 0;
    size_t m_ioFeatureDimension = 0;
    std::string m_featureKind;

    // A flag that indicates whether the utterance should be extended to match the lenght of the utterance from the primary deserializer.
    // TODO: This should be moved to the packers when deserializers work in sequence mode only.
    bool m_expandToPrimary;
};

typedef std::shared_ptr<HTKDataDeserializer> HTKDataDeserializerPtr;

}}}
