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
#include <boost/noncopyable.hpp>

namespace CNTK {

// Class represents an HTK deserializer.
// Provides a set of chunks/sequences to the upper layers.
class HTKDeserializer : public DataDeserializerBase, private boost::noncopyable
{
public:
    // Expects new configuration.
    HTKDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

    // TODO: Should be removed, when legacy config goes away, expects configuration in a legacy mode.
    HTKDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& featureConfig, const std::wstring& featureName, bool primary);

    // Get information about chunks.
    virtual std::vector<ChunkInfo> ChunkInfos() override;

    // Get information about particular chunk.
    virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override;

    // Retrieves data for a chunk.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Gets sequence description by the primary one.
    virtual bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo&) override;

private:
    class HTKChunk;

    // Initialization functions.
    void InitializeChunkInfos(ConfigHelper& config);
    void InitializeStreams(const std::wstring& featureName, bool definesMbSize);
    void InitializeFeatureInformation();
    void InitializeAugmentationWindow(const std::pair<size_t, size_t>& augmentationWindow);

    // Gets sequence by its chunk id and id inside the chunk.
    void GetSequenceById(ChunkIdType chunkId, size_t id, std::vector<SequenceDataPtr>&);

    // Dimension of features.
    size_t m_dimension;

    // Type of the features.
    DataType m_elementType;

    // Chunk descriptions.
    std::vector<HTKChunkInfo> m_chunks;

    // Augmentation window.
    std::pair<size_t, size_t> m_augmentationWindow;

    CorpusDescriptorPtr m_corpus;

    // General configuration
    int m_verbosity;

    // Flag that indicates whether a single speech frames should be exposed as a sequence.
    bool m_frameMode;

    // Used to correlate a sequence key with the sequence inside the chunk when deserializer is running not in primary mode.
    // <key, chunkid, offset inside chunk>, sorted by key to be able to retrieve by binary search.
    std::vector<std::tuple<size_t, ChunkIdType, uint32_t>> m_keyToChunkLocation;

    // Auxiliary data for checking against the data in the feature file.
    unsigned int m_samplePeriod = 0;
    size_t m_ioFeatureDimension = 0;
    std::string m_featureKind;

    // A flag that indicates whether the utterance should be extended to match the length of the utterance from the primary deserializer.
    // TODO: This should be moved to the packers when deserializers work in sequence mode only.
    bool m_expandToPrimary;

    // Upper limit of utterance lengths. Longer utterances are skipped.
    size_t m_maxSequenceSize;
};

typedef std::shared_ptr<HTKDeserializer> HTKDeserializerPtr;

}
