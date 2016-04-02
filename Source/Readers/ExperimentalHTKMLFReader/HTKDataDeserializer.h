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
    HTKDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& featureConfig, const std::wstring& featureName);

    // Get information about chunks.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Get information about particular chunk.
    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result) override;

    // Retrieves data for a chunk.
    virtual ChunkPtr GetChunk(size_t chunkId) override;

private:
    class HTKChunk;
    DISABLE_COPY_AND_MOVE(HTKDataDeserializer);

    // Initialization functions.
    void InitializeChunkDescriptions(ConfigHelper& config);
    void InitializeStreams(const std::wstring& featureName);
    void InitializeFeatureInformation();

    // Gets sequence by its chunk id and id inside the chunk.
    void GetSequenceById(size_t chunkId, size_t id, std::vector<SequenceDataPtr>&);

    // Dimension of features.
    size_t m_dimension;

    // Type of the features.
    ElementType m_elementType;

    // Chunk descriptions.
    std::vector<HTKChunkDescription> m_chunks;

    // Weak pointers on existing chunks.
    // If randomizer asks the same chunk twice we do not need to recreate
    // the chunk if we already uploaded it in memory.
    std::vector<std::weak_ptr<Chunk>> m_weakChunks;

    // Augmentation window.
    std::pair<size_t, size_t> m_augmentationWindow;

    CorpusDescriptorPtr m_corpus;

    int m_verbosity;

    // Total number of frames.
    size_t m_totalNumberOfFrames;

    // Flag that indicates whether a single speech frames should be exposed as a sequence.
    bool m_frameMode;

    // Auxiliary data for checking against the data in the feature file.
    unsigned int m_samplePeriod;
    size_t m_ioFeatureDimension;
    std::string m_featureKind;
};

typedef std::shared_ptr<HTKDataDeserializer> HTKDataDeserializerPtr;

}}}
