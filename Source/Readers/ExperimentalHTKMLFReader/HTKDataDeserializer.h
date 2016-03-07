//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Config.h"
#include "CorpusDescriptor.h"
#include "UtteranceDescription.h"
#include "ChunkDescription.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an HTK deserializer.
// Provides a set of chunks/sequences to the upper layers.
class HTKDataDeserializer : public IDataDeserializer
{
public:
    HTKDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& featureConfig, const std::wstring& featureName);

    // Describes streams this data deserializer can produce. Streams correspond to network inputs.
    // Produces a single stream of HTK features.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    virtual void GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result) override;

    // Retrieves a chunk with data.
    virtual ChunkPtr GetChunk(size_t chunkId) override;
    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result) override;
    virtual ChunkDescriptions GetChunkDescriptions() override;

private:
    DISABLE_COPY_AND_MOVE(HTKDataDeserializer);

    // Represents a frame.
    // TODO: Change the structure to descrease the memory footprint.
    // TOOD: SequenceDescription should become an interfaces and be requested only for current chunks.
    struct Frame : SequenceDescription
    {
        Frame(UtteranceDescription* u) : m_utterence(u), m_frameIndex(0)
        {
        }

        UtteranceDescription* m_utterence;
        size_t m_frameIndex;
    };

    class HTKChunk;
    void GetSequenceById(size_t chunkId, size_t id, std::vector<SequenceDataPtr>&);

    // Dimension of features.
    size_t m_dimension;

    // All utterance descriptions.
    std::vector<UtteranceDescription> m_utterances;

    // Type of the features.
    ElementType m_elementType;

    // Chunk descriptions.
    std::vector<HTKChunkDescription> m_chunks;
    // Weak pointers on existing chunks.
    std::vector<std::weak_ptr<Chunk>> m_weakChunks;

    // Augmentation window.
    std::pair<size_t, size_t> m_augmentationWindow;

    // Streams exposed by this deserializer.
    std::vector<StreamDescriptionPtr> m_streams;

    CorpusDescriptorPtr m_corpus;

    int m_verbosity;
    size_t m_totalNumberOfFrames;

    // Auxiliary data for checking against the data in the feature file.
    unsigned int m_samplePeriod;
    size_t m_ioFeatureDimension;
    std::string m_featureKind;
};

typedef std::shared_ptr<HTKDataDeserializer> HTKDataDeserializerPtr;

}}}
