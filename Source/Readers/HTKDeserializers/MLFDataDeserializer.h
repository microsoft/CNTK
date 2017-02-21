//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "HTKDataDeserializer.h"
#include "../HTKMLFReader/biggrowablevectors.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an MLF deserializer.
// Provides a set of chunks/sequences to the upper layers.
class MLFDataDeserializer : public DataDeserializerBase
{
public:
    // Expects new configuration.
    MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

    // TODO: Should be removed, when all readers go away, expects configuration in a legacy mode.
    MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    bool GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& s) override;

    // Gets description of all chunks.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Get sequence descriptions of a particular chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& s) override;

    // Retrieves a chunk with data.
    // TODO: Currently it is a single chunk => all labels are loaded into memory.
    // TODO: After we switch the timeline to work in chunks, we will also introduce chunking of labels.
    virtual ChunkPtr GetChunk(ChunkIdType) override;

private:
    class MLFChunk;
    DISABLE_COPY_AND_MOVE(MLFDataDeserializer);

    void InitializeChunkDescriptions(CorpusDescriptorPtr corpus, const ConfigHelper& config, const std::wstring& stateListPath, size_t dimension);
    void InitializeStream(const std::wstring& name, size_t dimension);

    void GetSequenceById(size_t sequenceId, std::vector<SequenceDataPtr>& result);

    // Vector that maps KeyType.m_sequence into an utterance ID (or SIZE_MAX if the key is not assigned).
    // This assumes that IDs introduced by the corpus are dense (which they right now, depending on the number of invalid / filtered sequences).
    // TODO compare perf to map we had before.
    std::vector<size_t> m_keyToSequence;

    // Number of sequences
    size_t m_numberOfSequences = 0;

    // Array of all labels.
    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;

    // Index of utterances in the m_classIds (index of the first frame of the utterance)
    msra::dbn::biggrowablevector<size_t> m_utteranceIndex;

    // Type of the data this serializer provides.
    ElementType m_elementType;

    // Total number of frames.
    size_t m_totalNumberOfFrames;

    // Array of available categories.
    // We do no allocate data for all input sequences, only returning a pointer to existing category.
    std::vector<SparseSequenceDataPtr> m_categories;

    // A list of category indices
    // (a list of numbers from 0 to N, where N = (number of categories -1))
    std::vector<IndexType> m_categoryIndices;

    // Flag that indicates whether a single speech frames should be exposed as a sequence.
    bool m_frameMode;
};

}}}
