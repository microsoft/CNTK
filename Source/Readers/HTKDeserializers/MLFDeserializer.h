//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "DataDeserializer.h"
#include "HTKDataDeserializer.h"
#include "CorpusDescriptor.h"
#include "MLFUtils.h"
#include "MLFIndexer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an MLF deserializer.
// Provides a set of chunks/sequences to the upper layers.
class MLFDeserializer : public DataDeserializerBase, boost::noncopyable
{
public:
    // Expects new configuration.
    MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

    // TODO: Should be removed, when all readers go away, expects configuration in a legacy mode.
    MLFDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    bool GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& s) override;

    // Gets description of all chunks.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Get sequence descriptions of a particular chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& s) override;

    // Retrieves a chunk with data.
    virtual ChunkPtr GetChunk(ChunkIdType) override;

private:
    class ChunkBase;
    class SequenceChunk;
    class FrameChunk;

    // Initializes chunk descriptions.
    void InitializeChunkDescriptions(CorpusDescriptorPtr corpus, const ConfigHelper& config, const std::wstring& stateListPath);

    // Initializes a single stream this deserializer exposes.
    void InitializeStream(const std::wstring& name);

    // In frame mode initializes data for all categories/labels in order to
    // avoid memory copy.
    void InitializeReadOnlyArrayOfLabels();

    // Vector that maps KeyType.m_sequence into an utterance ID (or type max() if the key is not assigned).
    // This assumes that IDs introduced by the corpus are dense (which they right now, depending on the number of invalid / filtered sequences).
    std::vector<std::pair<ChunkIdType, uint32_t>> m_keyToSequence;

    // Type of the data this serializer provides.
    ElementType m_elementType;

    // Array of available categories.
    // We do no allocate data for all input sequences, only returning a pointer to existing category.
    std::vector<SparseSequenceDataPtr> m_categories;

    // A list of category indices
    // (a list of numbers from 0 to N, where N = (number of categories -1))
    std::vector<IndexType> m_categoryIndices;

    // Flag that indicates whether a single speech frames should be exposed as a sequence.
    bool m_frameMode;

    CorpusDescriptorPtr m_corpus;

    std::vector<const ChunkDescriptor*> m_chunks;
    std::map<const ChunkDescriptor*, size_t> m_chunkToFileIndex;

    size_t m_dimension;
    size_t m_chunkSizeBytes;

    // Track phone boundaries
    bool m_withPhoneBoundaries;

    StateTablePtr m_stateTable;

    std::vector<std::pair<std::wstring, MLFIndexerPtr>> m_indexers;
    std::vector<std::wstring> m_mlfFiles;
};

}}}
