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
    MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    void GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& s) override;

    // Gets description of all chunks.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Get sequence descriptions of a particular chunk.
    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& s) override;

    // Retrieves a chunk with data.
    // TODO: Currenty it is a single chunk => all labels are loaded into memory.
    // TODO: After we switch the timeline to work in chunks, we will also introduce chunking of labels.
    virtual ChunkPtr GetChunk(size_t) override;

private:
    class MLFChunk;
    DISABLE_COPY_AND_MOVE(MLFDataDeserializer);

    // Inner class for a frame.
    struct MLFFrame : SequenceDescription
    {
        // Index of the frame in the utterance.
        size_t m_index;
    };

    void GetSequenceById(size_t sequenceId, std::vector<SequenceDataPtr>& result);

    // Key to sequence map.
    std::map<size_t, size_t> m_keyToSequence;

    // Array of all labels.
    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;

    // Index of utterances in the m_classIds.
    msra::dbn::biggrowablevector<size_t> m_utteranceIndex;

    // TODO: All sequences(currently frames), this deserializer provides.
    // This interface has to change when the randomizer asks timeline in chunks.
    msra::dbn::biggrowablevector<MLFFrame> m_frames;

    // Type of the data this serializer provdes.
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
