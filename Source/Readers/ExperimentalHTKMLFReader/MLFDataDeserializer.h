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
class MLFDataDeserializer : public IDataDeserializer
{
public:
    MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

    // Describes streams this data deserializer can produce. Streams correspond to network inputs.
    // Produces a single stream of MLF labels.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // Retrieves description of all sequences this data deserializer can produce, together with associated chunks.
    // TODO: For huge corpus, the memory footprint is too big. We adapt this interface to request timeline in chunks.
    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    const SequenceDescription* GetSequenceDescriptionByKey(const KeyType& key) override;

    // Retrieves total number of chunks this deserializer can produce.
    virtual size_t GetTotalNumberOfChunks() override;

    // Retrieves a chunk with data.
    // TODO: Currenty it is a single chunk => all labels are loaded into memory.
    // TODO: After we switch the timeline to work in chunks, we will also introduce chunking of labels.
    virtual ChunkPtr GetChunk(size_t) override;

private:
    DISABLE_COPY_AND_MOVE(MLFDataDeserializer);

    // Inner class for a frame.
    struct MLFFrame : SequenceDescription
    {
        // Index of the frame in the utterance.
        size_t m_index;
    };

    class MLFChunk;

    std::vector<SequenceDataPtr> GetSequenceById(size_t sequenceId);

    // Key to sequence map.
    std::map<wstring, size_t> m_keyToSequence;

    // Array of all labels.
    msra::dbn::biggrowablevector<msra::dbn::CLASSIDTYPE> m_classIds;
    // Index of utterances in the m_classIds.
    msra::dbn::biggrowablevector<size_t> m_utteranceIndex;

    // TODO: All sequences(currently frames), this deserializer provides.
    // This interface has to change when the randomizer asks timeline in chunks.
    msra::dbn::biggrowablevector<MLFFrame> m_frames;
    SequenceDescriptions m_sequences;

    // Type of the data this serializer provdes.
    ElementType m_elementType;

    // Streams, this deserializer provides. A single mlf stream.
    std::vector<StreamDescriptionPtr> m_streams;
};

}}}
