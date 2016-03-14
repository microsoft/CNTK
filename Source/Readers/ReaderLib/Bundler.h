//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an bundler of several deserializers.
// In case when only a single deserializer is used, the bundler can be omitted and 
// no performance penalty is paid.
// TODO: The interface will changed when the timeline will support chunking.
class Bundler : public IDataDeserializer
{
public:
    Bundler(const ConfigParameters& readerConfig, IDataDeserializerPtr driver, std::vector<IDataDeserializerPtr> deserializers);

    // Retrieves description of all sequences this data deserializer can produce, together with associated chunks.
    // TODO: For huge corpus, the memory footprint is too big. We adapt this interface to request timeline in chunks.
    virtual const SequenceDescriptions& GetSequenceDescriptions() const override;

    // Retrieves description of a single sequence given its key.
    virtual const SequenceDescription* GetSequenceDescriptionByKey(const KeyType& key) override;

    // Describes bundled streams of the underlying data deserializers.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // Retrieves a chunk with data.
    virtual ChunkPtr GetChunk(size_t) override;

    // Retrieves total number of chunks this deserializer can produce.
    virtual size_t GetTotalNumberOfChunks() override;

private:
    DISABLE_COPY_AND_MOVE(Bundler);

    void CreateSequenceDescriptions();

    // Exposed bundled streams.
    std::vector<StreamDescriptionPtr> m_streams;
    // Underlying deserializers.
    std::vector<IDataDeserializerPtr> m_deserializers;
    // Driving deserializer that defines chunks.
    IDataDeserializerPtr m_driver;

    // Seqeunce descriptions.
    std::vector<SequenceDescription> m_sequenceDescriptions;
    SequenceDescriptions m_sequences;

    // Exposed sequence id to chunk mapping.
    std::vector<std::vector<size_t>> m_sequenceToChunk;

    // Exposed sequence id to internal sequence id mapping.
    std::vector<std::vector<size_t>> m_sequenceToSequence;

    // Chunk offsets - m_chunkOffsets[chunkId] stores the index of 
    // the sequence in m_sequenceDescription where the chunk starts.
    std::vector<size_t> m_chunkOffsets;

    friend class BundlingChunk;
};

}}}
