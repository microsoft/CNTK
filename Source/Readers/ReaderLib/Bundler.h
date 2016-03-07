//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "Config.h"
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

struct BundlerChunkDescription : public ChunkDescription
{
    ChunkDescriptionPtr m_original;
    std::set<size_t> m_invalid;
};

typedef std::shared_ptr<BundlerChunkDescription> BundlerChunkDescriptionPtr;

struct BundlerSequenceDescription : public SequenceDescription
{
    BundlerChunkDescriptionPtr m_chunk;
};

typedef std::shared_ptr<BundlerSequenceDescription> BundlerSequenceDescriptionPtr;

// Class represents an bundler of several deserializers.
// In case when only a single deserializer is used, the bundler can be omitted and 
// no performance penalty is paid.
// TODO: The interface will changed when the timeline will support chunking.
class Bundler : public IDataDeserializer
{
public:
    Bundler(const ConfigParameters& readerConfig, IDataDeserializerPtr driver, std::vector<IDataDeserializerPtr> deserializers, bool cleanse);

    virtual ChunkDescriptions GetChunkDescriptions() override;
    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result) override;

    // Retrieves description of a single sequence given its key.
    virtual void GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result) override;

    // Describes bundled streams of the underlying data deserializers.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // Retrieves a chunk with data.
    virtual ChunkPtr GetChunk(size_t) override;

private:
    DISABLE_COPY_AND_MOVE(Bundler);

    void CreateChunkDescriptions();

    // Exposed bundled streams.
    std::vector<StreamDescriptionPtr> m_streams;

    // Underlying deserializers.
    std::vector<IDataDeserializerPtr> m_deserializers;

    // Driving deserializer that defines chunks.
    IDataDeserializerPtr m_driver;

    std::vector<BundlerChunkDescriptionPtr> m_chunks;

    // True if there should be cleaning of data between different deserializers.
    bool m_cleanse;

    friend class BundlingChunk;
};

}}}
