//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A cache to store the complete dataset (all chunks) in memory. The caching can
// be switched on/off by a boolean flag in the reader config section, independent 
// of the randomization and chunking parameters. The caching should only be enabled 
// when the whole dataset fits in memory.
// Implemented as a wrapping proxy around a deserializer that stores pointers to
// all chunks it sees in an internal map.
class ChunkCache : public IDataDeserializer
{
public:

    ChunkCache(IDataDeserializerPtr deserializer) : m_deserializer(deserializer) { }

    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_deserializer->GetStreamDescriptions();
    }

    virtual ChunkDescriptions GetChunkDescriptions() override
    {
        return m_deserializer->GetChunkDescriptions();
    }

    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& descriptions) override
    {
        return m_deserializer->GetSequencesForChunk(chunkId, descriptions);
    }

    virtual bool GetSequenceDescription(const SequenceDescription& primary, SequenceDescription& description) override
    {
        return m_deserializer->GetSequenceDescription(primary, description);
    }

    // Gets chunk data given its id.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId);

private:
    // A map of currently loaded chunks
    std::map<size_t, ChunkPtr> m_chunkMap;
    IDataDeserializerPtr m_deserializer;

    DISABLE_COPY_AND_MOVE(ChunkCache);
};

} } }
