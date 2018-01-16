//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include "DataDeserializer.h"

namespace CNTK {

// A cache to store the complete dataset (all chunks) in memory. The caching can
// be switched on/off by a boolean flag in the reader config section, independent 
// of the randomization and chunking parameters. The caching should only be enabled 
// when the whole dataset fits in memory.
// Implemented as a wrapping proxy around a deserializer that stores pointers to
// all chunks it sees in an internal map.
class ChunkCache : public DataDeserializer
{
public:

    ChunkCache(DataDeserializerPtr deserializer) : m_deserializer(deserializer) { }

    virtual std::vector<StreamInformation> StreamInfos() override
    {
        return m_deserializer->StreamInfos();
    }

    virtual std::vector<ChunkInfo> ChunkInfos() override
    {
        return m_deserializer->ChunkInfos();
    }

    virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& descriptions) override
    {
        return m_deserializer->SequenceInfosForChunk(chunkId, descriptions);
    }

    virtual bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo& description) override
    {
        return m_deserializer->GetSequenceInfo(primary, description);
    }

    // Gets chunk data given its id.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId);

private:
    // A map of currently loaded chunks
    std::map<size_t, ChunkPtr> m_chunkMap;
    DataDeserializerPtr m_deserializer;

    DISABLE_COPY_AND_MOVE(ChunkCache);
};

}
