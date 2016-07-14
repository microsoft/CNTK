//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "ChunkCache.h"

namespace Microsoft { namespace MSR { namespace CNTK {

ChunkPtr ChunkCache::GetChunk(ChunkIdType chunkId)
{
    auto it = m_chunkMap.find(chunkId);
    if (it != m_chunkMap.end())
    {
        return it->second;
    }
 
    ChunkPtr chunk = m_deserializer->GetChunk(chunkId);
    m_chunkMap[chunkId] = chunk;
 
    return chunk;
}

} } }
