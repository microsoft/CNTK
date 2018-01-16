//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "BinaryConfigHelper.h"
#include "BinaryChunkDeserializer.h"
#include "BinaryDataDeserializer.h"

namespace CNTK {

class BinaryDataChunk : public Chunk, public std::enable_shared_from_this<Chunk>
{
public:
    explicit BinaryDataChunk(ChunkIdType chunkId,
        size_t numSequences, 
        unique_ptr<byte[]> buffer, 
        std::vector<BinaryDataDeserializerPtr> deserializer)
        : m_chunkId(chunkId),
        m_numSequences(numSequences), 
        m_buffer(std::move(buffer)), 
        m_deserializers(deserializer)
    { }

    // Gets a sequence using its index inside the chunk.
    void GetSequence(size_t sequenceIdx, std::vector<SequenceDataPtr>& result) override
    {
        // Check if we've already parsed the chunk. If not, parse it.
        if (m_data.size() == 0)
            ParseChunk();

        assert(m_data.size() != 0);

        // resize the output to have the same dimensionality
        result.resize(m_data.size());
        // now copy the decoded sequences
        for (size_t i = 0; i < m_data.size(); i++)
            result[i] = m_data[i].at(sequenceIdx);
    }

    uint32_t GetNumSamples(size_t sequenceIdx)
    {
        uint32_t numSamples = 0;
        for (size_t i = 0; i < m_data.size(); i++)
            numSamples = max(numSamples, m_data[i].at(sequenceIdx)->m_numberOfSamples);
        return numSamples;
    }

protected:
    void ParseChunk()
    {
        m_data.resize(m_deserializers.size());

        // the number of bytes of buffer that have been processed by the deserializer so far
        size_t bytesProcessed = 0;
        // Now call all of the deserializers on the chunk, in order
        for (size_t i = 0; i < m_deserializers.size(); i++)
            bytesProcessed += m_deserializers[i]->GetSequenceDataForChunk(m_numSequences, m_buffer.get() + bytesProcessed, m_data[i]);
    }

    // chunk id (copied from the descriptor)
    ChunkIdType m_chunkId;

    // num sequences in this chunk. Note this should be in the chunk, but for simplicity it is in the offsets table
    // so we must tell the chunk where it starts.
    size_t m_numSequences;

    // This is the actual chunk read from disk. We will call back to the deserializer for it to be deserialized
    unique_ptr<byte[]> m_buffer;

    // This is the deserializer who knows how to interpret the m_data chunk that we read in
    std::vector<BinaryDataDeserializerPtr> m_deserializers;
    
    // The parsed data. We will parse each chunk once, and store the data here. 
    // If we want to delay parsing, we will add that later as/if needed.
    std::vector<std::vector<SequenceDataPtr>> m_data;
};

typedef shared_ptr<BinaryDataChunk> BinaryChunkPtr;

}
