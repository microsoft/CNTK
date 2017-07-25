//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "BinaryConfigHelper.h"
#include "BinaryDataChunk.h"
#include "BinaryDataDeserializer.h"

namespace CNTK {

// Chunk meta-info: byte offset in the inputfile, number of sequences and samples in the chunk.
struct BinaryChunkInfo 
{
    int64_t offset;
    uint32_t numSequences;
    uint32_t numSamples;
};

// Chunk table used to find the chunks in the binary file. Added some helper methods around the core data.
class ChunkTable {
public:

    ChunkTable(uint32_t numChunks, BinaryChunkInfo * offsetsTable) :
        m_numChunks(numChunks),
        m_diskOffsetsTable(offsetsTable),
        m_startIndex(numChunks)
    {
        uint64_t numSequences = 0;
        for (decltype(m_numChunks) i = 0; i < m_numChunks; i++)
        {
            m_startIndex[i] = numSequences;
            numSequences += m_diskOffsetsTable[i].numSequences;
        }
    }

    int64_t GetOffset(uint32_t index) 
    { 
        return m_diskOffsetsTable[index].offset; 
    }

    int64_t GetDataStartOffset(uint32_t index)
    {
        auto sequenceLengthPrefix = GetNumSequences(index) * sizeof(uint32_t);
        return GetOffset(index) + sequenceLengthPrefix;
    }

    uint32_t GetNumSequences(uint32_t index) 
    { 
        return m_diskOffsetsTable[index].numSequences;
    }

    uint32_t GetNumSamples(uint32_t index) 
    { 
        return m_diskOffsetsTable[index].numSamples; 
    }

    int64_t GetStartIndex(uint32_t index) 
    {
        return m_startIndex.at(index); 
    }

    uint64_t GetChunkSize(uint32_t index) 
    { 
        auto dataStartOffset = GetDataStartOffset(index);
        auto dataEndOffset = GetOffset(index + 1);
        return dataEndOffset - dataStartOffset;
    }

private:
    uint32_t m_numChunks;
    unique_ptr<BinaryChunkInfo[]> m_diskOffsetsTable;
    vector<uint64_t> m_startIndex;
};

typedef unique_ptr<ChunkTable> ChunkTablePtr;

class CorpusDescriptor;
typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

// TODO: more details when tracing warnings 
class BinaryChunkDeserializer : public DataDeserializerBase {
public:
    explicit BinaryChunkDeserializer(const BinaryConfigHelper& helper);

    BinaryChunkDeserializer(CorpusDescriptorPtr corpus, const BinaryConfigHelper& helper) = delete;

    ~BinaryChunkDeserializer();

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Get information about chunks.
    std::vector<ChunkInfo> ChunkInfos() override;

    // Get information about particular chunk.
    void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override;

private:
    // Builds an index of the input data.
    void Initialize(const std::map<std::wstring, std::wstring>& rename, DataType precision);

    // Reads the chunk table from disk into memory
    void ReadChunkTable(FILE* infile, uint32_t firstChunkIdx, uint32_t numChunks);
    void ReadChunkTable(FILE* infile);

    // Reads a chunk from disk into buffer
    unique_ptr<byte[]> ReadChunk(ChunkIdType chunkId);

    BinaryChunkDeserializer(const wstring& filename);

    void SetTraceLevel(unsigned int traceLevel);

private:
    const wstring m_filename;
    FILE* m_file;

    int64_t m_headerOffset, m_chunkTableOffset;

    std::vector<BinaryDataDeserializerPtr> m_deserializers;
    ChunkTablePtr m_chunkTable;
    void* m_chunkBuffer;

    
    uint32_t m_numChunks;
    uint32_t m_numInputs;
    
    unsigned int m_traceLevel;

    static const uint32_t s_currentVersion = 1;

    friend class CNTKBinaryReaderTestRunner;


    DISABLE_COPY_AND_MOVE(BinaryChunkDeserializer);
};
}
