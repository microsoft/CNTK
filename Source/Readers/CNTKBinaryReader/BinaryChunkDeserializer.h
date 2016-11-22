//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "BinaryConfigHelper.h"
#include "CorpusDescriptor.h"
#include "BinaryDataChunk.h"
#include "BinaryDataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Offsets table used to find the chunks in the binary file. Added some helper methods around the core data.
#pragma pack(push, 1)
struct DiskOffsetsTable 
{
    int64_t offset;
    int32_t numSequences;
    int32_t numSamples;
};
#pragma pack(pop)

    // Offsets table used to find the chunks in the binary file. Added some helper methods around the core data.
class OffsetsTable {
public:

    OffsetsTable(size_t numChunks, DiskOffsetsTable* offsetsTable) : m_numChunks(numChunks), m_diskOffsetsTable(offsetsTable)
    {
        Initialize();
    }

    int64_t GetOffset(size_t index) { return m_diskOffsetsTable[index].offset; }
    int32_t GetNumSequences(size_t index) { return m_diskOffsetsTable[index].numSequences; }
    int32_t GetNumSamples(size_t index) { return m_diskOffsetsTable[index].numSamples; }
    int64_t GetStartIndex(size_t index) { return m_startIndex[index]; }
    size_t GetChunkSize(size_t index) { return m_diskOffsetsTable[index + 1].offset - m_diskOffsetsTable[index].offset; }

private:
    void Initialize()
    {
        m_startIndex.resize(m_numChunks);
        m_startIndex[0] = 0;
        for (int64_t c = 1; c < m_numChunks; c++)
            m_startIndex[c] = m_startIndex[c-1] + m_diskOffsetsTable[c].numSequences;
    }

private:
    int64_t m_numChunks;
    DiskOffsetsTable* m_diskOffsetsTable;
    vector<size_t> m_startIndex;
};

typedef unique_ptr<OffsetsTable> OffsetsTablePtr;

// TODO: more details when tracing warnings 
class BinaryChunkDeserializer : public DataDeserializerBase {
public:
    explicit BinaryChunkDeserializer(const BinaryConfigHelper& helper);

    BinaryChunkDeserializer(CorpusDescriptorPtr corpus, const BinaryConfigHelper& helper);

    ~BinaryChunkDeserializer();

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Get information about chunks.
    ChunkDescriptions GetChunkDescriptions() override;

    // Get information about particular chunk.
    void GetSequencesForChunk(ChunkIdType chunkId, vector<SequenceDescription>& result) override;

    // Parses buffer into a BinaryChunkPtr
    void ParseChunk(ChunkIdType chunkId, unique_ptr<byte[]> const& buffer, std::vector<std::vector<SequenceDataPtr>>& data);

private:
    // Builds an index of the input data.
    void Initialize(const std::map<std::wstring, std::wstring>& rename);

    // Reads the offsets table from disk into memory
    void ReadOffsetsTable(FILE* infile, size_t startOffset, size_t numChunks);
    void ReadOffsetsTable(FILE* infile);

    // Reads a chunk from disk into buffer
    unique_ptr<byte[]> ReadChunk(ChunkIdType chunkId);

    BinaryChunkDeserializer(const wstring& filename);

    void SetTraceLevel(unsigned int traceLevel);

private:
    const wstring m_filename;
    FILE* m_file;

    int64_t m_offsetStart;
    int64_t m_dataStart;


    std::vector<BinaryDataDeserializerPtr> m_deserializers;
    OffsetsTablePtr m_offsetsTable;
    void* m_chunkBuffer;

    int64_t m_versionNumber = 1;
    int64_t m_numChunks;
    int32_t m_numInputs;
    
    unsigned int m_traceLevel;

    friend class CNTKBinaryReaderTestRunner;

    DISABLE_COPY_AND_MOVE(BinaryChunkDeserializer);
};
}}}
