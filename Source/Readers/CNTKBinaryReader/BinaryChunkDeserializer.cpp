//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "BinaryChunkDeserializer.h"
#include "BinaryDataChunk.h"
#include "CBFUtils.h"
#include "FileWrapper.h"
#include <vector>

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

enum class MatrixEncodingType : unsigned char
{
    dense = 0,
    sparse_csc = 1,
    // TODO: compressed_sparse_csc = 2, // indices are encoded as var-ints
};


void BinaryChunkDeserializer::ReadChunkTable()
{
    uint64_t firstChunkOffset = m_chunkTableOffset;

    // Seek to the start of the offset info
    m_file.SeekOrDie(firstChunkOffset, SEEK_SET);

    // Note we create m_numChunks + 1 since we want to be consistent with determining the size of each chunk.
    BinaryChunkInfo* chunks = new BinaryChunkInfo[m_numChunks + 1];

    // Read in all of the offsets for the chunks
    m_file.ReadOrDie(chunks, sizeof(BinaryChunkInfo), m_numChunks);

    // We fill the final entry with the current position of the file
    auto position = m_file.TellOrDie();
    chunks[m_numChunks].offset = position;
    chunks[m_numChunks].numSamples = 0;
    chunks[m_numChunks].numSequences = 0;

    m_chunkTable = make_unique<ChunkTable>(m_numChunks, chunks);
}

BinaryChunkDeserializer::BinaryChunkDeserializer(const BinaryConfigHelper& helper) :
    BinaryChunkDeserializer(helper.GetFilePath())
{
    SetTraceLevel(helper.GetTraceLevel());

    Initialize(helper.GetRename(), helper.GetElementType());
}


BinaryChunkDeserializer::BinaryChunkDeserializer(const std::wstring& filename) :
    DataDeserializerBase(true),
    m_file(FileWrapper::OpenOrDie(filename, L"rb")),
    m_headerOffset(0),
    m_chunkTableOffset(0),
    m_traceLevel(0)
{
}

void BinaryChunkDeserializer::Initialize(const std::map<std::wstring, std::wstring>& rename, DataType precision)
{
    m_file.CheckIsOpenOrDie();

    // First, verify the magic number.
    CBFUtils::FindMagicOrDie(m_file);
    
    // Second, read the version number of the data file, and (for now) make sure the reader version is the same.
    uint32_t versionNumber = CBFUtils::GetVersionNumber(m_file);
    if (versionNumber != s_currentVersion)
        LogicError("The reader version is %" PRIu32 ", but the data file was created for version %" PRIu32 ".",
            s_currentVersion, versionNumber);

    // Now, find where the header is.
    m_headerOffset = CBFUtils::GetHeaderOffset(m_file);
    m_file.SeekOrDie(m_headerOffset, SEEK_SET);

    // Once again, make sure that the header is well-formed and starts with a magic number.
    CBFUtils::FindMagicOrDie(m_file);

    // Next is the number of chunks in the input file.
    m_file.ReadOrDie(m_numChunks);

    // Next is the number of inputs
    m_file.ReadOrDie(m_numInputs);

    // Reserve space for all of the inputs, and then read them in.
    m_streams.resize(m_numInputs);
    m_deserializers.resize(m_numInputs);

    for (decltype(m_numInputs) i = 0; i < m_numInputs; i++)
    {
        MatrixEncodingType type;
        m_file.ReadOrDie(type);
        if (type == MatrixEncodingType::dense)
            m_deserializers[i] = make_shared<DenseBinaryDataDeserializer>(m_file, precision);
        else if (type == MatrixEncodingType::sparse_csc)
            m_deserializers[i] = make_shared<SparseBinaryDataDeserializer>(m_file, precision);
        else
            RuntimeError("Unknown encoding type %u requested.", (unsigned int)type);

        auto description = m_deserializers[i]->GetStreamDescription();
        description.m_id = i;
        // Check if we should rename this input based on the config
        auto it = rename.find(description.m_name);
        if (it != rename.end()) 
        {
            description.m_name = it->second;
        }

        m_streams[i] = description;
    }

    // We just finished the header. So we're now at the chunk table.
    m_chunkTableOffset = m_file.TellOrDie();

    // We only have to read in the offsets table once, so do that now.
    // Note it's possible in distributed reading mode to only want to read
    // a subset of the offsets table.
    ReadChunkTable();
}

std::vector<ChunkInfo> BinaryChunkDeserializer::ChunkInfos()
{
    assert(m_chunkTable);

    std::vector<ChunkInfo> result;
    result.reserve(m_numChunks);

    for (ChunkIdType i = 0; i < m_numChunks; i++ ) 
    {
        result.push_back(ChunkInfo{ i, m_chunkTable->GetNumSamples(i), m_chunkTable->GetNumSequences(i) });
    }

    return result;
}

void BinaryChunkDeserializer::SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result)
{
    // Reserve space for each sequence
    result.reserve(m_chunkTable->GetNumSequences(chunkId));

    auto offset = m_chunkTable->GetOffset(chunkId);
    auto numberOfSequences = m_chunkTable->GetNumSequences(chunkId);
    unique_ptr<uint32_t[]> numSamplesPerSequence(new uint32_t[numberOfSequences]);

    // Seek to the start of the chunk
    m_file.SeekOrDie(offset, SEEK_SET);
    // read 'numberOfSequences' unsigned ints
    m_file.ReadOrDie(numSamplesPerSequence.get(), sizeof(uint32_t), numberOfSequences);

    auto startId = m_chunkTable->GetStartIndex(chunkId);
    for (decltype(numberOfSequences) i = 0; i < numberOfSequences; i++)
    {
        SequenceInfo sd = {};
        sd.m_indexInChunk = i;
        sd.m_numberOfSamples = numSamplesPerSequence[i];
        sd.m_chunkId = chunkId;
        sd.m_key.m_sequence = startId + i;
        sd.m_key.m_sample = 0;

        result.push_back(sd);
    }
}

unique_ptr<byte[]> BinaryChunkDeserializer::ReadChunk(ChunkIdType chunkId)
{
    // Seek to the start of the data portion in the chunk
    m_file.SeekOrDie(m_chunkTable->GetDataStartOffset(chunkId), SEEK_SET);

    // Determine how big the chunk is.
    size_t chunkSize = m_chunkTable->GetChunkSize(chunkId);
    
    // Create buffer
    // TODO: use a pool of buffers instead of allocating a new one, each time a chunk is read.
    unique_ptr<byte[]> buffer(new byte[chunkSize]);

    // Read the chunk from disk
    m_file.ReadOrDie(buffer.get(), sizeof(byte), chunkSize);

    return buffer;
}


ChunkPtr BinaryChunkDeserializer::GetChunk(ChunkIdType chunkId)
{
    // Read the chunk into memory
    unique_ptr<byte[]> buffer = ReadChunk(chunkId);

    return make_shared<BinaryDataChunk>(chunkId, m_chunkTable->GetNumSequences(chunkId), std::move(buffer), m_deserializers);
}

void BinaryChunkDeserializer::SetTraceLevel(unsigned int traceLevel)
{
    m_traceLevel = traceLevel;
}

}
