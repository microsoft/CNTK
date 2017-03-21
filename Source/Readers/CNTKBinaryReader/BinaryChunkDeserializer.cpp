//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "BinaryChunkDeserializer.h"
#include "BinaryDataChunk.h"
#include "FileHelper.h"
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

enum class MatrixEncodingType : unsigned char
{
    dense = 0,
    sparse_csc = 1,
    // TODO: compressed_sparse_csc = 2, // indices are encoded as var-ints
};


void BinaryChunkDeserializer::ReadChunkTable(FILE* infile)
{
    ReadChunkTable(infile, 0, m_numChunks);
}

void BinaryChunkDeserializer::ReadChunkTable(FILE* infile, uint32_t firstChunkIdx, uint32_t numChunks)
{
    if (firstChunkIdx + numChunks > m_numChunks) 
    {
        RuntimeError("Requested chunks (from %" PRIu32 " to %" PRIu32 ") are out of bounds "
            "(the total number of chunks in the dataset is %" PRIu32 ").",
            firstChunkIdx, (firstChunkIdx + numChunks - 1), m_numChunks);
    }

    uint64_t firstChunkOffset = firstChunkIdx * sizeof(ChunkInfo) + m_chunkTableOffset;

    // Seek to the start of the offset info for the first requested chunk 
    CNTKBinaryFileHelper::SeekOrDie(infile, firstChunkOffset, SEEK_SET);

    // Note we create numChunks + 1 since we want to be consistent with determining the size of each chunk.
    ChunkInfo* chunks = new ChunkInfo[numChunks + 1];

    // Read in all of the offsets for the chunks of interest
    CNTKBinaryFileHelper::ReadOrDie(chunks, sizeof(ChunkInfo), numChunks, infile);

    // Now read the final entry. It is either the next offset entry (if we're reading a subset and the
    // entry exists), or we just fill it with the correct information based on file size if it doesn't
    if (firstChunkIdx + numChunks == m_numChunks)
    {
        auto position = CNTKBinaryFileHelper::TellOrDie(infile);
        chunks[numChunks].offset = position;
        chunks[numChunks].numSamples = 0;
        chunks[numChunks].numSequences = 0;
    }
    else
        CNTKBinaryFileHelper::ReadOrDie(chunks + numChunks, sizeof(ChunkInfo), 1, infile);

    m_chunkTable = make_unique<ChunkTable>(numChunks, chunks);

}

BinaryChunkDeserializer::BinaryChunkDeserializer(const BinaryConfigHelper& helper) :
    BinaryChunkDeserializer(helper.GetFilePath())
{
    SetTraceLevel(helper.GetTraceLevel());

    Initialize(helper.GetRename(), helper.GetElementType());
}


BinaryChunkDeserializer::BinaryChunkDeserializer(const std::wstring& filename) : 
    DataDeserializerBase(true),
    m_filename(filename),
    m_file(nullptr),
    m_headerOffset(0),
    m_chunkTableOffset(0),
    m_traceLevel(0)
{
}

BinaryChunkDeserializer::~BinaryChunkDeserializer()
{
    if (m_file)
        CNTKBinaryFileHelper::CloseOrDie(m_file);
}


void BinaryChunkDeserializer::Initialize(const std::map<std::wstring, std::wstring>& rename, ElementType precision)
{
    if (m_file)
        CNTKBinaryFileHelper::CloseOrDie(m_file);

    m_file = CNTKBinaryFileHelper::OpenOrDie(m_filename, L"rb");

    // First, verify the magic number.
    CNTKBinaryFileHelper::FindMagicOrDie(m_file, m_filename);
    
    // Second, read the version number of the data file, and (for now) make sure the reader version is the same.
    uint32_t versionNumber = CNTKBinaryFileHelper::GetVersionNumber(m_file);
    if (versionNumber != s_currentVersion)
        LogicError("The reader version is %" PRIu32 ", but the data file was created for version %" PRIu32 ".",
            s_currentVersion, versionNumber);

    // Now, find where the header is.
    m_headerOffset = CNTKBinaryFileHelper::GetHeaderOffset(m_file);
    CNTKBinaryFileHelper::SeekOrDie(m_file, m_headerOffset, SEEK_SET);
    // Once again, make sure that the header is well-formed and starts with a magic number.
    CNTKBinaryFileHelper::FindMagicOrDie(m_file, m_filename);

    // Next is the number of chunks in the input file.
    CNTKBinaryFileHelper::ReadOrDie(&m_numChunks, sizeof(m_numChunks), 1, m_file);

    // Next is the number of inputs
    CNTKBinaryFileHelper::ReadOrDie(&m_numInputs, sizeof(m_numInputs), 1, m_file);

    // Reserve space for all of the inputs, and then read them in.
    m_streams.resize(m_numInputs);
    m_deserializers.resize(m_numInputs);

    for (decltype(m_numInputs) i = 0; i < m_numInputs; i++)
    {
        MatrixEncodingType type;
        CNTKBinaryFileHelper::ReadOrDie(&type, sizeof(type), 1, m_file);
        if (type == MatrixEncodingType::dense)
            m_deserializers[i] = make_shared<DenseBinaryDataDeserializer>(m_file, precision);
        else if (type == MatrixEncodingType::sparse_csc)
            m_deserializers[i] = make_shared<SparseBinaryDataDeserializer>(m_file, precision);
        else
            RuntimeError("Unknown encoding type %u requested.", (unsigned int)type);

        auto description = m_deserializers[i]->GetStreamDescription();
        description->m_id = i;
        // Check if we should rename this input based on the config
        auto it = rename.find(description->m_name);
        if (it != rename.end()) 
        {
            description->m_name = it->second;
        }

        m_streams[i] = description;
    }

    // We just finished the header. So we're now at the chunk table.
    m_chunkTableOffset = CNTKBinaryFileHelper::TellOrDie(m_file);

    // We only have to read in the offsets table once, so do that now.
    // Note it's possible in distributed reading mode to only want to read
    // a subset of the offsets table.
    ReadChunkTable(m_file);
}

ChunkDescriptions BinaryChunkDeserializer::GetChunkDescriptions()
{
    assert(m_chunkTable);

    ChunkDescriptions result;
    result.reserve(m_numChunks);

    for (ChunkIdType i = 0; i < m_numChunks; i++ ) 
    {
        result.push_back(shared_ptr<ChunkDescription>(
            new ChunkDescription{ i, m_chunkTable->GetNumSamples(i), m_chunkTable->GetNumSequences(i) }));
    }

    return result;
}

void BinaryChunkDeserializer::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
{
    // Reserve space for each sequence
    result.reserve(m_chunkTable->GetNumSequences(chunkId));

    auto offset = m_chunkTable->GetOffset(chunkId);
    auto numberOfSequences = m_chunkTable->GetNumSequences(chunkId);
    unique_ptr<uint32_t[]> numSamplesPerSequence(new uint32_t[numberOfSequences]);

    // Seek to the start of the chunk
    CNTKBinaryFileHelper::SeekOrDie(m_file, offset, SEEK_SET);
    // read 'numberOfSequences' unsigned ints
    CNTKBinaryFileHelper::ReadOrDie(numSamplesPerSequence.get(), sizeof(uint32_t), numberOfSequences, m_file);

    auto startId = m_chunkTable->GetStartIndex(chunkId);
    for (decltype(numberOfSequences) i = 0; i < numberOfSequences; i++)
    {
        SequenceDescription sd = {};
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
    CNTKBinaryFileHelper::SeekOrDie(m_file, m_chunkTable->GetDataStartOffset(chunkId), SEEK_SET);

    // Determine how big the chunk is.
    size_t chunkSize = m_chunkTable->GetChunkSize(chunkId);
    
    // Create buffer
    // TODO: use a pool of buffers instead of allocating a new one, each time a chunk is read.
    unique_ptr<byte[]> buffer(new byte[chunkSize]);

    // Read the chunk from disk
    CNTKBinaryFileHelper::ReadOrDie(buffer.get(), sizeof(byte), chunkSize, m_file);

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

}}}
