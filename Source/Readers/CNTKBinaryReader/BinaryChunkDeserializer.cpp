//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BinaryChunkDeserializer.h"
#include "BinaryDataChunk.h"
#include "FileHelper.h"
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

void BinaryChunkDeserializer::ReadOffsetsTable(FILE* infile)
{
    ReadOffsetsTable(infile, 0, m_numBatches);
}

void BinaryChunkDeserializer::ReadOffsetsTable(FILE* infile, size_t startOffset, size_t numBatches)
{
    assert((int64_t)(startOffset + numBatches) <= m_numBatches);

    size_t offsetRowSize = OffsetsTable::GetOffsetRowSize();
    size_t startPos = startOffset * offsetRowSize + m_offsetStart;

    // Seek to the offsets table start
    CNTKBinaryFileHelper::seekOrDie(infile, startPos, SEEK_SET);

    // Note we create numBatches + 1 since we want to be consistent with determining the size of each chunk.
    byte* offsetsTable = new byte[(numBatches + 1)*offsetRowSize];

    CNTKBinaryFileHelper::readOrDie(offsetsTable, offsetRowSize, numBatches, infile);

    // Now read the final entry. It is either the next offset entry (if we're reading a subset and the
    // entry exists), or we just fill it with the correct information based on file size
    if ((int64_t)(startOffset + numBatches) != m_numBatches)
        CNTKBinaryFileHelper::readOrDie(offsetsTable + numBatches, offsetRowSize, 1, infile);

    m_offsetsTable = make_unique<OffsetsTable>(numBatches, offsetsTable);

    if ((int64_t)(startOffset + numBatches) == m_numBatches)
    {
        CNTKBinaryFileHelper::seekOrDie(infile, 0, SEEK_END);
        size_t maxOffset = CNTKBinaryFileHelper::tellOrDie(infile) - m_dataStart;
        m_offsetsTable->SetOffset(numBatches, maxOffset);
    }
    /*
        CNTKBinaryFileHelper::seekOrDie(infile, 0, SEEK_END);
        offsetsTable[numBatches].offset = CNTKBinaryFileHelper::tellOrDie(infile) - m_dataStart;
        offsetsTable[numBatches].numSamples = 0;
        offsetsTable[numBatches].numSequences = 0;
    }
    else
    */

}

BinaryChunkDeserializer::BinaryChunkDeserializer(const BinaryConfigHelper& helper) :
    BinaryChunkDeserializer(helper.GetFilePath())
{
    SetTraceLevel(helper.GetTraceLevel());

    // Rename/alias not yet implemented
    Initialize(helper.GetRename());
}


BinaryChunkDeserializer::BinaryChunkDeserializer(const std::wstring& filename) : 
    m_filename(filename),
    m_file(nullptr),
    m_offsetStart(0),
    m_dataStart(0),
    m_traceLevel(0)
{
    // streams will be used for rename when it's implemented.

}

BinaryChunkDeserializer::~BinaryChunkDeserializer()
{
    if (m_file)
        CNTKBinaryFileHelper::closeOrDie(m_file);
}

void BinaryChunkDeserializer::Initialize(const std::map<std::wstring, std::wstring>& rename)
{
    if (m_file)
        CNTKBinaryFileHelper::closeOrDie(m_file);

    m_file = CNTKBinaryFileHelper::openOrDie(m_filename, L"rb");

    // Parse the header
    CNTKBinaryFileHelper::seekOrDie(m_file, 0, SEEK_SET);

    // First read the version number of the data file, and make sure the reader version is the same.
    int64_t versionNumber;
    CNTKBinaryFileHelper::readOrDie(&versionNumber, sizeof(versionNumber), 1, m_file);
    if (versionNumber != m_versionNumber)
        LogicError("The reader version is %d, but the data file was created for version %d.", (int)m_versionNumber, (int)versionNumber);

    // Next is the number of batches in the input file.
    CNTKBinaryFileHelper::readOrDie(&m_numBatches, sizeof(m_numBatches), 1, m_file);

    // Next is the number of inputs
    CNTKBinaryFileHelper::readOrDie(&m_numInputs, sizeof(m_numInputs), 1, m_file);

    // Reserve space for all of the inputs, and then read them in.
    m_streams.resize(m_numInputs);
    m_deserializers.resize(m_numInputs);
    m_sequenceNum.resize(m_numInputs);
    size_t numSequences = 0;

    int32_t len;
    int32_t maxLen = 100;
    char* tempName = new char[maxLen];
    for (int32_t c = 0; c < m_numInputs; c++)
    {
        // Create our streamDescription for this input
        auto streamDescription = std::make_shared<StreamDescription>();

        // read the name
        CNTKBinaryFileHelper::readOrDie(&len, sizeof(len), 1, m_file);
        if (len + 1 > maxLen)
        {
            maxLen = len + 1;
            delete[] tempName;
            tempName = new char[maxLen];
        }
        CNTKBinaryFileHelper::readOrDie(tempName, sizeof(char), len, m_file);
        tempName[len] = '\0';
        wstring wname = msra::strfun::utf16(tempName);
        if (rename.find(wname) == rename.end())
            streamDescription->m_name = wname;
        else
            streamDescription->m_name = rename.at(wname);

        // Read the matrix type. Then instantiate the appropriate BinaryDataDeserializer, and have it read in its parameters
        // Note: Is there a better way to do this?
        int32_t matType;
        CNTKBinaryFileHelper::readOrDie(&matType, sizeof(matType), 1, m_file);
        if (matType == 0)
            m_deserializers[c] = make_shared<DenseBinaryDataDeserializer>(m_file);
        else if (matType == 1)
            m_deserializers[c] = make_shared<SparseBinaryDataDeserializer>(m_file);
        else
            RuntimeError("Unknown matrix type %d requested.", matType);

        streamDescription->m_id           = c;
        streamDescription->m_elementType  = m_deserializers[c]->GetElementType();
        streamDescription->m_storageType  = m_deserializers[c]->GetStorageType();
        streamDescription->m_sampleLayout = m_deserializers[c]->GetSampleLayout();
        m_streams[c]                      = streamDescription;

        if (m_deserializers[c]->IsSequence())
        {
            m_sequenceNum[c] = (int32_t)numSequences;
            numSequences++;
        }
        else
            m_sequenceNum[c] = -1;
    }
    delete[] tempName;

    // We just finished the header. So we're now at the offsets table.
    m_offsetStart = CNTKBinaryFileHelper::tellOrDie(m_file);

    // After the header is the data start. Compute that now.
    m_dataStart = m_offsetStart + m_numBatches * OffsetsTable::GetOffsetRowSize();

    // We only have to read in the offsets table once, so do that now.
    // Note it's possible in distributed reading mode to only want to read
    // a subset of the offsets table.
    ReadOffsetsTable(m_file);
}

ChunkDescriptions BinaryChunkDeserializer::GetChunkDescriptions()
{
    assert(m_offsetsTable);
    ChunkDescriptions result;
    result.reserve(m_numBatches);

    if (m_numBatches > CHUNKID_MAX)
        RuntimeError("Currently CNTK does not support %d batches. The maximum number of batches allowed is %d.", (int)m_numBatches, (int)CHUNKID_MAX);

    for (ChunkIdType c = 0; c < (ChunkIdType)m_numBatches; c++ ) 
    {
        result.push_back(shared_ptr<ChunkDescription>(
            new ChunkDescription {
                c,
                (size_t)m_offsetsTable->GetNumSamples(c),
                (size_t)m_offsetsTable->GetNumSequences(c)
        }));
    }

    return result;
}

void BinaryChunkDeserializer::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
{
    // Reserve space for each sequence
    result.reserve(m_offsetsTable->GetNumSequences(chunkId));

    // We don't store every piece of sequence information, so we have to read the chunk in, parse it, and then
    // find the information.
    ChunkPtr chunk = GetChunk(chunkId);

    size_t startId = m_offsetsTable->GetStartIndex(chunkId);
    std::vector<SequenceDataPtr> temp;
    for (size_t c = 0; c < m_offsetsTable->GetNumSequences(chunkId); c++)
    {
        // BUGBUG: This is inefficient, but we don't have a choice. Why do we need this at all? Why can't
        // this information just be gotten from the chunks? It's not clear.
        // Note numSamples is 1 if there are no sequences.
        uint32_t numSamples = 1;
        temp.clear();
        chunk->GetSequence(m_offsetsTable->GetStartIndex(chunkId) + c, temp);
        // Only take the max over streams that are actually in use.
        for (size_t i = 0; i < temp.size(); i++)
            if (m_sequenceNum[i] >= 0)
                numSamples = max(numSamples, temp[i]->m_numberOfSamples);

        SequenceDescription sd = {};
        sd.m_id = startId + c;
        sd.m_numberOfSamples = numSamples;
        sd.m_chunkId = chunkId;
        sd.m_key.m_sequence = startId + c;
        sd.m_key.m_sample = 0;

        result.push_back(sd);
    }
}

unique_ptr<byte[]> BinaryChunkDeserializer::ReadChunk(ChunkIdType chunkId)
{
    // Seek to the start of the chunk
    CNTKBinaryFileHelper::seekOrDie(m_file, m_dataStart + m_offsetsTable->GetOffset(chunkId), SEEK_SET);

    // Determine how big the chunk is.
    size_t chunkSize = m_offsetsTable->GetChunkSize(chunkId);
    
    // Create buffer
    unique_ptr<byte[]> buffer(new byte[chunkSize]);

    // Read the chunk from disk
    CNTKBinaryFileHelper::readOrDie(buffer.get(), sizeof(byte), chunkSize, m_file);

    return buffer;
}


ChunkPtr BinaryChunkDeserializer::GetChunk(ChunkIdType chunkId)
{
    // Read the chunk into memory
    unique_ptr<byte[]> chunkBuffer = ReadChunk(chunkId);

    return make_shared<BinaryDataChunk>(chunkId, m_offsetsTable->GetStartIndex(chunkId), m_offsetsTable->GetNumSequences(chunkId), std::move(chunkBuffer), m_deserializers);
}

void BinaryChunkDeserializer::SetTraceLevel(unsigned int traceLevel)
{
    m_traceLevel = traceLevel;
}

}}}
