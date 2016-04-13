//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <inttypes.h>
#include "Indexer.h"
#include "TextReaderConstants.h"

using std::string;

namespace Microsoft { namespace MSR { namespace CNTK {

Indexer::Indexer(FILE* file, bool skipSequenceIds, size_t chunkSize) :
    m_file(file),
    m_fileOffsetStart(0),
    m_fileOffsetEnd(0),
    m_buffer(new char[BUFFER_SIZE + 1]),
    m_bufferStart(nullptr),
    m_bufferEnd(nullptr),
    m_pos(nullptr),
    m_done(false),
    m_hasSequenceIds(!skipSequenceIds),
    m_maxChunkSize(chunkSize)
{
    if (m_file == nullptr)
    {
        RuntimeError("Input file not open for reading");
    }
}

void Indexer::RefillBuffer()
{
    if (!m_done)
    {
        size_t bytesRead = fread(m_buffer.get(), 1, BUFFER_SIZE, m_file);
        if (bytesRead == (size_t)-1)
            RuntimeError("Could not read from the input file.");
        if (bytesRead == 0)
        {
            m_done = true;
        }
        else
        {
            m_fileOffsetStart = m_fileOffsetEnd;
            m_fileOffsetEnd += bytesRead;
            m_bufferStart = m_buffer.get();
            m_pos = m_bufferStart;
            m_bufferEnd = m_bufferStart + bytesRead;
        }
    }
}

void Indexer::AddSequence(SequenceDescriptor& sd)
{
    assert(!m_chunks.empty());
    ChunkDescriptor* chunk = &m_chunks.back();
    if (chunk->m_byteSize > 0 && (chunk->m_byteSize + sd.m_byteSize) > m_maxChunkSize)
    {
        m_chunks.push_back({});
        chunk = &m_chunks.back();
        chunk->m_id = m_chunks.size() - 1;
    }
    chunk->m_byteSize += sd.m_byteSize;
    chunk->m_numberOfSequences++;
    chunk->m_numberOfSamples += sd.m_numberOfSamples;
    sd.m_chunkId = chunk->m_id;
    chunk->m_sequences.push_back(sd);
}

void Indexer::BuildFromLines()
{
    assert(m_pos == m_bufferStart);
    m_hasSequenceIds = false;
    size_t lines = 0;
    int64_t offset = GetFileOffset();
    while (!m_done)
    {
        m_pos = (char*)memchr(m_pos, ROW_DELIMITER, m_bufferEnd - m_pos);
        if (m_pos)
        {
            SequenceDescriptor sd = {};
            sd.m_id = lines;
            sd.m_numberOfSamples = 1;
            sd.m_isValid = true;
            sd.m_fileOffsetBytes = offset;
            offset = GetFileOffset() + 1;
            sd.m_byteSize = offset - sd.m_fileOffsetBytes;
            // TODO: ignore empty lines.
            AddSequence(sd);
            ++m_pos;
            ++lines;
        }
        else
        {
            RefillBuffer();
        }
    }
}

void Indexer::Build()
{
    if (!m_chunks.empty())
    {
        return;
    }

    if (m_maxChunkSize > 0)
    {
        auto fileSize = filesize(m_file);
        m_chunks.reserve((fileSize + m_maxChunkSize - 1) / m_maxChunkSize);
    }

    m_chunks.push_back({});

    RefillBuffer(); // read the first block of data
    if (m_done)
    {
        RuntimeError("Input file is empty");
    }

    if ((m_bufferEnd - m_bufferStart > 3) &&
        (m_bufferStart[0] == '\xEF' && m_bufferStart[1] == '\xBB' && m_bufferStart[2] == '\xBF'))
    {
        // input file contains UTF-8 BOM value, skip it.
        m_pos += 3;
        m_fileOffsetStart += 3;
        m_bufferStart += 3;
    }

    // check the first byte and decide what to do next
    if (!m_hasSequenceIds || m_bufferStart[0] == NAME_PREFIX)
    {
        // skip sequence id parsing, treat lines as individual sequences
        BuildFromLines();
        return;
    }

    size_t id = 0;
    int64_t offset = GetFileOffset();
    // read the very first sequence id
    if (!GetNextSequenceId(id))
    {
        RuntimeError("Expected a sequence id at the offset %" PRIi64 ", none was found.", offset);
    }

    SequenceDescriptor sd = {};
    sd.m_id = id;
    sd.m_fileOffsetBytes = offset;
    sd.m_isValid = true;

    while (!m_done)
    {
        SkipLine(); // ignore whatever is left on this line.
        offset = GetFileOffset(); // a new line starts at this offset;
        sd.m_numberOfSamples++;

        if (!m_done && GetNextSequenceId(id) && id != sd.m_id)
        {
            // found a new sequence, which starts at the [offset] bytes into the file
            sd.m_byteSize = offset - sd.m_fileOffsetBytes;
            AddSequence(sd);
            sd = {};
            sd.m_id = id;
            sd.m_fileOffsetBytes = offset;
            sd.m_isValid = true;
        }
    }

    // calculate the byte size for the last sequence
    sd.m_byteSize = m_fileOffsetEnd - sd.m_fileOffsetBytes;
    AddSequence(sd);
}


void Indexer::SkipLine()
{
    while (!m_done)
    {
        m_pos = (char*)memchr(m_pos, ROW_DELIMITER, m_bufferEnd - m_pos);
        if (m_pos)
        {
            //found a new-line character
            if (++m_pos == m_bufferEnd)
            {
                RefillBuffer();
            }
            return;
        }
        RefillBuffer();
    }
}

bool Indexer::GetNextSequenceId(size_t& id)
{
    bool found = false;
    id = 0;
    while (!m_done)
    {
        while (m_pos != m_bufferEnd)
        {
            char c = *m_pos;
            // a well-formed sequence id must end in either a column delimiter 
            // or a name prefix
            if (c == COLUMN_DELIMITER || c == NAME_PREFIX)
            {
                return found;
            }

            if (!isdigit(c))
            {
                // TODO: ignore malformed sequences
                RuntimeError("Unexpected character('%c')"
                    " while reading a sequence id"
                    " at the offset = %" PRIi64 "\n", c, GetFileOffset());
            }

            found |= true;
            size_t temp = id;
            id = id * 10 + (c - '0');
            if (temp > id)
            {
                // TODO: ignore malformed sequences
                RuntimeError("Size_t overflow while reading a sequence id"
                    " at the offset = %" PRIi64 "\n", GetFileOffset());
            }
            ++m_pos;
        }
        RefillBuffer();
    }

    // TODO: ignore malformed sequences
    // reached EOF without hitting the pipe character.
    RuntimeError("Reached the end of file "
        " while reading a sequence id"
        " at the offset = %" PRIi64 "\n", GetFileOffset());
}

}}}
