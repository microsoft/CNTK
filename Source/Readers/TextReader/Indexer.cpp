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

    //TODO: use fdadvise(fd, 0, 0, FADVISE_SEQUENTIAL) if possible
    // also, take at look at http://git.savannah.gnu.org/cgit/coreutils.git/tree/src/wc.c


    Indexer::Indexer(FILE* file, bool skipSequenceIds, int64_t chunkSize) : m_maxChunkSize(chunkSize) {
        if (!file) {
            RuntimeError("Input file not open for reading");
        }
        m_file = file;
        m_bufferStart = new char[BUFFER_SIZE + 1];
        m_fileOffsetStart = 0;
        m_fileOffsetEnd = 0;
        m_done = false;
        m_skipSequenceIds = skipSequenceIds;
        m_chunks.push_back({});
    }

    Indexer::~Indexer() {
        delete[] m_bufferStart;
    }

    void Indexer::Fill() {
        if (!m_done) {
            size_t bytesRead = fread(m_bufferStart, 1, BUFFER_SIZE, m_file);
            if (bytesRead == (size_t)-1)
                RuntimeError("Could not read from input file.");
            if (!bytesRead) {
                m_done = true;
            }
            else {
                m_fileOffsetStart = m_fileOffsetEnd;
                m_fileOffsetEnd += bytesRead;
                m_pos = m_bufferStart;
                m_bufferEnd = m_bufferStart + bytesRead;
            }
        }
    }

    void Indexer::UpdateTimeline(SequenceDescriptor& sd) {
        ChunkDescriptor* chunk = &m_chunks.back();
        TimelineOffset timelineOffset = m_timeline.size();
        if (chunk->m_byteSize > 0 && (chunk->m_byteSize + sd.m_byteSize) > m_maxChunkSize) {
            m_chunks.push_back({});
            chunk = &m_chunks.back();
            chunk->m_index = m_chunks.size() - 1;
            chunk->m_timelineOffset = timelineOffset;
        }
        chunk->m_byteSize += sd.m_byteSize;
        chunk->m_numSequences++;
        sd.m_chunkId = chunk->m_index;

        if (sd.m_id != timelineOffset) {
            m_idToOffsetMap[sd.m_id] = timelineOffset;
        }
        m_timeline.push_back(sd);
    }

    Index* Indexer::BuildFromLines() {
        m_skipSequenceIds = true;
        size_t lines = 0;
        int64_t offset = m_fileOffsetStart;
        while (!m_done)
        {
            m_pos = (char*)memchr(m_pos, ROW_DELIMETER, m_bufferEnd - m_pos);
            if (m_pos) {
                SequenceDescriptor sd = {};
                sd.m_id = lines;
                sd.m_numberOfSamples = 1;
                sd.m_isValid = true;
                sd.m_fileOffset = offset;
                sd.m_byteSize = (m_fileOffsetEnd - (m_bufferEnd - m_pos)) - offset + 1;
                offset += sd.m_byteSize;
                UpdateTimeline(sd);
                ++m_pos;
                ++lines;
            }
            else {
                Fill();
            }
        }

        return new Index
        { 
            m_skipSequenceIds,
            std::move(m_timeline), 
            std::move(m_chunks), 
            std::move(m_idToOffsetMap) 
        };
    }


    Index* Indexer::Build() {
        Fill(); // read the first block of data
        if (m_done) {
            RuntimeError("Input file is empty");
        }
        // check the first byte and decide what to do next
        // TODO: ignore BOM

        if (m_skipSequenceIds || m_bufferStart[0] == NAME_PREFIX) {
            // skip sequence id parsing, treat lines as individual sequences
            return BuildFromLines();
        }

        size_t id = 0;
        int64_t offset = m_fileOffsetStart;
        // read the very first sequence id
        if (!GetNextSequenceId(id)) {
            RuntimeError("Expected a sequence id at the offset %" PRIi64 ", none was found.",
                offset);
        }

        SequenceDescriptor sd = SequenceDescriptor();
        sd.m_id = id;
        sd.m_fileOffset = offset;
        sd.m_isValid = true;

        while (!m_done) {
            SkipLine(); // ignore whatever is left on this line.
            offset = m_fileOffsetStart + (m_pos - m_bufferStart); // a new line starts at this offset;
            sd.m_numberOfSamples++; 

            if (!m_done && GetNextSequenceId(id) && id != sd.m_id) {
                // found a new sequence, which starts at the [offset] bytes into the file
                sd.m_byteSize = offset - sd.m_fileOffset;
                UpdateTimeline(sd);
                sd = SequenceDescriptor();
                sd.m_id = id;
                sd.m_fileOffset = offset;
                sd.m_isValid = true;
            }
        }

        // calculate the byte size for the last sequence
        sd.m_byteSize = m_fileOffsetEnd - sd.m_fileOffset;
        UpdateTimeline(sd);
        return new Index
        {
            m_skipSequenceIds,
            std::move(m_timeline), // TODO: shrink_to_fit
            std::move(m_chunks),
            std::move(m_idToOffsetMap) // this map can be relatively large
        };
    }


    void Indexer::SkipLine() {
        while (!m_done)
        {
            m_pos = (char*)memchr(m_pos, ROW_DELIMETER, m_bufferEnd - m_pos);
            if (m_pos) {
                //found a new-line character
                if (++m_pos == m_bufferEnd) {
                    Fill();
                }
                return;
            }
            Fill();
        }
    }

    bool Indexer::GetNextSequenceId(size_t& id) {
        bool found = false;
        id = 0;
        while (!m_done)
        {
            while (m_pos != m_bufferEnd) 
            {
                char c = *m_pos;
                // a well-formed sequence id must end in either a column delimeter 
                // or a name prefix
                if (c == COLUMN_DELIMETER || c == NAME_PREFIX) 
                {
                    return found;
                }

                if (c < '0' || c > '9') 
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
            Fill();
        }

        // TODO: ignore malformed sequences
        // reached EOF without hitting the pipe character.
        RuntimeError("Reached the end of file "
            " while reading a sequence id"
            " at the offset = %" PRIi64 "\n", GetFileOffset());
    }

}}}