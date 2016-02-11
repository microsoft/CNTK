//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Indexer.h"

using std::string;

namespace Microsoft { namespace MSR { namespace CNTK {

    //TODO: throw proper exceptions

    //TODO: use fdadvise(fd, 0, 0, FADVISE_SEQUENTIAL) if possible
    // also, take at look at http://git.savannah.gnu.org/cgit/coreutils.git/tree/src/wc.c


    Indexer::Indexer(FILE* file, size_t chunkSize) : m_maxChunkSize(chunkSize) {
        if (!file) {
            throw "file not opened for reading";
        }
        m_file = file;
        m_bufferStart = new char[BUFFER_SIZE + 1];
        m_fileOffsetStart = 0;
        m_fileOffsetEnd = 0;
        m_done = false;
        m_chunks.push_back({});
    }

    Indexer::~Indexer() {
        delete[] m_bufferStart;
        //std::cout << "Filesize = " << m_fileOffsetEnd << std::endl;
    }

    void Indexer::Fill() {
        if (!m_done) {
            size_t bytesRead = fread(m_bufferStart, 1, BUFFER_SIZE, m_file);
            if (bytesRead == (size_t)-1)
                throw "read failed";
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
        if (chunk->m_byteSize > 0 && (chunk->m_byteSize + sd.m_byteSize) > m_maxChunkSize) {
            m_chunks.push_back({});
            chunk = &m_chunks.back();
            chunk->m_index = m_chunks.size();
            chunk->m_timelineOffset = m_timeline.size();
        }
        chunk->m_byteSize += sd.m_byteSize;
        chunk->m_numSequences++;
        sd.m_chunkId = chunk->m_index;

        m_timeline.push_back(sd);
    }

    Index* Indexer::BuildFromLines() {
        size_t lines = 0;
        int64_t offset = m_fileOffsetStart;
        while (!m_done)
        {
            while ((m_pos = (char*)memchr(m_pos, '\n', m_bufferEnd - m_pos))) {
                SequenceDescriptor sd = {};
                sd.m_id = lines;
                sd.m_numberOfSamples = 1;
                sd.m_fileOffset = offset;
                sd.m_byteSize = (m_fileOffsetEnd - (m_bufferEnd - m_pos)) - offset + 1;
                offset += sd.m_byteSize;
                UpdateTimeline(sd);
                ++m_pos;
                ++lines;
            }
            Fill();
        }

        return new Index{ true, m_timeline, m_chunks };
    }


    Index* Indexer::Build() {
        Fill(); // read the first block of data
        if (m_done) {
            throw "malformed input";
        }
        // check the first byte and decide what to do next
        // TODO: ignore BOM

        if (m_bufferStart[0] == '|') {
            // skip sequence id parsing, treat lines as individual sequences
            return BuildFromLines();
        }

        size_t id = 0;
        int64_t offset = m_fileOffsetStart;
        // read the very first sequence id
        if (!GetNextSequenceId(id)) {
            throw "malformed input";
        }

        SequenceDescriptor sd = {};
        sd.m_id = id;
        sd.m_fileOffset = offset;

        while (!m_done) {
            SkipLine(); // ignore whatever is left on this line.
            offset = m_fileOffsetStart + (m_pos - m_bufferStart); // a new line starts at this offset;
            sd.m_numberOfSamples++; 

            if (!m_done && GetNextSequenceId(id) && id != sd.m_id) {
                // found a new sequence, which starts at the [offset] bytes into the file
                sd.m_byteSize = offset - sd.m_fileOffset;
                UpdateTimeline(sd);
                sd = {};
                sd.m_id = id;
                sd.m_fileOffset = offset;
            }
        }

        // calculate the byte size for the last sequence
        sd.m_byteSize = m_fileOffsetEnd - sd.m_fileOffset;
        UpdateTimeline(sd);
        return new Index{ false, m_timeline, m_chunks };
    }


    void Indexer::SkipLine() {
        while (!m_done)
        {
            m_pos = (char*)memchr(m_pos, '\n', m_bufferEnd - m_pos);
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
            while (m_pos != m_bufferEnd) {
                char c = *m_pos;
                //remember offset;
                if (c == '|')  // TODO: replace with a constant
                {
                    return found;
                }

                if (c < '0' || c > '9') {
                    throw "malfomed input"; // TODO: add offset, char and other details
                }
                found |= true;
                // TODO: check for overflows
                id = id * 10 + (c - '0');
                ++m_pos;
            }
            Fill();
        }
        // reached EOF without hitting the pipe character.
        throw "malfomed input";
    }

}}}