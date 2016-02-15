//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "Descriptors.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class Indexer 
{
private:
    FILE* m_file = NULL;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    char* m_bufferStart = NULL;
    char* m_bufferEnd = NULL;
    char* m_pos = NULL; // buffer index

    bool m_done; // true, when all input was processed

    bool m_skipSequenceIds; // true, when input contains one sequence per line 
                            // and sequence id column can be skipped.

    const int64_t m_maxChunkSize; // maximum permited chunk size;

    std::vector<SequenceDescriptor> m_timeline;
    std::vector<ChunkDescriptor> m_chunks;
    std::map<SequenceId, TimelineOffset> m_idToOffsetMap;

    // assigns an appropriate chunk id to the sequence descriptor,
    // ensures that chunks do not exceed the maximum allowed size
    // (except when a sequence size is greater than the maximum chunk size)
    void UpdateTimeline(SequenceDescriptor& sd);

    // fills buffer with data, this method assumes that all buffered
    // data was already consumed.
    void Fill();

    // moves the buffer position to the beginning of the next line.
    void SkipLine();

    // reads the line until the next pipe character, parsing numerical characters into a sequence id.
    // Throws an exception if a non-numerical is read until the pipe character or 
    // EOF is reached without hitting the pipe character.
    // Returns false if no numerical characters are found preceeding the pipe.
    // Otherwise, writes sequence id value to the provided reference, returns true.
    bool GetNextSequenceId(size_t& id);

    // Builds timeline, treating each line as an individual sequence.
    // Does not do any sequence parsing, instead uses line number as the corresponding sequence id.
    Index* BuildFromLines();


    int64_t GetFileOffset() { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    Indexer(const Indexer&) = delete;
    Indexer& operator=(const Indexer&) = delete;

public:
    Indexer(FILE* file, bool skipSequenceIds, int64_t chunkSize = 32 * 1024 * 1024);

    ~Indexer();

    // Reads the input file building an index of sequence metadata.
    Index* Build();
};

}}}