//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "Descriptors.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A helper class that does a pass over the input file building up
// an index consisting of sequence and chunk descriptors (which among 
// others specify size and file offset of the respective structure).
// As opposed to the data deserializer, indexer performs almost no parsing 
// and therefore is several magnitudes faster.
class Indexer 
{
public:
    Indexer(FILE* file, bool skipSequenceIds, size_t chunkSize = 32 * 1024 * 1024);

    // Reads the input file building an index of sequence metadata.
    IndexPtr Build();

private:
    FILE* m_file;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    unique_ptr<char[]> m_buffer;
    const char* m_bufferStart;
    const char* m_bufferEnd;
    const char* m_pos; // buffer index

    bool m_done; // true, when all input was processed

    bool m_skipSequenceIds; // true, when input contains one sequence per line 
                            // and sequence id column can be skipped.

    const size_t m_maxChunkSize; // maximum permitted chunk size;

    std::vector<SequenceDescriptor> m_timeline; // a collection of sequence descriptors
    std::vector<ChunkDescriptor> m_chunks; // a collection of chunk descriptors

    // Assigns an appropriate chunk id to the sequence descriptor,
    // ensures that chunks do not exceed the maximum allowed size
    // (except when a sequence size is greater than the maximum chunk size)
    void UpdateTimeline(SequenceDescriptor& sd);

    // fills up the buffer with data from file, all previously buffered data
    // will be overwritten.
    void RefillBuffer();

    // Moves the buffer position to the beginning of the next line.
    void SkipLine();

    // Reads the line until the next pipe character, parsing numerical characters into a sequence id.
    // Throws an exception if a non-numerical is read until the pipe character or 
    // EOF is reached without hitting the pipe character.
    // Returns false if no numerical characters are found preceding the pipe.
    // Otherwise, writes sequence id value to the provided reference, returns true.
    bool GetNextSequenceId(size_t& id);

    // Builds timeline, treating each line as an individual sequence.
    // Does not do any sequence parsing, instead uses line number as the corresponding sequence id.
    IndexPtr BuildFromLines();

    // Returns current offset in the input file (in bytes). 
    int64_t GetFileOffset() const { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    DISABLE_COPY_AND_MOVE(Indexer);
};

}}}
