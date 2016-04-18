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
    Indexer(FILE* file, bool skipSequenceIds = false, size_t chunkSize = 32 * 1024 * 1024);

    // Reads the input file, building and index of chunks and corresponding
    // sequences.
    void Build();

    // Returns input data index (chunk and sequence metadata)
    const Index& GetIndex() const { return m_chunks; }

    // True, when input does not have the sequence id column
    // or when sequence id column was ignored during indexing
    // (by passing skipSequenceIds = true to the constructor).
    bool HasSequenceIds() const { return m_hasSequenceIds; }

private:
    FILE* m_file;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    unique_ptr<char[]> m_buffer;
    const char* m_bufferStart;
    const char* m_bufferEnd;
    const char* m_pos; // buffer index

    bool m_done; // true, when all input was processed

    bool m_hasSequenceIds; // true, when input contains one sequence per line 
                           // or when sequence id column was ignored during indexing.

    const size_t m_maxChunkSize; // maximum permitted chunk size;

    std::vector<ChunkDescriptor> m_chunks; // a collection of chunk descriptors

    // Adds sequence (metadata) to the index. Additionally, it
    // assigns an appropriate chunk id to the sequence descriptor,
    // ensures that chunks do not exceed the maximum allowed size
    // (except when a sequence size is greater than the maximum chunk size)
    void AddSequence(SequenceDescriptor& sd);

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
    bool TryGetSequenceId(size_t& id);

    // Build a chunk/sequence index, treating each line as an individual sequence.
    // Does not do any sequence parsing, instead uses line number as 
    // the corresponding sequence id.
    void BuildFromLines();

    // Returns current offset in the input file (in bytes). 
    int64_t GetFileOffset() const { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    DISABLE_COPY_AND_MOVE(Indexer);
};

}}}
