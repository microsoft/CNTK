//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "DataDeserializer.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Sequence metadata. This text-reader specific descriptor adds two additional
// fields: file offset and size in bytes. Both are required to efficiently
// locate and retrieve a sequence from file, given a sequence descriptor.
struct SequenceDescriptor : SequenceDescription
{
    SequenceDescriptor() : SequenceDescription({}), m_fileOffsetBytes(0),
        m_byteSize(0)
    {
    }
    // size_t m_numberOfSamples -- number of samples in the sequence (largest count among all inputs)
    // in case of text data this value == number of rows this sequence spans over.
    int64_t m_fileOffsetBytes; // sequence offset in the input file (in bytes)
    size_t m_byteSize; // size in bytes
};

// Chunk metadata, similar to the sequence descriptor above,
// but used to facilitate indexing and retrieval of blobs of input data of
// some user-specified size.
struct ChunkDescriptor : ChunkDescription
{
    ChunkDescriptor() : ChunkDescription({}), m_byteSize(0) {}
    // TODO: if we don't want to keep the whole index
    // (metadata for all sequences in memory), we should not
    // leave this empty when building a chunk index, and only
    // fill it out when the chunk needs to be loaded
    // (the indexer will have to do a second pass for this chunk).
    std::vector<SequenceDescriptor> m_sequences;

    size_t m_byteSize; // size in bytes
};

typedef shared_ptr<ChunkDescriptor> ChunkDescriptorPtr;

// A collection of chunk descriptors, each containing
// a collection of sequence descriptors for the corresponding
// chunk of the input data.
// It also stores a mapping of keys into sequence descriptors.
struct Index
{
    std::vector<ChunkDescriptor> m_chunks;                                  // chunks
    std::map<size_t, std::pair<size_t, size_t>> m_keyToSequenceInChunk;     // sequence key -> sequence location in chunk
    const size_t m_maxChunkSize;                                            // maximum chunk size in bytes
    bool m_primary;                                                         // index for primary deserializer

    Index(size_t chunkSize, bool primary) : m_maxChunkSize(chunkSize), m_primary(primary)
    {}

    // Adds sequence (metadata) to the index. Additionally, it
    // assigns an appropriate chunk id to the sequence descriptor,
    // ensures that chunks do not exceed the maximum allowed size
    // (except when a sequence size is greater than the maximum chunk size)
    void AddSequence(SequenceDescriptor& sd)
    {
        assert(!m_chunks.empty());
        ChunkDescriptor* chunk = &m_chunks.back();
        if (chunk->m_byteSize > 0 && (chunk->m_byteSize + sd.m_byteSize) > m_maxChunkSize)
        {
            // Creating a new chunk if the size is exceeded.
            chunk->m_sequences.shrink_to_fit();
            m_chunks.push_back({});
            chunk = &m_chunks.back();
            chunk->m_id = (ChunkIdType)(m_chunks.size() - 1);
            if (CHUNKID_MAX < m_chunks.size())
            {
                RuntimeError("Maximum number of chunks exceeded");
            }
        }

        chunk->m_byteSize += sd.m_byteSize;
        chunk->m_numberOfSequences++;
        chunk->m_numberOfSamples += sd.m_numberOfSamples;
        sd.m_chunkId = chunk->m_id;
        sd.m_indexInChunk = chunk->m_sequences.size();
        if (!m_primary)
        {
            auto location = std::make_pair(chunk->m_id, sd.m_indexInChunk);
            auto sequenceId = sd.m_key.m_sequence;
            m_keyToSequenceInChunk.insert(std::make_pair(sequenceId, location));
        }
        chunk->m_sequences.push_back(sd);
    }

    // Reserves inner structures for the specified number of bytes.
    void Reserve(size_t sizeInBytes)
    {
        if (m_maxChunkSize > 0)
        {
            m_chunks.reserve((sizeInBytes + m_maxChunkSize - 1) / m_maxChunkSize);
        }

        m_chunks.push_back({});
    }

    // Checks if the index is empty.
    bool IsEmpty() const
    {
        return m_chunks.empty();
    }

    DISABLE_COPY_AND_MOVE(Index);
};

// A helper class that does a pass over the input file building up
// an index consisting of sequence and chunk descriptors (which among 
// others specify size and file offset of the respective structure).
// As opposed to the data deserializer, indexer performs almost no parsing 
// and therefore is several magnitudes faster.
class Indexer
{
public:
    Indexer(FILE* file, bool isPrimary, bool skipSequenceIds = false, char streamPrefix = '|', size_t chunkSize = 32 * 1024 * 1024, size_t bufferSize = 2 * 1024 * 1024);

    // Reads the input file, building and index of chunks and corresponding
    // sequences.
    void Build(CorpusDescriptorPtr corpus);

    // Returns input data index (chunk and sequence metadata)
    const Index& GetIndex() const { return m_index; }

    // True, when input does not have the sequence id column
    // or when sequence id column was ignored during indexing
    // (by passing skipSequenceIds = true to the constructor).
    bool HasSequenceIds() const { return m_hasSequenceIds; }

private:
    FILE* m_file;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    std::unique_ptr<char[]> m_buffer;
    const size_t m_bufferSize;
    const char* m_bufferStart;
    const char* m_bufferEnd;
    const char* m_pos; // buffer index

    bool m_done; // true, when all input was processed

    bool m_hasSequenceIds; // true, when input contains one sequence per line 
                           // or when sequence id column was ignored during indexing.

    // a collection of chunk descriptors and sequence keys.
    Index m_index;

    const char m_streamPrefix;

    // fills up the buffer with data from file, all previously buffered data
    // will be overwritten.
    void RefillBuffer();

    // Moves the buffer position to the beginning of the next line.
    void SkipLine();

    // Tries to get numeric sequence id.
    // Throws an exception if a non-numerical is read until the pipe character or 
    // EOF is reached without hitting the pipe character.
    // Returns false if no numerical characters are found preceding the pipe.
    // Otherwise, writes sequence id value to the provided reference, returns true.
    bool TryGetNumericSequenceId(size_t& id);

    // Same as above but for symbolic ids.
    // It reads a symbolic key and converts it to numeric id using provided keyToId function.
    bool TryGetSymbolicSequenceId(size_t& id, std::function<size_t(const std::string&)> keyToId);


    // Build a chunk/sequence index, treating each line as an individual sequence.
    // Does not do any sequence parsing, instead uses line number as 
    // the corresponding sequence id.
    void BuildFromLines();

    // Returns current offset in the input file (in bytes). 
    int64_t GetFileOffset() const { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    DISABLE_COPY_AND_MOVE(Indexer);
};

}}}
