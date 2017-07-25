//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <memory>
#include <boost/noncopyable.hpp>
#include "Basics.h"

namespace CNTK {

class IndexedSequence;

// Metadata describing a sequence within a chunk.
struct SequenceDescriptor
{
    SequenceDescriptor(size_t key, uint32_t numberOfSamples, uint32_t size, size_t offset)
        : m_key(key),
        m_numberOfSamples(numberOfSamples),
        m_byteSize(size),
        m_offsetInChunk(static_cast<uint32_t>(offset))
    {
        if (m_offsetInChunk != offset)
            RuntimeError("Sequence offset overflows uint32_t type.");
    }

    const size_t m_key;                // Unique sequence identifier.
    const uint32_t m_numberOfSamples;  // Number of samples in this sequence.
    const uint32_t m_offsetInChunk;    // Sequence offset inside a chunk (in bytes).
    const uint32_t m_byteSize;         // Size in bytes.


    uint32_t NumberOfSamples() const { return m_numberOfSamples; }

    uint32_t OffsetInChunk() const { return m_offsetInChunk; }

    uint32_t SizeInBytes() const { return m_byteSize; }
};

// Chunk metadata, similar to the sequence descriptor above,
// but used to facilitate indexing and retrieval of blobs of input data of
// some user-specified size.
class ChunkDescriptor
{
    friend class Index;

public:
    // Start offset of the chunk in a file (in bytes)
    size_t StartOffset() const { return m_startOffset; }

    // End offset of the chunk in a file (in bytes)
    size_t EndOffset() const { return m_endOffset; }


    // This is the file size of a chunk (the length in bytes between 
    // the end and start offsets). Only if all sequences in this chunk are laid 
    // out contiguously with no gaps, this size will match the sum of 
    // sequence sizes.
    size_t SizeInBytes() const { return m_endOffset - m_startOffset; }

    size_t NumberOfSamples() const { return m_numberOfSamples; }

    size_t NumberOfSequences() const { return m_sequences.size(); }
    
    const std::vector<SequenceDescriptor>& Sequences() const { return m_sequences; }

    const SequenceDescriptor& operator[](size_t i) const
    {
        return m_sequences[i];
    }

private:

    ChunkDescriptor(size_t offset)
        : m_startOffset(offset),
        m_endOffset(offset)
    {}

    void AddSequence(const IndexedSequence& sequence);
    
    size_t m_startOffset, m_endOffset;
    size_t m_numberOfSamples {0};
    std::vector<SequenceDescriptor> m_sequences;
};

// A collection of chunk descriptors (each containing
// a collection of sequence descriptors) for the corresponding
// chunks of the input data.
// An index serves two purposes. First, it provides the minimum required
// information about an input data source to the randomizer, so that it
// can construct a global timeline needed for sliding-window randomization
// (specifically, when the window size is given in samples or sequences). 
// Second, an index facilitates random access to chunks on disk and 
// to sequences within a chunk (residing in memory or on disk). Towards 
// this end, it stores a mapping of sequence keys (unique size_t identifiers) 
// to corresponding chunk ids and positions within the chunk.
class Index : private boost::noncopyable
{
    friend class IndexBuilder;

public:
    Index(size_t chunkSize) : m_maxChunkSize(chunkSize)
    {}

    // Reserves inner structures for the specified number of bytes.
    void Reserve(size_t sizeInBytes)
    {
        if (m_maxChunkSize > sizeInBytes) 
        {
            m_maxChunkSize = sizeInBytes;
            m_chunks.reserve(1);
        }
        else if (m_maxChunkSize > 0)
            m_chunks.reserve((sizeInBytes + m_maxChunkSize - 1) / m_maxChunkSize);
    }

    // Checks if the index is empty.
    bool IsEmpty() const { return m_chunks.empty(); }

    // Returns true or false with chunk and sequence index depending if the key has been found.
    std::tuple<bool, uint32_t, uint32_t> GetSequenceByKey(size_t key) const;

    const std::vector<ChunkDescriptor>& Chunks() const { return m_chunks; }

    size_t NumberOfSamples() const { return m_numberOfSamples; }

    size_t NumberOfSequences() const { return m_numberOfSequences; }

    size_t NumberOfChunks() const { return m_chunks.size(); }

    size_t SizeInBytes() const { return m_sizeInBytes; }

    const ChunkDescriptor& operator[](size_t i) const
    {
        return m_chunks[i];
    }

    // Adds a new sequence (metadata) to the index.
    void AddSequence(const IndexedSequence& sequence);

private:
    // Vector containing <sequence key, chunk index, sequence index in chunk> tuples, 
    // sorted by sequence key and used for fast sequence metadata retrieval for 
    // non-primary deserializers.
    std::vector<std::tuple<size_t, uint32_t, uint32_t>> m_keyToSequenceInChunk;

    void MapSequenceKeyToLocation();

    size_t m_maxChunkSize; // maximum chunk size in bytes
    
    std::vector<ChunkDescriptor> m_chunks;
    size_t m_numberOfSamples {0};
    size_t m_numberOfSequences {0};
    size_t m_sizeInBytes {0};
};

}
