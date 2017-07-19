//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "DataDeserializer.h"
#include "CorpusDescriptor.h"
#include "MemoryBuffer.h"

namespace CNTK {

// Sequence metadata that allows indexing a sequence in a binary file.
struct SequenceDescriptor
{
    SequenceDescriptor(size_t key, uint32_t numberOfSamples)
        : m_offsetInChunk(0),
          m_byteSize(0), 
          m_numberOfSamples(numberOfSamples),
          m_key(key)
    {
    }

    const size_t m_key;                       // Sequence key, uniquely identifies the sequence.
    const uint32_t m_numberOfSamples;         // Number of samples in a sequence.

    uint32_t OffsetInChunk() const
    {
        return m_offsetInChunk;
    }

    uint32_t SizeInBytes() const
    {
        return m_byteSize;
    }

private:
    void SetSize(size_t size)
    {
        m_byteSize = static_cast<uint32_t>(size);
        if (m_byteSize != size)
            RuntimeError("Sequence size overflows uint32_t type.");
    }

    void SetOffsetInChunk(size_t offset)
    {
        m_offsetInChunk = static_cast<uint32_t>(offset);
        if (m_offsetInChunk != offset)
            RuntimeError("Chunk size overflows uint32_t type.");
    }

    uint32_t m_offsetInChunk;         // sequence offset in the chunk (in bytes)
    uint32_t m_byteSize;                 // size in bytes

    friend struct Index;
    friend class ChunkDescriptor;
};

// Chunk metadata, similar to the sequence descriptor above,
// but used to facilitate indexing and retrieval of blobs of input data of
// some user-specified size.
class ChunkDescriptor
{
    ChunkDescriptor() : m_maxSizeInBytes(0), m_offset(0) {}

public:
    const size_t m_maxSizeInBytes;

    // offset of the chunk in bytes
    const size_t m_offset;

    ChunkDescriptor(size_t maxSizeInBytes, size_t startOffset)
        : m_maxSizeInBytes(maxSizeInBytes), m_sizeInBytes(0),
          m_offset(startOffset), m_numberOfSamples(0)
    {}

    bool HasSpaceFor(const SequenceDescriptor& sd) const
    {
        return m_sizeInBytes == 0 || m_sizeInBytes + sd.m_byteSize <= m_maxSizeInBytes;
    }

    void AddSequence(SequenceDescriptor&& sd, bool trackFirstSample = false)
    {
        assert(HasSpaceFor(sd));
        if (trackFirstSample) // Adding number of samples where the new sequence starts.
            m_sequenceOffsetInChunkInSamples.push_back(static_cast<uint32_t>(m_numberOfSamples));

        m_sizeInBytes += sd.m_byteSize;
        m_numberOfSamples += sd.m_numberOfSamples;
        m_sequences.push_back(std::move(sd));

        if (m_sizeInBytes >= m_maxSizeInBytes) // Last one, finalizing.
            m_sequences.shrink_to_fit();

        if (m_sequences.size() > std::numeric_limits<uint32_t>::max())
            RuntimeError("Exceeded maximum number of sequences in a chunk");
    }

    size_t SizeInBytes() const { return m_sizeInBytes; }
    size_t NumSamples() const { return m_numberOfSamples; }
    const std::vector<SequenceDescriptor>& Sequences() const { return m_sequences; }

    // Offset of first sample of each sequence from the beginning of the chunk.
    const std::vector<uint32_t>& SequenceOffsetInSamples() const { return m_sequenceOffsetInChunkInSamples; }

private:
    // TODO: if we don't want to keep the whole index
    // (metadata for all sequences in memory), we should not
    // leave this empty when building a chunk index, and only
    // fill it out when the chunk needs to be loaded
    // (the indexer will have to do a second pass for this chunk).
    std::vector<SequenceDescriptor> m_sequences;

    size_t m_numberOfSamples;
    size_t m_sizeInBytes;

    // Offset of first sample of each sequence from the beginning of the chunk.
    // Optionally filled in by the indexer.
    std::vector<uint32_t> m_sequenceOffsetInChunkInSamples;
};

typedef shared_ptr<ChunkDescriptor> ChunkDescriptorPtr;

// A collection of chunk descriptors, each containing
// a collection of sequence descriptors for the corresponding
// chunk of the input data.
// It also stores a mapping of keys into sequence descriptors.
struct Index
{
private:
    std::vector<ChunkDescriptor> m_chunks;

public:
    const std::vector<ChunkDescriptor>& Chunks() const { return m_chunks; }

    // Sorted dictionary of <sequence key, chunk index, sequence index in chunk>
    // used for fast retrieval of sequence by key for non primary deserializers.
    std::vector<std::tuple<size_t, uint32_t, uint32_t>> m_keyToSequenceInChunk;

    const size_t m_maxChunkSize;                                            // maximum chunk size in bytes
    bool m_primary;                                                         // index for primary deserializer
    bool m_trackFirstSamples;                                               // flag indicating whether to build index of first samples
                                                                            // for sequences (m_firstSamples)
                                                                            // Used when deserializer operates in frame mode (i.e. MLF)
                                                                            // and needs to find a sequence by sample in the chunk.

    Index(size_t chunkSize, bool primary, bool trackFirstSamples = false)
        : m_maxChunkSize(chunkSize), m_primary(primary), m_trackFirstSamples(trackFirstSamples)
    {
    }

    // Adds sequence (metadata) to the index. Additionally, it
    // assigns an appropriate chunk id to the sequence descriptor,
    // ensures that chunks do not exceed the maximum allowed size
    // (except when a sequence size is greater than the maximum chunk size)
    void AddSequence(SequenceDescriptor&& sd, size_t startOffsetInFile, size_t endOffsetInFile);

    // Reserves inner structures for the specified number of bytes.
    void Reserve(size_t sizeInBytes)
    {
        if (m_maxChunkSize > 0)
            m_chunks.reserve((sizeInBytes + m_maxChunkSize - 1) / m_maxChunkSize);
    }

    // Checks if the index is empty.
    bool IsEmpty() const
    {
        return m_chunks.empty();
    }

    // Returns true or false with chunk and sequence index depending if the key has been found.
    std::tuple<bool, uint32_t, uint32_t> GetSequenceByKey(size_t key) const;

    void MapSequenceKeyToLocation();

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
    Indexer(FILE* file, bool isPrimary, bool skipSequenceIds = false, char streamPrefix = '|', size_t chunkSize = 32 * 1024 * 1024, const std::string& mainStream = "", size_t bufferSize = 2 * 1024 * 1024);

    // Reads the input file, building and index of chunks and corresponding
    // sequences.
    void Build(CorpusDescriptorPtr corpus);

    // Returns input data index (chunk and sequence metadata)
    const Index& GetIndex() const { return m_index; }

    // True, when input does not have the sequence id column
    // or when sequence id column was ignored during indexing
    // (by passing skipSequenceIds = true to the constructor).
    bool HasSequenceIds() const { return m_hasSequenceIds; }

    const std::string& MainStream() const
    {
        return m_mainStream;
    }

private:
    FILE* m_file;
    int64_t m_fileSize;
    MemoryBuffer m_buffer;
    bool m_hasSequenceIds; // true, when input contains one sequence per line 
                           // or when sequence id column was ignored during indexing.

    // Stream that defines the size of the sequence.
    std::string m_mainStream;

    // a collection of chunk descriptors and sequence keys.
    Index m_index;

    const char m_streamPrefix;

    // Moves the buffer position to the beginning of the next line.
    void SkipLine();

    // Moves the buffer position to the beginning of the next line.
    // Returns true if the current line containes m_mainStream.
    bool SkipLineWithCheck();

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

    DISABLE_COPY_AND_MOVE(Indexer);
};

}
