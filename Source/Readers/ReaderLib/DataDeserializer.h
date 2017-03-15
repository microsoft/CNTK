//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Sequence key, used for correlations of sequences between different deserializers.
// TODO: In many cases sequence keys share the same prefix. Splitting the sequence key on
// sequence prefix and suffix will allow us to store keys more efficiently.

// The sample identifies a particular sample inside the sequence. In the future it will be hidden, so that deserializers won't know about
// sequence or sample mode, exposing only sequences.
struct KeyType
{
    KeyType() : m_sequence(0), m_sample(0) {}

    KeyType(size_t sequence, size_t sample) : m_sequence(sequence), m_sample(sample) {}

    // Possible sequence common prefix.
    // size_t m_prefix;

    // Identifies sequence between different deserializers.
    size_t m_sequence : 40;

    // Sample id.
    size_t m_sample : 24;
};

class Chunk;
typedef std::shared_ptr<Chunk> ChunkPtr;

typedef uint32_t ChunkIdType;
#define CHUNKID_MAX ((ChunkIdType)(-1))

#define SEQUENCELEN_MAX ((uint32_t)(-1))

// Defines main properties of a sequence.
// Sequence descriptions are used by the randomizer to establish a global timeline for complete input.
// A sequence is defined as an ordered set of samples (size == 1 is used for sample training).
struct SequenceDescription
{
    size_t m_indexInChunk;                     // Sequence index in chunk.
    uint32_t m_numberOfSamples;                // Number of samples in a sequence.
    ChunkIdType m_chunkId;                     // Each sequence belongs to an I/O chunk, how chunk is defined is specific to a
                                               // particular data deserializer (or bundler). The randomizer guarantees to request
                                               // sequences from only limited subset of chunks at any moment in time.
    KeyType m_key;                             // Sequence key, uniquely identifies the sequence.
                                               // When data is coming from different deserializers
                                               // the reader will perform a form for SQL join operation on the m_key 
                                               // to correlated the data between different streams.
};

typedef std::shared_ptr<SequenceDescription> SequenceDescriptionPtr;

// Defines sequence data and its layout.
// Currently CNTK supports dense and sparse sequences (csc).
// The storageType in the corresponding stream description identifies what type of SequenceData
// data deserializer or transformer provides.
// The layout of samples are described in the sampleLayout.
// All samples in the sequence should have the same layout.
// TODO: add type casts (As<T>() or AsRef<>() or AsPtr<>()) to subclasses as members here.
struct SequenceDataBase
{
    SequenceDataBase() : m_numberOfSamples(0), m_elementType(ElementType::tvariant), m_isValid(true) {}
    virtual ~SequenceDataBase() = default;

    uint32_t m_numberOfSamples;      // Number of samples in the sequence

    // Returns a pointer to the data buffer.
    // The actual size is provided for particular sequences,i.e. see DenseSequenceData, or SparseSequenceData.
    virtual const void* GetDataBuffer() = 0;

    ElementType    m_elementType;     // Sequence element type.
    TensorShapePtr m_sampleLayout;    // Sample layout, can be shared by several sequences.
    bool           m_isValid;         // Flag indicating if sequence is valid.
    KeyType        m_key;             // Sequence key.
};
typedef std::shared_ptr<SequenceDataBase> SequenceDataPtr;

// Dense sequence. Should be returned by the deserializer for streams with storage type StorageType::dense.
// All samples are stored in the 'data' member as a contiguous array.
struct DenseSequenceData : SequenceDataBase
{
};
typedef std::shared_ptr<DenseSequenceData> DenseSequenceDataPtr;

// Sparse sequence. Should be returned by the deserializer for streams with storage type StorageType::csc_sparse.
// All non zero values are store in the 'data' member as a contiguous array.
// The corresponding row indices are stored in 'indices' per sample.
// All samples in the sequence should have the same layout.
struct SparseSequenceData : SequenceDataBase
{
    IndexType* m_indices; // an index for every value in the m_data array
    std::vector<IndexType> m_nnzCounts; // nnz count for each sample in the sequence
    IndexType m_totalNnzCount; // sum of all nzzCounts of all samples
    // Using IndexType for both properties above since the nnzCount should fit inside
    // the index type (in CSC format, the last value in the column index array == nnzCount)
};
typedef std::shared_ptr<SparseSequenceData> SparseSequenceDataPtr;

// A chunk represents a set of sequences.
// In order to enable efficient IO, the deserializer is asked to load a complete chunk in memory.
// Which chunks to load are controlled by the randomizer. The randomizer guarantees that at any point in time
// only a limited number of chunks is requested from the deserializer and uses for randomization only sequences
// from these chunks.
//
// In case when several deserializers provide data, the chunking of the "primary" deserializer defines
// which chunks are requested by the randomizer. Thus, if the deserializers are "aligned" as how they see chunks,
// the randomizer will access only a limited set. If the data between different randomizers is not aligned - this
// could lead to memory pressure caused by randomly accessed sequences in different chunks in secondary deserializers.
//
// The lifetime of chunk is controlled by the randomizer - when all sequences of the chunk are consumed, the randomizer
// releases the shared pointer to the chunk by that freeing the associated memory.
// Sequences are only pointers to the real data which is allocated on chunk basis.
class Chunk
{
public:
    // Gets a sequence per input by its index inside the chunk.
    virtual void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) = 0;

    virtual ~Chunk() {};

protected:
    Chunk() {}

private:
    DISABLE_COPY_AND_MOVE(Chunk);
};

// Represents a chunk description.
struct ChunkDescription
{
    // Chunk id.
    ChunkIdType m_id;
    // Number of samples in the chunk.
    size_t m_numberOfSamples;
    // Number of sequences in the chunk.
    size_t m_numberOfSequences;
};

typedef std::shared_ptr<ChunkDescription> ChunkDescriptionPtr;
typedef std::vector<ChunkDescriptionPtr> ChunkDescriptions;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Interface all data deserializers should implement.
// Data deserializers are intimately familiar with a particular input formats and responsible for bringing
// the serialized data into sequences in memory. Very often data for different streams (i.e. features/lattices)
// reside in the same physical storage (file), so the data deserializer can expose not a single but several
// streams. Examples of data include image data deserializer or htkmlf data deserializer.
// TODO: This interface will become ABI and deserializers can be implemented in different languages, i.e. Python.
//////////////////////////////////////////////////////////////////////////////////////////////////
class IDataDeserializer
{
public:
    // Gets stream descriptions for all streams this deserializer exposes.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const = 0;

    // Gets chunk descriptions this deserializer exposes.
    virtual ChunkDescriptions GetChunkDescriptions() = 0;

    // Gets sequence descriptions for a given a chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& descriptions) = 0;

    // Gets sequence description given the sequence description of the primary deserializer.
    // Used for deserializers not in driving/primary mode.
    // Returns false if the corresponding secondary sequence is not valid.
    // TODO: Possibly move this out into a separate interface.
    virtual bool GetSequenceDescription(const SequenceDescription& primary, SequenceDescription& description) = 0;

    // Gets chunk data given its id.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) = 0;

    virtual ~IDataDeserializer() {};
};

typedef std::shared_ptr<IDataDeserializer> IDataDeserializerPtr;

}}}
