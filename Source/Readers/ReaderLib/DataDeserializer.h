//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct SequenceDataBase;
typedef std::shared_ptr<SequenceDataBase> SequenceDataPtr;

class Chunk
{
public:
    // Gets sequences by id.
    virtual std::vector<SequenceDataPtr> GetSequence(const size_t& sequenceId) = 0;
    virtual ~Chunk() {};

protected:
    Chunk() {}

private:
    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;
};
typedef std::shared_ptr<Chunk> ChunkPtr;

// Defines main properties of a sequence.
// Sequence descriptions are used by the randomizer to establish a global timeline for complete input.
// A sequence is defined as an ordered set of samples (size == 1 is used for sample training).
struct SequenceDescription
{
    size_t m_id;              // Sequence id, uniquely identifies the sequence.
    size_t m_numberOfSamples; // Number of samples in a sequence.
    size_t m_chunkId;         // Each sequence belongs to an I/O chunk, how chunk is defined is specific to a
                              // particular data deserializer (or bundler). The randomizer guarantees to request
                              // sequences from only limited subset of chunks at any moment in time.
    bool m_isValid;           // Indicates whether the sequence is valid.
};
typedef std::vector<const SequenceDescription*> SequenceDescriptions;

// Defines sequence data and its layout.
// Currently CNTK supports dense and sparse sequences (csc).
// The storageType in the corresponding stream description identifies what type of SequenceData
// data deserializer or transformer can provide provides.
struct SequenceDataBase
{
    SequenceDataBase() : m_data(nullptr) { }
    virtual ~SequenceDataBase()
    {
        
    }

    ChunkPtr m_chunk;
    // A non-owned pointer. The actual size is provided for particular sequences,
    // i.e. see DenseSequenceData, or SparseSequenceData.
    void* m_data;
};
typedef std::shared_ptr<SequenceDataBase> SequenceDataPtr;

// Dense sequence. Should be returned by the deserializer for streams with storage type StorageType::dense.
// All samples are stored in the 'data' member as a contiguous array.
// The layout of samples are described in the sampleLayout.
// All samples in the sequence should have the same layout.
struct DenseSequenceData : SequenceDataBase
{
    DenseSequenceData() : m_numberOfSamples(0) { }

    TensorShapePtr m_sampleLayout; // Sample layout, can be shared by several sequences.
    size_t m_numberOfSamples;      // Number of samples in the sequence
};
typedef std::shared_ptr<DenseSequenceData> DenseSequenceDataPtr;

// Sparse sequence. Should be returned by the deserializer for streams with storage type StorageType::csc_sparse.
// All non zero values are store in the 'data' member as a contiguous array.
// The corresponding row indices are stored in 'indices' per sample.
// All samples in the sequence should have the same layout.
struct SparseSequenceData : SequenceDataBase
{
    std::vector<std::vector<size_t>> m_indices;
};
typedef std::shared_ptr<SparseSequenceData> SparseSequenceDataPtr;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Interface all data deserializers should implement.
// Data deserializers are intimately familiar with a particular input formats and responsible for bringing 
// the serialized data into sequences in memory. Very often data for different streams (i.e. features/lattices)
// reside in the same physical storage (file), so the data deserializer can expose not a single but several 
// streams. Examples of data include image data deserializer or htkmlf data deserializer.
// TODO: This interface will become ABI and deserializers can be implemented in different languages, i.e. Python.
//////////////////////////////////////////////////////////////////////////////////////////////////
class DataDeserializer
{
public:
    // Describes streams this data deserializer can produce. Streams correspond to network inputs.
    // TODO: Introduce the interface to reduce the size of the sequences available at any point in time (chunks/sequences).
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const = 0;

    // Retrieves description of all sequences this data deserializer can produce.
    virtual const SequenceDescriptions& GetSequenceDescriptions() const = 0;

    // Gets a chunk.
    virtual ChunkPtr GetChunk(size_t chunkId) = 0;

    virtual ~DataDeserializer() {};
};

typedef std::shared_ptr<DataDeserializer> DataDeserializerPtr;
} } }
