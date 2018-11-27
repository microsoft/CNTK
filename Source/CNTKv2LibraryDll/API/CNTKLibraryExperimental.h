//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This is the main header of the CNTK library API containing the entire public API definition. 
//

#pragma once

#ifdef SWIG
#define final
#define explicit
#define static_assert(condition, message)
#endif

#include "CNTKLibrary.h"

///
/// Experimental features in CNTK library. 
/// Please be aware that these are subject to frequent changes and even removal.
///

namespace CNTK {
    ///
    ///  Sequence key, used to correlate sequences across different deserializers.
    ///
    struct SequenceKey
    {
        SequenceKey() : SequenceKey(0, 0) {}
        SequenceKey(size_t sequence, unsigned int sample) : m_sequence(sequence), m_sample(sample) {}

        size_t m_sequence;    /// Sequence id - identifies sequence across different deserializers.
        unsigned int m_sample;/// Sample id - identifies a sample inside the sequence.
                              /// Used when the deserializer operates in sample mode.
    };

    typedef unsigned int ChunkIdType;

    static const ChunkIdType ChunkIdMax = (ChunkIdType)(-1);
    static const unsigned int SequenceLenMax = (unsigned int)(-1);

    ///
    /// Defines main properties of a sequence.
    /// Sequence information is used by the randomizer to establish a global timeline for complete input.
    /// A sequence is defined as an ordered collection of samples.
    ///
    struct SequenceInfo
    {
        size_t m_indexInChunk;                     /// Sequence index in chunk.
        unsigned int m_numberOfSamples;            /// Number of samples in a sequence.
        ChunkIdType m_chunkId;                     /// Each sequence belongs to an I/O chunk, how chunk is defined is specific to a
                                                   /// particular data deserializer (or bundler). The randomizer guarantees to request
                                                   /// sequences from only limited subset of chunks at any moment in time.
        SequenceKey m_key;                         /// Sequence key, uniquely identifies the sequence.
                                                   /// When data is coming from different deserializers it is used to correlated sequences
                                                   /// (the reader will perform a form of SQL join operation on the m_key
                                                   /// to correlated the data between different streams).
    };

    //forward declaration for Chunk/ChunkPtr
    class Chunk;
    typedef std::shared_ptr<Chunk> ChunkPtr;

    ///
    /// Defines sequence data and its layout.
    /// Currently CNTK supports dense and sparse sequences (csc).
    /// The storageType in the corresponding stream information identifies what type of SequenceData
    /// data deserializer or transformer provides.
    /// The layout of samples are described in the sampleLayout.
    /// All samples in the sequence should have the same layout.
    ///
    struct SequenceDataBase
    {
        SequenceDataBase(unsigned int numberOfSamples, bool isValid)
            : m_numberOfSamples(numberOfSamples), m_elementType(DataType::Unknown), m_isValid(isValid)
        {}

        virtual ~SequenceDataBase() = default;

        // Returns the shape of samples in the sequence.
        virtual const NDShape& GetSampleShape() = 0;

        // Returns a pointer to internal data buffer.
        virtual const void* GetDataBuffer() = 0;

        unsigned int m_numberOfSamples;  /// Number of samples in the sequence

        DataType m_elementType;   /// Sequence element type.
        bool m_isValid;           /// Flag indicating if sequence is valid.
        SequenceKey m_key;        /// Sequence key.
        std::shared_ptr<uint8_t> m_holdingBuffer; /// Hold reference to data buffer when sequence shares memory with it
    };
    typedef std::shared_ptr<SequenceDataBase> SequenceDataPtr;

    ///
    /// Dense sequence. Should be returned by the deserializer for streams with storage type StorageType::dense.
    /// All samples are stored in the 'data' member as a contiguous array.
    ///
    struct DenseSequenceData : SequenceDataBase
    {
        DenseSequenceData(unsigned int numberOfSamples = 0, bool isValid = true)
            : SequenceDataBase(numberOfSamples, isValid)
        {}
    };
    typedef std::shared_ptr<DenseSequenceData> DenseSequenceDataPtr;

    ///
    /// Sparse sequence. Should be returned by the deserializer for streams with storage type StorageType::csc_sparse.
    /// All non zero values are stored in the m_data member as a contiguous array.
    /// The corresponding row indices are stored in m_indices.
    /// All samples in the sequence should have the same layout.
    ///
    struct SparseSequenceData : SequenceDataBase
    {
        SparseSequenceData(unsigned int numberOfSamples = 0, bool isValid = true)
            : SequenceDataBase(numberOfSamples, isValid)
        {}

        SparseIndexType* m_indices {0};            /// an index for every value in the m_data array
        std::vector<SparseIndexType> m_nnzCounts;  /// nnz count for each sample in the sequence
        SparseIndexType m_totalNnzCount {0};       /// sum of all nzzCounts of all samples
                                                   /// Using IndexType for both properties above since the nnzCount should fit inside
                                                   /// the index type (in CSC format, the last value in the column index array == nnzCount)
    };
    typedef std::shared_ptr<SparseSequenceData> SparseSequenceDataPtr;

    ///
    /// A chunk represents a set of sequences.
    /// In order to enable efficient IO, the deserializer is asked to load a complete chunk in memory.
    /// Which chunks to load is controlled by the randomizer. The randomizer guarantees that at any point in time
    /// only a limited number of chunks are kept in memory. It uses only sequences from these chunks for randomization.
    ///
    /// In case when several deserializers provide data, the chunking of the "primary" deserializer defines
    /// which chunks are requested by the randomizer. If the order of sequences in the deserializers is the same,
    /// the randomizer will access only a limited set of chunks at any point in time. 
    /// If the sequences across the deserializers are not aligned, it can lead to increased memory pressure
    /// caused by randomly accessed sequences in different chunks in secondary deserializers.
    ///
    /// The lifetime of a chunk is controlled by the randomizer - when all sequences of the chunk are consumed, the randomizer
    /// releases the shared pointer to the chunk releasing the associated memory.
    ///
    class Chunk
    {
    public:
        ///
        /// Gets data for the sequence with the given index.
        /// result contains a SequenceDataPtr for every input stream declared by the
        /// deserializer that produced this chunk.
        ///
        virtual void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) = 0;

        ///
        /// Returns meta information about sequences that this chunk has.
        /// This allows deserialization of sequences on several threads afterwords.
        /// TODO: Currently chunk->SequenceInfos() == deserializer->SequenceInfo(chunk),
        /// this should be unified.
        ///
        virtual void SequenceInfos(std::vector<SequenceInfo>& /*result*/) { NOT_IMPLEMENTED; }

        virtual ~Chunk() = default;

    protected:
        Chunk() = default;
    };

    ///
    /// Meta information of a chunk.
    ///
    struct ChunkInfo
    {
        ChunkIdType m_id;            /// Chunk id.
        size_t m_numberOfSamples;    /// Number of samples in the chunk.
        size_t m_numberOfSequences;  /// Number of sequences in the chunk.
    };

    ///
    /// Interface all data deserializers should implement.
    /// Data deserializers are intimately familiar with a particular input formats and responsible for bringing
    /// the serialized data into sequences in memory. Very often data for different streams (i.e. features/lattices)
    /// reside in the same physical storage (file), so the data deserializer can expose not a single but several
    /// streams. Examples of data include image data deserializer or htkmlf data deserializer.
    ///
    class DataDeserializer
    {
    public:
        ///
        /// Gets stream information for all streams this deserializer exposes.
        ///
        virtual std::vector<StreamInformation> StreamInfos() = 0;

        ///
        /// Gets metadata for chunks this deserializer exposes.
        ///
        virtual std::vector<ChunkInfo> ChunkInfos() = 0;

        ///
        /// Gets sequence infos for a given a chunk.
        ///
        virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) = 0;

        ///
        /// Gets sequence information given the one of the primary deserializer.
        /// Used for non-primary deserializers.
        /// Returns false if the corresponding secondary sequence is not valid.
        ///
        virtual bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo& result) = 0;

        ///
        /// Gets chunk data given its id.
        ///
        virtual ChunkPtr GetChunk(ChunkIdType chunkId) = 0;

        virtual ~DataDeserializer() = default;

    protected:
        DataDeserializer() = default;
    };

    typedef std::shared_ptr<DataDeserializer> DataDeserializerPtr;
}
