//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <array>
#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "TextConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
struct Data 
{
    virtual ~Data() {};

    size_t m_numberOfSamples = 0;
    std::vector<ElemType> m_buffer;
};

template <class ElemType>
struct DenseData : Data<ElemType>
{
    // capacity = expected number of samples * sample size
    DenseData(size_t capacity)
    {
        Data<ElemType>::m_buffer.reserve(capacity);
    }
};

template <class ElemType>
struct SparseData : Data<ElemType>
{
    std::vector<std::vector<size_t>> m_indices;
};

template <class ElemType>
using Sequence = std::vector<std::unique_ptr<Data<ElemType>>>;

// TODO: more details when tracing warnings 
// (e.g., buffer content around the char that triggered the warning)
template <class ElemType>
class TextParser : public DataDeserializerBase {
private:
    class TextDataChunk : public Chunk, public std::enable_shared_from_this<TextDataChunk> {
    public:
        TextDataChunk(const ChunkDescriptor& descriptor);

        // Gets sequences by id.
        std::vector<SequenceDataPtr> GetSequence(size_t sequenceId) override;

        std::map<size_t, std::vector<SequenceDataPtr>> m_sequencePtrMap;
        // Buffer to store the actual data.
        std::vector<Sequence<ElemType>> m_sequenceData;

        // chunk id (copied from the descriptor)
        size_t m_id; 
        // Keeps track of how many times GetSequence was called.
        // When this counter value reaches the number of sequences in 
        // the this chunk, it can be safely unloaded.
        size_t m_sequenceRequestCount;
    };

    typedef std::shared_ptr<TextDataChunk> TextChunkPtr;

    enum TraceLevel {
        Error = 0,
        Warning = 1,
        Info = 2
    };

    struct StreamInfo {
        StorageType m_type;
        size_t m_sampleDimension;
    };

    const std::wstring m_filename;
    FILE* m_file = nullptr;

    const size_t m_numberOfStreams;
    std::vector<StreamInfo> m_streamInfos;

    size_t m_maxAliasLength;
    std::map<std::string, size_t> m_aliasToIdMap;

    std::shared_ptr<Index> m_index = nullptr;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    char* m_bufferStart = nullptr;
    char* m_bufferEnd = nullptr;
    char* m_pos = nullptr; // buffer index

    char* m_scratch; // local buffer for string parsing

    size_t m_chunkSizeBytes = 0;
    unsigned int m_chunkCacheSize = 0; // number of chunks to keep in the memory
    unsigned int m_traceLevel = 0;
    unsigned int m_numAllowedErrors = 0;
    bool m_skipSequenceIds;
    
    // All streams this reader provides.
    std::vector<StreamDescriptionPtr> m_streams;

    // A map of currently loaded chunks
    std::map<size_t, TextChunkPtr> m_chunkCache;

    // throws runtime exception when number of parsing erros is 
    // greater than the specified threshold
    void IncrementNumberOfErrorsOrDie();

    void SetFileOffset(int64_t position);

    void SkipToNextValue(size_t& bytesToRead);
    void SkipToNextInput(size_t& bytesToRead);

    bool Fill();

    int64_t GetFileOffset() { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    // reads an alias/name and converts it to an internal stream id (= stream index).
    bool GetInputId(size_t& id, size_t& bytesToRead);

    bool ReadRealNumber(ElemType& value, size_t& bytesToRead);

    bool ReadUint64(size_t& index, size_t& bytesToRead);

    bool ReadDenseSample(std::vector<ElemType>& values, size_t sampleSize, size_t& bytesToRead);

    bool ReadSparseSample(std::vector<ElemType>& values, std::vector<size_t>& indices, size_t& bytesToRead);

    // read one whole row (terminated by a row delimiter) of samples
    bool ReadRow(Sequence<ElemType>& sequence, size_t& bytesToRead);

    bool inline CanRead() { return m_pos != m_bufferEnd || Fill(); }

    Sequence<ElemType> LoadSequence(bool verifyId, const SequenceDescriptor& descriptor);

    TextParser(const std::wstring& filename, const vector<StreamDescriptor>& streams);

    DISABLE_COPY_AND_MOVE(TextParser);
protected:
    void FillSequenceDescriptions(SequenceDescriptions& timeline) const override;

public:
    TextParser(const TextConfigHelper& helper);

    ~TextParser();

    // Builds an index of the input data.
    void Initialize();

    // Description of streams that this data deserializer provides.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(size_t chunkId) override;

    // Retrieves total number of chunks this deserializer can produce.
    size_t GetTotalNumberOfChunks() override;

    void SetTraceLevel(unsigned int traceLevel);

    void SetMaxAllowedErrors(unsigned int maxErrors);

    void SetSkipSequenceIds(bool skip);

    void SetChunkSize(size_t size);

    void SetChunkCacheSize(unsigned int size);
};
}}}
