//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "TextConfigHelper.h"
#include "Indexer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CNTKTextFormatReaderTestRunner;

// TODO: more details when tracing warnings 
// (e.g., buffer content around the char that triggered the warning)
template <class ElemType>
class TextParser : public DataDeserializerBase {
public:
    explicit TextParser(const TextConfigHelper& helper);

    ~TextParser();

    // Builds an index of the input data.
    void Initialize();

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(size_t chunkId) override;

    // Get information about chunks.
    ChunkDescriptions GetChunkDescriptions() override;

    // Get information about particular chunk.
    void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result) override;

private:
    // A buffer to keep data for all samples in a (variable length) sequence 
    // from a single input stream.
    struct InputStreamBuffer
    {
        virtual ~InputStreamBuffer() { };

        size_t m_numberOfSamples = 0;
        std::vector<ElemType> m_buffer;
    };

    struct DenseInputStreamBuffer : InputStreamBuffer
    {
        // capacity = expected number of samples * sample size
        DenseInputStreamBuffer(size_t capacity)
        {
            InputStreamBuffer::m_buffer.reserve(capacity);
        }
    };

    // In case of sparse input, we also need a vector of 
    // indices (one index for each input value) and a vector 
    // of NNZ counts (one for each sample).
    struct SparseInputStreamBuffer : InputStreamBuffer
    {
        IndexType m_totalNnzCount = 0;
        std::vector<IndexType> m_indices;
        std::vector<IndexType> m_nnzCounts;
    };

    // A sequence buffer is a vector that contains an input buffer for each input stream.
    typedef std::vector<std::unique_ptr<InputStreamBuffer>> SequenceBuffer;

    // A chunk of input data in the text format.
    class TextDataChunk;
    
    typedef std::shared_ptr<TextDataChunk> TextChunkPtr;

    enum TraceLevel 
    {
        Error = 0,
        Warning = 1,
        Info = 2
    };

    const std::wstring m_filename;
    FILE* m_file;

    // An internal structure to assist with copying from input stream buffers into
    // into sequence data in a proper format.
    struct StreamInfo;
    std::vector<StreamInfo> m_streamInfos;

    size_t m_maxAliasLength;
    std::map<std::string, size_t> m_aliasToIdMap;

    std::unique_ptr<Indexer> m_indexer;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    // TODO: not DRY (same in the Indexer), needs refactoring
    unique_ptr<char[]> m_buffer;
    const char* m_bufferStart;
    const char* m_bufferEnd;
    const char* m_pos; // buffer index

    unique_ptr<char[]> m_scratch; // local buffer for string parsing

    size_t m_chunkSizeBytes;
    unsigned int m_chunkCacheSize; // number of chunks to keep in the memory
    unsigned int m_traceLevel;
    unsigned int m_numAllowedErrors;
    bool m_skipSequenceIds;

    // A map of currently loaded chunks
    // TODO: remove caching once partial randomization is in master.
    std::map<size_t, TextChunkPtr> m_chunkCache;

    // throws runtime exception when number of parsing errors is 
    // greater than the specified threshold
    void IncrementNumberOfErrorsOrDie();

    void SetFileOffset(int64_t position);

    void SkipToNextValue(size_t& bytesToRead);
    void SkipToNextInput(size_t& bytesToRead);

    bool TryRefillBuffer();

    int64_t GetFileOffset() const { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    // Reads an alias/name and converts it to an internal stream id (= stream index).
    bool TryGetInputId(size_t& id, size_t& bytesToRead);

    bool TryReadRealNumber(ElemType& value, size_t& bytesToRead);

    bool TryReadUint64(size_t& value, size_t& bytesToRead);

    // Reads dense sample values into the provided vector.
    bool TryReadDenseSample(std::vector<ElemType>& values, size_t sampleSize, size_t& bytesToRead);

    // Reads sparse sample values and corresponging indices into the provided vectors.
    bool TryReadSparseSample(std::vector<ElemType>& values, std::vector<IndexType>& indices, size_t& bytesToRead);

    // Reads one whole row (terminated by a row delimiter) of samples
    bool TryReadRow(SequenceBuffer& sequence, size_t& bytesToRead);

    // Returns true if there's still data available.
    bool inline CanRead() { return m_pos != m_bufferEnd || TryRefillBuffer(); }

    // Given a descriptor, retrieves the data for the corresponging sequence from the file.
    SequenceBuffer LoadSequence(bool verifyId, const SequenceDescriptor& descriptor);

    // Given a descriptor, retrieves the data for the corresponging chunk from the file.
    void LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor);

    TextParser(const std::wstring& filename, const vector<StreamDescriptor>& streams);

    void SetTraceLevel(unsigned int traceLevel);

    void SetMaxAllowedErrors(unsigned int maxErrors);

    void SetSkipSequenceIds(bool skip);

    void SetChunkSize(size_t size);

    void SetChunkCacheSize(unsigned int size);

    friend class CNTKTextFormatReaderTestRunner<ElemType>;

    DISABLE_COPY_AND_MOVE(TextParser);
};
}}}
