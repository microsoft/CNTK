//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "TextConfigHelper.h"
#include "Indexer.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CNTKTextFormatReaderTestRunner;

// TODO: more details when tracing warnings
// (e.g., buffer content around the char that triggered the warning)
template <class ElemType>
class TextParser : public DataDeserializerBase {
public:
    TextParser(CorpusDescriptorPtr corpus, const TextConfigHelper& helper, bool pimary);
    ~TextParser();

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Get information about chunks.
    ChunkDescriptions GetChunkDescriptions() override;

    // Get information about particular chunk.
    void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result) override;

    bool GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&) override;

private:
    TextParser(CorpusDescriptorPtr corpus, const std::wstring& filename, const vector<StreamDescriptor>& streams, bool primary = true);

    // Builds an index of the input data.
    void Initialize();

    struct DenseInputStreamBuffer : DenseSequenceData
    {
        // capacity = expected number of samples * sample size
        DenseInputStreamBuffer(size_t capacity)
        {
            m_buffer.reserve(capacity);
        }

        const void* GetDataBuffer() override
        {
            return m_buffer.data();
        }

        std::vector<ElemType> m_buffer;
    };

    // In case of sparse input, we also need a vector of
    // indices (one index for each input value) and a vector
    // of NNZ counts (one for each sample).
    struct SparseInputStreamBuffer : SparseSequenceData
    {
        SparseInputStreamBuffer()
        {
            m_totalNnzCount = 0;
        }

        const void* GetDataBuffer() override
        {
            return m_buffer.data();
        }

        std::vector<IndexType> m_indicesBuffer;
        std::vector<ElemType> m_buffer;
    };

    // A sequence buffer is a vector that contains sequence data for each input stream.
    typedef std::vector<SequenceDataPtr> SequenceBuffer;

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

    size_t m_fileOffsetStart;
    size_t m_fileOffsetEnd;

    // TODO: not DRY (same in the Indexer), needs refactoring
    unique_ptr<char[]> m_buffer;
    const char* m_bufferStart;
    const char* m_bufferEnd;
    const char* m_pos; // buffer index

    unique_ptr<char[]> m_scratch; // local buffer for string parsing

    size_t m_chunkSizeBytes;
    unsigned int m_traceLevel;
    bool m_hadWarnings;
    unsigned int m_numAllowedErrors;
    bool m_skipSequenceIds;
    unsigned int m_numRetries; // specifies the number of times an unsuccessful
                               // file operation should be repeated (default value is 5).

    // Corpus descriptor.
    CorpusDescriptorPtr m_corpus;

    // throws runtime exception when number of parsing errors is
    // greater than the specified threshold
    void IncrementNumberOfErrorsOrDie();

    // prints a messages that there were warnings which might
    // have been swallowed.
    void PrintWarningNotification();

    void SetFileOffset(int64_t position);

    void SkipToNextValue(size_t& bytesToRead);
    void SkipToNextInput(size_t& bytesToRead);

    bool TryRefillBuffer();

    int64_t GetFileOffset() const { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    // Returns a string containing input file information (current offset, file name, etc.),
    // which can be included as a part of the trace/log message.
    std::wstring GetFileInfo();

    // Reads an alias/name and converts it to an internal stream id (= stream index).
    bool TryGetInputId(size_t& id, size_t& bytesToRead);

    bool TryReadRealNumber(ElemType& value, size_t& bytesToRead);

    bool TryReadUint64(size_t& value, size_t& bytesToRead);

    // Reads dense sample values into the provided vector.
    bool TryReadDenseSample(std::vector<ElemType>& values, size_t sampleSize, size_t& bytesToRead);

    // Reads sparse sample values and corresponding indices into the provided vectors.
    bool TryReadSparseSample(std::vector<ElemType>& values, std::vector<IndexType>& indices,
        size_t sampleSize, size_t& bytesToRead);

    // Reads one sample (an input identifier followed by a list of values)
    bool TryReadSample(SequenceBuffer& sequence, size_t& bytesToRead);

    // Reads one whole row (terminated by a row delimiter) of samples
    bool TryReadRow(SequenceBuffer& sequence, size_t& bytesToRead);

    // Returns true if there's still data available.
    bool inline CanRead() { return m_pos != m_bufferEnd || TryRefillBuffer(); }

    // Returns true if the trace level is greater or equal to 'Warning'
    bool inline ShouldWarn() { m_hadWarnings = true; return m_traceLevel >= Warning; }

    // Given a descriptor and the file offset of the containing chunk,
    // retrieves the data for the corresponding sequence from the file.
    SequenceBuffer LoadSequence(const SequenceDescriptor& descriptor, size_t chunkOffset);

    // Given a descriptor, retrieves the data for the corresponding chunk from the file.
    void LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor);

    // Fills some metadata members to be conformant to the exposed SequenceData interface.
    void FillSequenceMetadata(SequenceBuffer& sequenceBuffer, const KeyType& sequenceKey);

    void SetTraceLevel(unsigned int traceLevel);

    void SetMaxAllowedErrors(unsigned int maxErrors);

    void SetSkipSequenceIds(bool skip);

    void SetChunkSize(size_t size);

    void SetNumRetries(unsigned int numRetries);

    friend class CNTKTextFormatReaderTestRunner<ElemType>;

    DISABLE_COPY_AND_MOVE(TextParser);
};
}}}
