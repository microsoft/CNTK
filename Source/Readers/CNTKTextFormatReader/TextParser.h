//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "TextConfigHelper.h"
#include "Index.h"
#include "CorpusDescriptor.h"

namespace CNTK {

template <class ElemType>
class CNTKTextFormatReaderTestRunner;

class FileWrapper;
class BufferedFileReader;

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
    std::vector<ChunkInfo> ChunkInfos() override;

    // Get information about particular chunk.
    void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override;

    bool GetSequenceInfoByKey(const SequenceKey&, SequenceInfo&) override;

private:
    TextParser(CorpusDescriptorPtr corpus, const std::wstring& filename, const vector<StreamDescriptor>& streams, bool primary = true);

    // Builds an index of the input data.
    void Initialize();

    struct DenseInputStreamBuffer : DenseSequenceData
    {
        // capacity = expected number of samples * sample size
        DenseInputStreamBuffer(size_t capacity, const NDShape& sampleShape) : m_sampleShape(sampleShape)
        {
            m_buffer.reserve(capacity);
        }

        const void* GetDataBuffer() override
        {
            return m_buffer.data();
        }

        const NDShape& GetSampleShape() override
        {
            return m_sampleShape;
        }

        const NDShape& m_sampleShape;
        std::vector<ElemType> m_buffer;
    };

    // In case of sparse input, we also need a vector of
    // indices (one index for each input value) and a vector
    // of NNZ counts (one for each sample).
    struct SparseInputStreamBuffer : SparseSequenceData
    {
        SparseInputStreamBuffer(const NDShape& sampleShape) : m_sampleShape(sampleShape)
        {
            m_totalNnzCount = 0;
        }

        const void* GetDataBuffer() override
        {
            return m_buffer.data();
        }

        const NDShape& GetSampleShape() override
        {
            return m_sampleShape;
        }

        const NDShape& m_sampleShape;
        std::vector<SparseIndexType> m_indicesBuffer;
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
    std::shared_ptr<FileWrapper> m_file;
    std::shared_ptr<BufferedFileReader> m_fileReader;

    // An internal structure to assist with copying from input stream buffers into
    // into sequence data in a proper format.
    struct StreamInfo;
    std::vector<StreamInfo> m_streamInfos;
    std::vector<StreamDescriptor> m_streamDescriptors;

    size_t m_maxAliasLength;
    std::map<std::string, size_t> m_aliasToIdMap;

    std::shared_ptr<Index> m_index;

    unique_ptr<char[]> m_scratch; // local buffer for string parsing

    // Indicates if the sequence length is computed as the maximum 
    // of number of samples across all streams (inputs).
    bool m_useMaximumAsSequenceLength;

    size_t m_chunkSizeBytes;
    unsigned int m_traceLevel;
    bool m_hadWarnings;
    unsigned int m_numAllowedErrors;
    bool m_skipSequenceIds;
    bool m_cacheIndex;
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

    int64_t GetFileOffset() const;

    void SkipToNextInput(size_t& bytesToRead);

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
    bool TryReadSparseSample(std::vector<ElemType>& values, std::vector<SparseIndexType>& indices,
        size_t sampleSize, size_t& bytesToRead);

    // Reads one sample (an input identifier followed by a list of values)
    bool TryReadSample(SequenceBuffer& sequence, size_t& bytesToRead);

    // Reads one whole row (terminated by a row delimiter) of samples
    bool TryReadRow(SequenceBuffer& sequence, size_t& bytesToRead);

    // Returns true if there's still data available.
    bool inline CanRead();

    // Returns true if the trace level is greater or equal to 'Warning'
    bool inline ShouldWarn() { m_hadWarnings = true; return m_traceLevel >= Warning; }

    // Given a descriptor and the file offset of the containing chunk,
    // retrieves the data for the corresponding sequence from the file.
    SequenceBuffer LoadSequence(const SequenceDescriptor& descriptor, size_t chunkOffset);

    // Given a descriptor, retrieves the data for the corresponding chunk from the file.
    void LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor);

    // Fills some metadata members to be conformant to the exposed SequenceData interface.
    void FillSequenceMetadata(SequenceBuffer& sequenceBuffer, const SequenceKey& sequenceKey);

    void SetTraceLevel(unsigned int traceLevel);

    void SetMaxAllowedErrors(unsigned int maxErrors);

    void SetSkipSequenceIds(bool skip);

    void SetChunkSize(size_t size);

    void SetNumRetries(unsigned int numRetries);

    void SetCacheIndex(bool value);

    friend class CNTKTextFormatReaderTestRunner<ElemType>;

    DISABLE_COPY_AND_MOVE(TextParser);
};
}
