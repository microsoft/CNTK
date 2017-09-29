//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "Index.h"
#include "TextParserInfo.h"

namespace CNTK {

template <class ElemType>
class TextDeserializer;

template <class ElemType>
class TextParser;

// A sequence buffer is a vector that contains sequence data for each input stream.
typedef std::vector<SequenceDataPtr> SequenceBuffer;

template <class ElemType>
class TextDataChunk : public Chunk, public std::enable_shared_from_this<Chunk>
{
public:
    explicit TextDataChunk(std::shared_ptr<TextParserInfo> parserInfo);

    // Gets sequences by id.
    void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override;

    // A map from sequence ids to the sequence data.
    std::vector<SequenceBuffer> m_sequenceMap;

    // A map from sequence ids to the the sequence descriptors.
    std::vector<SequenceDescriptor> m_sequenceDescriptors;

    // Chunk file offset in file
    size_t m_offsetInFile;

    std::shared_ptr<TextParserInfo> m_parserInfo;

    TextParser<ElemType> * m_parser;

    std::unique_ptr<char[]> m_buffer;
};


class BufferedFileReader;

// TODO: more details when tracing warnings
// (e.g., buffer content around the char that triggered the warning)
template <class ElemType>
class TextParser {
public:
    TextParser(std::shared_ptr<TextParserInfo> parserInfo, const char * bufferStart, const size_t& sequenceLength, const size_t& offsetInFile);

    void ParseSequence(SequenceBuffer& sequence, const SequenceDescriptor& sequenceDsc);

private:
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

    std::shared_ptr<TextParserInfo> m_parserInfo;

    bool m_hadWarnings;

    std::shared_ptr<BufferedReader> m_bufferedReader;

    unique_ptr<char[]> m_scratch; // local buffer for string parsing

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
    bool inline ShouldWarn() { m_hadWarnings = true; return m_parserInfo->m_traceLevel >= static_cast<unsigned int>(TraceLevel::Warning); }

    // Fills some metadata members to be conformant to the exposed SequenceData interface.
    void FillSequenceMetadata(SequenceBuffer& sequenceBuffer, const SequenceKey& sequenceKey);
};
}
