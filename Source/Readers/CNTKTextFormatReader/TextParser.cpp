//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <cfloat>
#include "Indexer.h"
#include "TextParser.h"
#include "TextReaderConstants.h"

#define isSign(c) ((c == '-' || c == '+'))
#define isE(c) ((c == 'e' || c == 'E'))

namespace Microsoft { namespace MSR { namespace CNTK {

inline bool IsDigit(char c)
{
    return '0' <= c && c <= '9';
}

enum State
{
    Init = 0,
    Sign,
    IntegralPart,
    Period,
    FractionalPart,
    TheLetterE,
    ExponentSign,
    Exponent
};

template <class ElemType>
class TextParser<ElemType>::TextDataChunk : public Chunk, public std::enable_shared_from_this<Chunk>
{
public:
    explicit TextDataChunk(const ChunkDescriptor& descriptor, TextParser* parser);

    // Gets sequences by id.
    void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override;

    // A map from sequence ids to the sequence data.
    std::vector<SequenceBuffer> m_sequenceMap;

    // chunk id (copied from the descriptor)
    ChunkIdType m_id;

    // a non-owned pointer to the parser that created this chunk
    TextParser* m_parser;
};


template <class ElemType>
struct TextParser<ElemType>::StreamInfo
{
    StorageType m_type;
    size_t m_sampleDimension;
};

template <class ElemType>
TextParser<ElemType>::TextParser(CorpusDescriptorPtr corpus, const TextConfigHelper& helper, bool isPrimary) :
TextParser(corpus, helper.GetFilePath(), helper.GetStreams(), isPrimary)
{
    SetTraceLevel(helper.GetTraceLevel());
    SetMaxAllowedErrors(helper.GetMaxAllowedErrors());
    SetChunkSize(helper.GetChunkSize());
    SetSkipSequenceIds(helper.ShouldSkipSequenceIds());

    Initialize();
}

// Internal, used for testing.
template <class ElemType>
TextParser<ElemType>::TextParser(CorpusDescriptorPtr corpus, const std::wstring& filename, const vector<StreamDescriptor>& streams, bool isPrimary) :
    m_filename(filename),
    m_file(nullptr),
    m_streamInfos(streams.size()),
    m_indexer(nullptr),
    m_fileOffsetStart(0),
    m_fileOffsetEnd(0),
    m_buffer(new char[BUFFER_SIZE + 1]),
    m_bufferStart(nullptr),
    m_bufferEnd(nullptr),
    m_pos(nullptr),
    m_chunkSizeBytes(0),
    m_traceLevel(TraceLevel::Error),
    m_hadWarnings(false),
    m_numAllowedErrors(0),
    m_skipSequenceIds(false),
    m_numRetries(5),
    m_corpus(corpus),
    m_isPrimary(isPrimary)
{
    assert(streams.size() > 0);

    m_maxAliasLength = 0;

    for (size_t i = 0; i < streams.size(); ++i)
    {
        const StreamDescriptor& stream = streams[i];
        const string& alias = stream.m_alias;
        if (m_maxAliasLength < alias.length())
        {
            m_maxAliasLength = alias.length();
        }
        m_aliasToIdMap[alias] = i;
        m_streamInfos[i].m_type = stream.m_storageType;
        m_streamInfos[i].m_sampleDimension = stream.m_sampleDimension;

        auto streamDescription = std::make_shared<StreamDescription>(stream);
        streamDescription->m_sampleLayout = std::make_shared<TensorShape>(stream.m_sampleDimension);
        m_streams.push_back(streamDescription);
    }

    assert(m_maxAliasLength > 0);

    m_scratch = unique_ptr<char[]>(new char[m_maxAliasLength + 1]);
}

template <class ElemType>
TextParser<ElemType>::~TextParser()
{
    if (m_file)
    {
        fclose(m_file);
    }
}

template <class ElemType>
void TextParser<ElemType>::PrintWarningNotification()
{
    if (m_hadWarnings && m_traceLevel < Warning)
    {
        fprintf(stderr,
            "A number of warnings were generated while reading input data, "
            "to see them please set 'traceLevel' to a value greater or equal to %d.\n", Warning);
    }
}

template <class ElemType>
void TextParser<ElemType>::Initialize()
{
    if (m_indexer != nullptr)
    {
        return;
    }

    attempt(m_numRetries, [this]()
    {
        if (m_file == nullptr)
        {
            m_file = fopenOrDie(m_filename, L"rbS");
        }
        else if (ferror(m_file) != 0)
        {
            fclose(m_file);
            m_file = fopenOrDie(m_filename, L"rbS");
        }
        
        if (funicode(m_file))
        {
            // Retrying won't help here, the file is UTF-16 encoded.
            m_numRetries = 0;
            RuntimeError("Found a UTF-16 BOM at the beginning of the input file (%ls). "
                "UTF-16 encoding is currently not supported.", m_filename.c_str());
        }

        m_indexer = make_unique<Indexer>(m_file, m_isPrimary, m_skipSequenceIds, NAME_PREFIX, m_chunkSizeBytes);

        m_indexer->Build(m_corpus);
    });

    assert(m_indexer != nullptr);

    int64_t position = _ftelli64(m_file);
    if (position == -1L)
    {
        RuntimeError("Error retrieving current position in the input file (%ls).", m_filename.c_str());
    }

    m_fileOffsetStart = position;
    m_fileOffsetEnd = position;
}

template <class ElemType>
ChunkDescriptions TextParser<ElemType>::GetChunkDescriptions()
{
    assert(m_indexer != nullptr);

    const auto& index = m_indexer->GetIndex();

    ChunkDescriptions result;
    result.reserve(index.m_chunks.size());
    for (auto const& chunk : index.m_chunks)
    {
        result.push_back(shared_ptr<ChunkDescription>(
            new ChunkDescription {
                chunk.m_id,
                chunk.m_numberOfSamples,
                chunk.m_numberOfSequences
        }));
    }

    return result;
}

template <class ElemType>
void TextParser<ElemType>::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
{
    const auto& index = m_indexer->GetIndex();
    const auto& chunk = index.m_chunks[chunkId];
    result.reserve(chunk.m_sequences.size());

    for (auto const& s : chunk.m_sequences)
    {
        result.push_back(
        {
            s.m_id,
            s.m_numberOfSamples,
            s.m_chunkId,
            s.m_key
        });
    }
}

template <class ElemType>
TextParser<ElemType>::TextDataChunk::TextDataChunk(const ChunkDescriptor& descriptor, TextParser* parser) :
    m_parser(parser)
{
    m_id = descriptor.m_id;
}

template <class ElemType>
void TextParser<ElemType>::TextDataChunk::GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result)
{
    assert(sequenceId < m_sequenceMap.size());
    result.reserve(m_parser->m_streamInfos.size());

    const auto& sequenceData = m_sequenceMap[sequenceId];
    result.insert(result.end(), sequenceData.begin(), sequenceData.end());
}

template <class ElemType>
ChunkPtr TextParser<ElemType>::GetChunk(ChunkIdType chunkId)
{
    const auto& chunkDescriptor = m_indexer->GetIndex().m_chunks[chunkId];
    auto textChunk = make_shared<TextDataChunk>(chunkDescriptor, this);

    attempt(m_numRetries, [this, &textChunk, &chunkDescriptor]()
    {
        if (ferror(m_file) != 0)
        {
            fclose(m_file);
            m_file = fopenOrDie(m_filename, L"rbS");
        }
        LoadChunk(textChunk, chunkDescriptor);
    });

    return textChunk;
}

template <class ElemType>
void TextParser<ElemType>::LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor)
{
    chunk->m_sequenceMap.resize(descriptor.m_sequences.size());
    for (const auto& sequenceDescriptor : descriptor.m_sequences)
    {
        chunk->m_sequenceMap[sequenceDescriptor.m_id] = LoadSequence(sequenceDescriptor);
    }
}

template <class ElemType>
void TextParser<ElemType>::IncrementNumberOfErrorsOrDie()
{
    if (m_numAllowedErrors == 0)
    {
        PrintWarningNotification();
        RuntimeError("Reached the maximum number of allowed errors"
            " while reading the input file (%ls).",
            m_filename.c_str());
    }
    --m_numAllowedErrors;
}

template <class ElemType>
bool TextParser<ElemType>::TryRefillBuffer()
{
    size_t bytesRead = fread(m_buffer.get(), 1, BUFFER_SIZE, m_file);

    if (bytesRead == (size_t)-1)
    {
        PrintWarningNotification();
        RuntimeError("Could not read from the input file (%ls).", m_filename.c_str());
    }

    if (!bytesRead)
    {
        return false;
    }

    m_fileOffsetStart = m_fileOffsetEnd;
    m_fileOffsetEnd += bytesRead;
    m_bufferStart = m_buffer.get();
    m_pos = m_bufferStart;
    m_bufferEnd = m_bufferStart + bytesRead;
    return true;
}

template <class ElemType>
void TextParser<ElemType>::SetFileOffset(int64_t offset)
{
    int rc = _fseeki64(m_file, offset, SEEK_SET);
    if (rc)
    {
        PrintWarningNotification();
        RuntimeError("Error seeking to position %" PRId64 " in the input file (%ls).",
            offset, m_filename.c_str());
    }

    m_fileOffsetStart = offset;
    m_fileOffsetEnd = offset;

    TryRefillBuffer();
}

template <class ElemType>
typename TextParser<ElemType>::SequenceBuffer TextParser<ElemType>::LoadSequence(const SequenceDescriptor& sequenceDsc)
{
    auto fileOffset = sequenceDsc.m_fileOffsetBytes;

    if (fileOffset < m_fileOffsetStart || fileOffset > m_fileOffsetEnd)
    {
        SetFileOffset(fileOffset);
    }

    size_t bufferOffset = fileOffset - m_fileOffsetStart;
    m_pos = m_bufferStart + bufferOffset;
    size_t bytesToRead = sequenceDsc.m_byteSize;

    SequenceBuffer sequence;

    // TODO: reuse loaded sequences instead of creating new ones!
    for (auto const & stream : m_streamInfos)
    {
        if (stream.m_type == StorageType::dense)
        {
            sequence.push_back(make_unique<DenseInputStreamBuffer>(
                stream.m_sampleDimension * sequenceDsc.m_numberOfSamples));
        }
        else
        {
            sequence.push_back(make_unique<SparseInputStreamBuffer>());
        }
    }

    size_t numRowsRead = 0, expectedRowCount = sequenceDsc.m_numberOfSamples;
    for (size_t i = 0; i < expectedRowCount; i++)
    {
        if ((TryReadRow(sequence, bytesToRead)))
        {
            ++numRowsRead;
        }
        else
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Could not read a row (# %" PRIu64 ")"
                    " while loading sequence (id = %" PRIu64 ") %ls.\n",
                    i + 1,
                    sequenceDsc.m_key.m_sequence,
                    GetFileInfo().c_str());
            }
            IncrementNumberOfErrorsOrDie();
        }

        if (!bytesToRead && numRowsRead < expectedRowCount)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Exhausted all input"
                    " expected for the current sequence (id = %" PRIu64 ") %ls,"
                    " but only read %" PRIu64 " out of %" PRIu64 " expected rows.\n",
                    sequenceDsc.m_key.m_sequence,
                    GetFileInfo().c_str(), numRowsRead, expectedRowCount);
            }
            break;
        }
    }

    // Double check if there are empty input streams.
    // TODO this handling needs to be graceful, but currently CNTK complains when we return empty sequences.
    bool hasEmptyInputs = false, hasDuplicateInputs = false;
    uint32_t maxInputLength = 0;
    for (size_t i = 0; i < sequence.size(); ++i)
    {
        if (sequence[i]->m_numberOfSamples == 0)
        {
            fprintf(stderr,
                "ERROR: Input ('%ls') is empty in sequence (id = %" PRIu64 ") %ls.\n",
                m_streams[i]->m_name.c_str(), sequenceDsc.m_key.m_sequence, GetFileInfo().c_str());
            hasEmptyInputs = true;
        }

        if (sequence[i]->m_numberOfSamples > expectedRowCount)
        {
            hasDuplicateInputs = true;
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input ('%ls') contains more samples than expected"
                    " (%u vs. %" PRIu64 ") for sequence (id = %" PRIu64 ") %ls.\n",
                    m_streams[i]->m_name.c_str(), sequence[i]->m_numberOfSamples, expectedRowCount,
                    sequenceDsc.m_key.m_sequence, GetFileInfo().c_str());
            }
        }
        maxInputLength = max(sequence[i]->m_numberOfSamples, maxInputLength);
    }

    if (hasEmptyInputs)
    {
        PrintWarningNotification();
        RuntimeError("Malformed input file. Bailing out.");
    }

    if (hasDuplicateInputs)
    {
        IncrementNumberOfErrorsOrDie();
    }
    else if (maxInputLength < expectedRowCount)
    {
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Maximum per-input number of samples for sequence (id = %" PRIu64 ") %ls"
                " is less than expected (%u vs. %" PRIu64 ").\n",
                sequenceDsc.m_key.m_sequence,
                GetFileInfo().c_str(), maxInputLength, expectedRowCount);
        }
        IncrementNumberOfErrorsOrDie();
    }

    if (m_traceLevel >= Info)
    {
        fprintf(stderr,
            "INFO: Finished loading sequence (id = %" PRIu64 ") %ls,"
            " successfully read %" PRIu64 " out of expected %" PRIu64 " rows.\n",
            sequenceDsc.m_key.m_sequence, GetFileInfo().c_str(), numRowsRead, expectedRowCount);
    }

    FillSequenceMetadata(sequence, sequenceDsc.m_id);
    return sequence;
}

template<class ElemType>
void TextParser<ElemType>::FillSequenceMetadata(SequenceBuffer& sequenceData, size_t sequenceId)
{
    for (size_t j = 0; j < m_streamInfos.size(); ++j)
    {
        const StreamInfo& stream = m_streamInfos[j];
        SequenceDataBase* data = sequenceData[j].get();
        if (stream.m_type == StorageType::dense)
        {
            auto denseData = static_cast<DenseInputStreamBuffer*>(data);
            denseData->m_sampleLayout = m_streams[j]->m_sampleLayout;
        }
        else
        {
            auto sparseData = static_cast<SparseInputStreamBuffer*>(data);
            sparseData->m_indices = sparseData->m_indicesBuffer.data();
            assert(data->m_numberOfSamples == sparseData->m_nnzCounts.size());
        }

        data->m_id = sequenceId;
    }
}

template <class ElemType>
bool TextParser<ElemType>::TryReadRow(SequenceBuffer& sequence, size_t& bytesToRead)
{
    while (bytesToRead && CanRead() && IsDigit(*m_pos))
    {
        // skip sequence ids
        ++m_pos;
        --bytesToRead;
    }

    size_t numSampleRead = 0;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (c == ROW_DELIMITER)
        {
            // found the end of row, skip the delimiter, return.
            ++m_pos;
            --bytesToRead;

            if (numSampleRead == 0 && ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Empty input row %ls.\n", GetFileInfo().c_str());
            }
            else if (numSampleRead > m_streams.size() && ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input row %ls contains more"
                    " samples than expected (%" PRIu64 " vs. %" PRIu64 ").\n",
                    GetFileInfo().c_str(), numSampleRead, m_streams.size());
            }

            return numSampleRead > 0;
        }

        if (isColumnDelimiter(c))
        {
            // skip column (input) delimiters.
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (TryReadSample(sequence, bytesToRead))
        {
            numSampleRead++;
        }
        else
        {
            // skip over until the next sample/end of row
            SkipToNextInput(bytesToRead);
        }
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Exhausted all input expected for the current sequence"
            " while reading an input row %ls."
            " Possibly, a trailing newline is missing.\n", GetFileInfo().c_str());
    }

    // Return true when we've consumed all expected input.
    return bytesToRead == 0;
}

// Reads one sample (an pipe-prefixed input identifier followed by a list of values)
template <class ElemType>
bool TextParser<ElemType>::TryReadSample(SequenceBuffer& sequence, size_t& bytesToRead)
{
    assert(m_pos < m_bufferEnd);

    // prefix check.
    if (*m_pos != NAME_PREFIX)
    {
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Unexpected character('%c') in place of a name prefix ('%c')"
                " in an input name %ls.\n",
                *m_pos, NAME_PREFIX, GetFileInfo().c_str());
        }
        IncrementNumberOfErrorsOrDie();
        return false;
    }

    // skip name prefix
    ++m_pos;
    --bytesToRead;

    if (bytesToRead && CanRead() && *m_pos == ESCAPE_SYMBOL)
    {
        // A vertical bar followed by the number sign (|#) is treated as an escape sequence, 
        // everything that follows is ignored until the next vertical bar or the end of 
        // row, whichever comes first.
        ++m_pos;
        --bytesToRead;
        return false;
    }

    size_t id;
    if (!TryGetInputId(id, bytesToRead))
    {
        return false;
    }

    const StreamInfo& stream = m_streamInfos[id];

    if (stream.m_type == StorageType::dense)
    {
        DenseInputStreamBuffer* data = reinterpret_cast<DenseInputStreamBuffer*>(sequence[id].get());
        vector<ElemType>& values = data->m_buffer;
        size_t size = values.size();
        assert(size % stream.m_sampleDimension == 0);
        if (!TryReadDenseSample(values, stream.m_sampleDimension, bytesToRead))
        {
            // expected a dense sample, but was not able to fully read it, ignore it.
            if (values.size() != size)
            {
                //clean up the buffer
                values.resize(size);
            }
            IncrementNumberOfErrorsOrDie();
            return false;
        }
        // everything went well, increment the number of samples.
        ++data->m_numberOfSamples;
    }
    else
    {
        SparseInputStreamBuffer* data = reinterpret_cast<SparseInputStreamBuffer*>(sequence[id].get());
        vector<ElemType>& values = data->m_buffer;
        vector<IndexType>& indices = data->m_indicesBuffer;
        assert(values.size() == indices.size());
        size_t size = values.size();
        if (!TryReadSparseSample(values, indices, stream.m_sampleDimension, bytesToRead))
        {
            // expected a sparse sample, but something went south, ignore it.
            if (values.size() != size)
            {
                //clean up the buffer
                values.resize(size);
            }
            if (indices.size() != size)
            {
                //clean up the buffer
                indices.resize(size);
            }

            IncrementNumberOfErrorsOrDie();
            return false;
        }
        assert(values.size() == indices.size());
        ++data->m_numberOfSamples;
        IndexType count = static_cast<IndexType>(values.size() - size);
        data->m_nnzCounts.push_back(count);
        data->m_totalNnzCount += count;
    }

    return true;
}

template <class ElemType>
bool TextParser<ElemType>::TryGetInputId(size_t& id, size_t& bytesToRead)
{
    char* scratchIndex = m_scratch.get();

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        // stop as soon as there's a value delimiter, an input prefix
        // or a non-printable character (e.g., newline, carriage return).
        if (isValueDelimiter(c) || c == NAME_PREFIX || isNonPrintable(c))
        {
            size_t size = scratchIndex - m_scratch.get();
            if (size)
            {
                string name(m_scratch.get(), size);
                auto it = m_aliasToIdMap.find(name);
                if (it != m_aliasToIdMap.end())
                {
                    id = it->second;
                    return true;
                }

                if (m_traceLevel >= Info)
                {
                    fprintf(stderr,
                        "INFO: Skipping unknown input ('%s') %ls. "
                        "Input name '%s' was not specified in the reader config section.\n",
                        name.c_str(), GetFileInfo().c_str(), name.c_str());
                }

                // return false here to skip this input, but do not call IncrementNumberOfErrorsOrDie()
                return false;
            }
            
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input name prefix ('%c') is followed by"
                    " an invalid character ('%c') %ls.\n",
                    NAME_PREFIX, c, GetFileInfo().c_str());
            }

            break;
        }
        else if (scratchIndex < (m_scratch.get() + m_maxAliasLength))
        {
            *scratchIndex = c;
            ++scratchIndex;
        }
        else
        {
            // the current string length is already equal to the maximum expected length,
            // yet it's not followed by a delimiter.
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Did not find a valid input name %ls.\n",
                    GetFileInfo().c_str());
            }
            break;
        }

        ++m_pos;
        --bytesToRead;
    }

    if (ShouldWarn()) {
        if (bytesToRead == 0)
        {
            fprintf(stderr,
                "WARNING: Exhausted all input expected for the current sequence"
                " while reading an input name %ls.\n", GetFileInfo().c_str());
        }
        else if (!CanRead()) 
        {
            fprintf(stderr,
                "WARNING: Expected %" PRIu64 " more bytes, but no more input is available for the current sequence"
                " while reading an input name %ls.\n", bytesToRead, GetFileInfo().c_str());
        }
    }
    
    // Sequence ends with a dangling input id.
    IncrementNumberOfErrorsOrDie();
    return false;
}

template <class ElemType>
bool TextParser<ElemType>::TryReadDenseSample(vector<ElemType>& values, size_t sampleSize, size_t& bytesToRead)
{
    size_t counter = 0;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (isValueDelimiter(c))
        {
            // skip value delimiters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        // return as soon as we hit a non-printable or a name prefix
        if (isNonPrintable(c) || c == NAME_PREFIX)
        {
            if (counter > sampleSize)
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: Dense sample (size = %" PRIu64 ") %ls"
                        " exceeds the expected size (%" PRIu64 ").\n",
                        counter, GetFileInfo().c_str(), sampleSize);
                }
                return false;
            }

            // For dense matrices, it should be possible to input only the left part
            // if the suffix is sparse. Fill up the rest with zeros.
            if (counter < sampleSize)
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: A dense sample %ls has a sparse suffix "
                        "(expected size = %" PRIu64 ", actual size = %" PRIu64 ").\n",
                        GetFileInfo().c_str(), sampleSize, counter);
                }
                for (; counter < sampleSize; ++counter)
                {
                    values.push_back(0.0f);
                }
            }

            return true;
        }

        if (!TryReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        ++counter;
    }

    if (ShouldWarn())
    {
        if (bytesToRead == 0)
        {
            fprintf(stderr,
                "WARNING: Exhausted all input expected for the current sequence"
                " while reading a dense sample %ls.\n", GetFileInfo().c_str());
        }
        else if (!CanRead())
        {
            fprintf(stderr,
                "WARNING: Expected %" PRIu64 " more bytes, but no more input is available for the current sequence"
                " while reading a dense sample %ls.\n", bytesToRead, GetFileInfo().c_str());
        }
    }

    // If we've consumed all expected input, return true when we've successfully read
    // at least a single value
    return bytesToRead > 0 || counter > 0;
}

template <class ElemType>
bool TextParser<ElemType>::TryReadSparseSample(std::vector<ElemType>& values, std::vector<IndexType>& indices,
    size_t sampleSize, size_t& bytesToRead)
{
    size_t index = 0;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (isValueDelimiter(c))
        {
            // skip value delimiters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        // return as soon as we hit a non-printable or a name prefix
        if (isNonPrintable(c) || c == NAME_PREFIX)
        {
            // empty sparse samples are allowed ("|InputeName_1|InputName2...")
            return true;
        }

        // read next sparse index
        if (!TryReadUint64(index, bytesToRead))
        {
            // bail out.
            return false;
        }

        if (index >= sampleSize)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Sparse index value (%" PRIu64 ") %ls"
                    " exceeds the maximum expected value (%" PRIu64 ").\n",
                    index, GetFileInfo().c_str(), sampleSize - 1);
            }
            // bail out.
            return false;
        }

        // an index must be followed by a delimiter
        c = *m_pos;
        if (c != INDEX_DELIMITER)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Unexpected character('%c')"
                    " in place of the index delimiter ('%c')"
                    " after a sparse value index (%" PRIu64 ") %ls.\n",
                    c, INDEX_DELIMITER, index, GetFileInfo().c_str());
            }
            return false;
        }

        // skip index delimiter
        ++m_pos;
        --bytesToRead;

        // read the corresponding value
        if (!TryReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        indices.push_back(static_cast<IndexType>(index));
    }

    if (ShouldWarn())
    { 
        if (bytesToRead == 0)
        {
            fprintf(stderr,
                "WARNING: Exhausted all input expected for the current sequence"
                " while reading a sparse sample %ls.\n", GetFileInfo().c_str());
        }
        else if (!CanRead())
        {
            fprintf(stderr,
                "WARNING: Expected %" PRIu64 " more bytes, but no more input is available for the current sequence"
                " while reading a sparse sample %ls.\n", bytesToRead, GetFileInfo().c_str());
        }
    }

    // If we've consumed all expected input, return true when we've successfully read
    // at least a single value
    return bytesToRead > 0 || values.size() > 0;
}

template <class ElemType>
void TextParser<ElemType>::SkipToNextValue(size_t& bytesToRead)
{
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;
        // skip everything until we hit either a value delimiter, an input marker or the end of row.
        if (isValueDelimiter(c) || c == NAME_PREFIX || c == ROW_DELIMITER)
        {
            return;
        }
        ++m_pos;
        --bytesToRead;
    }
}

template <class ElemType>
void TextParser<ElemType>::SkipToNextInput(size_t& bytesToRead)
{
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;
        // skip everything until we hit either an input marker or the end of row.
        if (c == NAME_PREFIX || c == ROW_DELIMITER)
        {
            return;
        }
        ++m_pos;
        --bytesToRead;
    }
}

template <class ElemType>
bool TextParser<ElemType>::TryReadUint64(size_t& value, size_t& bytesToRead)
{
    value = 0;
    bool found = false;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (!IsDigit(c))
        {
            if (!found && ShouldWarn()) 
            {
                fprintf(stderr,
                    "WARNING: Expected a uint64 value, but none found %ls.\n", 
                    GetFileInfo().c_str());
            }

            return found;
        }

        found |= true;

        size_t temp = value;
        value = value * 10 + (c - '0');
        if (temp > value)
        {
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Overflow while reading a uint64 value %ls.\n",
                    GetFileInfo().c_str());
            }

            return false;
        }

        ++m_pos;
        --bytesToRead;
    }

    if (ShouldWarn())
    {
        if (bytesToRead == 0) {
            fprintf(stderr,
                "WARNING: Exhausted all input expected for the current sequence"
                " while reading a uint64 value %ls.\n", GetFileInfo().c_str());
        }
        else if (!CanRead())
        {
            fprintf(stderr,
                "WARNING: Expected %" PRIu64 " more bytes, but no more input is available for the current sequence"
                " while reading a uint64 value %ls.\n", bytesToRead, GetFileInfo().c_str());
        }
        
    }
    
    // A well-formed input cannot end with a uint64 value.
    return false;
}



// TODO: better precision (at the moment we're at parity with UCIFast)?
// Assumes that bytesToRead is greater than the number of characters 
// in the string representation of the floating point number
// (i.e., the string is followed by one of the delimiters)
// Post condition: m_pos points to the first character that 
// cannot be parsed as part of a floating point number.
// Returns true if parsing was successful.
template <class ElemType>
bool TextParser<ElemType>::TryReadRealNumber(ElemType& value, size_t& bytesToRead)
{
    State state = State::Init;
    double coefficient = .0, number = .0, divider = .0;
    bool negative = false;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        switch (state)
        {
        case State::Init:
            // the number must either start with a number or a sign
            if (IsDigit(c))
            {
                state = IntegralPart;
                number = (c - '0');
            }
            else if (isSign(c))
            {
                state = Sign;
                negative = (c == '-');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: Unexpected character ('%c')"
                        " in a floating point value %ls.\n",
                        c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case Sign:
            // the sign must be followed by a number
            if (IsDigit(c))
            {
                state = IntegralPart;
                number = (c - '0');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: A sign symbol is followed by an invalid character('%c')"
                        " in a floating point value %ls.\n",
                        c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case IntegralPart:
            if (IsDigit(c))
            {
                number = number * 10 + (c - '0');
            }
            else if (c == '.')
            {
                state = Period;
            }
            else if (isE(c))
            {
                state = TheLetterE;
                coefficient = (negative) ? -number : number;
                number = 0;
            }
            else
            {
                value = static_cast<ElemType>((negative) ? -number : number);
                return true;
            }
            break;
        case Period:
            if (IsDigit(c))
            {
                state = FractionalPart;
                coefficient = number;
                number = (c - '0');
                divider = 10;
            }
            else
            {
                value = static_cast<ElemType>((negative) ? -number : number);
                return true;
            }
            break;
        case FractionalPart:
            if (IsDigit(c))
            {
                // TODO: ignore if number of precision digits > FLT_[MANT_]DIG/DBL_[MANT_]DIG
                // no state change
                number = number * 10 + (c - '0');
                divider *= 10;
            }
            else if (isE(c))
            {
                state = TheLetterE;
                coefficient += (number / divider);
                if (negative)
                {
                    coefficient = -coefficient;
                }
            }
            else
            {
                coefficient += (number / divider);
                value = static_cast<ElemType>((negative) ? -coefficient : coefficient);
                return true;
            }
            break;
        case TheLetterE:
            // followed with optional minus or plus sign and nonempty sequence of decimal digits
            if (IsDigit(c))
            {
                state = Exponent;
                negative = false;
                number = (c - '0');
            }
            else if (isSign(c))
            {
                state = ExponentSign;
                negative = (c == '-');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: An exponent symbol is followed by"
                        " an invalid character('%c')"
                        " in a floating point value %ls.\n", c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case ExponentSign:
            // exponent sign must be followed by a number
            if (IsDigit(c))
            {
                state = Exponent;
                number = (c - '0');
            }
            else
            {
                if (ShouldWarn())
                {
                    fprintf(stderr,
                        "WARNING: An exponent sign symbol followed by"
                        " an unexpected character('%c')"
                        " in a floating point value %ls.\n", c, GetFileInfo().c_str());
                }
                return false;
            }
            break;
        case Exponent:
            if (IsDigit(c))
            {
                // no state change
                number = number * 10 + (c - '0');
            }
            else
            {
                // TODO: check the exponent value (see FLT_[MAX/MIN]_10_EXP).
                double exponent = (negative) ? -number : number;
                value = static_cast<ElemType>(coefficient * pow(10.0, exponent));
                return true;
            }
            break;
        default:
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Reached an invalid state while reading a floating point value %ls.\n",
                    GetFileInfo().c_str());
            }
            return false;
        }

        ++m_pos;
        --bytesToRead;
    }

    // We've run out of input, see if we're in a valid state
    if (bytesToRead == 0)
    {
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Exhausted all input expected for the current sequence"
                " while reading an input row %ls."
                " Possibly, a trailing newline is missing.\n", GetFileInfo().c_str());
        }

        switch (state)
        {
        case IntegralPart:
        case Period:
            value = static_cast<ElemType>((negative) ? -number : number);
            return true;
        case FractionalPart:
            coefficient += (number / divider);
            value = static_cast<ElemType>((negative) ? -coefficient : coefficient);
            return true;
        case Exponent:
            double exponent = (negative) ? -number : number;
            value = static_cast<ElemType>(coefficient * pow(10.0, exponent));
            return true;
        }

        // The floating point number we're reading is malformed.
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Reached an invalid state while reading a floating point value %ls.\n",
                GetFileInfo().c_str());
        }
        return false;
    }

    if (ShouldWarn())
    {
        fprintf(stderr,
            "WARNING: Expected %" PRIu64 " more bytes, but no more input is available for the current sequence"
            " while reading an input row %ls.\n", bytesToRead, GetFileInfo().c_str());
    }

    return false;
}

template <class ElemType>
void TextParser<ElemType>::SetTraceLevel(unsigned int traceLevel)
{
    m_traceLevel = traceLevel;
}

template <class ElemType>
void TextParser<ElemType>::SetMaxAllowedErrors(unsigned int maxErrors)
{
    m_numAllowedErrors = maxErrors;
}

template <class ElemType>
void TextParser<ElemType>::SetSkipSequenceIds(bool skip)
{
    m_skipSequenceIds = skip;
}

template <class ElemType>
void TextParser<ElemType>::SetChunkSize(size_t size)
{
    m_chunkSizeBytes = size;
}

template <class ElemType>
void TextParser<ElemType>::SetNumRetries(unsigned int numRetries)
{
    m_numRetries = numRetries;
}

template <class ElemType>
std::wstring TextParser<ElemType>::GetFileInfo()
{
    std::wstringstream info;
    info << L"at offset " << GetFileOffset() << L" in the input file (" << m_filename << L")";
    return info.str();
}

template <class ElemType>
bool TextParser<ElemType>::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
{
    if (m_isPrimary)
        LogicError("Matching by sequence key is not supported for primary deserilalizer.");

    const auto& keys = m_indexer->GetIndex().m_keyToSequenceInChunk;
    auto sequenceLocation = keys.find(key.m_sequence);
    if (sequenceLocation == keys.end())
    {
        return false;
    }

    result = m_indexer->GetIndex().m_chunks[sequenceLocation->second.first].m_sequences[sequenceLocation->second.second];
    return true;
}

template class TextParser<float>;
template class TextParser<double>;
}}}
