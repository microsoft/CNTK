//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <inttypes.h>
#include <cfloat>
#include <limits>
#include "Indexer.h"
#include "TextParser.h"
#include "TextReaderConstants.h"

#undef max // max is defined in minwindef.h
#define isSign(c) ((c == '-' || c == '+'))
#define isE(c) ((c == 'e' || c == 'E'))

namespace Microsoft { namespace MSR { namespace CNTK {

enum State {
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
class TextParser<ElemType>::TextDataChunk : public Chunk, public std::enable_shared_from_this<TextDataChunk>
{
public:
    explicit TextDataChunk(const ChunkDescriptor& descriptor);

    // Gets sequences by id.
    void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override;

    std::map<size_t, std::vector<SequenceDataPtr>> m_sequencePtrMap;

    // Buffer to store the actual data.
    std::vector<SequenceBuffer> m_sequences;

    // chunk id (copied from the descriptor)
    size_t m_id;
    // Keeps track of how many times GetSequence was called.
    // When this counter value reaches the number of sequences in 
    // the this chunk, it can be safely unloaded.
    size_t m_sequenceRequestCount;
};


template <class ElemType>
struct TextParser<ElemType>::StreamInfo{
    StorageType m_type;
    size_t m_sampleDimension;
};

template <class ElemType>
TextParser<ElemType>::TextParser(const TextConfigHelper& helper)
    : TextParser(helper.GetFilePath(), helper.GetStreams())
{
    SetTraceLevel(helper.GetTraceLevel());
    SetMaxAllowedErrors(helper.GetMaxAllowedErrors());
    SetChunkCacheSize(helper.GetNumChunksToCache());
    SetChunkSize(helper.GetChunkSize());
    SetSkipSequenceIds(helper.ShouldSkipSequenceIds());

    Initialize();
}

template <class ElemType>
TextParser<ElemType>::TextParser(const std::wstring& filename, const vector<StreamDescriptor>& streams) : 
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
    m_chunkCacheSize(0),
    m_traceLevel(TraceLevel::Error),
    m_numAllowedErrors(0),
    m_skipSequenceIds(false)
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
void TextParser<ElemType>::Initialize()
{
    if (m_indexer != nullptr) 
    {
        return;
    }

    attempt(5, [this]()
    {
        m_file = fopenOrDie(m_filename, L"rbS");
    });
    
    if (funicode(m_file))
    {
        RuntimeError("Found a UTF-16 BOM at the beginning of the input file %ls. "
            "UTF-16 encoding is currently not supported.", m_filename.c_str());
    }

    m_indexer = make_unique<Indexer>(m_file, m_skipSequenceIds, m_chunkSizeBytes);

    attempt(5, [this]()
    {
        m_indexer->Build();
    });

    // it's still possible that the actual input data does not have sequence id column.
    m_skipSequenceIds = !m_indexer->HasSequenceIds();

    assert(m_indexer != nullptr);

    int64_t position = _ftelli64(m_file);
    if (position == -1L)
    {
        RuntimeError("Error retrieving file position in file %ls", m_filename.c_str());
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
    result.reserve(index.size());
    for (auto const& chunk : index)
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
void TextParser<ElemType>::GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result)
{
    const auto& index = m_indexer->GetIndex();
    const auto& chunk = index[chunkId];
    result.reserve(chunk.m_sequences.size());

    for (auto const& s : chunk.m_sequences)
    {
        result.push_back(
        {
            s.m_id, 
            s.m_numberOfSamples,
            s.m_chunkId,
            s.m_isValid,
            s.m_key
        });
    }
}

template <class ElemType>
TextParser<ElemType>::TextDataChunk::TextDataChunk(const ChunkDescriptor& descriptor)
{
    m_id = descriptor.m_id;
    m_sequenceRequestCount = 0;
    m_sequences.reserve(descriptor.m_numberOfSequences);
}

template <class ElemType>
void TextParser<ElemType>::TextDataChunk::GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result)
{
    auto it = m_sequencePtrMap.find(sequenceId);
    assert(it != m_sequencePtrMap.end());
//TODO: Remove pragma once new randomizer is in master.
#pragma omp atomic
    ++m_sequenceRequestCount;
    result.reserve(it->second.size());
    copy(it->second.begin(), it->second.end(), back_inserter(result));
}

template <class ElemType>
ChunkPtr TextParser<ElemType>::GetChunk(size_t chunkId)
{
    ChunkPtr chunk;
    //TODO: Remove pragma once new randomizer is in master.
#pragma omp critical
    {
        auto it = m_chunkCache.find(chunkId);
        if (it != m_chunkCache.end())
        {
            chunk = it->second;
        }
        else
        {
            const auto& chunkDescriptor = m_indexer->GetIndex()[chunkId];
            auto textChunk = make_shared<TextDataChunk>(chunkDescriptor);

            attempt(5, [this, &textChunk, &chunkDescriptor]()
            {
                LoadChunk(textChunk, chunkDescriptor);
            });

            if (m_chunkCacheSize > 0 && m_chunkCache.size() == m_chunkCacheSize)
            {
                size_t candidateId = SIZE_MAX;
                size_t minNumSequencesLeft = SIZE_MAX;
                for (const auto& it : m_chunkCache)
                {
                    const auto& chunk = *(it.second.get());
                    size_t numSequencesUsed = 0;
#pragma omp atomic
                    numSequencesUsed += chunk.m_sequenceRequestCount;
                    size_t numSequencesLeft = chunk.m_sequences.size() - numSequencesUsed;
                    if (numSequencesLeft < minNumSequencesLeft)
                    {
                        minNumSequencesLeft = numSequencesLeft;
                        candidateId = it.first;
                    }
                }
                assert(candidateId != SIZE_MAX);
                m_chunkCache.erase(candidateId);
            }

            if (m_chunkCacheSize > 0)
            {
                m_chunkCache[chunkId] = textChunk;
            }

            chunk = textChunk;
        }
    }
    return chunk;
}

template <class ElemType>
void TextParser<ElemType>::LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor)
{
    for (const auto& sequenceDescriptor : descriptor.m_sequences)
    {
        chunk->m_sequences.push_back(LoadSequence(!m_skipSequenceIds, sequenceDescriptor));
        const auto& sequenceData = chunk->m_sequences.back();
        vector<SequenceDataPtr> sequencePtrs(m_streamInfos.size());
        for (size_t j = 0; j < m_streamInfos.size(); ++j)
        {
            const StreamInfo& stream = m_streamInfos[j];
            if (stream.m_type == StorageType::dense)
            {
                DenseInputStreamBuffer* input = reinterpret_cast<DenseInputStreamBuffer*>(sequenceData[j].get());
                auto data = make_shared<DenseSequenceData>();
                data->m_data = input->m_buffer.data();
                data->m_sampleLayout = m_streams[j]->m_sampleLayout;
                data->m_numberOfSamples = input->m_numberOfSamples;
                data->m_chunk = chunk;
                data->m_id = sequenceDescriptor.m_id;
                sequencePtrs[j] = data;
            }
            else
            {
                SparseInputStreamBuffer* input = reinterpret_cast<SparseInputStreamBuffer*>(sequenceData[j].get());
                auto data = make_shared<SparseSequenceData>();
                data->m_data = input->m_buffer.data();
                data->m_indices = input->m_indices.data();
                data->m_nnzCounts.reserve(input->m_nnzCounts.size());
                copy(input->m_nnzCounts.begin(), input->m_nnzCounts.end(), back_inserter(data->m_nnzCounts));
                data->m_totalNnzCount = input->m_totalNnzCount;
                data->m_chunk = chunk;
                data->m_id = sequenceDescriptor.m_id;
                data->m_numberOfSamples = input->m_nnzCounts.size();
                sequencePtrs[j] = data;
            }
        }
        chunk->m_sequencePtrMap[sequenceDescriptor.m_id] = move(sequencePtrs);
    }
}

template <class ElemType>
void TextParser<ElemType>::IncrementNumberOfErrorsOrDie() 
{
    if (m_numAllowedErrors == 0)
    {
        RuntimeError("Reached maximum allowed number of reader errors");
    }
    --m_numAllowedErrors;
}

template <class ElemType>
bool TextParser<ElemType>::RefillBuffer()
{
    size_t bytesRead = fread(m_buffer.get(), 1, BUFFER_SIZE, m_file);
    
    if (bytesRead == (size_t)-1)
        RuntimeError("Could not read from the input file %ls", m_filename.c_str());

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
        RuntimeError("Error seeking to position %" PRId64 " in file %ls", offset, m_filename.c_str());
    }

    m_fileOffsetStart = offset;
    m_fileOffsetEnd = offset;

    RefillBuffer();
}

template <class ElemType>
typename TextParser<ElemType>::SequenceBuffer TextParser<ElemType>::LoadSequence(bool verifyId, const SequenceDescriptor& sequenceDsc)
{
    auto fileOffset = sequenceDsc.m_fileOffsetBytes;

    if (fileOffset < m_fileOffsetStart || fileOffset > m_fileOffsetEnd)
    {
        SetFileOffset(fileOffset);
    }

    size_t bufferOffset = fileOffset - m_fileOffsetStart;
    m_pos = m_bufferStart + bufferOffset;
    size_t bytesToRead = sequenceDsc.m_byteSize;

    if (verifyId) 
    {
        size_t id;
        if (!ReadUint64(id, bytesToRead) || id != sequenceDsc.m_id) 
        {
            RuntimeError("Did not find the expected sequence id ( %" PRIu64 ") "
                " at the file offset = %" PRId64 "\n", sequenceDsc.m_id, GetFileOffset());
        }
    }

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
        if ((ReadRow(sequence, bytesToRead)))
        {
            ++numRowsRead;
        }
        else 
        {
            IncrementNumberOfErrorsOrDie();
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: could not read a row (# %" PRIu64 ")"
                    " while loading sequence (id = %" PRIu64 ")"
                    " at the offset = %" PRId64 "\n", i, sequenceDsc.m_id, GetFileOffset());
            }
        } 

        if (!bytesToRead && numRowsRead < expectedRowCount)
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: exhausted all expected input"
                    " expected for the current sequence (id = %" PRIu64 ")"
                    " at the offset = %" PRId64 "\n", sequenceDsc.m_id, GetFileOffset());
            }
            break;
        }
    }

    if (m_traceLevel >= Info)
    {
        fprintf(stderr,
            "INFO: finished loading sequence (id = %" PRIu64 "),"
            " successfully read %" PRIu64 " out of expected %" PRIu64 " rows\n",
            sequenceDsc.m_id, numRowsRead, expectedRowCount);
    }

    // Double check if there are empty input streams.
    // TODO this handling needs to be graceful, but currently CNTK complains when we return empty sequences.
    bool hasEmptyInputs = false;

    for (size_t i = 0; i < sequence.size(); ++i)
    {
        if (sequence[i]->m_numberOfSamples == 0)
        {
            fprintf(stderr,
                "ERROR: While reading input %" PRIu64 ""
                " in sequence id = %" PRIu64 
                " at file offset = %" PRId64 ": Input is empty.\n", i + 1, sequenceDsc.m_id, GetFileOffset());
            hasEmptyInputs = true;
        }
    }

    if (hasEmptyInputs)
    {
        RuntimeError("Could not read input file. Bailing out.");
    }

    return sequence;
}

template <class ElemType>
bool TextParser<ElemType>::ReadRow(SequenceBuffer& sequence, size_t& bytesToRead)
{
    bool found = false;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (isdigit(c) || c == COLUMN_DELIMITER || c == CARRIAGE_RETURN)
        {
            // skip sequence ids, column separators and CRs.
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (c == ROW_DELIMITER)
        {
            // found the end of row, skip the delimiter, return.
            ++m_pos;
            --bytesToRead;
            return found;
        }

        size_t id;
        if (!GetInputId(id, bytesToRead))
        {
            IncrementNumberOfErrorsOrDie();
            SkipToNextInput(bytesToRead);
            continue;
        }

        const StreamInfo& stream = m_streamInfos[id];

        if (stream.m_type == StorageType::dense)
        {
            DenseInputStreamBuffer* data = reinterpret_cast<DenseInputStreamBuffer*>(sequence[id].get());
            vector<ElemType>& values = data->m_buffer;
            size_t size = values.size();
            assert(size % stream.m_sampleDimension == 0);
            if (!ReadDenseSample(values, stream.m_sampleDimension, bytesToRead))
            {
                // expected a dense sample, but was not able to fully read it, ignore it.
                if (values.size() != size)
                {
                    //clean up the buffer
                    values.resize(size);
                }
                IncrementNumberOfErrorsOrDie();
                SkipToNextInput(bytesToRead);
                continue;
            }
            // everything went well, increment the number of samples.
            ++data->m_numberOfSamples;
        }
        else
        {
            SparseInputStreamBuffer* data = reinterpret_cast<SparseInputStreamBuffer*>(sequence[id].get());
            vector<ElemType>& values = data->m_buffer;
            vector<IndexType>& indices = data->m_indices;
            assert(values.size() == indices.size());
            size_t size = values.size();
            if (!ReadSparseSample(values, indices, bytesToRead))
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
                SkipToNextInput(bytesToRead);
                continue;
            }
            assert(values.size() == indices.size());
            ++data->m_numberOfSamples;
            IndexType count = static_cast<IndexType>(values.size() - size);
            data->m_nnzCounts.push_back(count);
            data->m_totalNnzCount += count;
        }

        found |= true;
    }

    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhausted all input expected for the current sequence"
            " while reading an input row"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}

template <class ElemType>
bool TextParser<ElemType>::GetInputId(size_t& id, size_t& bytesToRead)
{
    char* scratchIndex = m_scratch.get();

    char c = *m_pos;

    // prefix check.
    if (c != NAME_PREFIX)
    {
        if (m_traceLevel >= Warning)
        {
            fprintf(stderr,
                "WARNING: unexpected character('%c') in place of a name prefix ('%c')"
                " while reading an input name"
                " at the offset = %" PRId64 "\n", c, NAME_PREFIX, GetFileOffset());
        }
        return false;
    }

    // skip name prefix
    ++m_pos;
    --bytesToRead;

    while (bytesToRead && CanRead())
    {
        c = *m_pos;

        // an input id can be followed by a value marker, end of line (also, carriage return),
        // column separator or the name prefix of the following input.
        if (c <= VALUE_DELIMITER || c == NAME_PREFIX)
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
                else if (m_traceLevel >= Warning)
                {
                    fprintf(stderr,
                        "WARNING: an invalid input name ('%s')"
                        " while reading an input name"
                        " at the offset = %" PRId64 "\n", name.c_str(), GetFileOffset());
                }
            }
            else if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: a name prefix is immediately  followed by a delimiter"
                    " while reading an input name"
                    " at the offset = %" PRId64 "\n", GetFileOffset());
            }

            return false;
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
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: was not able to find an input name"
                    " at the offset = %" PRId64 "\n", GetFileOffset());
            }
            return false;
        }

        ++m_pos;
        --bytesToRead;
    }

    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhausted all input expected for the current sequence"
            " while reading an input name"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}

template <class ElemType>
bool TextParser<ElemType>::ReadDenseSample(vector<ElemType>& values, size_t sampleSize, size_t& bytesToRead)
{
    size_t counter = 0;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        // return as soon as we hit a non-printable or a name prefix
        if (c < VALUE_DELIMITER || c == NAME_PREFIX)
        {
            // TODO: 
            // if (counter > sampleSize) -- drop the extra elements?
            // increment the number of errors by the diff(sampleSize and counter)
            // and return true?
            // what if counter == 0?

            if (counter > sampleSize)
            {
                RuntimeError("Encountered a sample (size = %" PRId64 ")"
                    " exceeding the expected size of %" PRId64 " at the offset  %" PRId64,
                    counter, sampleSize, GetFileOffset());
            }

            while (counter < sampleSize)
            {
                // For dense matrices, it should be possible to input only the left part
                // if the suffix is sparse. Fill up the rest with zeros.
                values.push_back(0.0f);
                ++counter;
            }

            return true;
        }

        if (c == VALUE_DELIMITER)
        {
            // skip value delimiters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (!ReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        ++counter;
    }

    IncrementNumberOfErrorsOrDie();
    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhausted all input expected for the current sequence"
            " while reading a dense sample"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}

template <class ElemType>
bool TextParser<ElemType>::ReadSparseSample(std::vector<ElemType>& values, std::vector<IndexType>& indices, size_t& bytesToRead)
{
    size_t index;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        // return as soon as we hit a non-printable or a name prefix
        if (c < VALUE_DELIMITER || c == NAME_PREFIX)
        {
            // empty sparse samples are allowed ("|InputeName_1|InputName2...")
            return true;
        }

        if (c == VALUE_DELIMITER)
        {
            // skip value delimiters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        // read next sparse index
        if (!ReadUint64(index, bytesToRead))
        {
            // bail out.
            return false;
        } 
        if (index > numeric_limits<IndexType>::max())
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: sparse index value(%" PRIu64 ") exceeds the maximum allowed "
                    " value (%" PRIu64 ")\n", index, (size_t)numeric_limits<IndexType>::max());
            }
            // bail out.
            return false;
        }

        // an index must be followed by a delimiter
        c = *m_pos;
        if (c == INDEX_DELIMITER)
        {
            // consume index delimiter
            ++m_pos;
            --bytesToRead;
        }
        else
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: unexpected character('%c') after a sparse value index"
                    " at the offset = %" PRId64 "\n", c, GetFileOffset());
            }
            return false;
        }

        // read the corresponding value
        if (!ReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        indices.push_back(static_cast<IndexType>(index));
    }

    IncrementNumberOfErrorsOrDie();
    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhausted all input expected for the current sequence"
            " while reading a sparse sample"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }

    return false;
}

template <class ElemType>
void TextParser<ElemType>::SkipToNextValue(size_t& bytesToRead)
{
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;
        // skip everything until we hit either a value marker, an input marker or the end of row.
        if (c == VALUE_DELIMITER || c == ROW_DELIMITER || c == NAME_PREFIX)
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
bool TextParser<ElemType>::ReadUint64(size_t& value, size_t& bytesToRead)
{
    value = 0;
    bool found = false;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (!isdigit(c))
        {
            return found;
        }

        found |= true;

        size_t temp = value;
        value = value * 10 + (c - '0');
        if (temp > value)
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: size_t overflow while reading a uint64"
                    " at the offset = %" PRId64 "\n", GetFileOffset());
            }

            return false;
        }

        ++m_pos;
        --bytesToRead;
    }

    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhausted all input expected for the current sequence"
            " while reading a uint64"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
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
bool TextParser<ElemType>::ReadRealNumber(ElemType& value, size_t& bytesToRead)
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
            if (isdigit(c))
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
                if (m_traceLevel >= Warning)
                {
                    fprintf(stderr,
                        "WARNING: unexpected prefix('%c') while reading a floating point value"
                        " at the offset = %" PRId64 "\n", c, GetFileOffset());
                }
                return false;
            }
            break;
        case Sign:
            // the sign must be followed by a number
            if (isdigit(c))
            {
                state = IntegralPart;
                number = (c - '0');
            }
            else
            {
                if (m_traceLevel >= Warning)
                {
                    fprintf(stderr,
                        "WARNING: a sign symbol is followed by an unexpected character('%c')"
                        " while reading a floating point value"
                        " at the offset = %" PRId64 "\n", c, GetFileOffset());
                }
                return false;
            }
            break;
        case IntegralPart:
            if (isdigit(c))
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
            if (isdigit(c))
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
            if (isdigit(c))
            {
                // TODO: ignore if number of precision digits > FLT_DIG/DBL_DIG
                // or check for overflows.
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
            if (isdigit(c))
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
                if (m_traceLevel >= Warning)
                {
                    fprintf(stderr,
                        "WARNING: the exponent symbol is followed by an unexpected character('%c')"
                        " while reading a floating point value"
                        " at the offset = %" PRId64 "\n", c, GetFileOffset());
                }
                return false;
            }
            break;
        case ExponentSign:
            // exponent sign must be followed by a number
            if (isdigit(c))
            {
                state = Exponent;
                number = (c - '0');
            }
            else
            {
                if (m_traceLevel >= Warning)
                {
                    fprintf(stderr,
                        "WARNING: an exponent sign symbol is followed by an unexpected character('%c')"
                        " while reading a floating point value"
                        " at the offset = %" PRId64 "\n", c, GetFileOffset());
                }
                return false;
            }
            break;
        case Exponent:
            if (isdigit(c))
            {
                // no state change
                number = number * 10 + (c - '0');
            }
            else
            {
                double exponent = (negative) ? -number : number;
                value = static_cast<ElemType>(coefficient * pow(10.0, exponent));
                return true;
            }
            break;
        default:
            LogicError("Invalid parsing state");
        }

        ++m_pos;
        --bytesToRead;
    }

    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhausted all input expected for the current sequence"
            " while reading  a floating point value"
            " at the offset = %" PRId64 "\n", GetFileOffset());
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
void TextParser<ElemType>::SetChunkCacheSize(unsigned int size)
{
    m_chunkCacheSize = size;
}

template <class ElemType>
void TextParser<ElemType>::SetChunkSize(size_t size)
{
    m_chunkSizeBytes = size;
}

template class TextParser<float>;
template class TextParser<double>;
}}}
