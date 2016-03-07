//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <cfloat>
#include <inttypes.h>
#include "Indexer.h"
#include "TextParser.h"
#include "TextReaderConstants.h"

#define isNumber(c) ((c >= '0' && c <= '9'))
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

TextParser::TextParser(const std::wstring& filename, const vector<StreamDescriptor>& streams) 
    : m_filename(filename),
    m_numberOfStreams(streams.size()),
    m_streamInfos(m_numberOfStreams)
{
    assert(streams.size() > 0);

    m_file = fopenOrDie(m_filename, L"rbS");
    if (funicode(m_file)) 
    {
        RuntimeError("Found a UTF-16 BOM at the beginning of the input file %ls. "
            "UTF-16 encoding is currently not supported.", m_filename.c_str());
    }

    m_numAllowedErrors = 0;
    m_traceLevel = TraceLevel::Error;
    m_skipSequenceIds = false;

    m_maxAliasLength = 0;

    for (size_t i = 0; i < m_numberOfStreams; ++i)
    {
        const StreamDescriptor& stream = streams[i];
        const string& alias = stream.m_alias;
        if (m_maxAliasLength < alias.length())
        {
            m_maxAliasLength = alias.length();
        }
        m_aliasToIdMap[alias] = i;
        m_streamInfos[i].m_type = stream.m_storageType;
        m_streamInfos[i].m_sampleDimension = stream.m_sampleSize;

        auto streamDescription = std::make_shared<StreamDescription>(stream);
        streamDescription->m_sampleLayout = std::make_shared<TensorShape>(stream.m_sampleSize);
        m_streams.push_back(streamDescription);
    }

    assert(m_maxAliasLength > 0);

    m_bufferStart = new char[BUFFER_SIZE + 1];
    m_scratch = new char[m_maxAliasLength + 1];

    Initialize();
}

TextParser::~TextParser() 
{
    delete[] m_bufferStart;
    delete[] m_scratch;
    if (m_file) 
    {
        fclose(m_file);
    }
}

void TextParser::Initialize()
{
    if (m_index) 
    {
        return;
    }

    m_index = Indexer(m_file, m_skipSequenceIds).Build();

    // it's still possible that the actual input data does not have sequence id column.
    m_skipSequenceIds = !m_index->m_hasSequenceIds; 

    assert(m_index);

    int64_t position = _ftelli64(m_file);
    if (position == -1L)
    {
        RuntimeError("Error retrieving file position in file %ls", m_filename.c_str());
    }

    m_fileOffsetStart = position;
    m_fileOffsetEnd = position;
}


vector<StreamDescriptionPtr> TextParser::GetStreamDescriptions() const
{
    return m_streams;
}

size_t TextParser::GetTotalNumberOfChunks() 
{
    return m_index->m_chunks.size();
}

void TextParser::FillSequenceDescriptions(SequenceDescriptions& timeline) const
{
    timeline.resize(m_index->m_timeline.size());
    std::transform(
        m_index->m_timeline.begin(),
        m_index->m_timeline.end(),
        timeline.begin(),
        [](const SequenceDescription& desc)
    {
        return &desc;
    });
}

TextParser::TextDataChunk::TextDataChunk(const ChunkDescriptor& descriptor, TextParser& parent) 
    : m_sequenceData(descriptor.m_numSequences),
    m_parent(parent)
{
    m_id = descriptor.m_id;
    m_sequenceRequestCount = 0;
}

vector<SequenceDataPtr> TextParser::TextDataChunk::GetSequence(size_t sequenceId)
{
    assert(m_sequenceRequestCount < m_sequencePtrMap.size());
    auto it = m_sequencePtrMap.find(sequenceId);
    assert(it != m_sequencePtrMap.end());
    if (++m_sequenceRequestCount == m_sequencePtrMap.size()) 
    {
        // all sequences in this chunk have been used up,
        // now it's time to remove it from the cache.
        // TODO: add to list of eviction candidates.
        m_parent.m_chunkCache.erase(m_id);
    }
    return it->second;
}

ChunkPtr TextParser::GetChunk(size_t chunkId)
{
    auto it = m_chunkCache.find(chunkId);
    if (it != m_chunkCache.end()) 
    {
        return it->second;
    }

    const auto& chunkDescriptor = m_index->m_chunks[chunkId];
    auto chunk = make_shared<TextDataChunk>(chunkDescriptor, *this);
    vector<Sequence> sequences(chunkDescriptor.m_numSequences);
    for (size_t i = 0; i < chunkDescriptor.m_numSequences; ++i)
    {
        size_t offset = chunkDescriptor.m_timelineOffset + i;
        const auto& sequenceDescriptor = m_index->m_timeline[offset];
        chunk->m_sequenceData[i] = move(LoadSequence(!m_skipSequenceIds, sequenceDescriptor));
        const auto& sequenceData = chunk->m_sequenceData[i];
        vector<SequenceDataPtr> sequencePtrs(m_numberOfStreams);
        for (size_t j = 0; j < m_numberOfStreams; ++j)
        {
            const StreamInfo& stream = m_streamInfos[j];
            if (stream.m_type == StorageType::dense)
            {
                DenseData* loadedData = (DenseData*)(sequenceData[j].get());
                auto data = make_shared<DenseSequenceData>();
                data->m_data = loadedData->m_buffer.data();
                data->m_sampleLayout = m_streams[j]->m_sampleLayout;
                data->m_numberOfSamples = loadedData->m_numberOfSamples;
                data->m_chunk = chunk;
                sequencePtrs[j] = data;
            }
            else
            {
                SparseData* loadedData = (SparseData*)(sequenceData[j].get());
                auto data = make_shared<SparseSequenceData>();
                data->m_data = loadedData->m_buffer.data();
                data->m_indices = move(loadedData->m_indices);
                data->m_chunk = chunk;
                sequencePtrs[j] = data;
            }
        }
        chunk->m_sequencePtrMap[sequenceDescriptor.m_id] = move(sequencePtrs);
    }

    // TODO: implement cache eviction. 
    m_chunkCache[chunkId] = chunk;

    return chunk;
}

void TextParser::IncrementNumberOfErrorsOrDie() 
{
    if (--m_numAllowedErrors <= 0)
    {
        RuntimeError("Reached maximum allowed number of reader errors");
    }
}

bool TextParser::Fill() 
{
    size_t bytesRead = fread(m_bufferStart, 1, BUFFER_SIZE, m_file);
    
    if (bytesRead == (size_t)-1)
        RuntimeError("Could not read from the input file %ls", m_filename.c_str());

    if (!bytesRead) 
    {
        return false;
    }

    m_fileOffsetStart = m_fileOffsetEnd;
    m_fileOffsetEnd += bytesRead;
    m_pos = m_bufferStart;
    m_bufferEnd = m_bufferStart + bytesRead;
    return true;
}

void TextParser::SetFileOffset(int64_t offset)
{
    int rc = _fseeki64(m_file, offset, SEEK_SET);
    if (rc) {
        RuntimeError("Error seeking to position %" PRId64 " in file %ls", offset, m_filename.c_str());
    }

    m_fileOffsetStart = offset;
    m_fileOffsetEnd = offset;

    Fill();
}

Sequence TextParser::LoadSequence(bool verifyId, const SequenceDescriptor& sequenceDsc) {
    auto fileOffset = sequenceDsc.m_fileOffset;

    if (fileOffset < m_fileOffsetStart || fileOffset > m_fileOffsetEnd)
    {
        SetFileOffset(fileOffset);
    }


    size_t bufferOffset = fileOffset - m_fileOffsetStart;
    m_pos = m_bufferStart + bufferOffset;
    int64_t bytesToRead = sequenceDsc.m_byteSize;


    if (verifyId) {
        size_t id;
        if (!ReadUint64(id, bytesToRead) || id != sequenceDsc.m_id) {
            RuntimeError("Did not find the expected sequence id ( %" PRIu64 ") "
                " at the file offset = %" PRId64 "\n", sequenceDsc.m_id, GetFileOffset());
        }
    }

    Sequence sequence;

    // TODO: reuse loaded sequences instead of creating new ones!
    for (auto const & stream : m_streamInfos) {
        if (stream.m_type == StorageType::dense)
        {
            sequence.push_back(unique_ptr<DenseData>(new DenseData(stream.m_sampleDimension * sequenceDsc.m_numberOfSamples)));
        }
        else
        {
            sequence.push_back(unique_ptr<SparseData>(new SparseData()));
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
                    "WARNING: exhaused all expected input"
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

    return sequence;
}

// read one whole line of input 
bool TextParser::ReadRow(Sequence& sequence, int64_t& bytesToRead) {
    bool found = false;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (isNumber(c) || c == COLUMN_DELIMITER || c == CARRIAGE_RETURN) {
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
        if (!GetInputId(id, bytesToRead)) {
            IncrementNumberOfErrorsOrDie();
            SkipToNextInput(bytesToRead);
            continue;
        }

        const StreamInfo& stream = m_streamInfos[id];

        if (stream.m_type == StorageType::dense)
        {
            DenseData* data = (DenseData*)(sequence[id].get());
            vector<float>& values = data->m_buffer;
            size_t size = values.size();
            // TODO: assert that size % stream.sampleSize == 0.
            if (!ReadDenseSample(values, stream.m_sampleDimension, bytesToRead)) {
                // expected a dense sample, but was not able to fully read it, ignore it.
                if (values.size() != size) {
                    //clean up the buffer
                    values.resize(size);
                }
                IncrementNumberOfErrorsOrDie();
                SkipToNextInput(bytesToRead);
                continue;
            }
            // everything went well, increment the number of samples.
            data->m_numberOfSamples++;
        }
        else
        {
            SparseData* data = (SparseData*)(sequence[id].get());
            vector<float>& values = data->m_buffer;
            size_t size = values.size();
            vector<size_t> indices;
            if (!ReadSparseSample(values, indices, bytesToRead)) {
                // expected a sparse sample, but something went south, ignore it.
                if (values.size() != size) {
                    //clean up the buffer
                    values.resize(size);
                }
                IncrementNumberOfErrorsOrDie();
                SkipToNextInput(bytesToRead);
                continue;
            }
            data->m_indices.push_back(indices);
            data->m_numberOfSamples++;
        }

        found |= true;
    }

    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhaused all input expected for the current sequence"
            " while reading an input row"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}

bool TextParser::GetInputId(size_t& id, int64_t& bytesToRead)
{
    char* scratchIndex = m_scratch;

    char c = *m_pos;

    // prefix check.
    if (c != NAME_PREFIX) {
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
            size_t size = scratchIndex - m_scratch;
            if (size)
            {
                string name(m_scratch, size);
                auto it = m_aliasToIdMap.find(name);
                if (it != m_aliasToIdMap.end()) {
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
        else if (scratchIndex < (m_scratch + m_maxAliasLength))
        {
            *scratchIndex = c;
            ++scratchIndex;
        }
        else
        {
            // the current string length already equal to the maximum expected length,
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
            "WARNING: exhaused all input expected for the current sequence"
            " while reading an input name"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}


bool TextParser::ReadDenseSample(vector<float>& values, size_t sampleSize, int64_t& bytesToRead)
{
    size_t counter = 0;
    float value;

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

        if (!ReadRealNumber(value, bytesToRead)) {
            // For the time being, return right away;
            return false;
            // TODO: figure out what to do here.
            //IncrementNumberOfErrorsOrDie();
            //SkipToNextValue(bytesToRead);
            // Add a zero instead (is this a valid strategy?)
            //value = 0.0f;
        }

        values.push_back(value);
        ++counter;
    }

    IncrementNumberOfErrorsOrDie();
    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhaused all input expected for the current sequence"
            " while reading a dense sample"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}

bool TextParser::ReadSparseSample(std::vector<float>& values, std::vector<size_t>& indices, int64_t& bytesToRead)
{
    size_t index;
    float value;

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
        if (!ReadUint64(index, bytesToRead)) {
            // For the time being, return right away;
            return false;
            // TODO: figure out what to do here.
            //IncrementNumberOfErrorsOrDie();
            //SkipToNextValue(bytesToRead);
            //continue;
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
        if (!ReadRealNumber(value, bytesToRead)) {
            // For the time being, return right away;
            return false;
            // TODO: figure out what to do here.
            //IncrementNumberOfErrorsOrDie();
            //SkipToNextValue(bytesToRead);
            //continue;
        }

        values.push_back(value);
        indices.push_back(index);
    }

    IncrementNumberOfErrorsOrDie();
    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhaused all input expected for the current sequence"
            " while reading a sparse sample"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }

    return false;
}

void TextParser::SkipToNextValue(int64_t& bytesToRead)
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

void TextParser::SkipToNextInput(int64_t& bytesToRead)
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

bool TextParser::ReadUint64(size_t& id, int64_t& bytesToRead) {
    id = 0;
    bool found = false;
    while (bytesToRead && CanRead())
    {
        char c = *m_pos;

        if (!isNumber(c))
        {
            return found;
        }

        found |= true;

        size_t temp = id;
        id = id * 10 + (c - '0');
        if (temp > id)
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
            "WARNING: exhaused all input expected for the current sequence"
            " while reading a uint64"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }
    return false;
}



// TODO: templatize, better precision?
// Assumes that bytesToRead is greater than the number of characters 
// in the string representation of the floating point number
// (i.e., the string is followed by one of the delimiters)
// Post condition: m_pos points to the first character that 
// cannot be parsed as part of a floating point number.
// Returns true if parsing was successful.
bool TextParser::ReadRealNumber(float& value, int64_t& bytesToRead)
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
            if (isNumber(c))
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
            if (isNumber(c))
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
            if (isNumber(c))
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
                value = static_cast<float>((negative) ? -number : number);
                return true;
            }
            break;
        case Period:
            if (isNumber(c))
            {
                state = FractionalPart;
                coefficient = number;
                number = (c - '0');
                divider = 10;
            }
            else
            {
                value = static_cast<float>((negative) ? -number : number);
                return true;
            }
            break;
        case FractionalPart:
            if (isNumber(c))
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
                value = static_cast<float>((negative) ? -coefficient : coefficient);
                return true;
            }
            break;
        case TheLetterE:
            // followed with optional minus or plus sign and nonempty sequence of decimal digits
            if (isNumber(c))
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
            if (isNumber(c))
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
            if (isNumber(c))
            {
                // no state change
                number = number * 10 + (c - '0');
            }
            else {
                double exponent = (negative) ? -number : number;
                value = static_cast<float>(coefficient * pow(10.0, exponent));
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
            "WARNING: exhaused all input expected for the current sequence"
            " while reading  a floating point value"
            " at the offset = %" PRId64 "\n", GetFileOffset());
    }

    return false;
}

void TextParser::SetTraceLevel(unsigned int traceLevel) {
    m_traceLevel = traceLevel;
}

void TextParser::SetMaxAllowedErrors(unsigned int maxErrors) {
    m_numAllowedErrors = maxErrors;
}

void TextParser::SetSkipSequenceIds(bool skip) {
    m_skipSequenceIds = skip;
}

void TextParser::SetChunkCacheSize(unsigned int size) {
    m_chunkCacheSize = size;
}

}}}
