//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <cfloat>
#include <inttypes.h>
#include "TextParser.h"
#include "Indexer.h"

#define isNumber(c) ((c >= '0' && c <= '9'))
#define isSign(c) ((c == '-' || c == '+'))
#define isE(c) ((c == 'e' || c == 'E'))

namespace Microsoft { namespace MSR { namespace CNTK {

const char COLUMN_DELIMETER = '\t';
const char NAME_PREFIX = '|';
const char VALUE_DELIMETER = ' ';
const char INDEX_DELIMETER = ':';
const char ROW_DELIMETER = '\n';


const unsigned int MAXIMUM_NUMBER_OF_ERRORS = 10;

TextParser::TextParser(const std::string& filename, const vector<StreamDescriptor>& streams) : m_streams(streams) {
    errno_t err = fopen_s(&m_file, filename.c_str(), "rb");
    if (err) {
        throw "error opening file";
    }
    m_maxAliasLength = 0;

    for (size_t i = 0; i < m_streams.size(); ++i) {
        const StreamDescriptor& stream = m_streams[i];
        const string& alias = stream.m_alias;
        if (m_maxAliasLength < alias.length()) {
            m_maxAliasLength = alias.length();
        }
        m_aliasToIdMap[stream.m_alias] = i;
    }

    m_bufferStart = new char[BUFFER_SIZE + 1];

    m_scratch = new char[m_maxAliasLength + 1];
}



TextParser::~TextParser() {
    delete[] m_bufferStart;
    delete[] m_scratch;
    if (m_index) {
        delete m_index;
    }
    if (m_file) {
        fclose(m_file);
    }
}

void TextParser::Initialize() {

    m_index = Indexer(m_file).Build();

    int64_t position = _ftelli64(m_file);
    if (position == -1L) {
        throw "error retrieving file position in file";
    }

    m_fileOffsetStart = position;
    m_fileOffsetEnd = position;
}

const vector<SequenceDescriptor>& TextParser::GetTimeline() {
    if (!m_index) {
        Initialize();
    }
    return m_index->m_timeline;
}

void TextParser::IncrementNumberOfErrors() {
    if (++m_numErrors >= MAXIMUM_NUMBER_OF_ERRORS) {
        // any clean up here?
        throw "Reached maximum allowed number of reader errors";
    }
}


bool TextParser::Fill() {
    size_t bytesRead = fread(m_bufferStart, 1, BUFFER_SIZE, m_file);
    if (bytesRead == (size_t)-1)
        // TODO throw a proper runtime exception
        throw "read failed";
    if (!bytesRead) {
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
    if (rc)
        throw "error seeking in file";

    m_fileOffsetStart = offset;
    m_fileOffsetEnd = offset;

    Fill();
}

vector<Sequence> TextParser::LoadChunk(const ChunkDescriptor& chunk) {
    vector<Sequence> chunkData(chunk.m_numSequences);
    for (size_t i = 0; i < chunk.m_numSequences; i++) {
        size_t sequenceIndex = chunk.m_timelineOffset + i;
        const SequenceDescriptor& sequenceDsc = m_index->m_timeline[sequenceIndex];
        chunkData[i] = LoadSequence(!m_index->noSequenceIds, sequenceDsc);
    }
    return chunkData;
}

Sequence TextParser::LoadSequence(bool verifyId, const SequenceDescriptor& sequenceDsc) {
    auto fileOffset = sequenceDsc.m_fileOffset;

    if (fileOffset < m_fileOffsetStart || fileOffset > m_fileOffsetEnd)
    {
        SetFileOffset(fileOffset);
    }


    size_t bufferOffset = fileOffset - m_fileOffsetStart;
    m_pos = m_bufferStart + bufferOffset;
    uint64_t bytesToRead = sequenceDsc.m_byteSize;


    if (verifyId) {
        size_t id;
        if (!ReadSequenceId(id, bytesToRead) || id != sequenceDsc.m_id) {
            throw "unexpected sequence id"; // todo add details
        }
    }

    Sequence sequence(m_streams.size());

    for (size_t i = 0; i < m_streams.size(); ++i) {
        const auto& stream = m_streams[i];
        if (stream.m_storageType == StorageType::dense)
        {
            sequence[i] = new DenseData(stream.m_sampleSize * sequenceDsc.m_numberOfSamples);

        }
        else
        {
            sequence[i] = new SparseData();
        }
    }

    size_t numRowsRead = 0, expectedRowCount = sequenceDsc.m_numberOfSamples;
    for (size_t i = 0; i < expectedRowCount; i++)
    {
        if ((ReadRow(sequence, bytesToRead)))
        {
            ++numRowsRead;
        }
        else if (m_traceLevel >= Warning)
        {
            fprintf(stderr,
                "WARNING: could not read a row (# %d)"
                " while loading sequence (id = %u)"
                " at the offset = %" PRIu64 "\n", i, sequenceDsc.m_id, GetFileOffset());
        }

        if (!bytesToRead && numRowsRead < expectedRowCount)
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: exhaused all expected input"
                    " expected for the current sequence (id = %u)"
                    " at the offset = %" PRIu64 "\n", sequenceDsc.m_id, GetFileOffset());
            }
            break;
        }
    }

    if (m_traceLevel >= Info)
    {
        fprintf(stderr,
            "INFO: finished loading sequence (id = %u),"
            " successfully read %u out of expected %u rows\n",
            sequenceDsc.m_id, numRowsRead, expectedRowCount);
    }

    return sequence;
}

// read one whole line of input 
bool TextParser::ReadRow(Sequence& sequence, uint64_t& bytesToRead) {
    bool found = false;
    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;

        if (c == ROW_DELIMETER)
        {
            // skip delimeter
            ++m_pos;
            --bytesToRead;
            return found;
        }

        if (c != NAME_PREFIX) {
            ++m_pos;
            --bytesToRead;
            continue;
        }

        size_t id;
        if (!GetInputId(id, bytesToRead)) {
            IncrementNumberOfErrors();
            SkipToNextInput(bytesToRead);
            continue;
        }

        const StreamDescriptor& stream = m_streams[id];

        if (stream.m_storageType == StorageType::dense)
        {
            DenseData* data = (DenseData*)(sequence[id]);
            //DenseData& streamData = (DenseData&)sequence[id];
            vector<float>& values = data->m_buffer;
            size_t size = values.size();
            // TODO: assert that size % stream.m_sampleSize == 0.
            if (!ReadDenseSample(values, stream.m_sampleSize, bytesToRead)) {
                // expected a dense sample, but was not able to fully read it, ignore it.
                if (values.size() != size) {
                    //clean up the buffer
                    values.resize(size);
                }
                IncrementNumberOfErrors();
                SkipToNextInput(bytesToRead);
                continue;
            }
            // everything went well, increment the number of samples.
            data->m_numberOfSamples++;
        }
        else
        {
            SparseData* data = (SparseData*)(sequence[id]);
            //SparseData& streamData = (SparseData&)sequence[id];
            vector<float>& values = data->m_buffer;
            size_t size = values.size();
            vector<size_t> indices;
            if (!ReadSparseSample(values, indices, bytesToRead)) {
                // expected a sparse sample, but something went south, ignore it.
                if (values.size() != size) {
                    //clean up the buffer
                    values.resize(size);
                }
                IncrementNumberOfErrors();
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
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }
    return false;
}

bool TextParser::ReadSequenceId(size_t& id, uint64_t& bytesToRead) {
    id = 0;
    bool found = false;
    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;
        if (c == NAME_PREFIX)
        {
            return found;
        }

        if (!isNumber(c))
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: unexpected character('%c')"
                    " while reading a sequence id"
                    " at the offset = %" PRIu64 "\n", c, GetFileOffset());
            }
            return false;
        }

        found |= true;

        size_t temp = id;
        id = id * 10 + (c - '0');
        if (temp > id)
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: size_t overflow while reading a sequence id"
                    " at the offset = %" PRIu64 "\n", GetFileOffset());
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
            " while reading a sequence id"
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }
    return false;
}

bool TextParser::GetInputId(size_t& id, uint64_t& bytesToRead)
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
                " at the offset = %" PRIu64 "\n", c, NAME_PREFIX, GetFileOffset());
        }
        return false;
    }

    // skip name prefix
    ++m_pos;
    --bytesToRead;

    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        c = *m_pos;

        if (c == VALUE_DELIMETER || c == COLUMN_DELIMETER || c == ROW_DELIMETER || c == NAME_PREFIX)
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
                        " at the offset = %" PRIu64 "\n", name.c_str(), GetFileOffset());
                }
            }
            else if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: a name prefix is immediately  followed by a delimeter"
                    " while reading an input name"
                    " at the offset = %" PRIu64 "\n", GetFileOffset());
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
                    " at the offset = %" PRIu64 "\n", GetFileOffset());
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
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }
    return false;
}


bool TextParser::ReadDenseSample(vector<float>& values, size_t sampleSize, uint64_t& bytesToRead)
{
    size_t counter = 0;
    float value;

    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;

        if (c == COLUMN_DELIMETER || c == ROW_DELIMETER || c == NAME_PREFIX)
        {
            // Another option:
            // if (counter < sampleSize) -- fill the rest of the buffer with zeros
            // if (counter > sampleSize) -- drop the extra elements
            // increment the number of errors by the diff(sampleSize and counter)
            //  and return true?
            return (counter == sampleSize);
        }

        if (c == VALUE_DELIMETER)
        {
            // skip value delimeters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (!ReadValue(value, bytesToRead)) {
            // Should it return false immediately?
            IncrementNumberOfErrors();
            SkipToNextValue(bytesToRead);
            // Add a zero instead (is this a valid strategy?)
            value = 0.0f;

        }

        values.push_back(value);
        ++counter;
    }

    IncrementNumberOfErrors();
    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhaused all input expected for the current sequence"
            " while reading a dense sample"
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }
    return false;
}

bool TextParser::ReadSparseSample(std::vector<float>& values, std::vector<size_t>& indices, uint64_t& bytesToRead)
{
    size_t index;
    float value;
    bool found = false;

    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;

        if (c == COLUMN_DELIMETER || c == ROW_DELIMETER || c == NAME_PREFIX)
        {
            // TODO: do we allow empty sparse samples ("|InputeName_1|InputName2...")?
            return found;
        }

        if (c == VALUE_DELIMETER)
        {
            // skip value delimeters
            ++m_pos;
            --bytesToRead;
            continue;
        }

        if (!ReadIndex(index, bytesToRead)) {
            // Should it return false immediately?
            IncrementNumberOfErrors();
            SkipToNextValue(bytesToRead);
            continue;
        }

        if (!ReadValue(value, bytesToRead)) {
            // Should it return false immediately?
            IncrementNumberOfErrors();
            SkipToNextValue(bytesToRead);
            continue;
        }

        found |= true; // this actually seems a little redundant
        values.push_back(value);
        indices.push_back(index);
    }

    IncrementNumberOfErrors();
    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhaused all input expected for the current sequence"
            " while reading a sparse sample"
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }

    return false;
}

void TextParser::SkipToNextValue(uint64_t& bytesToRead)
{
    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;

        if (c == VALUE_DELIMETER || c == COLUMN_DELIMETER || c == ROW_DELIMETER || c == NAME_PREFIX)
        {
            return;
        }
        ++m_pos;
        --bytesToRead;
    }
}

void TextParser::SkipToNextInput(uint64_t& bytesToRead)
{
    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;

        if (c == NAME_PREFIX || c == ROW_DELIMETER)
        {
            return;
        }
        ++m_pos;
        --bytesToRead;
    }
}



bool TextParser::ReadIndex(size_t& index, uint64_t& bytesToRead) {
    index = 0;
    bool found = false;
    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

        char c = *m_pos;
        if (c == INDEX_DELIMETER)
        {
            // consume index delimeter
            ++m_pos;
            --bytesToRead;
            return found;
        }

        if (!isNumber(c))
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: unexpected character('%c') while reading a sparse value index"
                    " at the offset = %" PRIu64 "\n", c, GetFileOffset());
            }

            return false;
        }
        found |= true;
        size_t temp = index;
        index = index * 10 + (c - '0');
        if (temp > index)
        {
            if (m_traceLevel >= Warning)
            {
                fprintf(stderr,
                    "WARNING: size_t overflow while reading a sparse value index"
                    " at the offset = %" PRIu64 "\n", GetFileOffset());
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
            " while reading a sparse value index"
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }

    return false;
}



//TODO: better precision?
//XXX: assumens that bytesToRead is greater than the number of characters 
// in the string representation of the floating point number
// (i.e., the string is followed by one of the delimeters)
bool TextParser::ReadValue(float& value, uint64_t& bytesToRead)
{
    State state = State::Init;
    double coefficient = .0, number = .0, divider = .0;
    bool negative = false;

    while (bytesToRead)
    {
        if (m_pos == m_bufferEnd && !Fill())
        {
            break;
        }

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
                        " at the offset = %" PRIu64 "\n", c, GetFileOffset());
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
                        " at the offset = %" PRIu64 "\n", c, GetFileOffset());
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
                value = (float)((negative) ? -number : number);
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
                value = (float)((negative) ? -number : number);
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
                value = (float)((negative) ? -coefficient : coefficient);
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
                        " at the offset = %" PRIu64 "\n", c, GetFileOffset());
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
                        " at the offset = %" PRIu64 "\n", c, GetFileOffset());
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
                value = (float) (coefficient * pow(10.0, exponent));
                return true;
            }
            break;
        default:
            // TODO: throw a proper runtime exception
            throw "Invalid state";
        }

        ++m_pos;
        --bytesToRead;
    }

    if (m_traceLevel >= Warning)
    {
        fprintf(stderr,
            "WARNING: exhaused all input expected for the current sequence"
            " while reading  a floating point value"
            " at the offset = %" PRIu64 "\n", GetFileOffset());
    }

    return false;
}

void TextParser::SetTraceLevel(unsigned int traceLevel) {
    m_traceLevel = traceLevel;
}

}}}
