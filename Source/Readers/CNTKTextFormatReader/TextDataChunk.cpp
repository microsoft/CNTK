//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <cfloat>
#include "BufferedFileReader.h"
#include "IndexBuilder.h"
#include "TextDataChunk.h"
#include "TextDeserializer.h"
#include "TextReaderConstants.h"
#include "TextParserInfo.h"
#include "File.h"

#define isSign(c) ((c == '-' || c == '+'))
#define isE(c) ((c == 'e' || c == 'E'))

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

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
TextDataChunk<ElemType>::TextDataChunk(std::shared_ptr<TextParserInfo> parserInfo) :
    m_parserInfo(parserInfo)
{

}

template <class ElemType>
void TextDataChunk<ElemType>::GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result)
{
    assert(sequenceId < m_sequenceMap.size());

    if (m_sequenceMap[sequenceId].size() == 0)
    {
        SequenceBuffer& sequence = m_sequenceMap[sequenceId];

        // Get a pointer at sequence start in chunk and create a buffered reader from it
        auto sequenceStart = m_buffer.get() + m_sequenceDescriptors[sequenceId].OffsetInChunk();

        TextParser<ElemType> parser(m_parserInfo, sequenceStart, m_sequenceDescriptors[sequenceId].SizeInBytes(), m_offsetInFile + m_sequenceDescriptors[sequenceId].OffsetInChunk());

        parser.ParseSequence(sequence, m_sequenceDescriptors[sequenceId]);

    }

    const auto& sequenceData = m_sequenceMap[sequenceId];
    result.insert(result.end(), sequenceData.begin(), sequenceData.end());
}

template class TextDataChunk<float>;
template class TextDataChunk<double>;

template<class ElemType>
TextParser<ElemType>::TextParser(std::shared_ptr<TextParserInfo> parserInfo, const char * bufferStart, const size_t& sequenceLength, const size_t& offsetInFile) :
    m_parserInfo(parserInfo),
    m_bufferedReader(make_shared<BufferedReader>(sequenceLength, bufferStart, offsetInFile))
{
    m_scratch = unique_ptr<char[]>(new char[m_parserInfo->m_maxAliasLength + 1]);
}

template <class ElemType>
void TextParser<ElemType>::PrintWarningNotification()
{
    if (m_hadWarnings && m_parserInfo->m_traceLevel < static_cast<unsigned int>(TraceLevel::Warning))
    {
        fprintf(stderr,
            "A number of warnings were generated while reading input data, "
            "to see them please set 'traceLevel' to a value greater or equal to %d.\n", TraceLevel::Warning);
    }
}

template <class ElemType>
void TextParser<ElemType>::IncrementNumberOfErrorsOrDie()
{
    if (m_parserInfo->m_numAllowedErrors == 0)
    {
        PrintWarningNotification();
        RuntimeError("Reached the maximum number of allowed errors"
            " while reading the input file (%ls).",
            m_parserInfo->m_filename.c_str());
    }
    --m_parserInfo->m_numAllowedErrors;
}

template<class ElemType>
void TextParser<ElemType>::FillSequenceMetadata(SequenceBuffer& sequenceData, const SequenceKey& sequenceKey)
{
    for (size_t j = 0; j < m_parserInfo->m_streamInfos.size(); ++j)
    {
        const StreamInformation& stream = m_parserInfo->m_streamInfos[j];
        SequenceDataBase* data = sequenceData[j].get();
        if (stream.m_storageFormat == StorageFormat::SparseCSC)
        {
            auto sparseData = static_cast<SparseInputStreamBuffer*>(data);
            sparseData->m_indices = sparseData->m_indicesBuffer.data();
            assert(data->m_numberOfSamples == sparseData->m_nnzCounts.size());
        }

        data->m_key = sequenceKey;
    }
}

template<class ElemType>
void TextParser<ElemType>::ParseSequence(SequenceBuffer& sequence, const SequenceDescriptor& sequenceDsc)
{
    for (auto const & stream : m_parserInfo->m_streamInfos)
    {
        if (stream.m_storageFormat == StorageFormat::Dense)
        {
            sequence.push_back(make_unique<DenseInputStreamBuffer>(
                stream.m_sampleLayout.Dimensions()[0] * sequenceDsc.NumberOfSamples(), stream.m_sampleLayout));
        }
        else
        {
            sequence.push_back(make_unique<SparseInputStreamBuffer>(stream.m_sampleLayout));
        }
    }

    size_t bytesToRead = sequenceDsc.SizeInBytes();

    size_t numRowsRead = 0, expectedRowCount = sequenceDsc.NumberOfSamples();

    size_t rowNumber = 1;
    while (bytesToRead)
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
                    rowNumber,
                    sequenceDsc.m_key,
                    GetFileInfo().c_str());
            }
            IncrementNumberOfErrorsOrDie();
        }
        rowNumber++;
    }

    if (ShouldWarn() && numRowsRead < expectedRowCount)
    {
        fprintf(stderr,
            "WARNING: Exhausted all input"
            " expected for the current sequence (id = %" PRIu64 ") %ls,"
            " but only read %" PRIu64 " out of %" PRIu64 " expected rows.\n",
            sequenceDsc.m_key,
            GetFileInfo().c_str(), numRowsRead, expectedRowCount);
    }

    // Double check if there are empty input streams.
    // TODO this handling needs to be graceful, but currently CNTK complains when we return empty sequences.
    bool hasEmptyInputs = false, hasDuplicateInputs = false;
    uint32_t overallSequenceLength = 0; // the resulting sequence length across all inputs.
    for (size_t i = 0; i < sequence.size(); ++i)
    {
        if (sequence[i]->m_numberOfSamples == 0)
        {
            fprintf(stderr,
                "ERROR: Input ('%ls') is empty in sequence (id = %" PRIu64 ") %ls.\n",
                m_parserInfo->m_streamInfos[i].m_name.c_str(), sequenceDsc.m_key, GetFileInfo().c_str());
            hasEmptyInputs = true;
        }

        bool definesSequenceLength = (m_parserInfo->m_useMaxAsSequenceLength || m_parserInfo->m_streamInfos[i].m_definesMbSize);

        if (!definesSequenceLength)
            continue;

        overallSequenceLength = max(sequence[i]->m_numberOfSamples, overallSequenceLength);

        if (sequence[i]->m_numberOfSamples > expectedRowCount)
        {
            hasDuplicateInputs = true;
            if (ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input ('%ls') contains more samples than expected"
                    " (%u vs. %" PRIu64 ") for sequence (id = %" PRIu64 ") %ls.\n",
                    m_parserInfo->m_streamInfos[i].m_name.c_str(), sequence[i]->m_numberOfSamples,
                    expectedRowCount, sequenceDsc.m_key, GetFileInfo().c_str());
            }
        }

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
    else if (overallSequenceLength < expectedRowCount)
    {
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Number of samples for sequence (id = %" PRIu64 ") %ls"
                " is less than expected (%u vs. %" PRIu64 ").\n",
                sequenceDsc.m_key,
                GetFileInfo().c_str(), overallSequenceLength, expectedRowCount);
        }
        IncrementNumberOfErrorsOrDie();
    }

    if (m_parserInfo->m_traceLevel >= static_cast<unsigned int>(TraceLevel::Info))
    {
        fprintf(stderr,
            "INFO: Finished loading sequence (id = %" PRIu64 ") %ls,"
            " successfully read %" PRIu64 " out of expected %" PRIu64 " rows.\n",
            sequenceDsc.m_key, GetFileInfo().c_str(), numRowsRead, expectedRowCount);
    }

    FillSequenceMetadata(sequence, { sequenceDsc.m_key, 0 });
}

template <class ElemType>
bool TextParser<ElemType>::TryReadRow(SequenceBuffer& sequence, size_t& bytesToRead)
{
    while (bytesToRead && CanRead() && IsDigit(m_bufferedReader->Peek()))
    {
        // skip sequence ids
        m_bufferedReader->Pop();
        --bytesToRead;
    }

    size_t numSampleRead = 0;

    while (bytesToRead && CanRead())
    {
        char c = m_bufferedReader->Peek();

        if (c == ROW_DELIMITER)
        {
            // found the end of row, skip the delimiter, return.
            m_bufferedReader->Pop();
            --bytesToRead;

            if (numSampleRead == 0 && ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Empty input row %ls.\n", GetFileInfo().c_str());
            }
            else if (numSampleRead > m_parserInfo->m_streamInfos.size() && ShouldWarn())
            {
                fprintf(stderr,
                    "WARNING: Input row %ls contains more"
                    " samples than expected (%" PRIu64 " vs. %" PRIu64 ").\n",
                    GetFileInfo().c_str(), numSampleRead, m_parserInfo->m_streamInfos.size());
            }

            return numSampleRead > 0;
        }

        if (isColumnDelimiter(c))
        {
            // skip column (input) delimiters.
            m_bufferedReader->Pop();
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
    // prefix check.
    if (m_bufferedReader->Peek() != NAME_PREFIX)
    {
        if (ShouldWarn())
        {
            fprintf(stderr,
                "WARNING: Unexpected character('%c') in place of a name prefix ('%c')"
                " in an input name %ls.\n",
                m_bufferedReader->Peek(), NAME_PREFIX, GetFileInfo().c_str());
        }
        IncrementNumberOfErrorsOrDie();
        return false;
    }

    // skip name prefix
    m_bufferedReader->Pop();
    --bytesToRead;

    if (bytesToRead && CanRead() && m_bufferedReader->Peek() == ESCAPE_SYMBOL)
    {
        // A vertical bar followed by the number sign (|#) is treated as an escape sequence, 
        // everything that follows is ignored until the next vertical bar or the end of 
        // row, whichever comes first.
        m_bufferedReader->Pop();
        --bytesToRead;
        return false;
    }

    size_t id;
    if (!TryGetInputId(id, bytesToRead))
    {
        return false;
    }

    const StreamInformation& stream = m_parserInfo->m_streamInfos[id];

    if (stream.m_storageFormat == StorageFormat::Dense)
    {
        DenseInputStreamBuffer* data = reinterpret_cast<DenseInputStreamBuffer*>(sequence[id].get());
        vector<ElemType>& values = data->m_buffer;
        size_t size = values.size();
        assert(size % stream.m_sampleShape.Dimensions()[0] == 0);
        if (!TryReadDenseSample(values, stream.m_sampleLayout.Dimensions()[0], bytesToRead))
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
        vector<SparseIndexType>& indices = data->m_indicesBuffer;
        assert(values.size() == indices.size());
        size_t size = values.size();
        if (!TryReadSparseSample(values, indices, stream.m_sampleLayout.Dimensions()[0], bytesToRead))
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
        SparseIndexType count = static_cast<SparseIndexType>(values.size() - size);
        data->m_nnzCounts.push_back(count);
        data->m_totalNnzCount += count;
    }

    return true;
}

template <class ElemType>
bool TextParser<ElemType>::TryGetInputId(size_t& id, size_t& bytesToRead)
{
    char* scratchIndex = m_scratch.get();

    for (; bytesToRead && CanRead(); m_bufferedReader->Pop(), --bytesToRead)
    {
        unsigned char c = m_bufferedReader->Peek();

        // stop as soon as there's a value delimiter, an input prefix
        // or a non-printable character (e.g., newline, carriage return).
        if (isValueDelimiter(c) || c == NAME_PREFIX || isNonPrintable(c))
        {
            size_t size = scratchIndex - m_scratch.get();
            if (size)
            {
                string name(m_scratch.get(), size);
                auto it = m_parserInfo->m_aliasToIdMap.find(name);
                if (it != m_parserInfo->m_aliasToIdMap.end())
                {
                    id = it->second;
                    return true;
                }

                if (m_parserInfo->m_traceLevel >= static_cast<unsigned int>(TraceLevel::Info))
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
        else if (scratchIndex < (m_scratch.get() + m_parserInfo->m_maxAliasLength))
        {
            *scratchIndex = c;
            ++scratchIndex;
        }
        else
        {
            // the current string length is already equal to the maximum expected length,
            // yet it's not followed by a delimiter.
            if (m_parserInfo->m_traceLevel >= static_cast<unsigned int>(TraceLevel::Info))
            {
                string namePrefix(m_scratch.get(), m_parserInfo->m_maxAliasLength);
                fprintf(stderr,
                    "INFO: Skipping unknown input %ls. "
                    "Input name (with the %" PRIu64 "-character prefix '%s') "
                    "exceeds the maximum expected length (%" PRIu64 ").\n",
                    GetFileInfo().c_str(), m_parserInfo->m_maxAliasLength, namePrefix.c_str(), m_parserInfo->m_maxAliasLength);
            }
            return false;
        }
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
        char c = m_bufferedReader->Peek();

        if (isValueDelimiter(c))
        {
            // skip value delimiters
            m_bufferedReader->Pop();
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
bool TextParser<ElemType>::TryReadSparseSample(std::vector<ElemType>& values, std::vector<SparseIndexType>& indices,
    size_t sampleSize, size_t& bytesToRead)
{
    size_t index = 0;
    ElemType value;

    while (bytesToRead && CanRead())
    {
        char c = m_bufferedReader->Peek();

        if (isValueDelimiter(c))
        {
            // skip value delimiters
            m_bufferedReader->Pop();
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
        c = m_bufferedReader->Peek();
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
        m_bufferedReader->Pop();
        --bytesToRead;

        // read the corresponding value
        if (!TryReadRealNumber(value, bytesToRead))
        {
            // bail out.
            return false;
        }

        values.push_back(value);
        indices.push_back(static_cast<SparseIndexType>(index));
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
void TextParser<ElemType>::SkipToNextInput(size_t& bytesToRead)
{
    for (; bytesToRead && CanRead(); m_bufferedReader->Pop(), --bytesToRead)
    {
        char c = m_bufferedReader->Peek();
        // skip everything until we hit either an input marker or the end of row.
        if (c == NAME_PREFIX || c == ROW_DELIMITER)
        {
            return;
        }
    }
}

template <class ElemType>
bool TextParser<ElemType>::TryReadUint64(size_t& value, size_t& bytesToRead)
{
    value = 0;
    bool found = false;
    for (; bytesToRead && CanRead(); m_bufferedReader->Pop(), --bytesToRead)
    {
        char c = m_bufferedReader->Peek();

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

    for (; bytesToRead && CanRead(); m_bufferedReader->Pop(), --bytesToRead)
    {
        char c = m_bufferedReader->Peek();

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

template<class ElemType>
inline bool TextParser<ElemType>::CanRead()
{
    return !m_bufferedReader->Empty();
}

template <class ElemType>
int64_t TextParser<ElemType>::GetFileOffset() const
{
    return m_bufferedReader->GetFileOffset();
}

template <class ElemType>
std::wstring TextParser<ElemType>::GetFileInfo()
{
    std::wstringstream info;
    info << L"at offset " << GetFileOffset() << L" in the input file (" << m_parserInfo->m_filename << L")";
    return info.str();
}

template class TextParser<float>;
template class TextParser<double>;
}
