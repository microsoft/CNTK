//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "SequenceEnumerator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ConfigParameters;

size_t GetRandomizationWindowFromConfig(const ConfigParameters& config);

// Returns the size of the type.
inline size_t GetSizeByType(ElementType type)
{
    switch (type)
    {
    case ElementType::tfloat:
        return sizeof(float);
    case ElementType::tdouble:
        return sizeof(double);
    default:
        RuntimeError("Unsupported type '%d'", static_cast<int>(type));
    }
}

static std::vector<unsigned char> FillIndexTable()
{
    std::vector<unsigned char> indexTable;
    indexTable.resize(std::numeric_limits<unsigned char>().max());
    char value = 0;
    for (unsigned char i = 'A'; i <= 'Z'; i++)
        indexTable[i] = value++;
    assert(value == 26);

    for (unsigned char i = 'a'; i <= 'z'; i++)
        indexTable[i] = value++;
    assert(value == 52);

    for (unsigned char i = '0'; i <= '9'; i++)
        indexTable[i] = value++;
    assert(value == 62);
    indexTable['+'] = value++;
    indexTable['/'] = value++;
    assert(value == 64);
    return indexTable;
}

const static char* base64IndexTable = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::vector<unsigned char> base64DecodeTable = FillIndexTable();

inline bool IsBase64Char(char c)
{
    return isalnum(c) || c == '/' || c == '+' || c == '=';
}

inline bool DecodeBase64(const char* begin, const char* end, std::vector<char>& result)
{
    assert(std::find_if(begin, end, [](char c) { return !IsBase64Char(c); }) == end);

    size_t length = end - begin;
    if (length % 4 != 0)
        return false;

    result.resize((length * 3) / 4); // Upper bound on the max number of decoded symbols.
    size_t currentDecodedIndex = 0;
    while (begin < end)
    {
        result[currentDecodedIndex++] = base64DecodeTable[*begin] << 2 | base64DecodeTable[*(begin + 1)] >> 4;
        result[currentDecodedIndex++] = base64DecodeTable[*(begin + 1)] << 4 | base64DecodeTable[*(begin + 2)] >> 2;
        result[currentDecodedIndex++] = base64DecodeTable[*(begin + 2)] << 6 | base64DecodeTable[*(begin + 3)];
        begin += 4;
    }

    // In Base 64 each 3 characters are encoded with 4 bytes. Plus there could be padding (last two bytes)
    size_t resultingLength = (length * 3) / 4 - (*(end - 2) == '=' ? 2 : (*(end - 1) == '=' ? 1 : 0));
    result.resize(resultingLength);
    return true;
}

// Class to clean/keep track of invalid sequences.
class SequenceCleaner
{
public:
    SequenceCleaner(size_t maxNumberOfInvalidSequences)
        : m_maxNumberOfInvalidSequences(maxNumberOfInvalidSequences),
        m_numberOfCleanedSequences(0)
    {}

    // Removes invalid sequences in place.
    void Clean(Sequences& sequences)
    {
        if (sequences.m_data.empty())
            return;

        size_t clean = 0;
        for (size_t i = 0; i < sequences.m_data.front().size(); ++i)
        {
            bool invalid = false;
            for (const auto& s : sequences.m_data)
            {
                if (!s[i]->m_isValid)
                {
                    invalid = true;
                    break;
                }
            }

            if (invalid)
            {
                m_numberOfCleanedSequences++;
                continue;
            }

            // For all streams reassign the sequence.
            for (auto& s : sequences.m_data)
                s[clean] = s[i];
            clean++;
        }

        if (clean == 0)
        {
            sequences.m_data.resize(0);
            return;
        }

        // For all streams set new size.
        for (auto& s : sequences.m_data)
            s.resize(clean);

        if (m_numberOfCleanedSequences > m_maxNumberOfInvalidSequences)
            RuntimeError("Number of invalid sequences '%d' in the input exceeded the specified maximum number '%d'",
                (int)m_numberOfCleanedSequences,
                (int)m_maxNumberOfInvalidSequences);
    }

private:
    // Number of sequences cleaned.
    size_t m_numberOfCleanedSequences;

    // Max number of allowed invalid sequences.
    size_t m_maxNumberOfInvalidSequences;
};

}}}
