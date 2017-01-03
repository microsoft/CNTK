//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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

inline std::vector<char> DecodeBase64(const char* begin, const char* end)
{
    assert(std::find_if(begin, end, [](char c) { return !IsBase64Char(c); }) == end);

    size_t length = end - begin;
    if (length % 4 != 0)
        RuntimeError("Invalid base64 data, length '%d' is not divisible by 4.", (int)length);
    std::vector<char> result;
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
    return result;
}

}}}
