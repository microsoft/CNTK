//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "BufferedReader.h"

namespace CNTK {


using namespace std;

bool BufferedReaderBase::TryMoveToNextLine()
{
    for (; !m_done; Refill())
    {
        auto start = m_buffer.data() + m_index;
        auto found = (char*)memchr(start, g_eol, m_buffer.size() - m_index);
        if (found)
        {
            m_index = (found - m_buffer.data());
            // At this point, m_index points to the end of line, try moving it to the next line.
            return Pop();
        }
    }

    return false;
}

// Reads the current line (i.e., everything that's left on the current line) into the provided
// string reference (omitting the trailing EOL). Returns false upon reaching the EOF.
bool BufferedReaderBase::TryReadLine(string& str)
{
    if (m_done)
        return false;

    str.resize(0);
    bool result = false;
    for (; !m_done; Refill())
    {
        auto start = m_buffer.data() + m_index;
        auto found = (char*)memchr(start, g_eol, m_buffer.size() - m_index);
        if (found)
        {
            m_index = (found - m_buffer.data());
            str.append(start, found - start);
            // At this point, m_index points to the end of line, try moving it to the next line.
            Pop();
            return true;
        }

        if (m_index < m_buffer.size())
        {
            // The current buffer doe not contain an end of line (for instance, when the line is so huge,
            // it does not fit in a single buffer). Append the remainder of the buffer to the string and refill.
            str.append(start, m_buffer.size() - m_index);
            result = true;
        }
    }

    return result;
}

BufferedReader::BufferedReader(size_t maxSize, const char * buffer, const size_t& offsetInFile) :
    BufferedReaderBase(maxSize, buffer)
{
    m_buffer.reserve(maxSize);
    SetFileOffset(offsetInFile);
}

}