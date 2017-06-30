//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "BufferedFileReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    BufferedFileReader::BufferedFileReader(size_t maxSize, FILE* file) 
        : m_maxSize(maxSize), m_buffer(new char[maxSize]), m_file(file)
    {
        if (maxSize == 0)
            RuntimeError("Max buffer size cannot be zero.");
        Refill();
    }

    inline void BufferedFileReader::Refill()
    {
        if (m_done)
            return;

        m_index = 0;
        m_fileOffset += m_size;

        size_t bytesRead = fread(m_buffer, 1, m_maxSize, m_file);

        if (bytesRead != m_maxSize && !feof(m_file))
            RuntimeError("Error reading file: %s.", strerror(errno));
        
        m_size = bytesRead;
        m_done = (m_size == 0);
    }

    bool BufferedFileReader::TryMoveToNextLine()
    {
        for (;!m_done; Refill())
        {
            auto start = m_buffer + m_index;
            auto found = (char*)memchr(start, g_eol, m_size - m_index);
            if (found)
            {
                m_index = (found - m_buffer);
                // At this point, m_index points to the end of line, try moving it to the next line.
                return Pop();
            }
        }

        return false;
    }

    bool BufferedFileReader::TryReadLine(string& str)
    {
        str.resize(0);
        bool result = false;
        for (; !m_done; Refill())
        {
            auto start = m_buffer + m_index;
            auto found = (char*)memchr(start, g_eol, m_size - m_index);
            if (found)
            {
                m_index = (found - m_buffer);
                str.append(start, found-start);
                // At this point, m_index points to the end of line, try moving it to the next line.
                Pop();
                return true;
            }
            
            if (m_index < m_size) 
            {
                str.append(start, m_size - m_index);
                result = true;
            }
        }

        return result;
    }
}}}
