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

        size_t bytesRead = fread(m_buffer.get(), 1, m_maxSize, m_file);

        if (bytesRead != m_maxSize && !feof(m_file))
            RuntimeError("Error reading file: %s.", strerror(errno));
        
        m_size = bytesRead;
        m_done = (m_size == 0);
    }

    bool BufferedFileReader::TryGetNext(char& c)
    {
        if (Pop())
        {
            c = Peek();
            return true;
        }

        return false;
    }

    bool BufferedFileReader::TryMoveToNextLine()
    {
        while (!m_done) 
        {
            auto start = m_buffer.get() + m_index;
            auto found = (char*)memchr(start, g_eol, m_size - m_index);
            if (found)
            {
                m_index = (found - m_buffer.get());
                // At this point, m_index points to the end of line, try moving it to the next line.
                return Pop();
            }
            Refill();
        }

        return false;
    }
}}}
