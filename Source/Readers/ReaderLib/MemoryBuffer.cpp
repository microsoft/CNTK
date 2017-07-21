//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "MemoryBuffer.h"
#include <boost/utility/string_ref.hpp>
#include <boost/algorithm/string.hpp>

namespace CNTK {

    using namespace std;

    MemoryBuffer::MemoryBuffer(size_t maxSize, bool useCompleteLines) 
        : m_maxSize(maxSize), m_useCompleteLines(useCompleteLines), m_line(0){}

    void MemoryBuffer::RefillFrom(FILE* file)
    {
        if (m_done)
            return;

        m_fileOffsetStart += m_data.size();
        m_data.resize(m_maxSize);

        if (!m_useCompleteLines)
        {
            size_t bytesRead = fread(m_data.data(), 1, m_maxSize, file);
            if (bytesRead == (size_t)-1)
                RuntimeError("Could not read from the input file.");
            m_data.resize(bytesRead);
            if (!bytesRead)
                m_done = true;
            m_current = m_data.data();
        }
        else // Need to keep track of the last partial string.
        {
            if (m_lastPartialLineInBuffer.size() >= m_maxSize)
                RuntimeError("Length of a sequence cannot exceed '%zu' bytes.", m_maxSize);

            // Copy last partial line if it was left during the last read.
            memcpy(&m_data[0], m_lastPartialLineInBuffer.data(), m_lastPartialLineInBuffer.size());

            size_t bytesRead = fread(&m_data[0] + m_lastPartialLineInBuffer.size(), 1, m_data.size() - m_lastPartialLineInBuffer.size(), file);
            if (bytesRead == (size_t)-1)
                RuntimeError("Could not read from the input file.");

            size_t readBufferSize = m_lastPartialLineInBuffer.size() + bytesRead;
            m_data.resize(readBufferSize);
            m_current = m_data.data();

            if (bytesRead == 0) // End of file reached.
            {
                if (m_data.empty())
                    m_done = true;
                m_lastPartialLineInBuffer.clear();
                return;
            }

            // Let's find the last LF.
            int lastLF = 0;
            {
                // Let's find the latest \n if exists.
                for (lastLF = (int)readBufferSize - 1; lastLF >= 0; lastLF--)
                {
                    if (m_data[lastLF] == g_rowDelimiter)
                        break;
                }

                // Could not find, seems not enough data, or end of file.
                if (lastLF < 0)
                {
                    m_lastPartialLineInBuffer.resize(readBufferSize);
                    memcpy(&m_lastPartialLineInBuffer[0], m_data.data(), readBufferSize);
                    m_data.clear();
                    return;
                }
            }

            // Let's cut the buffer at the last EOL and save partial string
            // in m_lastPartialLineInBuffer.
            auto logicalBufferSize = lastLF + 1;
            auto lastPartialLineSize = readBufferSize - logicalBufferSize;

            // Remember the last parital line.
            m_lastPartialLineInBuffer.resize(lastPartialLineSize);
            if (lastPartialLineSize)
                memcpy(&m_lastPartialLineInBuffer[0], m_data.data() + logicalBufferSize, lastPartialLineSize);
            m_data.resize(logicalBufferSize);
        }
    }

    void MemoryBuffer::SkipBOMIfPresent()
    {
        assert(m_current == m_data.data());
        if ((m_data.size() > 3) &&
            (m_data[0] == '\xEF' && m_data[1] == '\xBB' && m_data[2] == '\xBF'))
        {
            m_current += 3;
        }
    }

}
