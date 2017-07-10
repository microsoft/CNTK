//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "Basics.h"
#include "ReaderConstants.h"

namespace CNTK {

class MemoryBuffer
{
public:
    MemoryBuffer(size_t maxSize, bool useCompleteLines = false);

    // Pointer to the start of the buffer.
    const char* Start() const { return m_data.data(); }

    // Number of bytes left in the buffer
    size_t Left() const { return End() - m_current; }

    // Pointer to the end of the buffer.
    const char* End() const { return m_data.data() + m_data.size(); }

    // Current position in the buffer.
    const char* m_current = 0;

    // File offset that correspond to the current position.
    int64_t GetFileOffset() const { return m_fileOffsetStart + (m_current - Start()); }

    // Skips UTF-8 BOM value, if it is present at current position.
    void SkipBOMIfPresent();

    // Refills the buffer from the file.
    void RefillFrom(FILE* file);

    // Moves the current position to the next line.
    // If no new lines is present, returns null, otherwise returns a new position.
    const char* MoveToNextLine()
    {
        assert(!Eof());
        m_current = (char*)memchr(m_current, g_rowDelimiter, End() - m_current);
        if (m_current)
        {
            ++m_line;
            ++m_current;
            return m_current;
        }
        else
        {
            m_current = End();
            return nullptr;
        }
    }

    size_t CurrentLine() const { return m_line; }

    // Returns true if no more data available.
    bool Eof() const { return m_done; }

private:
    const size_t m_maxSize;                      // Max size of the buffer.
    std::vector<char> m_data;                    // Buffer.
    int64_t m_fileOffsetStart = 0;               // Current file offset that the buffer is associated with.
    bool m_done = false;                         // Flag indicating whether there is more data.
    bool m_useCompleteLines;                     // Flag indicating whether the buffer should only contain complete lines.
    std::string m_lastPartialLineInBuffer;       // Buffer for the partial string to preserve them between two sequential Refills.
    size_t m_line;                               // Current line.
};

}
