//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include <memory>
#include "Basics.h"
#include "ReaderConstants.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class BufferedFileReader
{
public:
    BufferedFileReader(size_t maxSize, FILE* file);

    // File offset that correspond to the current position.
    size_t GetFileOffset() const { return m_fileOffset + m_index; }

    // Returns the character at the current buffer position.
    char Peek()
    {
        if (m_done)
            RuntimeError("Buffer is empty.");

        assert(m_index < m_size);

        return m_buffer[m_index];
    }

    // Advances the current position to the next character.
    // Returns true, unless EOF has been reached.
    bool Pop() 
    {
        if (m_done)
            return false;

        assert(m_index < m_size);

        if (m_buffer[m_index] == g_eol)
            m_lineNumber++;

        if (++m_index == m_size)
            Refill();

        return !m_done;
    }

    bool TryGetNext(char& c);

    // Moves the current position to the next line (the position following an EOL delimiter).
    // Returns true, unless EOF has been reached.
    bool TryMoveToNextLine();
    
    // Returns the current line number.
    size_t CurrentLineNumber() const { return m_lineNumber; }

    // Returns true if no more data is available (reached EOF).
    bool Empty() const { return m_done; }

private:
    // Read up to m_maxSize bytes from file into the buffer.
    void Refill();

    // Maximum allowed buffer size. 
    // Also, it defines the maximum number of bytes that we'll attempt to read at one time.
    const size_t m_maxSize{ 0 };

    // Buffer.
    std::unique_ptr<char[]> m_buffer; 

    // current buffer size (number of valid chars in the buffer, m_size is always LE m_maxSize)
    size_t m_size{ 0 };

    // Current position in the buffer.
    size_t m_index{ 0 };

    // File offset at which the buffer was fill out in the last call to Refill();
    size_t m_fileOffset{ 0 };

    // Flag indicating whether there is more data (set to true once EOF is reached).
    bool m_done{ false };

    // Current line number;
    size_t m_lineNumber{ 0 };

    FILE* m_file{ nullptr };
};

}}}
