//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include <memory>
#include "ReaderConstants.h"
#include "BufferedReader.h"

namespace CNTK {

class BufferedFileReader : public BufferedReaderBase
{
public:
    BufferedFileReader(size_t maxSize, const FileWrapper& file);

    void SetFileOffset(const size_t& fileOffset) override
    {
        // We reset the current buffer only if the new fileOffset is out of the buffer limits.
        // If not, we just go to the index corresponding to the offset.
        if (fileOffset >= (m_buffer.size() + m_fileOffset) || fileOffset < m_fileOffset) {
            m_file.SeekOrDie(fileOffset, SEEK_SET);
            Reset();
        }
        else
        {
            m_index = fileOffset - m_fileOffset;
            m_done = false;
        }
    }

private:
    // Read up to m_maxSize bytes from file into the buffer.
    void Refill() override;

    // Resets the buffer: clears the current buffer content and refills starting at the current file position.
    void Reset()
    {
        m_buffer.clear();
        m_index = 0;
        m_lineNumber = 0;
        m_done = false;
        Refill();
    }

    FileWrapper m_file;
};

}
