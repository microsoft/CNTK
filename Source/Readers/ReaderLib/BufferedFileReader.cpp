//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include "BufferedFileReader.h"

namespace CNTK {

using namespace std;

BufferedFileReader::BufferedFileReader(size_t maxSize, const FileWrapper& file) :
    BufferedReaderBase(maxSize),
    m_file(file)
{
    m_file.CheckIsOpenOrDie();

    if (maxSize == 0)
        RuntimeError("Max buffer size cannot be zero.");

    m_buffer.reserve(maxSize);

    Refill();
}

void BufferedFileReader::Refill()
{
    if (m_done)
        return;

    m_index = 0;
    m_fileOffset = m_file.TellOrDie();

    m_buffer.resize(m_maxSize);
    size_t bytesRead = m_file.Read(m_buffer.data(), 1, m_maxSize);

    if (bytesRead != m_maxSize && !m_file.ReachedEOF())
        RuntimeError("Error reading file '%ls': %s.", m_file.Filename().c_str(), strerror(errno));
        
    m_buffer.resize(bytesRead);
    m_done = (bytesRead == 0);
}

}
