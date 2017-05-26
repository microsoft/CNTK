//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/noncopyable.hpp>

#include "Indexer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MLFIndexer : boost::noncopyable
    {
    public:
        MLFIndexer(FILE* file, bool frameMode, size_t chunkSize = 64 * 1024 * 1024, size_t bufferSize = 64 * 1024 * 1024);

        void Build(CorpusDescriptorPtr corpus);

        // Returns input data index (chunk and sequence metadata)
        const Index& GetIndex() const { return m_index; }

    private:
        enum class State
        {
            Header,
            UtteranceKey,
            UtteranceFrames
        };

        FILE* m_file;  // MLF file descriptor
        bool m_done;   // true, when all input was processed

        const size_t m_maxBufferSize;             // Max allowed buffer size.
        std::vector<char> m_buffer;               // Buffer for data.
        int64_t m_fileOffsetStart;                // Current start offset in file that is mapped to m_buffer.
        std::string m_lastPartialLineInBuffer;    // Partial string from the previous read of m_buffer.

        Index m_index;

        std::string m_lastNonEmptyLine;           // Last non empty estring, used for parsing sequence length.

        // fills up the buffer with data from file, all previously buffered data
        // will be overwritten.
        void RefillBuffer();

        // Read lines from the buffer.
        void ReadLines(vector<char>& buffer, vector<boost::iterator_range<char*>>& lines);
        bool TryParseSequenceKey(const boost::iterator_range<char*>& line, size_t& id, std::function<size_t(const std::string&)> keyToId);
    };

    typedef std::shared_ptr<MLFIndexer> MLFIndexerPtr;

}}} // namespace
