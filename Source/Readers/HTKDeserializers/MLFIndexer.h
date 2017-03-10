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
        FILE* m_file;

        int64_t m_fileOffsetStart;
        int64_t m_fileOffsetEnd;

        std::vector<char> m_buffer;
        size_t m_bufferSize;
        bool m_done; // true, when all input was processed
        bool m_frameMode;

        Index m_index;

        // fills up the buffer with data from file, all previously buffered data
        // will be overwritten.
        void RefillBuffer();

        void ReadLines(vector<char>& buffer, vector<boost::iterator_range<char*>>& lines);
        bool TryParseSequenceId(const boost::iterator_range<char*>& line, size_t& id, std::function<size_t(const std::string&)> keyToId);

        std::string m_lastLineInBuffer;
        std::string m_lastNonEmptyLine;
    };

    typedef std::shared_ptr<MLFIndexer> MLFIndexerPtr;

}}} // namespace
