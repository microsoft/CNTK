//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/noncopyable.hpp>

#include "IndexBuilder.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MLFIndexBuilder : public IndexBuilder, boost::noncopyable
    {
    public:
        MLFIndexBuilder(const std::wstring& filename, FILE* file, CorpusDescriptorPtr corpus);

        MLFIndexBuilder& SetFrameMode(bool frameMode) { m_frameMode = frameMode; return *this; }

    private:

        virtual std::wstring GetCacheFilename() override;
        virtual void Populate(shared_ptr<Index>& index) override;

        enum class State
        {
            Header,
            UtteranceKey,
            UtteranceFrames
        };


        bool m_done;   // true, when all input was processed

        bool m_frameMode;

        std::vector<char> m_buffer;               // Buffer for data.
        std::vector<boost::iterator_range<char*>> m_lines; 
        int64_t m_fileOffsetStart;                // Current start offset in file that is mapped to m_buffer.
        std::string m_lastPartialLineInBuffer;    // Partial string from the previous read of m_buffer.
        std::string m_lastNonEmptyLine;           // Last non empty estring, used for parsing sequence length.
        



        // fills up the buffer with data from file, all previously buffered data
        // will be overwritten.
        void RefillBuffer();

        // Read lines from the buffer.
        void ReadLines();
        bool TryParseSequenceKey(const boost::iterator_range<char*>& line, size_t& id, std::function<size_t(const std::string&)> keyToId);
    };

}}} // namespace
