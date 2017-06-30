//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "IndexBuilder.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MLFIndexBuilder : public IndexBuilder, boost::noncopyable
    {
    public:
        MLFIndexBuilder(const std::wstring& filename, FILE* file, CorpusDescriptorPtr corpus);

    private:

        virtual std::wstring GetCacheFilename() override;
        virtual void Populate(shared_ptr<Index>& index) override;

        enum class State
        {
            Header,
            UtteranceKey,
            UtteranceFrames
        };

        inline bool TryParseSequenceKey(const std::string& line, size_t& id, std::function<size_t(const std::string&)> keyToId);
    };

}}} // namespace
