//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "IndexBuilder.h"

namespace CNTK {

    class MLFIndexBuilder : public IndexBuilder
    {
    public:
        MLFIndexBuilder(const FileWrapper& input, CorpusDescriptorPtr corpus);

        virtual std::wstring GetCacheFilename() override;

    private:

        virtual void Populate(std::shared_ptr<Index>& index) override;

        enum class State
        {
            Header,
            UtteranceKey,
            UtteranceFrames
        };

        inline bool TryParseSequenceKey(const std::string& line, size_t& id, std::function<size_t(const std::string&)> keyToId);
    };

} // namespace
