//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "IndexBuilder.h"
#include <limits>

namespace CNTK {

    typedef unsigned int uint;
    typedef unsigned short ushort;

    const uint MAX_UTT_ID = std::numeric_limits<uint>::max();
    const uint MAX_SENONE_COUNT = std::numeric_limits<ushort>::max();
    const std::string MLF_BIN_LABEL = "MLF";
    const size_t SENONE_ZEROS = 100000;
    const ushort MAX_UTTERANCE_LABEL_LENGTH = 256;

    class MLFBinaryIndexBuilder : public IndexBuilder
    {
    public:
        MLFBinaryIndexBuilder(const FileWrapper& input, CorpusDescriptorPtr corpus);

        virtual std::wstring GetCacheFilename() override;

    private:

        virtual void Populate(std::shared_ptr<Index>& index) override;

        enum class State
        {
            Header,
            UtteranceKey,
            UtteranceFrames
        };
    };

} // namespace
