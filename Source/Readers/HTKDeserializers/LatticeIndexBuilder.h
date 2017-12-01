//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "IndexBuilder.h"

namespace CNTK {

    class LatticeIndexBuilder : public IndexBuilder
    {
    public:
        LatticeIndexBuilder(const FileWrapper& latticeFile, const std::vector<std::string>& latticeToc, CorpusDescriptorPtr corpus);

        virtual std::wstring GetCacheFilename() override;

    private:
        virtual void Populate(std::shared_ptr<Index>& index) override;
        std::vector<std::string> m_latticeToc;
    };

} // namespace
