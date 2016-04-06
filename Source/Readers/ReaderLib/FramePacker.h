//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "PackerBase.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A sample packer that densely packs samples in parallel for GPU consumptions.
class FramePacker : public PackerBase
{
public:
    FramePacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        const std::vector<StreamDescriptionPtr>& streams) :
        PackerBase(memoryProvider, transformer, minibatchSize, streams)
    {

    }

private:
    MBLayoutPtr CreateMBLayout(const StreamBatch& batch) override;
};

typedef std::shared_ptr<FramePacker> FramePackerPtr;
} } }
