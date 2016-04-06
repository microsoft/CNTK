//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "PackerBase.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A sequence packer that packs dense or sparse samples in dense minibatch for parallel GPU consumption.
class SequencePacker : public PackerBase
{
public:
    SequencePacker(
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

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}}}
