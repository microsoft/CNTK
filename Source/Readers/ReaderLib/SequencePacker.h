//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "PackerBase.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This packer generates minibatches containing full sequences packed for 
// efficient (concurrent) consumption on a GPU.
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

    virtual Minibatch ReadMinibatch() override;

protected:
    virtual MBLayoutPtr PackDenseStream(const StreamBatch& batch, size_t streamIndex);

    virtual MBLayoutPtr PackSparseStream(const StreamBatch& batch, size_t streamIndex);

    // Given a number of sequences, creates an MB layout that is used to guide
    // the actual packing.
    virtual MBLayoutPtr CreateMBLayout(const StreamBatch& batch);
};

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}}}
