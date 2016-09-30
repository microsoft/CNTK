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
        SequenceEnumeratorPtr sequenceEnumerator,
        const std::vector<StreamDescriptionPtr>& streams,
        size_t numberOfBuffers = 2) :
        PackerBase(sequenceEnumerator, streams, numberOfBuffers)
    {}

    virtual Minibatch ReadMinibatch() override;

protected:
    virtual MBLayoutPtr PackDenseStream(const StreamBatch& batch, size_t streamIndex);

    virtual MBLayoutPtr PackSparseStream(const StreamBatch& batch, size_t streamIndex);

    // Given a number of sequences, creates an MB layout that is used to guide
    // the actual packing.
    virtual MBLayoutPtr CreateMBLayout(const StreamBatch& batch);

    // Helper function to check the sample shape of input samples.
    void CheckSampleShape(const std::vector<SequenceDataPtr>& minibatch, StreamDescriptionPtr outputStream);
};

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}}}
