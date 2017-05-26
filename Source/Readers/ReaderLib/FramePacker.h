//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SequencePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

//A packer optimized for the case of single-frame sequences.
class FramePacker : public SequencePacker
{
public:
    FramePacker(
        SequenceEnumeratorPtr sequenceEnumerator,
        const std::vector<StreamDescriptionPtr>& streams,
        size_t numberOfBuffers = 2,
        bool useLocalTimeline = false,
        CorpusDescriptorPtr corpus = nullptr) :
        SequencePacker(sequenceEnumerator, streams, numberOfBuffers, useLocalTimeline, corpus)
    {}

protected:
    MBLayoutPtr CreateMBLayout(const StreamBatch& batch) override;
};

typedef std::shared_ptr<FramePacker> FramePackerPtr;
} } }
