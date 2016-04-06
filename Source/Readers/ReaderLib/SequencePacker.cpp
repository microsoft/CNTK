//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "SequencePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

MBLayoutPtr SequencePacker::CreateMBLayout(const StreamBatch& batch)
{
    vector<MBLayout::SequenceInfo> infos;
    for (size_t index = 0; index < batch.size(); ++index)
    {
        MBLayout::SequenceInfo info;

        info.seqId = index;
        info.tBegin = 0;
        info.tEnd = batch[index]->m_numberOfSamples;
        infos.push_back(info);
    }

    vector<pair<size_t, size_t>> placement;
    vector<size_t> rowAllocations;

    // Creating the minibatch layout.
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    pMBLayout->InitAsPackedSequences(infos, placement, rowAllocations);
    return pMBLayout;
}

}}}
