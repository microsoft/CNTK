//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include "XPacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {


MBLayoutPtr XPacker::CreateMBLayout(const StreamBatch& batch)
{
    size_t maxLength = 0;

    vector<MBLayout::SequenceInfo> infos;
    for (size_t index = 0; index < batch.size(); ++index)
    {
        MBLayout::SequenceInfo info;

        info.seqId = index;
        info.s = index;
        info.tBegin = 0;
        info.tEnd = batch[index]->m_numberOfSamples;
        infos.push_back(info);

        if (maxLength < info.tEnd)
        {
            maxLength = info.tEnd;
        }
    }

    // Creating the minibatch layout.
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    pMBLayout->Init(infos.size(), maxLength);

    for (const auto& info : infos)
    {
        pMBLayout->AddSequence(info.seqId, info.s, (ptrdiff_t)info.tBegin, info.tBegin + info.GetNumTimeSteps());
        //if (info.tEnd < maxLength) 
    }

    for (const auto& info : infos)
    {
        pMBLayout->AddGap(info.s, (size_t)info.GetNumTimeSteps(), maxLength);
    }
    
    return pMBLayout;
}

} } }
