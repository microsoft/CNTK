//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include "FramePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {


MBLayoutPtr FramePacker::CreateMBLayout(const StreamBatch& batch)
{
    auto violation = find_if(batch.begin(), batch.end(), [](const SequenceDataPtr& s){ return s->m_numberOfSamples > 1; });
    if (violation != batch.end())
    {
        RuntimeError("Detected a non-frame sequence of size %d in frame mode.", 
            (int)(*violation)->m_numberOfSamples);
    }
    // Creating the minibatch layout.
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    pMBLayout->InitAsFrameMode(batch.size());
    return pMBLayout;
}

} } }
