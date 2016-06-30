//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class SwapInAction : public SyncAction
{

public:

    SwapInAction::SwapInAction()
    {
        m_bufferCPU = NULL;
        m_bufferGPU = NULL;
        m_isAsynchronous = true;
    }

    //implementation of abstract method
    void SwapInAction::executeAction();

};
}}}
