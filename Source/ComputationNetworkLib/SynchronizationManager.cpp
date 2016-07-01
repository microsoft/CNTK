//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings


#include "SynchronizationManager.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::shared_ptr<SynchronizationManager> SynchronizationManager::s_SynchronizationManager = nullptr;

    void SynchronizationManager::SynchronizeState(ComputationNodeBasePtr node)
{
    
    std::vector<SyncActionPtr> actionsToDo = m_actionTable[node];

    if (actionsToDo.size() == 0){ return; }

    // 1. first execute all asynchronous actions
    // 2. then execute all synchronous actions (while the asynchronous are already running)

    for (int i = 0; i < actionsToDo.size(); i++)
    {
        if (actionsToDo[i]->GetIsAsynchronous())
            actionsToDo[i]->executeAction();
    }

    for (int i = 0; i < actionsToDo.size(); i++)
    {
        if (!actionsToDo[i]->GetIsAsynchronous())
            actionsToDo[i]->executeAction();
    }

}

}}}
