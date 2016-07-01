//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"

#include <unordered_map>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {


class ComputationNodeBase;
typedef std::shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

class SynchronizationManager
{

private:
    typedef std::shared_ptr<SyncAction> SyncActionPtr;
    std::unordered_map<ComputationNodeBasePtr, std::vector<SyncActionPtr> > m_actionTable;
    SynchronizationManager(){};
    static std::shared_ptr<SynchronizationManager> s_SynchronizationManager;

    

public:
    ~SynchronizationManager(){};
    static std::shared_ptr<SynchronizationManager> GetSynchronizationManager()
    {
        if (SynchronizationManager::s_SynchronizationManager == NULL)
        {
            SynchronizationManager::s_SynchronizationManager = std::shared_ptr<SynchronizationManager>(new SynchronizationManager());
        }

        return SynchronizationManager::s_SynchronizationManager;
    }

    void SynchronizeState(ComputationNodeBasePtr node);



	
};


}}}

