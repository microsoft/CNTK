//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"
#include <unordered_map>
#include <memory>
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {


// forward declarations
class ComputationNodeBase;
typedef std::shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

class SynchronizationManager
{

private:
    typedef std::shared_ptr<SyncAction> SyncActionPtr;
    std::unordered_map<ComputationNodeBasePtr, std::vector<SyncActionPtr> > m_actionTable;
    SynchronizationManager(){};
    static std::shared_ptr<SynchronizationManager> s_synchronizationManager;
    std::unordered_map<std::string, int> m_stepName2StepNumber;

    enum SynchronizationState
    {
        Uninitialized = 0,
        RegisteringBuffers = 1,
        GatheringRuntimeStatistics = 2,
        FindingSwapOrder = 4,
        GeneratingAllocationScheme = 8,
        ExecutingActions = 16
    };

    SynchronizationState m_currentState;
    int m_currentStepNumber;

    void ExecuteActions(ComputationNodeBasePtr node);
    void RegisterBuffers(ComputationNodeBasePtr node);

    std::string GetStepName(ComputationNodeBasePtr node);
public:
    ~SynchronizationManager(){};
    static std::shared_ptr<SynchronizationManager> GetSynchronizationManager()
    {
        if (SynchronizationManager::s_synchronizationManager == NULL)
        {
            SynchronizationManager::s_synchronizationManager = std::shared_ptr<SynchronizationManager>(new SynchronizationManager());
            SynchronizationManager::s_synchronizationManager->m_currentState = Uninitialized;
        }

        return SynchronizationManager::s_synchronizationManager;
    }

    void SynchronizeState(ComputationNodeBasePtr node);
	
};


}}}

