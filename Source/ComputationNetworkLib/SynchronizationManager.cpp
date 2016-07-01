//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings


#include "SynchronizationManager.h"
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {



std::shared_ptr<SynchronizationManager> SynchronizationManager::s_synchronizationManager = nullptr;

void SynchronizationManager::SynchronizeState(ComputationNodeBasePtr node)
{
    switch(m_currentState)
    {
        case Uninitialized:
            m_currentState = RegisteringBuffers;
            SynchronizeState(node);
            break;
        case RegisteringBuffers:
            if(m_stepName2StepNumber.count(GetStepName(node)) > 0)
            {
                m_currentState = GatheringRuntimeStatistics;
                SynchronizeState(node);
                break;
            }
            else
            {
                //register node
                m_stepName2StepNumber[GetStepName(node)] = m_currentStepNumber;

                RegisterBuffers(node);
                m_currentStepNumber++;
            }
            break;
    }
}


std::string SynchronizationManager::GetStepName(ComputationNodeBasePtr node)
{
    
    int inputCount = node->GetNumInputs();
    std::string name = "";
    for(int i = 0; i < inputCount; i++)
    {
       name += std::to_string((long)node->Input(i)->ValuePtr().get()); 
    }

   name += std::to_string((long)node->ValuePtr().get()); 
   //TODO: do I need this to make the hash unique?
   //name += std::to_string((long)(&(node->Gradient()))); 
   

   return name;
}


void SynchronizationManager::RegisterBuffers(ComputationNodeBasePtr node)
{
}

void SynchronizationManager::ExecuteActions(ComputationNodeBasePtr node)
{
    std::vector<SyncActionPtr> actionsToDo = m_actionTable[node];

    if (actionsToDo.size() == 0){ return; }

    // 1. first execute all asynchronous actions
    // 2. then execute all synchronous actions (while the asynchronous are already running)

    for (int i = 0; i < actionsToDo.size(); i++)
    {
        // async actions
        if (actionsToDo[i]->GetIsAsynchronous())
            actionsToDo[i]->executeAction();
    }

    for (int i = 0; i < actionsToDo.size(); i++)
    {
        // sync actions
        if (!actionsToDo[i]->GetIsAsynchronous())
            actionsToDo[i]->executeAction();
    }
}

}}}
