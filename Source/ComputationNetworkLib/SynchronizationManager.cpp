//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings


#include "SynchronizationManager.h"
#include "ComputationNode.h"
#include "Sequences.h"
#include <iostream>
#include "SwapInAction.h"
#include "SwapOutAction.h"

namespace Microsoft { namespace MSR { namespace CNTK {


using std::cout;
using std::endl;

SynchronizationManager* SynchronizationManager::s_synchronizationManager = nullptr;

SynchronizationManager* SynchronizationManager::GetSynchronizationManager()
{
    if (SynchronizationManager::s_synchronizationManager == NULL)
    {
        SynchronizationManager::s_synchronizationManager = new SynchronizationManager();
        SynchronizationManager::s_synchronizationManager->m_currentState = Uninitialized;
        SynchronizationManager::s_synchronizationManager->m_currentStepNumber = 0;
        SynchronizationManager::s_synchronizationManager->m_timer = CUDATimer();
    }

    return SynchronizationManager::s_synchronizationManager;
}


void SynchronizationManager::SynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
    switch(m_currentState)
    {
        case Uninitialized:
            m_currentState = RegisteringBuffers;
            SynchronizeState(node, idx, fr, isForward);
            break;
        case RegisteringBuffers:
            if(m_stepName2StepNumber.count(GetStepName(node, isForward)) > 0)
            {
                m_currentState = GatheringRuntimeStatistics;
                SynchronizeState(node, idx, fr, isForward);
                break;
            }
            else
            {
                //register node
                m_stepName2StepNumber[GetStepName(node, isForward)] = m_currentStepNumber;
                  
                std::string name = GetStepName(node, isForward);
                fprintf(stdout, "%s stepNo: %i \n", name.c_str(),m_currentStepNumber) ;
                cout << GetStepName(node, isForward) << " is forward: " << isForward << endl;

                RegisterBuffers(node);
                m_currentStepNumber++;
            }
            break;
        case GatheringRuntimeStatistics:
            GatherRuntimeStatistics(node, idx, fr, isForward);
            break;
        case Benchmarking:
        default:
            break;
    }
}


inline std::string BoolToString(bool b){ return b ? std::string("true") : std::string("false"); }
std::string SynchronizationManager::GetStepName(ComputationNodeBase *node, bool isForward)
{
    
    int inputCount = node->GetNumInputs();
    std::string name = "";
    for(int i = 0; i < inputCount; i++)
    {
       name += std::to_string((long)node->Input(i)->ValuePtr().get()); 
    }

    name += std::to_string((long)node->ValuePtr().get()); 
    name += std::to_string((long)node->GradientPtr().get()); 

    name += BoolToString(isForward);
   

    return name;
}


void SynchronizationManager::RegisterBuffers(ComputationNodeBase *node)
{
    
    int inputCount = node->GetNumInputs();
    for(int i = 0; i < inputCount; i++)
    {
       m_stepNumber2Buffer[m_currentStepNumber].push_back(node->Input(i)->ValuePtr().get());
       m_buffer2StepNumbers[node->Input(i)->ValuePtr().get()].push_back(m_currentStepNumber);
    }

    m_stepNumber2Buffer[m_currentStepNumber].push_back(node->ValuePtr().get());
    m_stepNumber2Buffer[m_currentStepNumber].push_back(node->GradientPtr().get());

    m_buffer2StepNumbers[node->ValuePtr().get()].push_back(m_currentStepNumber);
    m_buffer2StepNumbers[node->GradientPtr().get()].push_back(m_currentStepNumber);
}  


void SynchronizationManager::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
    int sampleSize = 100;
    m_timer.tick(GetStepName(node, isForward));
    for(int i = 0; i < sampleSize; i++)
    {
        if(isForward)
        {
            m_currentState = Benchmarking;
            node->ForwardProp(fr);
            m_currentState = GatheringRuntimeStatistics;
        }
        else
        {
            m_currentState = Benchmarking;
            node->BackpropTo(idx, fr);
            m_currentState = GatheringRuntimeStatistics;
        }
    }
    std::string name = GetStepName(node, isForward);
    float t = m_timer.tock(name);

    if(m_stepName2Stats.count(name) == 0){ m_stepName2Stats[name] = Stats(); }
    else
    {
        m_currentState = FindingSwapOrder;
    }


    Stats s = m_stepName2Stats[name];
    if(isForward){ s.forwardTime = t/sampleSize; }
    else{ s.backpropTime = t/100.0f; }

    int inputCount = node->GetNumInputs();
    for(int i = 0; i < inputCount; i++)
    {
       Matrix<float> *input = (Matrix<float>*)node->Input(i)->ValuePtr().get();
       if(input != NULL)
       {
           SwapOutAction *out = new SwapOutAction(input);
           SyncAction *in =  new SwapInAction(out->GetCPUMatrix(), out->GetGPUMatrix(), out->GetSwapSteam());


           m_timer.tick("Swap out");
           for(int i = 0; i < sampleSize ; i++)
           {
               out->executeAction();
               cudaStreamSynchronize(out->GetSwapSteam());
           }
           t = m_timer.tock("Swap out");
           s.swapOutTimes.push_back(t/sampleSize);
           
           cout << "out: " << t << endl;

           m_timer.tick("Swap in");
           for(int i = 0; i < sampleSize; i++)
           {
               // swap in
               in->executeAction();
               // synchronize
               in->executeAction();
           }
           t = m_timer.tock("Swap in");
           s.swapInTimes.push_back(t/100.0f);

           cout << "in: " << t << endl;
       }
    }

    std::wstring wname = node->GetName();

    cout << std::string(wname.begin(), wname.end()) << endl;
    s.PrintStats();

    

}

void SynchronizationManager::ExecuteActions(ComputationNodeBase *node)
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
