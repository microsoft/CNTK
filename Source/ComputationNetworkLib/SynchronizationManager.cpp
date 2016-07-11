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

inline int SampleSize(){ return 1; }

SynchronizationManager* SynchronizationManager::s_synchronizationManager = nullptr;


SynchronizationManager* SynchronizationManager::GetSynchronizationManager(float performanceCostLimit)
{
    if (SynchronizationManager::s_synchronizationManager == NULL)
    {
        SynchronizationManager::s_synchronizationManager = new SynchronizationManager();
        SynchronizationManager::s_synchronizationManager->m_currentState = Uninitialized;
        SynchronizationManager::s_synchronizationManager->m_currentStepNumber = 0;
        SynchronizationManager::s_synchronizationManager->m_timer = CUDATimer();
        SynchronizationManager::s_synchronizationManager->m_performanceCostLimit = performanceCostLimit;
    }

    return SynchronizationManager::s_synchronizationManager;
}

void SynchronizationManager::CheckForStateTransitions(ComputationNodeBase *node, bool isForward)
{
    std::string name = GetStepName(node, isForward);
    switch(m_currentState)
    {
        case Uninitialized:
            m_currentState = RegisteringBuffers;
            break;
        case RegisteringBuffers:
            if(m_stepName2StepNumber.count(name) > 0)
            {

                if(m_stepName2StepNumber[name] == 0)
                {
                    // we already encountered this node, and it is the first node
                    // so we made a full roundtrip and can change the state
                    cout << "-----------------" << endl;
                    cout << "SWITCHING TO STATS" << endl;
                    cout << "-----------------" << endl;
                    cout << GetStepName(node, isForward);
                    m_currentState = GatheringRuntimeStatistics;
                }
            }
            break;
        case GatheringRuntimeStatistics:
            if(m_stepName2Stats.count(name) > 0)
            {
                if(m_stepName2StepNumber[name] == 0)
                {
                    // we already recorded stats of this node and it is the first node
                    // thus we made a full roundtrip and change the state
                    cout << "-----------------" << endl;
                    cout << "SWITCHING TO FINDING SWAP ORDER" << endl;
                    cout << "-----------------" << endl;
                    cout << GetStepName(node, isForward);
                    m_currentState = FindingSwapOrder;
                }
            }
            break;
        case FindingSwapOrder:
        default:
            break;
    }
 
}

void PrintFreeMemory()
{
    size_t free, total;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    cout << "free memory: " << free << endl;
}

void SynchronizationManager::FindSwapOrder()
{
    

    
}

void SynchronizationManager::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
    CheckForStateTransitions(node, isForward);

    if(m_currentState <= RegisteringBuffers)
        RegisterBuffers(node);
    if(m_currentState <= GatheringRuntimeStatistics)
    {
        SwapInFreedBuffers(node, isForward);
        GatherRuntimeStatistics(node, idx, fr, isForward);
    }
    if(m_currentState == FindingSwapOrder)

    PrintFreeMemory();
}

void SynchronizationManager::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
    
    if(m_currentState <= GatheringRuntimeStatistics)
    {
        // end synchronize is called after the forward / backward pass, thus we can free
        // the memory now
        FreeBuffersForDryRun(node, isForward);
    }
}

void PrintPtrAttributes(float *ptr)
{
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att,ptr);
    cout << "memtype: " << att.memoryType << endl;
    cout << "device: " << att.device << endl;
    cout << "host: " << att.hostPointer << endl;
    cout << "device ptr: " << att.devicePointer << endl;
}


void SynchronizationManager::SwapInFreedBuffers(ComputationNodeBase *node, bool isForward)
{
    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    {
        Matrix<float> *buffer = m_stepNumber2Buffer[stepNumber][i]; 
        if(m_buffer2IsFreed[buffer])
        {
            SwapInAction *swp = m_buffer2SwapIn[buffer];
            float* ptr = buffer->Data();
            int rows = m_buffer2Dim[buffer].first;
            int cols = m_buffer2Dim[buffer].second;

            buffer->Resize(rows,cols);
            swp->BeginAction(); // initiate swapping
            swp->endAction(); // synchronize swapping
            m_buffer2IsFreed[buffer] = false;
        }
    }
}

void SynchronizationManager::FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward)
{
    // if a value is marked as shareable, it will be used right after, thus it makes
    // no sense to swap out these values (it just makes it more complicated), so instead
    // we just swap out the non-sharable ones (used in backprop). This will give us enough
    // memory to perform the dry run
    if(node->IsValueSharable()){ return; }

    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    {
        Matrix<float> *buffer = m_stepNumber2Buffer[stepNumber][i];
        if(buffer == NULL){ continue; }

        if(m_buffer2IsFreed.count(buffer) > 0)
            if(m_buffer2IsFreed[buffer])
                continue;

        m_buffer2Dim[buffer] = std::make_pair<int,int>(buffer->GetNumRows(), buffer->GetNumCols());
        buffer->Resize(0,0,false); // force a shrink to 0 memory
        m_buffer2IsFreed[buffer] = true;
    }
}

inline std::string BoolToString(bool b){ return b ? std::string("_forward") : std::string("_backprop"); }
std::string SynchronizationManager::GetStepName(ComputationNodeBase *node, bool isForward)
{
    std::wstring wname = node->GetName();
    return std::string(wname.begin(), wname.end()) + BoolToString(isForward);
}


void SynchronizationManager::RegisterBuffers(ComputationNodeBase *node)
{
    
    int inputCount = node->GetNumInputs();
    // 0 == special value for flow control node, who do not have any buffers
    cout << inputCount << endl;
    if(inputCount == 0){ return; }
    for(int i = 0; i < inputCount; i++)
    {
       m_stepNumber2Buffer[m_currentStepNumber].push_back((Matrix<float>*)node->Input(i)->ValuePtr().get());
       m_buffer2StepNumbers[(Matrix<float>*)node->Input(i)->ValuePtr().get()].push_back(m_currentStepNumber);
    }

    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->ValuePtr().get());
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->GradientPtr().get());

    //m_buffer2StepNumbers[node->ValuePtr().get()].push_back(m_currentStepNumber);
    //m_buffer2StepNumbers[node->GradientPtr().get()].push_back(m_currentStepNumber);
}  


void SynchronizationManager::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
    cout << "stats" << endl;
    m_timer.tick(GetStepName(node, isForward));
    for(int i = 0; i < SampleSize(); i++)
    {
        if(isForward)
        {
            node->ForwardPropSpecialization(fr);
        }
        else
        {
            node->BackpropToSpecialization(idx, fr);
        }
    }
    std::string name = GetStepName(node, isForward);
    float t = m_timer.tock(name);

        cout << m_stepName2StepNumber[name] << ": " << name << endl;
    Stats *s = new Stats();
    m_stepName2Stats[name] = s;
    s->name = name;
    if(isForward){ s->forwardTime = t/SampleSize(); }
    else{ s->backpropTime = t/SampleSize(); }

    MeasureSwapTime(node, name);


    //cout << name << endl;
    //s->PrintStats();
}


void SynchronizationManager::MeasureSwapTime(ComputationNodeBase *node, std::string name)
{
    float t = 0.0f;
    Stats *s = m_stepName2Stats[name];

    int inputCount = node->GetNumInputs();
    if(inputCount == 0){ return; }
    for(int i = 0; i < inputCount; i++)
    {
       Matrix<float> *input = (Matrix<float>*)node->Input(i)->ValuePtr().get();
       if(input != NULL)
       {
           SwapOutAction *out = new SwapOutAction(input);
           SwapInAction *in =  new SwapInAction(out->GetCPUMatrix(), out->GetGPUMatrix());
           m_buffer2SwapOut[input] = out;
           m_buffer2SwapIn[input] = in;
           cout << "ADDED SWAP FOR: " << input << endl;

           m_timer.tick("Swap out");
           for(int i = 0; i < SampleSize(); i++)
           {
               out->BeginAction();
               out->endAction();
           }
           t = m_timer.tock("Swap out");
           s->swapOutTimes.push_back(t/SampleSize());

           m_timer.tick("Swap in");
           for(int i = 0; i < SampleSize(); i++)
           {
               in->BeginAction();
               in->endAction();
           }
           t = m_timer.tock("Swap in");
           s->swapInTimes.push_back(t/SampleSize());
           size_t rows = out->GetGPUMatrix()->GetNumRows();
           size_t cols = out->GetGPUMatrix()->GetNumCols();
           s->dim.push_back(std::to_string(rows) + "x" + std::to_string(cols));
       }
       else{ cout << "IS NULL: " << input << endl; }
    }
    //cout << name << endl;
    //s->PrintStats();
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
        {
            actionsToDo[i]->BeginAction();
            actionsToDo[i]->endAction();
        }
    }

    for (int i = 0; i < actionsToDo.size(); i++)
    {
        // sync actions
        if (!actionsToDo[i]->GetIsAsynchronous())
        {
            actionsToDo[i]->BeginAction();
            actionsToDo[i]->endAction();
        }
    }
}

}}}
