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
            cout << "Registering swap buffers..." << endl;
            break;
        case RegisteringBuffers:
            if(m_stepName2StepNumber.count(name) > 0)
            {
                if(m_stepName2StepNumber[name] == 0)
                {
                    // we already encountered this node, and it is the first node
                    // so we made a full roundtrip and can change the state
                    cout << "Gathering swapping stats..." << endl;
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
                    cout << "Finding swap order..." << endl;
                    m_currentState = FindingSwapOrder;
                }
            }
            break;
        case FindingSwapOrder:
            if(m_buffer2Swappable.size() > 0)
            {
                m_currentState = CleaningUp;
                cout << "Cleaning up unused buffers..." << endl;
            }
        case CleaningUp:
            if(m_buffer2Swappable.size() == m_buffer2SwapIn.size())
                m_currentState = ExecutingActions;
        default:
            break;
    }
 
}

void SynchronizationManager::CleanUp()
{
   // 1. remove all swap actions which are not needed, that is which are too slow
   // 2. remove stats and other temporary structures which are not needed for execution

    Matrix<float> *buffer;
    for(int i = 0; i < m_stepNumber2Buffer.size(); i++)
    {
        std::vector<Matrix<float> *> buffers = m_stepNumber2Buffer[i];

        for(int j = 0; j < buffers.size(); j++)
        {
            buffer = buffers[j];

            if(m_buffer2Swappable.count(buffer) == 0)
            {
                //m_buffer2SwapOut[buffer]->deallocatePinnedBuffer();

                m_buffer2SwapOut.erase(buffer);
                m_buffer2IsFreed.erase(buffer);
                m_buffer2SwapIn.erase(buffer);
                m_buffer2Dim.erase(buffer);
            }
            else
            {
            }
        }
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
    //TODO: refactor this, current is not needed at all
    Stats *prev = NULL;
    Stats *current = NULL;
    float totalMemorySwappedInMB = 0.0f;
    float totalMemoryNotSwappedInMB = 0.0f;
    for(int i = 0; i < m_vecStats.size(); i++)
    {
        current = m_vecStats[i];
        if(prev == NULL){ prev = current; continue; }


        float computeTime = prev->forwardTime > prev->backpropTime ? prev->forwardTime : prev->backpropTime;
        Matrix<float> *buffer;
        for(int j = 0 ; j < prev->buffers.size(); j++)
        {
            buffer = prev->buffers[j];
            float swapTime = prev->swapInTimes[j] > prev->swapOutTimes[j] ? prev->swapInTimes[j] : prev->swapOutTimes[j]; 

            //cout << "compute vs. swap: " << computeTime << " vs. " << swapTime << endl;
            if(swapTime < computeTime + (computeTime *m_performanceCostLimit))
            {
                //cout << "swap it" << endl;
                m_buffer2Swappable[buffer] = true;
                std::string name = prev->name;

                m_stepName2Actions[name].push_back(m_buffer2SwapOut[buffer]); 
                m_stepName2Actions[name].push_back(m_buffer2SwapIn[buffer]); 
                totalMemorySwappedInMB += buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float)*8.0/1024/1024;
            }
            else
            {
                m_buffer2Swappable[buffer] = false;
                totalMemoryNotSwappedInMB += buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float)*8.0/1024/1024;
            }

           //cout << "is swappable: " << m_buffer2Swappable[buffer] << " for buffer: " << buffer << endl;
        }
             
        prev = current;
    }
    cout << "Total swappable memory: " << totalMemorySwappedInMB << "MB" << endl;
    cout << "Total unswappable memory: " << totalMemoryNotSwappedInMB << "MB" << endl;
}

void SynchronizationManager::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{

    std::string name = GetStepName(node, isForward);




    if(m_stepName2StepNumber.count(name) > 0)
    {

    	if(m_stepName2StepNumber[name] == 0 && isForward == true)
    	{
    		cout << "NAME: " << name << endl;
        	m_currentState = FindingSwapOrder;
    	}
    }

    if(m_currentState <= FindingSwapOrder)
    {
    	RegisterBuffers(node, isForward);
    	SwapInFreedBuffers(node, isForward);
    	GatherRuntimeStatistics(node, idx, fr, isForward);
    }

    if(m_currentState == FindingSwapOrder)
    {
        FindSwapOrder();
        //CleanUp();
        m_currentState = ExecutingActions;
    }

    SwapInFreedBuffers(node, isForward);
    if(m_currentState == ExecutingActions)
    {
        std::string name = GetStepName(node, isForward);
        std::vector<SyncAction*> actionsToDo = m_stepName2Actions[name];
        if (actionsToDo.size() == 0){ return; }

        for (int i = 0; i < actionsToDo.size(); i++)
                actionsToDo[i]->BeginAction();
    }




    //PrintFreeMemory();
}

void SynchronizationManager::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{

    if(m_currentState <= GatheringRuntimeStatistics)
    {
        // end synchronize is called after the forward / backward pass, thus we can free
        // the memory now
        FreeBuffersForDryRun(node, isForward);
    }

    if(m_currentState == ExecutingActions)
    {
        std::string name = GetStepName(node, isForward);
        std::vector<SyncAction*> actionsToDo = m_stepName2Actions[name];
        if (actionsToDo.size() == 0){ return; }

        for (int i = 0; i < actionsToDo.size(); i++)
                actionsToDo[i]->EndAction();
    }
    m_currentStepNumber++;
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

            //TODO: set GPU id
            buffer->Resize(rows,cols);
            swp->BeginAction(); // initiate swapping
            swp->EndAction(); // synchronize swapping
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
    cout << "FREEING: " << name << endl;
    int stepNumber = m_stepName2StepNumber[name];
    for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    {
        Matrix<float> *buffer = m_stepNumber2Buffer[stepNumber][i];
        if(buffer == NULL){ continue; }

        if(m_buffer2IsFreed.count(buffer) > 0)
            if(m_buffer2IsFreed[buffer])
                continue;

        m_buffer2Dim[buffer] = std::make_pair<int,int>(buffer->GetNumRows(), buffer->GetNumCols());
        //TODO: set GPU id -> resize does this automatically?
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


void SynchronizationManager::RegisterBuffers(ComputationNodeBase *node, bool isForward)
{
    
    int inputCount = node->GetNumInputs();
    std::string name = GetStepName(node, isForward);
    m_stepName2StepNumber[name] = m_currentStepNumber;
    cout << name << " to " << m_currentStepNumber << endl;
    // 0 == special value for flow control node, who does not have any buffers
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

    Stats *s = new Stats();
    m_stepName2Stats[name] = s;
    m_vecStats.push_back(s);
    
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
       s->buffers.push_back(input); 
       if(input != NULL)
       {
           SwapOutAction *out = new SwapOutAction(input);
           SwapInAction *in =  new SwapInAction(out->GetCPUMatrix(), out->GetGPUMatrix());
           m_buffer2SwapOut[input] = out;
           m_buffer2SwapIn[input] = in;

           m_timer.tick("Swap out");
           for(int i = 0; i < SampleSize(); i++)
           {
               out->BeginAction();
               out->EndAction();
           }
           t = m_timer.tock("Swap out");
           s->swapOutTimes.push_back(t/SampleSize());

           m_timer.tick("Swap in");
           for(int i = 0; i < SampleSize(); i++)
           {
               in->BeginAction();
               in->EndAction();
           }
           t = m_timer.tock("Swap in");
           s->swapInTimes.push_back(t/SampleSize());
           size_t rows = out->GetGPUMatrix()->GetNumRows();
           size_t cols = out->GetGPUMatrix()->GetNumCols();
           s->dim.push_back(std::to_string(rows) + "x" + std::to_string(cols));
       }
    }
    //cout << name << endl;
    //s->PrintStats();
}

}}}
