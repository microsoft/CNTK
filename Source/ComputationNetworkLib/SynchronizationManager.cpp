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

inline int SampleSize(){ return 100; }
void PrintPtrAttributes(float *ptr)
{
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att,ptr);
    cout << "memtype: " << att.memoryType << endl;
    cout << "device: " << att.device << endl;
    cout << "host: " << att.hostPointer << endl;
    cout << "device ptr: " << att.devicePointer << endl;
}

void PrintFreeMemory()
{
    size_t free, total;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    cout << "free memory: " << free << endl;
}

SynchronizationManager* SynchronizationManager::s_synchronizationManager = nullptr;
SynchronizationManager* SynchronizationManager::GetSynchronizationManager(float performanceCostLimit)
{
    if (SynchronizationManager::s_synchronizationManager == NULL)
    {
        SynchronizationManager::s_synchronizationManager = new SynchronizationManager();
        SynchronizationManager::s_synchronizationManager->m_currentState = Uninitialized;
        SynchronizationManager::s_synchronizationManager->m_currentStepNumber = 0;
        SynchronizationManager::s_synchronizationManager->m_timer = CUDATimer();
        SynchronizationManager::s_synchronizationManager->m_performanceCostLimit = 0.15f;
        SynchronizationManager::s_synchronizationManager->m_isExecuting = false;
    }

    return SynchronizationManager::s_synchronizationManager;
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
        }
    }
}

void SynchronizationManager::ClearActionsAndTheirMemory()
{
    for(std::pair<std::string, std::vector<SyncAction*> > pair : m_stepName2Actions)
    {
       for(SyncAction *action : pair.second)
           action->ReleaseMemory();
       pair.second.clear();
    }
    m_stepName2Actions.clear();

    m_isExecuting = false;
}


void SynchronizationManager::FindSwapOrder()
{
    //TODO: refactor this, current is not needed at all
    float totalMemorySwappedInMB = 0.0f;
    float totalMemoryInMB  = 0.0f;
    for(Matrix<float>* buffer : m_bufferSet)
    {
        totalMemoryInMB += buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float)/1024.0f/1024;
        for(int i = 0; i < m_buffer2StepNumbers[buffer].size(); i++)
        {
            bool isSwappable = true;
            // by default we want to swap out the inputs to the current operation
            // this means the inputs to a matrix multiplication or convolution operation
            // that is we swap out the activities of the previous layer
            int swapOutStepNumber = m_buffer2StepNumbers[buffer][i];
            // when the buffer is needed for the next time
            int neededByStepNumber = (i+1) >= m_buffer2StepNumbers[buffer].size() ?
                                             m_buffer2StepNumbers[buffer][0] :
                                             m_buffer2StepNumbers[buffer][i+1];

            cout << "swap out: " << swapOutStepNumber << " needed by step: " << neededByStepNumber << endl;

            // by default we try to swap one timestep before the buffer is needed
            // if this does not work out we will look for timesteps before that and
            // try to fit the swap-in operation there
            int swapInStepNumber = neededByStepNumber-1; 
            // we get memory benefits only if we at least have one step with no swap operation
            if(swapInStepNumber < swapOutStepNumber+1){ continue; } 
            // this is a special case not worth handling
            if(swapOutStepNumber > m_currentStepNumber-1){ continue; }
            if(m_stepNumber2Stats.count(swapOutStepNumber) == 0)
            {
              cout << "no stats" << endl;
            }
            if(m_stepNumber2Stats.count(swapOutStepNumber) == 0){ continue; }

            float swapOutTime = m_buffer2SwapTime[buffer].first;
            float swapInTime = m_buffer2SwapTime[buffer].second;
            float computationTimeOut = m_stepNumber2Stats[swapOutStepNumber]->computationTime;
            
            if(m_stepNumber2CumulativeSwapInTime.count(swapOutStepNumber) == 0)
            { m_stepNumber2CumulativeSwapInTime[swapOutStepNumber] = std::make_pair(0.0f,0.0f); }
            float cumulativeSwapOutTime = m_stepNumber2CumulativeSwapInTime[swapOutStepNumber].first;
            // we just look if we can swap under the current operation
            // we can make it more complex anytime later to also look at other timesteps
            if(swapOutTime + cumulativeSwapOutTime > 
               computationTimeOut + (computationTimeOut * m_performanceCostLimit))
            {
                cout << "------------------" << endl;
                cout << m_stepNumber2StepName[swapOutStepNumber] << endl;
                cout << swapOutTime << " vs. " << computationTimeOut << endl;
                cout << swapOutTime + cumulativeSwapOutTime << " vs. " << computationTimeOut << endl;
                cout << "Memory: " << ((float)buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float))/1024.0f/1024.0f << endl;
                cout << "------------------" << endl;
            }
            else
            {
                cout << "------------------" << endl;
                cout << "FITS" << endl;
                cout << m_stepNumber2StepName[swapOutStepNumber] << endl;
                cout << swapOutTime << " vs. " << computationTimeOut << endl;
                cout << swapOutTime + cumulativeSwapOutTime << " vs. " << computationTimeOut << endl;
                cout << "Memory: " << ((float)buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float))/1024.0f/1024.0f << endl;
                cout << "------------------" << endl;

            }

            if(swapOutTime + cumulativeSwapOutTime > 
               computationTimeOut + (computationTimeOut * m_performanceCostLimit))
            { continue; }
            // find a place where we can swap-in the buffer just in time when it is needed
            while(swapInStepNumber > swapOutStepNumber+1)
            {
                if(m_stepNumber2Stats.count(swapInStepNumber) == 0){ swapInStepNumber--; continue; }

                Stats *s = m_stepNumber2Stats[swapInStepNumber];
                float computationTimeIn = s->computationTime;
                float cumulativeSwapInTime = m_stepNumber2CumulativeSwapInTime[swapInStepNumber].second;
                if(swapInTime + cumulativeSwapInTime < computationTimeIn  + (computationTimeIn * m_performanceCostLimit))
                { break; }
                
                swapInStepNumber--;
            }

            if(swapInStepNumber < swapOutStepNumber+1){ continue; } 

            // we found a suitable pair of swap-in and swap-out
            // 1. create swap actions and register them with a step-name (=step-number)
            // 2. add to the cumulative swap time, that is additional swap operations
            //    under the same time steps are more expensive
            SwapOutAction *swpOut = new SwapOutAction(buffer);
            SwapInAction *swpIn = new SwapInAction(swpOut->GetCPUMatrix(), buffer);

            m_stepName2Actions[m_stepNumber2StepName[swapOutStepNumber]].push_back(swpOut);
            m_stepName2Actions[m_stepNumber2StepName[swapInStepNumber]].push_back(swpIn);

            m_stepNumber2CumulativeSwapInTime[swapOutStepNumber].first += swapOutTime;
            m_stepNumber2CumulativeSwapInTime[swapInStepNumber].second += swapInTime;

            totalMemorySwappedInMB += buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float)/1024.0f/1024;
        }

    }
    cout << "Total swappable memory: " << totalMemorySwappedInMB << "MB" << endl;
    cout << "Total memory: " << totalMemoryInMB << "MB" << endl;
}

void SynchronizationManager::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{

    std::string name = GetStepName(node, isForward);
    if(m_currentState < ExecutingActions)
    {
        if(m_stepName2StepNumber.count(name) > 0)
        {

            if(m_stepName2StepNumber[name] == 0 && isForward == true)
            {
                m_currentState = FindingSwapOrder;
            }
        }
    }

    if(m_currentState <= FindingSwapOrder)
    {
    	RegisterBuffers(node, isForward);
    	//SwapInFreedBuffers(node, isForward);
    	GatherRuntimeStatistics(node, idx, fr, isForward);
    }

    if(m_currentState == FindingSwapOrder)
    {
        cout << "SWAP CALL " << endl;
        FindSwapOrder();
        CleanUp();
        m_currentState = ExecutingActions;
        m_isExecuting = true;
    }

    //SwapInFreedBuffers(node, isForward);
    if(m_currentState == ExecutingActions)
    {
        std::string name = GetStepName(node, isForward);
        std::vector<SyncAction*> actionsToDo = m_stepName2Actions[name];
        for (int i = 0; i < actionsToDo.size(); i++)
        {
               //cout << name << endl;
               actionsToDo[i]->BeginAction();
        }

    }
}

void SynchronizationManager::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{

    if(m_currentState <= GatheringRuntimeStatistics)
    {
        // end synchronize is called after the forward / backward pass, thus we can free
        // the memory now
        //FreeBuffersForDryRun(node, isForward);
    }

    if(m_currentState == ExecutingActions)
    {
        std::string name = GetStepName(node, isForward);
        std::vector<SyncAction*> actionsToDo = m_stepName2Actions[name];
        if (actionsToDo.size() == 0){ return; }

        for (int i = 0; i < actionsToDo.size(); i++)
        {
            actionsToDo[i]->EndAction();
        }
    }
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

            //TODO: set GPU id -> resize does this automatically?
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
    if(m_stepName2StepNumber.count(name) > 0)
    {
        // already registered
        return;
    }
    if(m_stepNumber2StepName.count(m_currentStepNumber) > 0)
    { m_currentStepNumber++; }

    m_stepName2StepNumber[name] = m_currentStepNumber;
    m_stepNumber2StepName[m_currentStepNumber] = name;
    m_stepNumber2IsForward[m_currentStepNumber] = isForward;
    // 0 == special value for flow control node, who does not have any buffers
    if(inputCount == 0){ return; }
    for(int i = 0; i < inputCount; i++)
    {
       Matrix<float> *buffer = (Matrix<float>*)node->Input(i)->ValuePtr().get();
       m_stepNumber2Buffer[m_currentStepNumber].push_back(buffer);
       m_buffer2StepNumbers[buffer].push_back(m_currentStepNumber);

       m_bufferSet.insert(buffer);
    }

    cout << m_currentStepNumber << " " << name << endl;

    //TODO: Are these buffers needed?
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->ValuePtr().get());
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->GradientPtr().get());

    //m_buffer2StepNumbers[node->ValuePtr().get()].push_back(m_currentStepNumber);
    //m_buffer2StepNumbers[node->GradientPtr().get()].push_back(m_currentStepNumber);
}  


void SynchronizationManager::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
    // swapping sharable nodes does not save any memory
    //if(node->IsValueSharable()){ return; }

    std::string name = GetStepName(node, isForward);
    // it is difficult to sample these operations as the CUDA compiler will remove duplicate
    // operations within a loop; so instead we synchronize the device and hope that our
    // measurement is quite reliable (it often is)
    CUDA_CALL(cudaDeviceSynchronize());
    m_timer.tick(name);
    if(isForward)
    {
        node->ForwardPropSpecialization(fr);
    }
    else
    {
        node->BackpropToSpecialization(idx, fr);
    }
    float t = m_timer.tock(name);
    Stats *s = new Stats();

    m_stepName2Stats[name] = s;
    m_stepNumber2Stats[m_stepName2StepNumber[name]] = s;
    
    s->name = name;
    s->computationTime = t;

    MeasureSwapTime(node, name);
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
           float swapOutTime = t/SampleSize();
           s->swapOutTimes.push_back(swapOutTime);

           m_timer.tick("Swap in");
           for(int i = 0; i < SampleSize(); i++)
           {
               in->BeginAction();
               in->EndAction();
           }
           t = m_timer.tock("Swap in");
           float swapInTime = t/SampleSize();
           s->swapInTimes.push_back(swapInTime);
           size_t rows = out->GetGPUMatrix()->GetNumRows();
           size_t cols = out->GetGPUMatrix()->GetNumCols();
           s->dim.push_back(std::to_string(rows) + "x" + std::to_string(cols));
           m_buffer2SwapTime[input] = std::make_pair(swapOutTime, swapInTime);
       }
    }
}

}}}
