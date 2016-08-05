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
#include "ComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {

SynchronizationManager<float>* g_floatSynchronizationManager = new SynchronizationManager<float>();
SynchronizationManager<double>* g_doubleSynchronizationManager = new SynchronizationManager<double>();

using std::cout;
using std::endl;

inline int SampleSize(){ return 100; }
inline float MeasurementUncertainty(){ return 1.15f; }


// this fetches the singleton instance
template <typename ElemType>
SynchronizationManager<ElemType>::SynchronizationManager()
{
    this->m_currentStepNumber = 0;
    this->m_timer = CUDATimer();
    this->m_isExecuting = false;
    this->m_useMemorySwapping = false;
    this->m_isInTrainingMode = false;
}

template <typename ElemType>
void SynchronizationManager<ElemType>::CleanUp()
{
   // 1. remove all swap actions which are not needed, that is which are too slow
   // 2. remove stats and other temporary structures which are not needed for execution
    Matrix<ElemType> *buffer;
    for(int i = 0; i < m_stepNumber2Buffer.size(); i++)
    {
        std::vector<Matrix<ElemType> *> buffers = m_stepNumber2Buffer[i];

        for(int j = 0; j < buffers.size(); j++)
        {
            buffer = buffers[j];

            if(m_buffer2SwapOut.count(buffer) == 0){ continue; }

            m_buffer2SwapOut[buffer]->ReleaseMemory();
            m_buffer2SwapOut.erase(buffer);
            m_buffer2SwapIn.erase(buffer);

            // adding this line causes and error during the 2nd actions, and I do not know why
            //m_buffer2IsFreed.erase(buffer); 
        }
    }
}

template<typename ElemType>
void SynchronizationManager<ElemType>::FindSwapOrder()
{
#ifndef CPUONLY
    float totalMemorySwappedInMB = 0.0f;
    float totalMemoryInMB  = 0.0f;
    for(Matrix<ElemType>* buffer : m_bufferSet)
    {
        totalMemoryInMB += buffer->GetNumRows()*buffer->GetNumCols()*sizeof(ElemType)/1024.0f/1024;
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

            //cout << "swap out: " << swapOutStepNumber << " needed by step: " << neededByStepNumber << endl;

            // by default we try to swap one timestep before the buffer is needed
            // if this does not work out we will look for timesteps before that and
            // try to fit the swap-in operation there
            int swapInStepNumber = neededByStepNumber-1; 
            // we get memory benefits only if we at least have one step with no swap operation
            if(swapInStepNumber < swapOutStepNumber+1){ continue; } 
            // this is a special case not worth handling
            if(swapOutStepNumber > m_currentStepNumber-1){ continue; }
            if(m_stepNumber2ComputationTime.count(swapOutStepNumber) == 0){ continue; } // do we have stats?

            float swapOutTime = m_buffer2SwapTime[buffer].first;
            float swapInTime = m_buffer2SwapTime[buffer].second;
            float computationTimeOut = m_stepNumber2ComputationTime[swapOutStepNumber];
            
            if(m_stepNumber2CumulativeSwapInTime.count(swapOutStepNumber) == 0)
            { m_stepNumber2CumulativeSwapInTime[swapOutStepNumber] = std::make_pair(0.0f,0.0f); }
            float cumulativeSwapOutTime = m_stepNumber2CumulativeSwapInTime[swapOutStepNumber].first;
            // we just look if we can swap under the current operation
            // we can make it more complex anytime later to also look at other timesteps
            /*
            if(m_performanceCostLimit*(swapOutTime + cumulativeSwapOutTime) > 
               computationTimeOut)
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
            */

            if(MeasurementUncertainty()*(swapOutTime + cumulativeSwapOutTime) > 
               computationTimeOut)
            { continue; }
            // find a place where we can swap-in the buffer just in time when it is needed
            while(swapInStepNumber > swapOutStepNumber+1)
            {
                if(m_stepNumber2ComputationTime.count(swapInStepNumber) == 0){ swapInStepNumber--; continue; }

                float computationTime = m_stepNumber2ComputationTime[swapInStepNumber];
                float computationTimeIn = computationTime;
                float cumulativeSwapInTime = m_stepNumber2CumulativeSwapInTime[swapInStepNumber].second;
                if(MeasurementUncertainty()*(swapInTime + cumulativeSwapInTime) < computationTimeIn)
                { break; }
                
                swapInStepNumber--;
            }

            if(swapInStepNumber < swapOutStepNumber+1){ continue; } 

            // we found a suitable pair of swap-in and swap-out
            // 1. create swap actions and register them with a step-name (=step-number)
            // 2. add to the cumulative swap time, that is additional swap operations
            //    under the same time steps are more expensive
            SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
            SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);

            m_stepNumber2Actions[swapOutStepNumber].push_back(swpOut);
            m_stepNumber2Actions[swapInStepNumber].push_back(swpIn);

            m_stepNumber2CumulativeSwapInTime[swapOutStepNumber].first += swapOutTime;
            m_stepNumber2CumulativeSwapInTime[swapInStepNumber].second += swapInTime;
            //m_stepNumber2CumulativeSwapInTime[swapOutStepNumber].first += 9999999.0f;
            //m_stepNumber2CumulativeSwapInTime[swapInStepNumber].second += 9999999.0f;

            totalMemorySwappedInMB += buffer->GetNumRows()*buffer->GetNumCols()*sizeof(float)/1024.0f/1024;
            fprintf(stderr, "Swapping buffer: %p with dim %zux%zu out at step %i and in at step %i\n", buffer, buffer->GetNumRows(), buffer->GetNumCols(), swapOutStepNumber, swapInStepNumber);
        }

    }

    fprintf(stderr, "Total swapped memory: %fMB\n", totalMemorySwappedInMB);
    fprintf(stderr, "Total swappable memory: %fMB\n", totalMemoryInMB);
#endif
}

template<typename ElemType>
void SynchronizationManager<ElemType>::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{

#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    if(!m_isExecuting)
    {
        bool allStatsGathered = false;
        std::string name = GetStepName(node, isForward);

        // the stats gathering ends when we are back at stepNumber 0, that is in the forward pass
        if(m_stepName2StepNumber.count(name) > 0)
            if(m_stepName2StepNumber[name] == 0 && isForward == true)
                allStatsGathered = true;
        
        //if(!allStatsGathered || !m_isInTrainingMode)
        if(!allStatsGathered)
        {
            RegisterBuffers(node, isForward);
            SwapInFreedBuffers(node, isForward);
            if(m_isInTrainingMode)
            {
            	GatherRuntimeStatistics(node, idx, fr, isForward);
            	CUDA_CALL(cudaDeviceSynchronize());
            	MeasureSwapTime(node, isForward);
            }
        }
        else
        {
            FindSwapOrder();
            CleanUp(); // release all cpu memory that was used in the dry run
            m_isExecuting = true;
        }
    }

    if(m_isExecuting)
    {
        std::string name = GetStepName(node, isForward);
        int stepNumber = m_stepName2StepNumber[name];
        std::vector<SyncAction<ElemType>*> actionsToDo = m_stepNumber2Actions[stepNumber];
        for (int i = 0; i < actionsToDo.size(); i++)
        {
               // criteron, evaluation and input nodes do not have a MB layout?
               // does not make sense to free those anyway
               if(node->HasMBLayout()) 
                   actionsToDo[i]->BeginAction();
        }

    }
#endif
}


template<typename ElemType>
void SynchronizationManager<ElemType>::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    if(!m_isExecuting)
    {
        // end synchronize is called after the forward / backward pass, thus we can free
        // the memory now
        FreeBuffersForDryRun(node, isForward);
    }
    else
    {
        std::string name = GetStepName(node, isForward);
        int stepNumber = m_stepName2StepNumber[name];
        std::vector<SyncAction<ElemType>*> actionsToDo = m_stepNumber2Actions[stepNumber];
        if (actionsToDo.size() == 0){ return; }

        for (int i = 0; i < actionsToDo.size(); i++)
        {
           // criteron, evaluation and input nodes do not have a MB layout?
           // does not make sense to free those anyway
           if(node->HasMBLayout()) 
                actionsToDo[i]->EndAction();
        }
    }
#endif
}

template<typename ElemType>
void SynchronizationManager<ElemType>::SwapInFreedBuffers(ComputationNodeBase *node, bool isForward)
{
    //if(node->IsValueSharable()){ return; }

    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    {
        Matrix<ElemType> *buffer = m_stepNumber2Buffer[stepNumber][i]; 

        if(m_buffer2IsFreed.count(buffer) == 0){ continue; }
        if(m_buffer2IsFreed[buffer])
        {
            SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
            swp->BeginAction(); // begin swap in
            swp->EndAction(); // end swap in
            m_buffer2IsFreed[buffer] = false;
        }
    }
}

template<typename ElemType>
void SynchronizationManager<ElemType>::FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward)
{
#ifndef CPUONLY
    // if a value is marked as shareable, it will be used right after, thus it makes
    // no sense to swap out these values (it just makes it more complicated), so instead
    // we just swap out the non-sharable ones (used in backprop). This will give us enough
    // memory to perform the dry run
    if(node->IsValueSharable()){ return; }

    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    {
        Matrix<ElemType> *buffer = m_stepNumber2Buffer[stepNumber][i];
        if(buffer == NULL){ continue; }

        if(m_buffer2IsFreed.count(buffer) > 0)
            if(m_buffer2IsFreed[buffer])
                continue; // buffer is already freed


        if(m_buffer2SwapIn.count(buffer) == 0)
        {
            SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
            SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
            m_buffer2SwapOut[buffer] = swpOut;
            m_buffer2SwapIn[buffer] = swpIn;
        }

        m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
        m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
        m_buffer2IsFreed[buffer] = true;

    }
#endif
}

inline std::string IsForwardToString(bool b){ return b ? std::string("_forward") : std::string("_backprop"); }
template<typename ElemType>
std::string SynchronizationManager<ElemType>::GetStepName(ComputationNodeBase *node, bool isForward)
{
    std::wstring wname = node->GetName();
    return std::string(wname.begin(), wname.end()) + IsForwardToString(isForward);
}


template<typename ElemType>
void SynchronizationManager<ElemType>::RegisterBuffers(ComputationNodeBase *node, bool isForward)
{
    
    int inputCount = node->GetNumInputs();
    std::string name = GetStepName(node, isForward);
    if(m_stepName2StepNumber.count(name) > 0)
    {
        // already registered
        return;
    }
    if(m_stepNumber2Buffer.count(m_currentStepNumber) > 0)
    { m_currentStepNumber++; }

    m_stepName2StepNumber[name] = m_currentStepNumber;
    // 0 == special value for flow control node, who does not have any buffers
    if(inputCount == 0){ return; }
    for(int i = 0; i < inputCount; i++)
    {
       //cout << "IS SHARABLE: " << node->Input(i)->IsValueSharable() << endl;;
       Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
       if((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
          buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
          buffer->GetMatrixType() != MatrixType::DENSE)
            { continue; }

       m_stepNumber2Buffer[m_currentStepNumber].push_back(buffer);
       m_buffer2StepNumbers[buffer].push_back(m_currentStepNumber);

       if(!node->Input(i)->IsValueSharable())
           m_bufferSet.insert(buffer);
    }

    fprintf(stderr, "Step number: %i step name: %s\n", m_currentStepNumber, name.c_str());

    //TODO: Are these buffers needed? -> we have one shared gradient for all nodes
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->ValuePtr().get());
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->GradientPtr().get());

    //m_buffer2StepNumbers[node->ValuePtr().get()].push_back(m_currentStepNumber);
    //m_buffer2StepNumbers[node->GradientPtr().get()].push_back(m_currentStepNumber);
}  


template<typename ElemType>
void SynchronizationManager<ElemType>::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
#ifndef CPUONLY
    //special nodes with no inputs can be ignored
    if(node->GetNumInputs() == 0){ return; }

    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    // it is difficult to sample these operations as the CUDA compiler will remove duplicate
    // operations within a loop; so instead we synchronize the device and hope that our
    // measurement is quite reliable (it often is)
    CUDA_CALL(cudaDeviceSynchronize());
    for(int i = 0; i < SampleSize(); i++)
    {
        // CUDA makes sure that calculations are only done once. 
        // we have to do this, otherwise the calls will be aborted because the values were already
        // calculated. This is at least so for GEMM operations.
        Matrix<ElemType> *output = (Matrix<ElemType>*)node->ValuePtr().get(); 
        ElemType *data = output->Data();
        CUDA_CALL(cudaMemset(data, 0, output->BufferSize()));
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
        m_timer.tick(name);

    }

    
    float t = m_timer.tock(name)/(float)SampleSize();


    m_stepNumber2ComputationTime[stepNumber] = t;
#endif
}


template<typename ElemType>
void SynchronizationManager<ElemType>::MeasureSwapTime(ComputationNodeBase *node, bool isForward) 
{
#ifndef CPUONLY
    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    float t = 0.0f;
    int inputCount = node->GetNumInputs();
    if(inputCount == 0){ return; }
    for(int i = 0; i < inputCount; i++)
    {
       Matrix<ElemType> *input = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
       if((input->GetDataLocation() != CurrentDataLocation::GPU &&
          input->GetDataLocation() != CurrentDataLocation::BOTH) ||
          input->GetMatrixType() != MatrixType::DENSE)
          { continue; }

       if(input != NULL)
       {
           SwapOutAction<ElemType> *out = new SwapOutAction<ElemType>(input);
           SwapInAction<ElemType> *in =  new SwapInAction<ElemType>(out, out->GetGPUMatrix());
           m_buffer2SwapOut[input] = out;
           m_buffer2SwapIn[input] = in;

           for(int i = 0; i < SampleSize(); i++)
           {

               m_timer.tick("Swap out");
               out->BeginAction();
               out->EndAction();
               m_timer.tick("Swap out");
               m_timer.tick("Swap in");
               in->BeginAction();
               in->EndAction();
               m_timer.tick("Swap in");
               CUDA_CALL(cudaDeviceSynchronize());
           }
           t = m_timer.tock("Swap out");
           float swapOutTime = t/SampleSize();

           t = m_timer.tock("Swap in");
           float swapInTime = t/SampleSize();
           m_buffer2SwapTime[input] = std::make_pair(swapOutTime, swapInTime);
       }
    }
#endif
}

template<typename ElemType>
void SynchronizationManager<ElemType>::ClearActionsAndTheirMemory()
{
    for(std::pair<int, std::vector<SyncAction<ElemType>*> > pair : m_stepNumber2Actions)
    {
       for(SyncAction<ElemType> *action : pair.second)
           action->ReleaseMemory();
       pair.second.clear();
    }
    m_stepName2StepNumber.clear();
    m_buffer2StepNumbers.clear();
    m_stepNumber2ComputationTime.clear();
 
    m_buffer2SwapIn.clear();
    m_buffer2SwapOut.clear();
    m_buffer2IsFreed.clear();
    m_stepNumber2Buffer.clear();
    
    m_buffer2SwapTime.clear();
    m_stepNumber2CumulativeSwapInTime.clear();

    m_stepNumber2Actions.clear();
    m_bufferSet.clear();

    m_currentStepNumber = 0;
    m_isExecuting = false;
    m_isInTrainingMode = false;
}



}}}
