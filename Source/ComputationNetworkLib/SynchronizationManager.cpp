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
#include <cmath> 

namespace Microsoft { namespace MSR { namespace CNTK {


using std::cout;
using std::endl;

inline int SampleSize(){ return 100; }
inline int SwapSampleSize(){ return 10; }
inline float MeasurementUncertainty(){ return 1.15f; }


template SynchronizationManager<double>::SynchronizationManager();
template SynchronizationManager<float>::SynchronizationManager();
template <typename ElemType> SynchronizationManager<ElemType>::SynchronizationManager()
{
        m_currentStepNumber = 0;
        m_currentIteration = 0;
        m_maxStepNumber = 0;
        m_maxTimestep = 0;
        m_GBFreed = 0.0f;
        m_timer = CUDATimer();
        m_isExecuting = false;
        m_useMemorySwapping = true;
        m_registeringBuffers = true;
        m_maxIsInitialized = false;
}

template void SynchronizationManager<double>::CleanUp();
template void SynchronizationManager<float>::CleanUp();
template <typename ElemType> void SynchronizationManager<ElemType>::CleanUp()
{
   // 1. remove all swap actions which are not needed, that is which are too slow
   // 2. remove stats and other temporary structures which are not needed for execution
    //Matrix<ElemType> *buffer;
    //for(auto pair : m_stepNumber2Buffer) // auto = <int, Matrix<ElemType>>
    //{
    //    std::vector<Matrix<ElemType> *> buffers = pair.second;
    //    for(int j = 0; j < buffers.size(); j++)
    //    {
    //        buffer = buffers[j];

    //        if(m_buffer2SwapOut.count(buffer) == 0){ continue; }

    //        m_buffer2SwapOut[buffer]->ReleaseMemory();
    //        m_buffer2SwapOut.erase(buffer);
    //        m_buffer2SwapIn.erase(buffer);

    //        // adding this line causes and error during the 2nd actions, and I do not know why
    //        //m_buffer2IsFreed.erase(buffer); 
    //    }
    //}

    // sanity check for a bug
    Matrix<ElemType> *buffer;
    for(std::pair<Matrix<ElemType>*, bool> pair : m_buffer2IsFreed)
    {
        if(pair.second)
        {
            SwapInAction<ElemType> *swpIn = m_buffer2SwapIn[pair.first];
            Matrix<ElemType>* buffer = swpIn->GetGPUMatrix();

            // gradients get deleted after training, this checks if
            // we still have the memory somewhere, otherwise swapping makes no sense
            if(buffer->GetDataLocation() == CurrentDataLocation::GPU ||
               buffer->GetDataLocation() == CurrentDataLocation::BOTH)
            {
                swpIn->BeginAction();
                swpIn->EndAction();
            }
        }
    }
}


template int SynchronizationManager<double>::GetStepNumber(int baseStep, int additionalSteps);
template int SynchronizationManager<float>::GetStepNumber(int baseStep, int additionalSteps);
template <typename ElemType> int SynchronizationManager<ElemType>::GetStepNumber(int baseStep, int additionalSteps)
{
    if(additionalSteps < 0)
        return baseStep + additionalSteps < 0 ? m_maxStepNumber-baseStep+additionalSteps : baseStep + additionalSteps;
    else
        return baseStep + additionalSteps > m_maxStepNumber ? m_maxStepNumber-baseStep-additionalSteps+1 : baseStep + additionalSteps;
}

template int SynchronizationManager<double>::GetStepDistance(int step1, int step2);
template int SynchronizationManager<float>::GetStepDistance(int step1, int step2);
template <typename ElemType> int SynchronizationManager<ElemType>::GetStepDistance(int step1, int step2)
{
    return step1 > step2 ? m_maxStepNumber - step2 + step1 : step2 - step1;
}


template void SynchronizationManager<double>::FindSwapOrder();
template void SynchronizationManager<float>::FindSwapOrder();
template<typename ElemType> void SynchronizationManager<ElemType>::FindSwapOrder()
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
            // do not swap in error functions
            if(swapOutStepNumber > m_maxStepNumber-2){ continue; } 

            //TODO: this is an error?!
            // when the buffer is needed for the next time
            int neededByStepNumber = (i+1) == m_buffer2StepNumbers[buffer].size() ?
                                             m_buffer2StepNumbers[buffer][0] :
                                             m_buffer2StepNumbers[buffer][i+1];
            // if the buffers are needed at a certain timestep, 
            // we need to swap them in at least one timestep before
            neededByStepNumber -= 1; 
            swapOutStepNumber +=1;
            if(neededByStepNumber <= 0){ continue; }

            //cout << "swap out: " << swapOutStepNumber << " needed by step: " << neededByStepNumber << endl;
            cout << "---------------" << endl;
            cout << "swapout: " << swapOutStepNumber << endl;
            cout << "need by: " << neededByStepNumber << endl;
            cout << "---------------" << endl;
            // by default we try to swap one timestep before the buffer is needed
            // if this does not work out we will look for timesteps before that and
            // try to fit the swap-in operation there
            int swapInStepNumber = neededByStepNumber-1; 
            // we get memory benefits only if we at least have one step with no swap operation
            if(GetStepDistance(swapOutStepNumber, swapInStepNumber) < 2){ continue; }
            if(m_stepNumber2ComputationTime.count(swapOutStepNumber) == 0){ continue; } // do we have stats?
            float swapOutTime = m_buffer2SwapTime[buffer].first;
            float swapInTime = m_buffer2SwapTime[buffer].second;
            float computationTimeOut = m_stepNumber2ComputationTime[swapOutStepNumber];
            
            if(m_stepNumber2CumulativeSwapInTime.count(swapOutStepNumber) == 0)
            { m_stepNumber2CumulativeSwapInTime[swapOutStepNumber] = std::make_pair(0.0f,0.0f); }
            float cumulativeSwapOutTime = m_stepNumber2CumulativeSwapInTime[swapOutStepNumber].first;
            cout << "------------------" << endl;
            cout << i << endl;
            cout << "Swap time: " << swapOutTime << endl;
            cout << "Cumulative swap time: " << cumulativeSwapOutTime << endl;
            cout << "Computation time out: " << computationTimeOut << endl;
            cout << "------------------" << endl;

            if(MeasurementUncertainty()*(swapOutTime + cumulativeSwapOutTime) > 
               computationTimeOut)
            { continue; }
            // find a place where we can swap-in the buffer just in time when it is needed
            while(GetStepDistance(swapOutStepNumber, swapInStepNumber) > 2)
            {
                if(m_stepNumber2ComputationTime.count(swapInStepNumber) == 0)
                { 
                    swapInStepNumber = GetStepNumber(swapInStepNumber, -1);
                    continue;
                }

                float computationTime = m_stepNumber2ComputationTime[swapInStepNumber];
                float computationTimeIn = computationTime;
                float cumulativeSwapInTime = m_stepNumber2CumulativeSwapInTime[swapInStepNumber].second;
                if(MeasurementUncertainty()*(swapInTime + cumulativeSwapInTime) < computationTimeIn)
                { break; }
                
                cout << "swapinstepnumber pre " << swapInStepNumber << endl;
                swapInStepNumber = GetStepNumber(swapInStepNumber, -1);
                cout << "swapinstepnumber post " << swapInStepNumber << endl;
            }

            if(GetStepDistance(swapOutStepNumber, swapInStepNumber) < 2){ continue; }
            if(swapInStepNumber == 0){ continue; }

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

            break;
        }

    }

    fprintf(stderr, "Total swapped memory: %fMB\n", totalMemorySwappedInMB);
    fprintf(stderr, "Total swappable memory: %fMB\n", totalMemoryInMB);
#endif
}


template std::vector<Matrix<double>*> SynchronizationManager<double>::GetBuffersForNode(ComputationNodeBase *node, bool isForward, bool getAll);
template std::vector<Matrix<float>*> SynchronizationManager<float>::GetBuffersForNode(ComputationNodeBase *node, bool isForward, bool getAll);
template <typename ElemType> std::vector<Matrix<ElemType>*> SynchronizationManager<ElemType>::GetBuffersForNode(ComputationNodeBase *node, bool isForward, bool getAll)
{

    std::vector<Matrix<ElemType>*> nodeBuffers;
    int inputCount = node->GetNumInputs();

    //if(isForward || node->OutputUsedInComputingInputNodesGradients())
    if(node->ValuePtr() != NULL)
    {
        nodeBuffers.push_back((Matrix<ElemType>*)node->ValuePtr().get());
        if(inputCount == 0){ m_bannedBuffers2bool[nodeBuffers.back()] = true; }
    }

    if(!isForward || getAll)
    if(node->GradientPtr() != NULL)
        nodeBuffers.push_back((Matrix<ElemType>*)node->GradientPtr().get());

    for(int i = 0; i < inputCount; i++)
    {
        ComputationNodeBase *parentNode = node->Input(i).get();

        if(isForward || node->InputUsedInComputingInputNodesGradients(i) || getAll)
            if(parentNode->ValuePtr() != NULL)
            {
                nodeBuffers.push_back((Matrix<ElemType>*)parentNode->ValuePtr().get());
                if(parentNode->GetNumInputs() == 0){ m_bannedBuffers2bool[nodeBuffers.back()] = true; }
            }
        if(!isForward || getAll)
        if(parentNode->GradientPtr() != NULL)
            nodeBuffers.push_back((Matrix<ElemType>*)parentNode->GradientPtr().get());
    }



    

    return nodeBuffers;
}


template void SynchronizationManager<double>::BeginSynchronizeState(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<float>::BeginSynchronizeState(ComputationNodeBase *node, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::BeginSynchronizeState(ComputationNodeBase *node, bool isForward)
{

#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    cout << nodename << " + " << isForward << endl;

    if(!isForward)
        for(auto action : m_node2BackwardSwapin[node])
        {
            action->BeginAction();
            action->EndAction();
        }
#endif
}


template void SynchronizationManager<double>::EndSynchronizeState(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<float>::EndSynchronizeState(ComputationNodeBase *node, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::EndSynchronizeState(ComputationNodeBase *node, bool isForward)
{
#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    cout << nodename << " + " << isForward << endl;

    if(isForward)
        for(auto action : m_node2ForwardSwapOut[node])
        {
            CUDA_CALL(cudaDeviceSynchronize());
            action->BeginAction();
            action->EndAction();
        }
    else
        for(auto matrix : m_node2BackwardFree[node])
        {
            cout << "Freeing matrix during backprop: " << matrix << " " << matrix->GetNumRows() << "x" << matrix->GetNumCols() << endl;
            matrix->Resize(0,0,0,false);
        }
        
        


    //std::vector<SyncAction<ElemType>*> actionsToDo = isForward ? m_stepNumber2SwapIn[stepNumber] : m_stepNumber2ActionsBackprop[stepNumber];
    //for (int i = 0; i < actionsToDo.size(); i++)
    //{
    //       // criteron, evaluation and input nodes do not have a MB layout?
    //       // does not make sense to free those anyway
    //       if(m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()])
    //       {
    //       actionsToDo[i]->EndAction();
    //       m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()] = false;
    //       }
    //}


    //actionsToDo = isForward ? m_stepNumber2SwapOut[stepNumber] : m_stepNumber2ActionsBackprop[stepNumber];
    //for (int i = 0; i < actionsToDo.size(); i++)
    //{
    //       // criteron, evaluation and input nodes do not have a MB layout?
    //       // does not make sense to free those anyway
    //       if(!m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()])
    //       {
    //           actionsToDo[i]->EndAction();
    //           m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()] = true;
    //       }
    //}



    //if(!isForward)
    //{
    //    std::unordered_map<Matrix<ElemType>*, bool> swapCandidates;
    //    int offset = 5;

    //    for(int i = m_maxTimestep; i > stepNumber; i--)
    //    {
    //        if(m_timestep2Buffers.count(i) == 0){ continue; }

    //        for(auto buffer : m_timestep2Buffers[i])
    //        {
    //        
    //            if(m_buffer2IsFreed.count(buffer) > 0)
    //                if(!m_buffer2IsFreed[buffer])
    //                    swapCandidates[buffer] = true;
    //        }
    //    }

    //    for(int i = stepNumber; i > stepNumber-1; i--)
    //    {
    //        cout << "test " << i << endl;
    //        if(m_timestep2Buffers.count(i) == 0){ continue; }

    //        for(auto buffer : m_timestep2Buffers[i])
    //                swapCandidates[buffer] = false;
    //    }


    //    for(auto pair : swapCandidates)
    //    {
    //        
    //        //if(m_bannedBuffers2bool.count(pair.first) > 0)
    //        //    cout << "BANNED: " << pair.first << ", " << pair.first->BufferSize()/1024./1024./1024. << "GB" << endl;
    //        if(pair.second)// && m_bannedBuffers2bool.count(pair.first) == 0)
    //        {
    //            SwapOutAction<ElemType> *swapOut = m_buffer2SwapOut[pair.first];
    //            swapOut->BeginAction();
    //            swapOut->EndAction();
    //            m_buffer2IsFreed[pair.first] = true;
    //            cout << "swap out for timestep: " << stepNumber+offset << " buffer: " << pair.first << endl;
    //        }
    //    }
    //}

        //for(int i = stepNumber-1; i < stepNumber; i++)
        //    for(auto buffer : m_timestep2Buffers[i])
        //    {
        //        cout << "swap in for timestep: " << i << endl;
        //        if(m_buffer2IsFreed.count(buffer) > 0)
        //            if(m_buffer2IsFreed[buffer])
        //            {
        //                SwapInAction<ElemType> *swapIn = m_buffer2SwapIn[buffer];
        //                swapIn->BeginAction();
        //                swapIn->EndAction();
        //                m_buffer2IsFreed[buffer] = false;
        //                cout << "swapped in: " << swapIn->GetGPUMatrix() << endl;
        //            }
        //    }
    //}
    //else
    //{
    //    m_timestep2Buffers[stepNumber] = GetBuffersForNode(node);
    //}



    //if(!m_isExecuting)
    //{
    //    // end synchronize is called after the forward / backward pass, thus we can free
    //    // the memory now
    //    GatherRuntimeStatistics(node, idx, fr, isForward, false);
    //    FreeBuffersForDryRun(node, isForward);
    //}
    //else
    //{
    //    std::string name = GetStepName(node, isForward);
    //    int stepNumber = m_stepName2StepNumber[name];
    //    std::vector<SyncAction<ElemType>*> actionsToDo = m_stepNumber2Actions[stepNumber];
    //    if (actionsToDo.size() == 0){ return; }

    //    for (int i = 0; i < actionsToDo.size(); i++)
    //    {
    //       // criteron, evaluation and input nodes do not have a MB layout?
    //       // does not make sense to free those anyway
    //       if(node->HasMBLayout()) 
    //            actionsToDo[i]->EndAction();
    //    }
    //}
#endif
}

template void SynchronizationManager<double>::SwapInFreedBuffers(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<float>::SwapInFreedBuffers(ComputationNodeBase *node, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::SwapInFreedBuffers(ComputationNodeBase *node, bool isForward)
{

    std::string name = GetStepName(node, isForward);
    int offset = 2;
    int from = isForward ? 0 : -1;
    int to = isForward ? 1 : 0;
    //for(int i = from; i <= to; i++)
    //{
        int stepNumber = m_stepName2StepNumber[name];
        stepNumber += isForward ? 1 : -1;
        //cout << "pre " << stepNumber << endl;
        //stepNumber += i;
        //cout << "post i " << stepNumber << endl;
        //stepNumber = stepNumber > m_maxStepNumber ? m_maxStepNumber + (m_maxStepNumber - stepNumber) : stepNumber; // wrap into backprop
        //stepNumber = stepNumber < 0 ? abs(1) : stepNumber; // wrap into forward-prop
        //cout << "post transform " << stepNumber << endl;
        if(m_timestep2node.count(stepNumber) == 0){ return; }
        for(auto buffer : GetBuffersForNode(m_timestep2node[stepNumber], isForward, false))
        {
            if(m_buffer2IsFreed.count(buffer) == 0){ continue; }
            if(!m_buffer2IsFreed[buffer]){ continue; }

            //bool doSwap = true;
            //for(auto currentBuffer : GetBuffersForNode(node, isForward, false))
            //{
            //    if(currentBuffer == buffer){ cout << "IS IN USE!" << endl; doSwap = false; }
            //}
            //if(!doSwap){ continue; }

            SwapInAction<ElemType>* swapIn = m_buffer2SwapIn[buffer];
            swapIn->BeginAction();
            swapIn->EndAction();
            m_buffer2IsFreed[buffer] = false;
            cout << "swap in for timestep: " << stepNumber <<  " buffer: " << buffer << endl;
        }
    //}
    //}
    
    //for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    //{
    //    Matrix<ElemType> *buffer = m_stepNumber2Buffer[stepNumber][i]; 

    //    if(m_buffer2IsFreed.count(buffer) == 0){ continue; }
    //    if(m_buffer2IsFreed[buffer])
    //    {
    //        SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
    //        swp->BeginAction(); // begin swap in
    //        swp->EndAction(); // end swap in



    //        //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //        m_buffer2IsFreed[buffer] = false;
    //        cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //    }
    //}

    //int inputCount = node->GetNumInputs();
    //std::string name = GetStepName(node, isForward);
    //for(int i = 0; i < inputCount; i++)
    //{

    //   if(node->Input(i)->ValuePtr() == NULL){ continue; }
    //   Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
    //   if((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //      buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //      buffer->GetMatrixType() != MatrixType::DENSE)
    //        { continue; }

    //        if(m_buffer2IsFreed.count(buffer) == 0){ continue; }
    //        if(m_buffer2IsFreed[buffer])
    //        {
    //            SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
    //            swp->BeginAction(); // begin swap in
    //            swp->EndAction(); // end swap in


    //            m_GBFreed -= buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;
    //            cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;

    //            //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //            m_buffer2IsFreed[buffer] = false;
    //            //cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //        }
    //    

    //}


    //if(node->ValuePtr() != NULL)
    //{
    //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->ValuePtr().get();

    //    if(!((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //          buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //          buffer->GetMatrixType() != MatrixType::DENSE))
    //    {

    //        if(m_buffer2IsFreed.count(buffer) > 0)
    //        if(m_buffer2IsFreed[buffer])
    //        {
    //            SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
    //            swp->BeginAction(); // begin swap in
    //            swp->EndAction(); // end swap in


    //            m_GBFreed -= buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;

    //            cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;

    //            //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //            m_buffer2IsFreed[buffer] = false;
    //            //cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //        }
    //    

    //    }
    //}


    //if(node->GradientPtr() != NULL)
    //{
    //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->GradientPtr().get();

    //    if(!((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //          buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //          buffer->GetMatrixType() != MatrixType::DENSE))
    //    {

    //        if(m_buffer2IsFreed.count(buffer) > 0)
    //        if(m_buffer2IsFreed[buffer])
    //        {
    //            SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
    //            swp->BeginAction(); // begin swap in
    //            swp->EndAction(); // end swap in


    //            m_GBFreed -= buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;

    //            cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;

    //            //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //            m_buffer2IsFreed[buffer] = false;
    //            //cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //        }
    //    

    //    }
    //}




}

template void SynchronizationManager<float>::FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<double>::FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward);
template <typename ElemType> void SynchronizationManager<ElemType>::FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward)
{
#ifndef CPUONLY
    // if a value is marked as shareable, it will be used right after, thus it makes
    // no sense to swap out these values (it just makes it more complicated), so instead
    // we just swap out the non-sharable ones (used in backprop). This will give us enough
    // memory to perform the dry run

    std::string name = GetStepName(node, isForward);
    // free those used two layers below
    int from = isForward ? -5 : 1;
    int to = isForward ? -1 : 5;
    for(int i = from; i < to; i++)
    {
        int stepNumber = m_stepName2StepNumber[name];
        stepNumber += i;
        stepNumber = stepNumber > m_maxStepNumber ? m_maxStepNumber + (m_maxStepNumber-stepNumber) : stepNumber; // wrap into backprop
        stepNumber = stepNumber < 0 ? abs(stepNumber) : stepNumber; // wrap into forward-prop
        if(stepNumber < 3){ continue; }
        //if(abs(stepNumber-m_maxStepNumber) < 3){ return; }
        //if(stepNumber < 3){ return; }
        if(m_timestep2Buffers.count(stepNumber) == 0){ return; }
        for(auto buffer : m_timestep2Buffers[stepNumber])
        {
            // we do not care about buffers with size less than 1 MB - too much overhead
            if(buffer->BufferSize() < 1024*1024){ continue; } 
            if(m_bannedBuffers2bool.count(buffer) > 0){ continue; }
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

            bool doSwap = true;
            for(auto currentBuffer : GetBuffersForNode(node, isForward, false))
            {
                if(currentBuffer == buffer){ cout << "IS IN USE!" << endl; doSwap = false; }
            }

            if(!doSwap){ continue; }

            cout << "freeing " << buffer << " at timestep " << stepNumber << endl;
            m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
            m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
            m_buffer2IsFreed[buffer] = true;

        }
    }

    //std::string name = GetStepName(node, isForward);
    //int timeStep = m_nodes2timestep[node];
    //timeStep = GetStepNumber(timeStep, -2);
    //cout << "freeing timestep: " << timeStep << endl;
    //if(m_timestep2nodes.count(timeStep) == 0){ return; }
    //ComputationNodeBase *swapNode = m_timestep2nodes[timeStep];

    //if(swapNode->GradientPtr() != NULL)
    //{
    //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)swapNode->ValuePtr().get();

    //    if(m_bannedBuffers2bool.count(buffer) == 0)
    //        m_bannedBuffers2bool[buffer] = true;
    //}

    //int inputCount = swapNode->GetNumInputs();
    //for(int i = 0; i < inputCount; i++)
    //{

    //   if(swapNode->Input(i)->ValuePtr() == NULL){ cout << "NULL" << endl; continue; }
    //   Matrix<ElemType> *buffer = (Matrix<ElemType>*)swapNode->Input(i)->ValuePtr().get();

    //    if(m_bannedNodes2Bool.count(swapNode->Input(i).get()) > 0){ cout << "BANNED" << endl; continue; }
    //    if(buffer == NULL){ cout << "NULL2" << endl; continue; }
    //    if(buffer->GetNumElements() < 1024){ cout << "small" << endl; continue; }
    //    if(m_bannedBuffers2bool.count(buffer) > 0){ cout << "BANNED2" << endl; continue; }
    //    //if(swapNode->Input(i)->IsValueSharable()){ cout << "SHARED" << endl; continue; }

    //    if(m_buffer2IsFreed.count(buffer) > 0)
    //        if(m_buffer2IsFreed[buffer])
    //            continue; // buffer is already freed


    //    if(m_buffer2SwapIn.count(buffer) == 0)
    //    {
    //        SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
    //        SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
    //        m_buffer2SwapOut[buffer] = swpOut;
    //        m_buffer2SwapIn[buffer] = swpIn;
    //    }

    //    //if(buffer->GetNumRows() == 10 && buffer->GetNumCols() == 200){ continue; }
    //    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    //    //if(buffer->GetNumRows() == 784 && buffer->GetNumCols() == 32){ cout << "INPUT: " << nodename << endl; continue; }
    //    //cout << "freeing buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //    m_GBFreed += buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;
    //    cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;
    //    m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
    //    m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
    //    m_buffer2IsFreed[buffer] = true;

    //}


    //if(m_bannedNodes2Bool.count(swapNode) > 0){ return; }
    //if(swapNode->ValuePtr() != NULL)
    //{
    //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)swapNode->ValuePtr().get();

    //    if(m_bannedBuffers2bool.count(buffer) > 0){ cout << "banned3" << endl; return; }
    //    if(buffer->GetNumElements() < 1024){ cout << "small2" << endl; return; }
    //    //if(swapNode->IsValueSharable()){ cout << "SHARED" << endl; return; }

    //    if(m_buffer2IsFreed.count(buffer) > 0)
    //        if(m_buffer2IsFreed[buffer])
    //            return; // buffer is already freed


    //    if(m_buffer2SwapIn.count(buffer) == 0)
    //    {
    //        SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
    //        SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
    //        m_buffer2SwapOut[buffer] = swpOut;
    //        m_buffer2SwapIn[buffer] = swpIn;
    //    }

    //    //cout << "freeing buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //    m_GBFreed += buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;
    //    cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;
    //    m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
    //    m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
    //    m_buffer2IsFreed[buffer] = true;
    //  
    //}
#endif
}

inline std::string IsForwardToString(bool b){ return b ? std::string("_forward") : std::string("_backprop"); }
template std::string SynchronizationManager<float>::GetStepName(ComputationNodeBase *node, bool isForward);
template std::string SynchronizationManager<double>::GetStepName(ComputationNodeBase *node, bool isForward);
template<typename ElemType> std::string SynchronizationManager<ElemType>::GetStepName(ComputationNodeBase *node, bool isForward)
{
    std::wstring wname = node->GetName();
    return std::string(wname.begin(), wname.end());
}


template void SynchronizationManager<float>::RegisterBuffers(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<double>::RegisterBuffers(ComputationNodeBase *node, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::RegisterBuffers(ComputationNodeBase *node, bool isForward)
{
    
    
    std::string name = GetStepName(node, isForward);
    if(m_stepName2StepNumber.count(name) > 0){ return; }

    m_stepName2StepNumber[name] = m_currentStepNumber;

    m_timestep2Buffers[m_currentStepNumber] = GetBuffersForNode(node, false, true);
    //m_nodes2timestep[node] = m_currentStepNumber;
    //m_timestep2nodes[m_currentStepNumber] = node;
    m_timestep2node[m_currentStepNumber] = node;
    m_maxStepNumber = m_currentStepNumber;
    m_currentStepNumber++;


    //int inputCount = node->GetNumInputs();
    //std::string name = GetStepName(node, isForward);
    //// already registered?
    //if(m_stepName2StepNumber.count(name) > 0){ return; }

    //if(m_stepNumber2Buffer.count(m_currentStepNumber) > 0){ m_currentStepNumber++; }
    //m_stepName2StepNumber[name] = m_currentStepNumber;
    //for(int i = 0; i < inputCount; i++)
    //{

    //   if(node->Input(i)->ValuePtr() == NULL){ continue; }
    //   // we do not track these
    //   if(node->Input(i)->IsValueSharable()){ continue; }

    //   Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
    //   if(m_bannedBuffers2bool.count(buffer) > 0){ continue; }
    //   if((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //      buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //      buffer->GetMatrixType() != MatrixType::DENSE)
    //        { continue; }

    //   
    //   cout << "REGISTER: " << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //   m_stepNumber2Buffer[m_currentStepNumber].push_back(buffer);
    //   m_buffer2StepNumbers[buffer].push_back(m_currentStepNumber);
    //   m_bufferSet.insert(buffer);
    //}

    //fprintf(stderr, "Step number: %i step name: %s\n", m_currentStepNumber, name.c_str());

    //if(node->ValuePtr() == NULL){ return; }
    //Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->ValuePtr().get();
    //   if(m_bannedBuffers2bool.count(buffer) > 0){ return; }
    //if((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //      buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //      buffer->GetMatrixType() != MatrixType::DENSE)
    //        { return; }

    //if(node->IsValueSharable()){ return; }
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(buffer);
    //m_bufferSet.insert(buffer);
    //cout << "REGISTER: " << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //TODO: Are these buffers needed? -> we have one shared gradient for all nodes
    //m_stepNumber2Buffer[m_currentStepNumber].push_back(node->GradientPtr().get());
    //m_buffer2StepNumbers[node->GradientPtr().get()].push_back(m_currentStepNumber);
}  


template void SynchronizationManager<float>::RegisterWeight(Matrix<float> *weight);
template void SynchronizationManager<double>::RegisterWeight(Matrix<double> *weight);
template <typename ElemType> void SynchronizationManager<ElemType>::RegisterWeight(Matrix<ElemType> *weight)
{
    // already registered
        //cout << "weight with: " << weight->GetNumRows()  << "x" << weight->GetNumCols() << endl;
        //cout << m_currentIteration << endl;
    if(m_buffer2IsFreed.count(weight) == 0){ return; }
    if(m_buffer2IsFreed[weight])
    {
        SwapInAction<ElemType> *swp = m_buffer2SwapIn[weight];
        swp->BeginAction(); // begin swap in
        swp->EndAction(); // end swap in
        //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
        m_buffer2IsFreed[weight] = false;
        //cout << "swapped in buffer: " << weight->GetNumRows()  << "x" << weight->GetNumCols() << endl;
    }

}


template void SynchronizationManager<float>::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward, bool isBeforeComputation);
template void SynchronizationManager<double>::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward, bool isBeforeComputation);
template<typename ElemType> void SynchronizationManager<ElemType>::GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward, bool isBeforeComputation)
{
#ifndef CPUONLY
    // we need to warm up the GPU so we get reliable estimates of computation
    if(m_currentIteration < 5){ return; }
    if(m_currentIteration >= SampleSize()){ return; }
    //special nodes with no inputs can be ignored
    if(node->GetNumInputs() == 0){ return; }

    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    // it is difficult to sample these operations as the CUDA compiler will remove duplicate
    // operations within a loop; so instead we synchronize the device and hope that our
    // measurement is quite reliable (it often is)
    if(isBeforeComputation)
    {
        CUDA_CALL(cudaDeviceSynchronize());
        m_timer.tick(name);
    }
    else
    {
        float t = m_timer.tock(name)/(float)SampleSize();
        m_stepNumber2ComputationTime[stepNumber] += t;
    }
#endif
}

template void SynchronizationManager<float>::InitializeSwapping(
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<float>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<float>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<float>*> > lastBackwardNodes2matrices);
template void SynchronizationManager<double>::InitializeSwapping(
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<double>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<double>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<double>*> > lastBackwardNodes2matrices);
template <typename ElemType> void SynchronizationManager<ElemType>::InitializeSwapping(
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > lastBackwardNodes2matrices)
{

    for(auto pair : forwardSwapOutNodes2matrices)
    {
        for(auto buffer : pair.second)
        {
            if(m_buffer2SwapOut.count(buffer) == 0)
            {
                SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
                SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
                m_buffer2SwapOut[buffer] = swpOut;
                m_buffer2SwapIn[buffer] = swpIn;
            }

            m_node2ForwardSwapOut[pair.first].push_back(m_buffer2SwapOut[buffer]);
        }
    }


    for(auto pair : backwardSwapInNodes2matrices)
    {
        for(auto buffer : pair.second)
        {
            if(m_buffer2SwapIn.count(buffer) == 0)
            {
                SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
                SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
                m_buffer2SwapOut[buffer] = swpOut;
                m_buffer2SwapIn[buffer] = swpIn;
            }

            m_node2BackwardSwapin[pair.first].push_back(m_buffer2SwapIn[buffer]);
        }
    }

    m_node2BackwardFree = lastBackwardNodes2matrices;

}


template int SynchronizationManager<double>::DetermineCurrentTimestep(ComputationNodeBase *node);
template int SynchronizationManager<float>::DetermineCurrentTimestep(ComputationNodeBase *node);
template <typename ElemType> int SynchronizationManager<ElemType>::DetermineCurrentTimestep(ComputationNodeBase *node)
{
    if(node->ValuePtr() == NULL){ return -1; }
    Matrix<ElemType>* buffer = (Matrix<ElemType>*)node->ValuePtr().get();
    if(m_buffer2Timesteps.count(buffer) == 0){ return -1; }
    if(m_buffer2Timesteps.count(buffer) == 1){ return m_buffer2Timesteps[buffer].front(); }
    
    std::cout << "USED MORE THAN ONCE" << std::endl;

    return -1;

}


template void SynchronizationManager<double>::MeasureSwapTime(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<float>::MeasureSwapTime(ComputationNodeBase *node, bool isForward); 
template<typename ElemType> void SynchronizationManager<ElemType>::MeasureSwapTime(ComputationNodeBase *node, bool isForward) 
{
#ifndef CPUONLY

    if(m_currentIteration >= 1){ return; }    
    std::string name = GetStepName(node, isForward);
    int stepNumber = m_stepName2StepNumber[name];
    float t = 0.0f;
    int inputCount = node->GetNumInputs();
    if(inputCount == 0){ return; }
    for(int i = 0; i < inputCount; i++)
    {
       Matrix<ElemType> *input = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
       // did we already process these buffers?
       if(m_buffer2SwapIn.count(input) > 0){ break; }

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

           for(int i = 0; i < SwapSampleSize(); i++)
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
           float swapOutTime = t/SwapSampleSize();

           t = m_timer.tock("Swap in");
           float swapInTime = t/SwapSampleSize();
           m_buffer2SwapTime[input] = std::make_pair(swapOutTime, swapInTime);
       }
    }
#endif
}

template void SynchronizationManager<float>::ClearActionsAndTheirMemory();
template void SynchronizationManager<double>::ClearActionsAndTheirMemory();
template<typename ElemType> void SynchronizationManager<ElemType>::ClearActionsAndTheirMemory()
{

    cout << "Cleaning up!" << endl;
    CleanUp();

    for(std::pair<int, std::vector<SyncAction<ElemType>*> > pair : m_stepNumber2Actions)
    {
       for(SyncAction<ElemType> *action : pair.second)
           action->ReleaseMemory();
       pair.second.clear();
    }

    m_stepName2StepNumber.clear();
    m_buffer2StepNumbers.clear();
    m_stepNumber2ComputationTime.clear();
    m_timestep2Buffers.clear();
 
    m_buffer2SwapIn.clear();
    m_buffer2SwapOut.clear();
    m_buffer2IsFreed.clear();
    m_stepNumber2Buffer.clear();
    m_bannedBuffers2bool.clear();
    m_bannedNodes2Bool.clear();
    
    m_buffer2SwapTime.clear();
    m_stepNumber2CumulativeSwapInTime.clear();

    m_stepNumber2Actions.clear();
    m_bufferSet.clear();

    m_currentStepNumber = 0;
    m_isExecuting = false;
    m_registeringBuffers = true;
    m_maxTimestep = -1;
}



}}}
