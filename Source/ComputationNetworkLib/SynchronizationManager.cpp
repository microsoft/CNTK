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


using std::cout;
using std::endl;

inline int SampleSize(){ return 100; }
inline int SwapSampleSize(){ return 10; }
inline float MeasurementUncertainty(){ return 1.15f; }


// this fetches the singleton instance
template  SynchronizationManager<double>* SynchronizationManager<double>::GetSynchronizationManager();
template  SynchronizationManager<float>* SynchronizationManager<float>::GetSynchronizationManager();
template <typename ElemType> SynchronizationManager<ElemType>* SynchronizationManager<ElemType>::GetSynchronizationManager()
{
    if (SynchronizationManager<ElemType>::s_synchronizationManager == NULL)
    {
        SynchronizationManager<ElemType>::s_synchronizationManager = new SynchronizationManager();
        SynchronizationManager<ElemType>::s_synchronizationManager->m_currentStepNumber = 0;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_currentIteration = 0;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_maxStepNumber = 0;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_maxTimestep = 0;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_GBFreed = 0.0f;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_timer = CUDATimer();
        SynchronizationManager<ElemType>::s_synchronizationManager->m_isExecuting = false;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_useMemorySwapping = false;
        SynchronizationManager<ElemType>::s_synchronizationManager->m_registeringBuffers = true;
    }

    return SynchronizationManager<ElemType>::s_synchronizationManager;
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
        buffer = pair.first;
        if(pair.second)
        {
            SwapInAction<ElemType> *swpIn = m_buffer2SwapIn[buffer];
            swpIn->BeginAction();
            swpIn->EndAction();
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
        return baseStep + additionalSteps > m_maxStepNumber ? m_maxStepNumber-baseStep-additionalSteps-1 : baseStep + additionalSteps;
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


template std::vector<Matrix<double>*> SynchronizationManager<double>::GetBuffersForNode(ComputationNodeBase *node);
template std::vector<Matrix<float>*> SynchronizationManager<float>::GetBuffersForNode(ComputationNodeBase *node);
template <typename ElemType> std::vector<Matrix<ElemType>*> SynchronizationManager<ElemType>::GetBuffersForNode(ComputationNodeBase *node)
{

    bool isForward = false;
    std::vector<Matrix<ElemType>*> nodeBuffers;

    //if(isForward || node->OutputUsedInComputingInputNodesGradients())
    if(node->ValuePtr() != NULL)
    {
        nodeBuffers.push_back((Matrix<ElemType>*)node->ValuePtr().get());
    }

    if(node->GradientPtr() != NULL)
    {
        nodeBuffers.push_back((Matrix<ElemType>*)node->GradientPtr().get());
        m_bannedBuffers2bool[nodeBuffers.back()] = true;
    }

    int inputCount = node->GetNumInputs();
    for(int i = 0; i < inputCount; i++)
    {
        ComputationNodeBase *parentNode = node->Input(i).get();

        if(isForward || node->InputUsedInComputingInputNodesGradients(i))
            if(parentNode->ValuePtr() != NULL)
                nodeBuffers.push_back((Matrix<ElemType>*)parentNode->ValuePtr().get());

        if(parentNode->GradientPtr() != NULL)
        {
            nodeBuffers.push_back((Matrix<ElemType>*)parentNode->GradientPtr().get());
            m_bannedBuffers2bool[nodeBuffers.back()] = true;
        }
    }


    

    return nodeBuffers;
}


template void SynchronizationManager<double>::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
template void SynchronizationManager<float>::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{

#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    int stepNumber = DetermineCurrentTimestep(node);
    cout << stepNumber << endl;


    

    if(!isForward)
    {

        //if(stepNumber > 3)
        //    if(m_timestep2Buffers.count(stepNumber+3) > 0)
        //        for(auto buffer : m_timestep2Buffers[stepNumber+3])
        //        {
        //        
        //                cout << "swap out for timestep: " << stepNumber+3 << endl;
        //                if(m_buffer2IsFreed.count(buffer) > 0)
        //                    if(!m_buffer2IsFreed[buffer])
        //                    {
        //                        SwapOutAction<ElemType> *swapOut = m_buffer2SwapOut[buffer];
        //                        swapOut->BeginAction();
        //                        swapOut->EndAction();
        //                        m_buffer2IsFreed[buffer] = true;
        //                        cout << "swapped out: " << swapOut->GetGPUMatrix() << endl;
        //                    }
        //        }

        for(int i = stepNumber; i < stepNumber+1; i++)
            for(auto buffer : m_timestep2Buffers[i])
            {
                if(m_buffer2IsFreed.count(buffer) > 0)
                    if(m_buffer2IsFreed[buffer])
                    {
                        SwapInAction<ElemType> *swapIn = m_buffer2SwapIn[buffer];
                        swapIn->BeginAction();
                        swapIn->EndAction();
                        m_buffer2IsFreed[buffer] = false;
                        cout << "swap in for timestep: " << i <<  " buffer: " << buffer << endl;
                    }
            }
    }
    else
    {
        m_timestep2Buffers[stepNumber] = GetBuffersForNode(node);
    }

    std::vector<SyncAction<ElemType>*> actionsToDo = isForward ? m_stepNumber2SwapIn[stepNumber] : m_stepNumber2ActionsBackprop[stepNumber];
    for (int i = 0; i < actionsToDo.size(); i++)
    {
           // criteron, evaluation and input nodes do not have a MB layout?
           // does not make sense to free those anyway
           if(m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()])
               actionsToDo[i]->BeginAction();
    }


    actionsToDo = isForward ? m_stepNumber2SwapOut[stepNumber] : m_stepNumber2ActionsBackprop[stepNumber];
    for (int i = 0; i < actionsToDo.size(); i++)
    {
           // criteron, evaluation and input nodes do not have a MB layout?
           // does not make sense to free those anyway
           if(!m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()])
               actionsToDo[i]->BeginAction();
    }



    //size_t free, total;
    //CUDA_CALL(cudaMemGetInfo(&free, &total));

    //cout << "FREE MEMORY: " << free/1024.0f/1024.0f/1024.0f << endl;

    //cout << "current timestep: " << DetermineCurrentTimestep(node) << endl;


    //if(node->ValuePtr() != NULL)
    //{
    //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->ValuePtr().get();

    //    //if(isForward)
    //    //    m_forwardGraph[buffer] << endl;
    //}

    //if(!m_isExecuting)
    //{
    //    bool allStatsGathered = false;
    //    std::string name = GetStepName(node, isForward);

    //    // the stats gathering ends when we are back at stepNumber 0, that is in the forward pass
    //    if(m_stepName2StepNumber.count(name) > 0)
    //        if(m_stepName2StepNumber[name] == 0 && isForward == true)
    //        {
    //            m_currentIteration += 1;
    //            //cout << "CURRENT ITERATION: " << m_currentIteration << endl;
    //            if(m_maxStepNumber == 0)
    //                m_maxStepNumber = m_currentStepNumber;

    //            //float t = m_timer.tock("swap");
    //            //cout << t << endl;
    //        }

    //    
    //    //if(m_currentIteration == SampleSize() + 1)
    //        //allStatsGathered = true;
    //    //if(!allStatsGathered || !m_isInTrainingMode)
    //    if(!allStatsGathered)
    //    {
    //        RegisterBuffers(node, isForward);


    //        int stepNumber = m_stepName2StepNumber[name];
    //        //SwapInFreedBuffers(GetStepNumber(stepNumber,-1));
    //        //SwapInFreedBuffers(stepNumber);
    //        //SwapInFreedBuffers(GetStepNumber(stepNumber,1));
    //        SwapInFreedBuffers(node, isForward);





    //        //int inputCount = node->GetNumInputs();
    //        //std::string name = GetStepName(node, isForward);
    //        //for(int i = 0; i < inputCount; i++)
    //        //{

    //        //   if(node->Input(i)->ValuePtr() == NULL){ continue; }
    //        //   Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
    //        //   if((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //        //      buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //        //      buffer->GetMatrixType() != MatrixType::DENSE)
    //        //        { continue; }

    //        //   
    //        //   cout << "NEEDED: " << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //        //}


    //        //if(node->ValuePtr() != NULL)
    //        //{
    //        //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->ValuePtr().get();

    //        //    if(!((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //        //          buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //        //          buffer->GetMatrixType() != MatrixType::DENSE))
    //        //    {

    //        //           cout << "NEEDED: " << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //        //    }
    //        //}



    //        //if(node->GradientPtr() != NULL)
    //        //{
    //        //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->GradientPtr().get();

    //        //    if(!((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
    //        //          buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
    //        //          buffer->GetMatrixType() != MatrixType::DENSE))
    //        //    {

    //        //           cout << "NEEDED: " << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
    //        //    }
    //        //}






    //        MeasureSwapTime(node, isForward);
    //        GatherRuntimeStatistics(node, idx, fr, isForward, true);
    //    }
    //    else
    //    {
    //        //CleanUp(); // release all cpu memory that was used in the dry run
    //        //FindSwapOrder();
    //        //m_isExecuting = true;
    //    }
    //}

    //if(m_isExecuting)
    //{
    //    std::string name = GetStepName(node, isForward);
    //    int stepNumber = m_stepName2StepNumber[name];
    //    std::vector<SyncAction<ElemType>*> actionsToDo = m_stepNumber2Actions[stepNumber];
    //    for (int i = 0; i < actionsToDo.size(); i++)
    //    {
    //           // criteron, evaluation and input nodes do not have a MB layout?
    //           // does not make sense to free those anyway
    //           if(node->HasMBLayout()) 
    //               actionsToDo[i]->BeginAction();
    //    }

    //}
#endif
}


template void SynchronizationManager<double>::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
template void SynchronizationManager<float>::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward)
{
#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    int stepNumber = DetermineCurrentTimestep(node);
    std::vector<SyncAction<ElemType>*> actionsToDo = isForward ? m_stepNumber2SwapIn[stepNumber] : m_stepNumber2ActionsBackprop[stepNumber];
    for (int i = 0; i < actionsToDo.size(); i++)
    {
           // criteron, evaluation and input nodes do not have a MB layout?
           // does not make sense to free those anyway
           if(m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()])
           {
           actionsToDo[i]->EndAction();
           m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()] = false;
           }
    }


    actionsToDo = isForward ? m_stepNumber2SwapOut[stepNumber] : m_stepNumber2ActionsBackprop[stepNumber];
    for (int i = 0; i < actionsToDo.size(); i++)
    {
           // criteron, evaluation and input nodes do not have a MB layout?
           // does not make sense to free those anyway
           if(!m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()])
           {
               actionsToDo[i]->EndAction();
               m_buffer2IsFreed[actionsToDo[i]->GetGPUMatrix()] = true;
           }
    }



    if(!isForward)
    {
        std::unordered_map<Matrix<ElemType>*, bool> swapCandidates;
        int offset = 5;

        for(int i = m_maxTimestep; i > stepNumber; i--)
        {
            if(m_timestep2Buffers.count(i) == 0){ continue; }

            for(auto buffer : m_timestep2Buffers[i])
            {
            
                if(m_buffer2IsFreed.count(buffer) > 0)
                    if(!m_buffer2IsFreed[buffer])
                        swapCandidates[buffer] = true;
            }
        }

        for(int i = stepNumber; i > stepNumber-1; i--)
        {
            cout << "test " << i << endl;
            if(m_timestep2Buffers.count(i) == 0){ continue; }

            for(auto buffer : m_timestep2Buffers[i])
                    swapCandidates[buffer] = false;
        }


        for(auto pair : swapCandidates)
        {
            
            //if(m_bannedBuffers2bool.count(pair.first) > 0)
            //    cout << "BANNED: " << pair.first << ", " << pair.first->BufferSize()/1024./1024./1024. << "GB" << endl;
            if(pair.second)// && m_bannedBuffers2bool.count(pair.first) == 0)
            {
                SwapOutAction<ElemType> *swapOut = m_buffer2SwapOut[pair.first];
                swapOut->BeginAction();
                swapOut->EndAction();
                m_buffer2IsFreed[pair.first] = true;
                cout << "swap out for timestep: " << stepNumber+offset << " buffer: " << pair.first << endl;
            }
        }
    }

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


        m_timestep2Buffers[stepNumber] = GetBuffersForNode(node);

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

    int inputCount = node->GetNumInputs();
    std::string name = GetStepName(node, isForward);
    for(int i = 0; i < inputCount; i++)
    {

       if(node->Input(i)->ValuePtr() == NULL){ continue; }
       Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->Input(i)->ValuePtr().get();
       if((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
          buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
          buffer->GetMatrixType() != MatrixType::DENSE)
            { continue; }

            if(m_buffer2IsFreed.count(buffer) == 0){ continue; }
            if(m_buffer2IsFreed[buffer])
            {
                SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
                swp->BeginAction(); // begin swap in
                swp->EndAction(); // end swap in


                m_GBFreed -= buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;
                cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;

                //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
                m_buffer2IsFreed[buffer] = false;
                //cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
            }
        

    }


    if(node->ValuePtr() != NULL)
    {
        Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->ValuePtr().get();

        if(!((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
              buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
              buffer->GetMatrixType() != MatrixType::DENSE))
        {

            if(m_buffer2IsFreed.count(buffer) > 0)
            if(m_buffer2IsFreed[buffer])
            {
                SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
                swp->BeginAction(); // begin swap in
                swp->EndAction(); // end swap in


                m_GBFreed -= buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;

                cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;

                //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
                m_buffer2IsFreed[buffer] = false;
                //cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
            }
        

        }
    }


    if(node->GradientPtr() != NULL)
    {
        Matrix<ElemType> *buffer = (Matrix<ElemType>*)node->GradientPtr().get();

        if(!((buffer->GetDataLocation() != CurrentDataLocation::GPU &&
              buffer->GetDataLocation() != CurrentDataLocation::BOTH) ||
              buffer->GetMatrixType() != MatrixType::DENSE))
        {

            if(m_buffer2IsFreed.count(buffer) > 0)
            if(m_buffer2IsFreed[buffer])
            {
                SwapInAction<ElemType> *swp = m_buffer2SwapIn[buffer];
                swp->BeginAction(); // begin swap in
                swp->EndAction(); // end swap in


                m_GBFreed -= buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;

                cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;

                //cout << buffer->GetNumRows() << "x" << buffer->GetNumCols() << endl;
                m_buffer2IsFreed[buffer] = false;
                //cout << "swapped in buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
            }
        

        }
    }




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

    //std::string name = GetStepName(node, isForward);
    //int stepNumber = m_stepName2StepNumber[name];
    //// free those used two layers below
    //stepNumber = GetStepNumber(stepNumber,-2);
    //for(int i = 0; i < m_stepNumber2Buffer[stepNumber].size(); i++) 
    //{
    //    Matrix<ElemType> *buffer = m_stepNumber2Buffer[stepNumber][i];
    //    if(buffer == NULL){ continue; }
    //    if(buffer->GetNumElements() < 10){ continue; }
    //    if(m_bannedBuffers2bool.count(buffer) > 0){ continue; }

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

    //    if(buffer->GetNumRows() == 10 && buffer->GetNumCols() == 200){ continue; }
    //    if(buffer->GetNumRows() == 784 && buffer->GetNumCols() == 32){ continue; }
    //    if(buffer->GetNumRows() == 10 && buffer->GetNumCols() == 32){ continue; }
    //    cout << "freeing buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
    //    m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
    //    m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
    //    m_buffer2IsFreed[buffer] = true;

    //}

    std::string name = GetStepName(node, isForward);
    int timeStep = m_nodes2timestep[node];
    timeStep = GetStepNumber(timeStep, -2);
    cout << "freeing timestep: " << timeStep << endl;
    if(m_timestep2nodes.count(timeStep) == 0){ return; }
    ComputationNodeBase *swapNode = m_timestep2nodes[timeStep];

    //if(swapNode->GradientPtr() != NULL)
    //{
    //    Matrix<ElemType> *buffer = (Matrix<ElemType>*)swapNode->ValuePtr().get();

    //    if(m_bannedBuffers2bool.count(buffer) == 0)
    //        m_bannedBuffers2bool[buffer] = true;
    //}

    int inputCount = swapNode->GetNumInputs();
    for(int i = 0; i < inputCount; i++)
    {

       if(swapNode->Input(i)->ValuePtr() == NULL){ cout << "NULL" << endl; continue; }
       Matrix<ElemType> *buffer = (Matrix<ElemType>*)swapNode->Input(i)->ValuePtr().get();

        if(m_bannedNodes2Bool.count(swapNode->Input(i).get()) > 0){ cout << "BANNED" << endl; continue; }
        if(buffer == NULL){ cout << "NULL2" << endl; continue; }
        if(buffer->GetNumElements() < 1024){ cout << "small" << endl; continue; }
        if(m_bannedBuffers2bool.count(buffer) > 0){ cout << "BANNED2" << endl; continue; }
        //if(swapNode->Input(i)->IsValueSharable()){ cout << "SHARED" << endl; continue; }

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

        //if(buffer->GetNumRows() == 10 && buffer->GetNumCols() == 200){ continue; }
        std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
        //if(buffer->GetNumRows() == 784 && buffer->GetNumCols() == 32){ cout << "INPUT: " << nodename << endl; continue; }
        //cout << "freeing buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
        m_GBFreed += buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;
        cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;
        m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
        m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
        m_buffer2IsFreed[buffer] = true;

    }


    if(m_bannedNodes2Bool.count(swapNode) > 0){ return; }
    if(swapNode->ValuePtr() != NULL)
    {
        Matrix<ElemType> *buffer = (Matrix<ElemType>*)swapNode->ValuePtr().get();

        if(m_bannedBuffers2bool.count(buffer) > 0){ cout << "banned3" << endl; return; }
        if(buffer->GetNumElements() < 1024){ cout << "small2" << endl; return; }
        //if(swapNode->IsValueSharable()){ cout << "SHARED" << endl; return; }

        if(m_buffer2IsFreed.count(buffer) > 0)
            if(m_buffer2IsFreed[buffer])
                return; // buffer is already freed


        if(m_buffer2SwapIn.count(buffer) == 0)
        {
            SwapOutAction<ElemType> *swpOut = new SwapOutAction<ElemType>(buffer);
            SwapInAction<ElemType> *swpIn = new SwapInAction<ElemType>(swpOut, buffer);
            m_buffer2SwapOut[buffer] = swpOut;
            m_buffer2SwapIn[buffer] = swpIn;
        }

        //cout << "freeing buffer: " << buffer->GetNumRows()  << "x" << buffer->GetNumCols() << endl;
        m_GBFreed += buffer->GetNumElements()*sizeof(ElemType)/1024.0f/1024.0f/1024.0f;
        cout << "Currently: " << m_GBFreed << "GB of memory is swapped out." << endl;
        m_buffer2SwapOut[buffer]->BeginAction(); // begin swap out
        m_buffer2SwapOut[buffer]->EndAction(); // complete swap out
        m_buffer2IsFreed[buffer] = true;
      
    }
#endif
}

inline std::string IsForwardToString(bool b){ return b ? std::string("_forward") : std::string("_backprop"); }
template std::string SynchronizationManager<float>::GetStepName(ComputationNodeBase *node, bool isForward);
template std::string SynchronizationManager<double>::GetStepName(ComputationNodeBase *node, bool isForward);
template<typename ElemType> std::string SynchronizationManager<ElemType>::GetStepName(ComputationNodeBase *node, bool isForward)
{
    std::wstring wname = node->GetName();
    return std::string(wname.begin(), wname.end()) + IsForwardToString(isForward);
}


template void SynchronizationManager<float>::RegisterBuffers(ComputationNodeBase *node, bool isForward);
template void SynchronizationManager<double>::RegisterBuffers(ComputationNodeBase *node, bool isForward);
template<typename ElemType> void SynchronizationManager<ElemType>::RegisterBuffers(ComputationNodeBase *node, bool isForward)
{
    
    
    std::string name = GetStepName(node, isForward);
    if(m_stepName2StepNumber.count(name) > 0){ return; }

    m_stepName2StepNumber[name] = m_currentStepNumber;
    m_nodes2timestep[node] = m_currentStepNumber;
    m_timestep2nodes[m_currentStepNumber] = node;
    m_currentStepNumber++;

    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    cout << m_currentStepNumber << " " << nodename << endl;

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

template void SynchronizationManager<double>::InitializeSwapping(
                          std::unordered_map<Matrix<double>*, std::vector<int> > buffer2Timesteps,
                          std::unordered_map<int, std::vector<Matrix<double>*> > timesteps2Buffers,
                          std::unordered_map<Matrix<double>*, bool> buffer2IsNeededDuringBackprop);
template void SynchronizationManager<float>::InitializeSwapping(
                          std::unordered_map<Matrix<float>*, std::vector<int> > buffer2Timesteps,
                          std::unordered_map<int, std::vector<Matrix<float>*> > timesteps2Buffers,
                          std::unordered_map<Matrix<float>*, bool>  buffer2IsNeededDuringBackprop);
template <typename ElemType> void SynchronizationManager<ElemType>::InitializeSwapping(
                          std::unordered_map<Matrix<ElemType>*, std::vector<int> > buffer2Timesteps,
                          std::unordered_map<int, std::vector<Matrix<ElemType>*> > timesteps2Buffers,
                          std::unordered_map<Matrix<ElemType>*, bool> buffer2IsNeededDuringBackprop)
{

    int maxTimestep = -1;
    for(auto pair : timesteps2Buffers)
        maxTimestep = pair.first > maxTimestep ? pair.first : maxTimestep;

    int maxSwapTimeStep = maxTimestep - 2;

    m_maxTimestep = maxTimestep;
    m_buffer2Timesteps = buffer2Timesteps;
    m_buffer2IsUsedInBackprop = buffer2IsNeededDuringBackprop;

    for(auto pair : m_bannedBuffers2bool)
        cout << "BANNED: " << pair.first << endl;

    for(auto pair : timesteps2Buffers) // int, vector<buffer>
    {
        for(auto buffer : pair.second)
        {
            if(m_bannedBuffers2bool.count(buffer) > 0){ continue; }
            // we do not swap the first 2 and last 2 layers
            if(pair.first+2 >= m_maxTimestep ||
               pair.first-2 <= 0)
                   continue;

            int lastUsage = -1;
            int earliestUsage = 99999999;
            if(buffer2Timesteps[buffer].size() > 1)
            {
                cout << "MULTI-USE: " << buffer << endl;
                for(auto timeStep : buffer2Timesteps[buffer])
                {
                    lastUsage = timeStep > lastUsage ? timeStep : lastUsage;
                    earliestUsage = timeStep < earliestUsage ? timeStep : earliestUsage;
                }

                if(lastUsage != pair.first){ continue; }
            }
            else
            {
                lastUsage = pair.first;
                earliestUsage = pair.first;
            }


            cout << "buffer: " << buffer << ", last usage: " << lastUsage << ", earliestUsage: " << earliestUsage << endl;


            SwapOutAction<ElemType> *out = new SwapOutAction<ElemType>(buffer);
            SwapInAction<ElemType> *in =  new SwapInAction<ElemType>(out, out->GetGPUMatrix());

            assert(m_buffer2SwapIn.count(buffer) == 0);

            m_buffer2SwapOut[buffer] = out;
            m_buffer2SwapIn[buffer] = in;
            //if(m_buffer2IsUsedInBackprop.count(buffer) > 0)
            //{
            //    m_stepNumber2SwapOut[lastUsage+2].push_back(out); 
            //    m_stepNumber2SwapIn[m_maxTimestep - 1].push_back(in); // swap in before backprop
            //    cout << "forward: swap out: " << buffer << " at timestep " << lastUsage+3 << endl;
            //    cout << "forward: swap in: " << buffer << " at timestep " << m_maxTimestep-1 << endl;
            //}
            //else
            {
                // forward
                m_stepNumber2SwapOut[lastUsage+2].push_back(out); 
                m_stepNumber2SwapIn[earliestUsage - 2].push_back(in); 
                cout << "forward: swap out: " << buffer << " at timestep " << lastUsage+2 << endl;
                cout << "forward: swap in: " << buffer << " at timestep " << earliestUsage-2 << endl;
                // backward
                //m_stepNumber2ActionsBackprop[lastUsage+3].push_back(in);
                //m_stepNumber2ActionsBackprop[earliestUsage-3].push_back(out);
                //cout << "backward: swap out: " << buffer << " at timestep " << earliestUsage-3 << endl;
                //cout << "backward: swap in: " << buffer << " at timestep " << lastUsage+3 << endl;
            }

            

        }
    }


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
}



}}}
