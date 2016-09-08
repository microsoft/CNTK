//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings


#include "SwapManager.h"
#include "SwapAction.h"
#include "ComputationNode.h"
#include "Sequences.h"
#include "SwapInAction.h"
#include "SwapOutAction.h"
#include "ComputationNetwork.h"
#include "GPUMatrix.h"

#include <iostream>
#include <cmath>

bool g_useMemorySwapping = false;

namespace Microsoft { namespace MSR { namespace CNTK {


using std::cout;
using std::endl;

inline int SampleSize(){ return 100; }
inline int SwapSampleSize(){ return 10; }
inline float MeasurementUncertainty(){ return 1.15f; }


template <typename ElemType> SwapManager<ElemType>::SwapManager()
{
        m_timer = CUDATimer();
        m_useMemorySwapping = g_useMemorySwapping;
        m_minFreeMemory = FreeGPUMemoryInGB();
        cout << "FREE: " << m_minFreeMemory << endl;
        m_freed = 0.0f;
        m_swappedOut = 0.0f;
        m_swappedIn = 0.0f;
        m_wasForward = true;
}

template <typename ElemType> void SwapManager<ElemType>::CleanUp()
{
    for(auto pair : m_buffer2SwapOut)
        pair.second->ReleaseMemory();
}

template <typename ElemType> void SwapManager<ElemType>::SwapOutNodes(ComputationNodeBase* node, bool isForward, bool isTraining, int n)
{
    if(m_node2Timestep.count(node) == 0){ return; }
    int currentTimeStep = m_node2Timestep[node];
    while(n > 0) 
    {
        // free / swapout previous layers, which are to the back for backprop, and in the front
        // for forward prop
        currentTimeStep += isForward ? -1 : 1;
        n--;
        if(m_timeStep2Node.count(currentTimeStep) == 0){ continue; }

        BeginSynchronizeState(m_timeStep2Node[currentTimeStep], isForward, isTraining);
        EndSynchronizeState(m_timeStep2Node[currentTimeStep], isForward, isTraining);
    }
    m_useMemorySwapping = true;
    
}

template<typename ElemType> void SwapManager<ElemType>::BeginSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining)
{

#ifndef CPUONLY
    if(!m_wasForward && isForward)
    {
        m_freed = 0.0f;
        m_swappedOut = 0.0f;
        m_swappedIn = 0.0f;
    }

    cout << "free: " << m_freed << " swapped out: " << m_swappedOut << " swapped in: " << m_swappedIn << endl;


	if(!m_useMemorySwapping)
    {
       //cout << m_minFreeMemory << endl;
        m_minFreeMemory = m_minFreeMemory < FreeGPUMemoryInGB() ? m_minFreeMemory : FreeGPUMemoryInGB();
        return;
    }

    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    //cout << nodename << " + " << isForward << endl;

    if(!isForward)
        for(auto action : m_node2BackwardSwapin[node])
        {
            cout << FreeGPUMemoryInGB() << endl;
            action->BeginAction();
            action->EndAction();
            cout << FreeGPUMemoryInGB() << endl;
            m_swappedIn += ((float)action->GetGPUMatrix()->BufferSize())/1024./1024./1024.;
        }
    m_minFreeMemory = m_minFreeMemory < FreeGPUMemoryInGB() ? m_minFreeMemory : FreeGPUMemoryInGB();
#endif
}


template<typename ElemType> void SwapManager<ElemType>::EndSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining)
{
#ifndef CPUONLY
	if(!m_useMemorySwapping){ return; }

    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    //cout << nodename << " + " << isForward << endl;
    //cout << "FORWARD: " << isForward << " " << " TRAINING: " << isTraining << endl;

    if(isForward && isTraining)
        for(auto action : m_node2ForwardSwapOut[node])
        {
            CUDA_CALL(cudaDeviceSynchronize());
            m_swappedOut += ((float)action->GetGPUMatrix()->BufferSize())/1024./1024./1024.;
            action->BeginAction();
            action->EndAction();
        }
    else if(isTraining)
        for(auto matrix : m_node2BackwardFree[node])
        {
            cout << "Freeing matrix during backprop: " << matrix << " " << matrix->GetNumRows() << "x" << matrix->GetNumCols() << endl;
            m_freed += ((float)matrix->BufferSize())/1024./1024./1024.;
            cout << FreeGPUMemoryInGB() << endl;
            matrix->Resize(0,0,0,false);
            cout << FreeGPUMemoryInGB() << endl;
        }
    m_wasForward = isForward;
#endif
}


template <typename ElemType> void SwapManager<ElemType>::InitializeSwapping(
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > lastBackwardNodes2matrices)
{

    ClearActionsAndTheirMemory();
    int timeStep = 0;
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

            // build timeline of nodes for later reference
            m_node2ForwardSwapOut[pair.first].push_back(m_buffer2SwapOut[buffer]);
            m_node2Timestep[pair.first] = timeStep;
            m_timeStep2Node[timeStep] = pair.first;
            timeStep++;
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


template<typename ElemType> void SwapManager<ElemType>::ClearActionsAndTheirMemory()
{
    cout << "Cleaning up!" << endl;
    cout << "FREE: " << m_minFreeMemory << endl;
    CleanUp();

    m_buffer2SwapIn.clear();
    m_buffer2SwapOut.clear();

    m_node2ForwardSwapOut.clear();
    m_node2BackwardSwapin.clear();
    m_node2BackwardFree.clear();
}

template <typename ElemType> float SwapManager<ElemType>::FreeGPUMemoryInGB()
{
#ifndef CPUONLY
	size_t free = 0, total = 0;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
	return free / 1024.0f / 1024.0f / 1024.0f;
#else
	return 0.0f;
#endif
    
}

template class SwapManager<double>;
template class SwapManager<float>;

}}}
