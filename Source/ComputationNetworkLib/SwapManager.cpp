//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapManager.h"
#include "SwapAction.h"
#include "../ComputationNetworkLib/ComputationNode.h"
#include "Sequences.h"
#include "SwapInAction.h"
#include "SwapOutAction.h"

#include <cmath>
#include <iostream>

bool g_useMemorySwapping = false;

namespace Microsoft { namespace MSR { namespace CNTK {


using std::cout;
using std::endl;

inline int SampleSize(){ return 100; }
inline int SwapSampleSize(){ return 10; }
inline float MeasurementUncertainty(){ return 1.15f; }


template <typename ElemType> SwapManager<ElemType>::SwapManager()
{
        m_useMemorySwapping = g_useMemorySwapping;
        m_minFreeMemory = FreeGPUMemoryInGB();
        cout << "FREE: " << m_minFreeMemory << endl;
}

template <typename ElemType> void SwapManager<ElemType>::CleanUp()
{
    for(auto pair : m_buffer2SwapOut)
        pair.second->ReleaseMemory();
}


template<typename ElemType> void SwapManager<ElemType>::BeginSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining)
{

#ifndef CPUONLY
	if(!m_useMemorySwapping || !isTraining)
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
            action->BeginAction();
            action->EndAction();
        }
    m_minFreeMemory = m_minFreeMemory < FreeGPUMemoryInGB() ? m_minFreeMemory : FreeGPUMemoryInGB();
#endif
}


template<typename ElemType> void SwapManager<ElemType>::EndSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining)
{
#ifndef CPUONLY
	if(!m_useMemorySwapping || !isTraining){ return; }

    std::string nodename = std::string(node->NodeName().begin(), node->NodeName().end());
    //cout << nodename << " + " << isForward << endl;
    //cout << "FORWARD: " << isForward << " " << " TRAINING: " << isTraining << endl;

    if(isForward)
        for(auto action : m_node2ForwardSwapOut[node])
        {
            //CUDA_CALL(cudaDeviceSynchronize());
            action->BeginAction();
            action->EndAction();
        }
    else
        for(auto matrix : m_node2BackwardFree[node])
        {
            cout << "Freeing matrix during backprop: " << matrix << " " << matrix->GetNumRows() << "x" << matrix->GetNumCols() << endl;
            matrix->Resize(0,0,0,false);
        }
#endif
}


template <typename ElemType> void SwapManager<ElemType>::InitializeSwapping(
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > lastBackwardNodes2matrices)
{

    ClearActionsAndTheirMemory();
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
	/*
#ifndef CPUONLY
	size_t free = 0, total = 0;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
	return free / 1024.0f / 1024.0f / 1024.0f;
#else
	return 0.0f;
#endif
	*/
	return 0.0f;
    
}

template class SwapManager<double>;
template class SwapManager<float>;

}}}
