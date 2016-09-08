//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include "CUDATimer.h"
#include <utility>
#include <set>
#include <vector>


extern bool g_useMemorySwapping;
extern bool g_shareNodeValueMatrices;

namespace Microsoft { namespace MSR { namespace CNTK {


// forward declarations
class ComputationNodeBase;
class FrameRange;
template <typename ElemType> class SwapInAction;
template <typename ElemType> class SwapOutAction;
template <typename ElemType> class SwapAction;
template <typename ElemType> class Matrix;

template <typename ElemType>
class SwapManager
{

private:
    std::unordered_map<Matrix<ElemType>*, SwapInAction<ElemType>*> m_buffer2SwapIn;
    std::unordered_map<Matrix<ElemType>*, SwapOutAction<ElemType>*> m_buffer2SwapOut;

    std::unordered_map<ComputationNodeBase*, std::vector<SwapAction<ElemType>*> > m_node2ForwardSwapOut;
    std::unordered_map<ComputationNodeBase*, std::vector<SwapAction<ElemType>*> > m_node2BackwardSwapin;
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > m_node2BackwardFree;
    // singleton constructor

    std::unordered_map<ComputationNodeBase*, int> m_node2Timestep;
    std::unordered_map<int, ComputationNodeBase*> m_timeStep2Node;

    CUDATimer m_timer;
    void CleanUp();
    float m_minFreeMemory;

    float  m_freed;
    float  m_swappedOut;
    float  m_swappedIn;
    bool m_wasForward;

public:
    SwapManager();
    ~SwapManager(){};
    float FreeGPUMemoryInGB();
    // this is called BEFORE a ForwardProp / BackpropTo method call
    void BeginSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining);
    // this is called AFTER a ForwardProp / BackpropTo method call
    void EndSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining);
    void SwapOutNodes(ComputationNodeBase* node, bool isForward, bool isTraining, int n);
    bool m_useMemorySwapping;
    void ClearActionsAndTheirMemory();
    void InitializeSwapping(std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > lastBackwardNodes2matrices);

};




}}}

