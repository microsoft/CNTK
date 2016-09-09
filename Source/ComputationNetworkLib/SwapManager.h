//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once


#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include <set>
#include <vector>

extern bool g_useMemorySwapping;

#ifdef CPUONLY
#pragma comment(lib, "Math.lib") // built by CNTKMathCUDA project
#else
#endif

namespace Microsoft { namespace MSR { namespace CNTK {


// forward declarationsc
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

    void CleanUp();
    float FreeGPUMemoryInGB();
    float m_minFreeMemory;

public:
    SwapManager();
    ~SwapManager(){};
    // this is called BEFORE a ForwardProp / BackpropTo method call
    void BeginSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining);
    // this is called AFTER a ForwardProp / BackpropTo method call
    void EndSynchronizeState(ComputationNodeBase *node, bool isForward, bool isTraining);
    bool m_useMemorySwapping;
    void ClearActionsAndTheirMemory();
    void InitializeSwapping(std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > forwardSwapOutNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > backwardSwapInNodes2matrices,
    std::unordered_map<ComputationNodeBase*, std::vector<Matrix<ElemType>*> > lastBackwardNodes2matrices);

};




}}}

