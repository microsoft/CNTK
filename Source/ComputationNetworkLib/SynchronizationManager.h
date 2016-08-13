//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"
#include <unordered_map>
#include <memory>
#include <string>
#include "CUDATimer.h"
#include <utility>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {


// forward declarations
class ComputationNodeBase;
class FrameRange;
template <typename ElemType> class SwapInAction;
template <typename ElemType> class SwapOutAction;

template <typename ElemType>
class SynchronizationManager
{

private:
    std::unordered_map<int, std::vector<SyncAction<ElemType>*> > m_stepNumber2Actions;
    // singleton constructor

    static SynchronizationManager* s_synchronizationManager;
    bool m_isExecuting;
    CUDATimer m_timer;
    int m_currentStepNumber;
    int m_currentIteration;
    int m_maxStepNumber;
    float m_GBFreed;

    std::unordered_map<Matrix<ElemType>*, std::vector<ComputationNodeBase*>> m_buffers2Nodes;

    std::unordered_map<ComputationNodeBase*,int> m_nodes2timestep;
    std::unordered_map<int, ComputationNodeBase*> m_timestep2nodes;
    // needed to identify a step
    std::unordered_map<std::string, int> m_stepName2StepNumber; 
    // steps to buffers; all these buffers need to be handled during one synchronization call for a given timestep
    // needed in order to determine dependencies
    std::unordered_map<Matrix<ElemType>*, std::vector<int> > m_buffer2StepNumbers; 
    std::unordered_map<int, ElemType> m_stepNumber2ComputationTime; 

    //these are for managing full memory swapping during the dryrun
    std::unordered_map<Matrix<ElemType>*, SwapInAction<ElemType>*> m_buffer2SwapIn;
    std::unordered_map<Matrix<ElemType>*, SwapOutAction<ElemType>*> m_buffer2SwapOut;
    std::unordered_map<Matrix<ElemType>*, bool> m_buffer2IsFreed;
    std::unordered_map<int, std::vector<Matrix<ElemType>*> > m_stepNumber2Buffer; 

    std::unordered_map<int, std::vector<Matrix<ElemType>*> > m_forwardGraph;
    std::unordered_map<int, std::vector<Matrix<ElemType>*> > m_backwardGraph;

    std::unordered_map<Matrix<ElemType>*, std::pair<ElemType, ElemType> > m_buffer2SwapTime;
    std::set<Matrix<ElemType> *> m_bufferSet; // contains all buffers which have the potential to be swapped (that is they are non-sharable)
    std::unordered_map<int, std::pair<ElemType,ElemType> > m_stepNumber2CumulativeSwapInTime;

    // during the dry run only one layer (and its input and output) are active at any time
    void FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward);
    void SwapInFreedBuffers(ComputationNodeBase *node, bool isForward);
    void RegisterBuffers(ComputationNodeBase *node, bool isForward);
    void GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward, bool isBeforeComputation);
    void FindSwapOrder();
    void CleanUp();
    int GetStepNumber(int baseStep, int additionalSteps);
    int GetStepDistance(int step1, int step2);

    void MeasureSwapTime(ComputationNodeBase *node, bool isForward);
    std::string GetStepName(ComputationNodeBase *node, bool isForward);

public:
    SynchronizationManager(){};
    // we use a singleton here; we could also injected the manager during node creation, but
    // sometimes this also makes sure that there is only a single instance available
    // which is quite handy for such a critical resource as memory
    static SynchronizationManager* GetSynchronizationManager();

    ~SynchronizationManager(){};
    // this is called BEFORE a ForwardProp / BackpropTo method call
    void BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
    // this is called AFTER a ForwardProp / BackpropTo method call
    void EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
    // indicates the current state of the manager: 
    // (1) gather stats and determine swap order, or 
    // (2) active usage of swapping (=Executing==true)
    bool IsExecuting(){ return m_isExecuting; }
    // the config sets this to false by default
    bool m_useMemorySwapping;
    bool m_isFloat;
    bool m_registeringBuffers;
    std::unordered_map<Matrix<ElemType>*, bool> m_bannedBuffers2bool;
    std::unordered_map<ComputationNodeBase*,bool> m_bannedNodes2Bool;
    // this cleans the SynchronizationManager up after a action completes
    void ClearActionsAndTheirMemory();
    void RegisterWeight(Matrix<ElemType> *weight);
    void InitializeSwapping(std::unordered_map<int, std::set<Matrix<ElemType>*> > forwardGraph,
                            std::unordered_map<int, std::set<Matrix<ElemType>*> > backwardGraph);	
};

template class SynchronizationManager<float>;
template class SynchronizationManager<double>;
template <typename ElemType> 
SynchronizationManager<ElemType>* SynchronizationManager<ElemType>::s_synchronizationManager = nullptr;


}}}

