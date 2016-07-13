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

namespace Microsoft { namespace MSR { namespace CNTK {


// forward declarations
class ComputationNodeBase;
class FrameRange;
class SwapInAction;
class SwapOutAction;

class Stats
{
public:
    std::string name;
    float forwardTime;
    float backpropTime;
    std::vector<float> swapInTimes;
    std::vector<float> swapOutTimes;
    std::vector<Matrix<float>*> buffers;
    std::vector<std::string> dim;
    void PrintStats()
    {
       fprintf(stdout, "%s: Forward: %f, Backprop: %f. ", name.c_str(), forwardTime, backpropTime);
       fprintf(stdout, "Swap times: \n");
       for(int i = 0; i < swapInTimes.size(); i++)
            fprintf(stdout, "For input %s idx %i: (in %f, out %f)\n", dim[i].c_str(), i, swapInTimes[i], swapOutTimes[i]);
    }
};

class SynchronizationManager
{

private:
    std::unordered_map<std::string, std::vector<SyncAction*> > m_stepName2Actions;
    SynchronizationManager(float performanceCosts);
    SynchronizationManager(){};
    static SynchronizationManager* s_synchronizationManager;

    // needed to identify a step
    std::unordered_map<std::string, int> m_stepName2StepNumber; 
    // steps to buffers; all these buffers need to be handled during one synchronization call for a given timestep
    std::unordered_map<int, std::vector<Matrix<float>*> > m_stepNumber2Buffer; 
    // needed in order to determine dependencies
    std::unordered_map<Matrix<float>*, std::vector<int> > m_buffer2StepNumbers; 
    std::unordered_map<std::string, Stats*> m_stepName2Stats; 
    std::vector<Stats*> m_vecStats; 

    std::unordered_map<Matrix<float>*, SwapInAction*> m_buffer2SwapIn;
    std::unordered_map<Matrix<float>*, SwapOutAction*> m_buffer2SwapOut;
    std::unordered_map<Matrix<float>*, bool> m_buffer2IsFreed;
    std::unordered_map<Matrix<float>*, std::pair<int, int> > m_buffer2Dim;
    std::unordered_map<Matrix<float>*, bool> m_buffer2Swappable;
    float m_performanceCostLimit;

    CUDATimer m_timer;


    enum SynchronizationState
    {
        Uninitialized = 0,
        RegisteringBuffers = 1,
        GatheringRuntimeStatistics = 2,
        FindingSwapOrder = 4,
        CleaningUp = 8,
        ExecutingActions = 16,
    };

    SynchronizationState m_currentState;
    int m_currentStepNumber;

    void FreeBuffersForDryRun(ComputationNodeBase *node, bool isForward);
    void SwapInFreedBuffers(ComputationNodeBase *node, bool isForward);
    void CheckForStateTransitions(ComputationNodeBase *node, bool isForward);
    void RegisterBuffers(ComputationNodeBase *node, bool isForward);
    void GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
    void FindSwapOrder();
    void CleanUp();

    void MeasureSwapTime(ComputationNodeBase *node, std::string name);
    std::string GetStepName(ComputationNodeBase *node, bool isForward);
    std::string GetBufferName(ComputationNodeBase *node, bool isForward);


public:
    ~SynchronizationManager(){};
    static SynchronizationManager* GetSynchronizationManager(float performanceCostLimit);
    void BeginSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
    void EndSynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
	
};


}}}

