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

namespace Microsoft { namespace MSR { namespace CNTK {


// forward declarations
class ComputationNodeBase;
class FrameRange;

class Stats
{
public:
    float forwardTime;
    float backpropTime;
    std::vector<float> swapInTimes;
    std::vector<float> swapOutTimes;
    void PrintStats()
    {
       fprintf(stdout, "Forward: %f, Backprop: %f\n", forwardTime, backpropTime);
       fprintf(stdout, "Swap times: \n");
       for(int i = 0; i < swapInTimes.size(); i++)
            fprintf(stdout, "For input idx %i: (in %f, out %f)\n", i, swapInTimes[i], swapOutTimes[i]);
    }
};

class SynchronizationManager
{

private:
    typedef std::shared_ptr<SyncAction> SyncActionPtr;
    std::unordered_map<ComputationNodeBase *, std::vector<SyncActionPtr> > m_actionTable;
    SynchronizationManager(){};
    static SynchronizationManager* s_synchronizationManager;

    // needed to identify a step
    std::unordered_map<std::string, int> m_stepName2StepNumber; 
    // steps to buffers; all these buffers need to be handled during one synchronization call for a given timestep
    std::unordered_map<int, std::vector<MatrixBase*> > m_stepNumber2Buffer; 
    // needed in order to determine dependencies
    std::unordered_map<MatrixBase*, std::vector<int> > m_buffer2StepNumbers; 
    std::unordered_map<std::string, Stats> m_stepName2Stats; 

    CUDATimer m_timer;


    enum SynchronizationState
    {
        Uninitialized = 0,
        RegisteringBuffers = 1,
        GatheringRuntimeStatistics = 2,
        Benchmarking = 4,
        FindingSwapOrder = 8,
        GeneratingAllocationScheme = 16,
        ExecutingActions = 32
    };

    SynchronizationState m_currentState;
    int m_currentStepNumber;

    void ExecuteActions(ComputationNodeBase *node);
    void RegisterBuffers(ComputationNodeBase *node);
    void GatherRuntimeStatistics(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);

    void MeasureSwapTime(ComputationNodeBase *node, std::string name);
    std::string GetStepName(ComputationNodeBase *node, bool isForward);
    std::string GetBufferName(ComputationNodeBase *node, bool isForward);

public:
    ~SynchronizationManager(){};
    static SynchronizationManager* GetSynchronizationManager();
    void SynchronizeState(ComputationNodeBase *node, const size_t idx, const FrameRange& fr, bool isForward);
	
};


}}}

