//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <list>
#include "ComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// class AdjustLearningRateAtBeginning
//       Providing option for DataParallelASGD training. so that every nodes
//       could adjust learning rate every minibatch at first N epochs.
// -----------------------------------------------------------------------
// TODO: We can removed these options once we can adjust learning rate at minibatches level
enum class AdjustLearningRateAtBeginning : int
{
    None = 0,  // default, don't adjust learning rate
    Linearly = 1, // using linear adjustment, learning rate will from 0 to learningRatesPerMB
    Staircase = (1 << 1), // using staircased adjustment, learning rate will from 0 to learningRatesPerMB every adjustNbMinibatch
};

template<class ElemType = float>
class ASGDHelper
{
public:
    virtual ~ASGDHelper() { }
    // -----------------------------------------------------------------------
    // InitModel() -- Upload initialized model (, which was pre-computed by CNTK logic) .
    // to the parameter servers, so that every node could start training from same model
    // -----------------------------------------------------------------------
    virtual void InitModel(const std::list<ComputationNodeBasePtr> & learnableNodes) = 0;

    // -----------------------------------------------------------------------
    // PushAndPullModel() -- Push parameters of learnableNodes to parameter servers, then get the latests model back.
    // -----------------------------------------------------------------------
    virtual bool PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNodes, size_t sampleSinceLastSynced = 0) = 0;

    // -----------------------------------------------------------------------
    // WaitAll() -- Wait(Barrier) all the other nodes to process
    // -----------------------------------------------------------------------
    virtual void WaitAll() = 0;

    // -----------------------------------------------------------------------
    // WaitAsyncBuffer() -- Wait pipeline thread to finish job when useAsyncBuffer is true
    // -----------------------------------------------------------------------
    virtual void WaitAsyncBuffer() = 0;

};  // Class ASGDHelper

// Factory method to create a ASGDHelper instance
template<class ElemType = float>
ASGDHelper<ElemType>* NewASGDHelper(
    const std::list<ComputationNodeBasePtr> & learnableNodes,                // Parameters that needs to be train
    size_t nodeNumRanks,                                                     // Number of working nodes
    bool useAsyncBuffered = true,                                            // Using asynchonous buffer to hide communication cost
    bool isSimulatedModelAveragingSGD = false,                               // Using parameter server-based MA rather than ASGD
    AdjustLearningRateAtBeginning adjusttype =
    AdjustLearningRateAtBeginning::None,                                     // Adjust learning per minibatches at very beginning of training process
    double adjustCoef = 0.2,                                                 // see in DecayCoefficient()
    size_t adjustPerMinibatches = 600,                                       //
    int traceLevel = 0,                                                      // log level
    int syncPerfStats = 0);                                                  // shown perf data every syncPerfStats

}}}
