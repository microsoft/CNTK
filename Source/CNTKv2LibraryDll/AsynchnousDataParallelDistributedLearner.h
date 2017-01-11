//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "CNTKLibrary.h"
#include "DistributedLearnerBase.h"
#include <numeric>

namespace CNTK
{
    ///
    /// Asynchnous Distributed Trainer.
    ///
    class AsynchnousDataParallelDistributedLearner : public DistributedLearnerBase
    {
    public:
        AsynchnousDataParallelDistributedLearner(ParameterServerCommunicatorPtr communicator, LearnerPtr learner, size_t distributedAfterSamples, bool useAsyncBufferedParameterUpdate);

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& trainingSampleCount) override;
    };
}