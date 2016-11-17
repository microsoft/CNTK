//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"

namespace CNTK
{
    ///
    /// Base class for distributed trainers.
    /// TODO: will be switched to Distributed Learner soon.
    ///
    class DistributedTrainerBase : public DistributedTrainer
    {
    public:
        // Callback that gets called before each minbatch during training
        void PreMinibatchCallback(const Trainer& trainer) override;

        // Gets checkpoint state associated with this distributed trainer
        Dictionary CreateCheckpoint(const Trainer& trainer, const Dictionary& localStateToShare) override;

        // Restores the trainer from the state.
        Dictionary RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        void Shutdown(const Trainer&) override {}

        DistributedCommunicatorPtr GetCommunicator() override
        {
            return m_communicator;
        }

    protected:
        explicit DistributedTrainerBase(DistributedCommunicatorPtr communicator, size_t distributedAfterSampleCount);
        Dictionary CreateCheckpoint(const Dictionary& localStateToShare);

        static void HandleEmptyMinibatch(std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info);

        DistributedCommunicatorPtr m_communicator;
    };
}