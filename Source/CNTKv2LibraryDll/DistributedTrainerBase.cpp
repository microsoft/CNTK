//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DistributedTrainerBase.h"
#include "DistributedCommunicator.h"

namespace CNTK
{
    DistributedTrainerBase::DistributedTrainerBase(DistributedCommunicatorPtr communicator, size_t distributedAfterSampleCount)
        : DistributedTrainer(distributedAfterSampleCount),
          m_communicator(communicator)
    {
    }

    // Optional override that gets called before each minbatch during training
    void DistributedTrainerBase::PreMinibatchCallback(const Trainer& /*trainer*/)
    {
    }

    // Get checkpoint state associated with distributed trainer
    Dictionary DistributedTrainerBase::CreateCheckpoint(const Trainer&, const Dictionary& localStateToShare)
    {
        return CreateCheckpoint(localStateToShare);
    }

    Dictionary DistributedTrainerBase::CreateCheckpoint(const Dictionary& localStateToShare)
    {
        std::vector<DictionaryPtr> remoteState;
        m_communicator->Gather(localStateToShare, remoteState, m_communicator->Workers());

        Dictionary result;
        for (size_t i = 0; i < m_communicator->Workers().size(); ++i)
        {
            result[std::to_wstring(i)] = *remoteState[i];
        }

        return result;
    }

    // Restores the state associated with distributed trainer
    Dictionary DistributedTrainerBase::RestoreFromCheckpoint(const Dictionary& checkpoint)
    {
        auto key = std::to_wstring(m_communicator->CurrentWorker().m_globalRank);
        if (checkpoint.Contains(key))
            return checkpoint[key].Value<Dictionary>();

        // Return 0 rank if possible.
        key = std::to_wstring(0);
        if (!checkpoint.Contains(key))
            RuntimeError("Cannot restore from the checkpoint, 0 rank is missing.");
        return checkpoint[key].Value<Dictionary>();
    }

    void DistributedTrainerBase::HandleEmptyMinibatch(std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info)
    {
        if (info.numberOfSamples == 0)
        {
            // Need to intialize gradients to 0 in case when it is an empty minibatch.
            for (auto& g : gradientValues)
            {
                auto weights = g.first.Value();
                g.second = MakeSharedObject<NDArrayView>(0, weights->GetDataType(), weights->Shape(), weights->Device());
            }

            // TODO: what if in the future the type is different?
            auto dataType = gradientValues.front().first.GetDataType();
            info.evalCriterionValue = MakeSharedObject<NDArrayView>(0, dataType, NDShape{ 1 }, DeviceDescriptor::CPUDevice());
            info.trainingLossValue = MakeSharedObject<NDArrayView>(0, dataType, NDShape{ 1 }, DeviceDescriptor::CPUDevice());
        }
    }
}
