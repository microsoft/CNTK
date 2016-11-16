//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DistributedTrainerBase.h"
#include "DistributedCommunicator.h"

namespace CNTK
{
    DistributedTrainerBase::DistributedTrainerBase(DistributedCommunicatorPtr communicator)
        : m_communicator(communicator)
    {
    }

    // Optional override that gets called before each minbatch during training
    void DistributedTrainerBase::PreMinibatchCallback(const Trainer& /*trainer*/)
    {
    }

    // Get checkpoint state associated with distributed trainer
    Dictionary DistributedTrainerBase::CreateCheckpoint(const Trainer&, const Dictionary& localStateToShare)
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
}
