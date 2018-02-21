//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DistributedLearnerBase.h"
#include "Learner.h"

namespace CNTK
{
    DistributedLearnerBase::DistributedLearnerBase(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool convertSparseToDense)
        : DistributedLearner(communicator, learner, distributeAfterSamples),
          m_convertSparseToDense(convertSparseToDense)
    {
        if (!m_learner)
            InvalidArgument("Learner cannot be null.");

        if (!m_communicator)
            InvalidArgument("Communicator of a DistributedLearner cannot be null.");
    }

    // Get checkpoint state associated with distributed trainer
    Dictionary DistributedLearnerBase::CreateCheckpoint()
    {
        Dictionary result;
        result[L"localLearners"] = m_learner->CreateCheckpoint();
        result[L"totalNumberOfSamplesSeen"] = m_sampleCount;
        return result;
    }

    // Restores the state associated with distributed trainer
    void DistributedLearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint)
    {
        m_learner->RestoreFromCheckpoint(checkpoint[L"localLearners"].Value<Dictionary>());
        m_sampleCount = checkpoint[L"totalNumberOfSamplesSeen"].Value<size_t>();
    }

    void DistributedLearnerBase::PrepaireZeroGradients(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues)
    {
        // Need to initialize gradients to 0 in case when it is an empty minibatch.
        for (auto& g : gradientValues)
        {
            auto weights = g.first.Value();
            g.second = MakeSharedObject<NDArrayView>(0, weights->GetDataType(), weights->Shape(), weights->Device());
        }
    }

    void DistributedLearnerBase::ConvertToOrdered(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, std::vector<std::pair<Parameter, NDArrayViewPtr>>& result, std::unordered_map<Parameter, NDArrayViewPtr>* convertedGradientValues)
    {
        result.reserve(gradientValues.size());
        result.clear();

        if (convertedGradientValues)
            convertedGradientValues->clear();

        for (auto g : gradientValues)
        {
            NDArrayViewPtr p = g.second;
            // convert sparse gradient to dense for accumulation
            if (m_convertSparseToDense && p->GetStorageFormat() != StorageFormat::Dense)
            {
                NDArrayViewPtr pDense = MakeSharedObject<NDArrayView>(0, p->GetDataType(), p->Shape(), p->Device());
                pDense->CopyFrom(*p);
                p = pDense;
            }
            auto pair = std::make_pair(g.first, p);
            result.push_back(pair);

            if (convertedGradientValues)
                convertedGradientValues->insert(pair);
        }

        std::sort(result.begin(), result.end(),
            [](const std::pair<Parameter, NDArrayViewPtr>& a, const std::pair<Parameter, NDArrayViewPtr>& b) { return a.first.Uid() < b.first.Uid(); });
    }
}
