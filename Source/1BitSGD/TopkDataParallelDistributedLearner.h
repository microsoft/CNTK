//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include <vector>
#include "CNTKLibrary.h"
#include "DistributedLearnerBase.h"
#include <numeric>
#include <iostream>

namespace CNTK
{
    ///
    /// TopK data parallel
    ///
    class TopkDataParallelDistributedLearner : public DataParallelDistributedLearner
    {
    private:
        using Base = DataParallelDistributedLearner;
        template<class T> using Matrix = Microsoft::MSR::CNTK::Matrix<T>;

    public:
        TopkDataParallelDistributedLearner(TopkDistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributedAfterSamples, bool useAsyncBufferedParameterUpdate)
            : DataParallelDistributedLearner(communicator, learner, distributedAfterSamples, useAsyncBufferedParameterUpdate)
        {
            std::cout << "Constructing TopkDataParallelDistributedLearner" << endl;
        }

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary CreateCheckpoint() override
        {
            // Resetting the residuals.
            // We do this to make sure that the returned checkpoint state is consistent with the in - memory state, since we do not checkpoint the residues.
            for (size_t i = 0; i < m_residuals.size(); ++i)
                if (m_residuals[i]->GetDataType() == DataType::Double)
                    m_residuals[i]->SetValue(0.0);
                else
                    m_residuals[i]->SetValue(0.0f);

            return Base::CreateCheckpoint();
        }

    protected:
        void AggregateInPlace(std::vector<NDArrayViewPtr>& valuesToAggregate) override
        {
            dynamic_cast<TopkDistributedCommunicator*>(m_communicator.get())->TopKAggregateInPlace(valuesToAggregate, m_residuals, m_communicator->Workers());
        }

    private:
        // Residuals of topK gradients.
        std::vector<NDArrayViewPtr> m_residuals;

        DISABLE_COPY_AND_MOVE(TopkDataParallelDistributedLearner);
    };
}
