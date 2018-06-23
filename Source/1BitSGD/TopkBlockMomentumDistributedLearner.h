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
#include <sstream>

namespace CNTK
{
    ///
    /// Block Momentum Trainer.
    ///
    class TopkBlockMomentumDistributedLearner : public BlockMomentumDistributedLearner
    {
    private:
        using Base = BlockMomentumDistributedLearner;
        template<class T> using Matrix = Microsoft::MSR::CNTK::Matrix<T>;

    public:
        TopkBlockMomentumDistributedLearner(
            TopkDistributedCommunicatorPtr communicator,
            LearnerPtr learner,
            size_t distributedAfterSamples,
            size_t globalModelAggregationBlockSize,
            bool useNesterovMomentum,
            bool resetSGDMomentumAfterAggregation,
            double blockLearningRate)
            : TopkBlockMomentumDistributedLearner(
                  communicator,
                  learner,
                  distributedAfterSamples,
                  globalModelAggregationBlockSize,
                  useNesterovMomentum,
                  resetSGDMomentumAfterAggregation,
                  blockLearningRate,
                  Momentum2TimeConstant(1.0 - 1.0 / (double)communicator->Workers().size(), globalModelAggregationBlockSize))
        {}

        TopkBlockMomentumDistributedLearner(
            TopkDistributedCommunicatorPtr communicator,
            LearnerPtr learner,
            size_t distributedAfterSamples,
            size_t globalModelAggregationBlockSize,
            bool useNesterovMomentum,
            bool resetSGDMomentumAfterAggregation,
            double blockLearningRate,
            double blockMomentumAsTimeConstant)
            : BlockMomentumDistributedLearner(
                communicator,
                learner,
                distributedAfterSamples,
                globalModelAggregationBlockSize,
                useNesterovMomentum,
                resetSGDMomentumAfterAggregation,
                blockLearningRate,
                blockMomentumAsTimeConstant)
        {
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
        void AggregateBlockGradientsInPlace() override
        {
            dynamic_cast<TopkDistributedCommunicator*>(m_communicator.get())->TopKAggregateInPlace(m_tempBlockGradient, m_residuals, m_communicator->Workers());
        }

    private:
        // Residuals of topK gradients.
        std::vector<NDArrayViewPtr> m_residuals;

        DISABLE_COPY_AND_MOVE(TopkBlockMomentumDistributedLearner);
    };
}
