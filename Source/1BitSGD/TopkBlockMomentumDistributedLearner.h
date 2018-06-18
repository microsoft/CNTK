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

        void Reset(const std::vector<NDArrayViewPtr>& parameters) override
        {
            if (m_residuals.size() != m_tempBlockGradient.size())
                m_residuals.resize(m_tempBlockGradient.size());


            Base::Reset(parameters);

            for (size_t i = 0; i < parameters.size(); ++i)
            {
                auto& p = parameters[i];

                if (p->GetDataType() == DataType::Double)
                    ResetBuffer1<double>(i, p);
                else if (p->GetDataType() == DataType::Float)
                    ResetBuffer1<float>(i, p);
                else
                    RuntimeError("Unsupported type.");
            }
        }

        template<class ElemType>
        void ResetBuffer1(size_t index, const NDArrayViewPtr& p)
        {
            if (!m_residuals[index])
            {
                m_residuals[index] = std::make_shared<NDArrayView>(AsDataType<ElemType>(), p->Shape(), AsDeviceDescriptor(p->Device().Id()));
            }
        }

    private:
        // Residuals of topK gradients.
        std::vector<NDArrayViewPtr> m_residuals;

        DISABLE_COPY_AND_MOVE(TopkBlockMomentumDistributedLearner);
    };
}
