//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// 1-bit Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include <vector>
#include "CNTKLibrary.h"
#include "DistributedTrainerBase.h"

namespace CNTK
{
    ///
    /// Quantized Distributed Trainer.
    ///
    class QuantizedDataParallelDistributedTrainer : public DistributedTrainerBase
    {
    public:
        QuantizedDataParallelDistributedTrainer(QuantizedDistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate, size_t distributedAfterSampleCount)
            : DistributedTrainerBase(communicator, distributedAfterSampleCount)
        {
            if (useAsyncBufferedParameterUpdate)
                LogicError("Asynchronous parameter update is not yet supported.");
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool PreParameterUpdateCallback(const Trainer& /*trainer*/, std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info) override
        {
            HandleEmptyMinibatch(gradientValues, info);

            std::vector<NDArrayViewPtr> headerToAggregate;
            headerToAggregate.push_back(info.evalCriterionValue);
            headerToAggregate.push_back(info.trainingLossValue);

            auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{ 1 }, DeviceDescriptor::CPUDevice());
            headerToAggregate.push_back(value);

            m_communicator->AggregateInPlace(headerToAggregate, m_communicator->Workers());

            info.numberOfSamples = static_cast<size_t>(*headerToAggregate.back()->DataBuffer<double>());

            std::vector<NDArrayViewPtr> gradients;
            for (const auto& i : gradientValues)
                gradients.push_back(i.second);

            dynamic_cast<QuantizedDistributedCommunicator*>(m_communicator.get())->QuantizedAggregateInPlace(
                gradients,
                m_residuals,
                m_stripeResiduals,
                m_communicator->Workers());

            return info.numberOfSamples == 0;
        }

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary CreateCheckpoint(const Trainer& trainer, const Dictionary& localStateToShare) override
        {
            // Resetting the residuals.
            // We do this to make sure that the returned checkpoint state is consistent with the in - memory state, since we do not checkpoint the residues.
            for (size_t i = 0; i < m_residuals.size(); ++i)
                if (m_residuals[i]->GetDataType() == DataType::Double)
                    m_residuals[i]->SetValue(0.0);
                else
                    m_residuals[i]->SetValue(0.0f);

            for (size_t i = 0; i < m_stripeResiduals.size(); ++i)
                if (m_stripeResiduals[i])
                    if (m_stripeResiduals[i]->GetDataType() == DataType::Double)
                        m_stripeResiduals[i]->SetValue(0.0);
                    else
                        m_stripeResiduals[i]->SetValue(0.0f);

            return DistributedTrainerBase::CreateCheckpoint(trainer, localStateToShare);
        }

    private:
        // Residuals of quantized gradients.
        std::vector<NDArrayViewPtr> m_residuals;
        // Residuals of quantized aggregated stripes this node is responsible for.
        std::vector<NDArrayViewPtr> m_stripeResiduals;
    };
}