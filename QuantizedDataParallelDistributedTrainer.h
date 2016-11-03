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

namespace CNTK
{
    ///
    /// Quantized Distributed Trainer.
    ///
    class QuantizedDataParallelDistributedTrainer : public DistributedTrainer
    {
    public:
        QuantizedDataParallelDistributedTrainer(QuantizedDistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate, size_t parallelizationStartAfterSampleCount)
            : DistributedTrainer(parallelizationStartAfterSampleCount),
            m_communicator(communicator),
            m_useAsyncBufferedParameterUpdate(useAsyncBufferedParameterUpdate)
        {
            if (m_useAsyncBufferedParameterUpdate)
                LogicError("Asynchronous parameter update is not yet supported.");
        }

        // Optional override that gets called before each minbatch during training
        void PreMinibatchCallback(const Trainer& /*trainer*/) override
        {
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        void PreParameterUpdateCallback(const Trainer& /*trainer*/, std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info) override
        {
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
            m_communicator->QuantizedAggregateInPlace(
                gradients,
                m_residuals,
                m_stripeResiduals,
                m_communicator->Workers());
        }

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary GetCheckpointState() const override
        {
            // Resetting the residuals. TODO: Does not seem right though... Get should not change the state.
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

            return Dictionary();
        }

        // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
        void RestoreFromCheckpoint(const Dictionary& /*checkpoint*/) override
        {
        }

        DistributedCommunicatorPtr GetCommunicator() override
        {
            return m_communicator;
        }

    private:
        QuantizedDistributedCommunicatorPtr m_communicator;
        bool m_useAsyncBufferedParameterUpdate;

        // Residuals of quantized gradients.
        std::vector<::CNTK::NDArrayViewPtr> m_residuals;
        // Residuals of quantized aggregated stripes this node is responsible for.
        std::vector<::CNTK::NDArrayViewPtr> m_stripeResiduals;
    };
}