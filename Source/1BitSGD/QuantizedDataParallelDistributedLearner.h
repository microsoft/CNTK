//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include <vector>
#include "CNTKLibrary.h"
#include "DistributedLearnerBase.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    ///
    /// Quantized Distributed Trainer.
    ///
    class QuantizedDataParallelDistributedLearner : public DistributedLearnerBase
    {
    public:
        QuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate)
            : DistributedLearnerBase(communicator, learner, distributeAfterSamples)
        {
            if (useAsyncBufferedParameterUpdate)
                LogicError("Asynchronous parameter update is not yet supported.");
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info) override
        {
            if (m_sampleCount >= m_distributeAfterSamples)
            {
                auto profGradientAgg = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainGradient);

                if (info.IsEmpty())
                    PrepaireZeroGradients(gradientValues);

                ConvertToOrdered(gradientValues, m_gradientBuffer);

                std::vector<NDArrayViewPtr> headerToAggregate;
                headerToAggregate.push_back(info.evalCriterionValue);
                headerToAggregate.push_back(info.trainingLossValue);

                auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{ 1 }, DeviceDescriptor::CPUDevice());
                headerToAggregate.push_back(value);

                m_communicator->AggregateInPlace(headerToAggregate, m_communicator->Workers());

                info.numberOfSamples = static_cast<size_t>(*headerToAggregate.back()->DataBuffer<double>());

                std::vector<NDArrayViewPtr> gradients;
                for (const auto& i : m_gradientBuffer)
                    gradients.push_back(i.second);
                m_gradientBuffer.clear();

                dynamic_cast<QuantizedDistributedCommunicator*>(m_communicator.get())->QuantizedAggregateInPlace(
                    gradients,
                    m_residuals,
                    m_stripeResiduals,
                    m_communicator->Workers());
            }

            auto profWeights = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainWeights);

            m_sampleCount += info.numberOfSamples;
            if (info.IsEmpty())
                return false;

            return m_learner->Update(gradientValues, info.numberOfSamples, info.atEndOfSweep);
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

            for (size_t i = 0; i < m_stripeResiduals.size(); ++i)
                if (m_stripeResiduals[i])
                    if (m_stripeResiduals[i]->GetDataType() == DataType::Double)
                        m_stripeResiduals[i]->SetValue(0.0);
                    else
                        m_stripeResiduals[i]->SetValue(0.0f);

            return DistributedLearnerBase::CreateCheckpoint();
        }

    private:
        // Residuals of quantized gradients.
        std::vector<NDArrayViewPtr> m_residuals;
        // Residuals of quantized aggregated stripes this node is responsible for.
        std::vector<NDArrayViewPtr> m_stripeResiduals;
    };
}
