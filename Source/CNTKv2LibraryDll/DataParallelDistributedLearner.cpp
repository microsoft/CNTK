//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedLearner.h"
#include "DistributedCommunicator.h"
#include "Learner.h"
#include "PerformanceProfiler.h"

#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
#include "QuantizedDistributedCommunicator.h"
#include "QuantizedDataParallelDistributedLearner.h"
#include "BlockMomentumDistributedLearner.h"
#endif

namespace CNTK
{
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits)
    {
        return MakeSharedObject<QuantizedMPICommunicatorImpl>(zeroThresholdFor1Bit, useQuantizationForSelfStripe, numQuantizationBits);
    }

    DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(
        QuantizedDistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<QuantizedDataParallelDistributedLearner>(communicator, learner, distributeAfterSamples, useAsyncBufferedParameterUpdate);
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        size_t blockSize,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedLearner>(
            communicator,
            learner,
            distributeAfterSamples,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate);
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedLearner>(
            communicator,
            learner,
            distributeAfterSamples,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate,
            blockMomentumAsTimeConstant);
    }

#else
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool, bool, size_t)
    {
        LogicError("Quantized MPI Communicator is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr, LearnerPtr, size_t, bool)
    {
        LogicError("Quantized Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr /*communicator*/,
        LearnerPtr,
        size_t /*distributeAfterSamples*/,
        size_t /*blockSize*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr /*communicator*/,
        LearnerPtr,
        size_t /*distributeAfterSamples*/,
        size_t /*blockSize*/,
        double /*blockMomentumAsTimeConstant*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }
#endif

    DistributedLearnerPtr CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributedAfterSamples, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedLearner>(communicator, learner, distributedAfterSamples, useAsyncBufferedParameterUpdate);
    }

    DataParallelDistributedLearner::DataParallelDistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributedAfterSamples, bool useAsyncBufferedParameterUpdate)
        : DistributedLearnerBase(communicator, learner, distributedAfterSamples, !Internal::ShouldUseSparseGradientAggregationInDataParallelSGD())
    {
        if (useAsyncBufferedParameterUpdate)
            LogicError("Asynchronous parameter update is not yet supported for the DataParallelDistributedLearner.");
    }

    bool DataParallelDistributedLearner::Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info)
    {
        // sparse gradient may be converted to dense for aggregation
        std::unordered_map<Parameter, NDArrayViewPtr> convertedGradientValues = gradientValues;

        if (m_sampleCount >= m_distributeAfterSamples && m_communicator->Workers().size() > 1)
        {
#ifndef  CNTK_UWP
            auto profGradientAgg = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainGradient);
#endif

            if (info.IsEmpty())
                PrepaireZeroGradients(gradientValues);

            // sorts gradient buffers according to parameter uid, and perform sparse to dense conversion
            // if !UseSparseGradientAggregationInDataParallelSGD()
            ConvertToOrdered(gradientValues, m_gradientBuffer, &convertedGradientValues);

            std::vector<NDArrayViewPtr> valuesToAggregate;
            std::vector<NDArrayViewPtr> sparseValuesToAggregate;
            for (const auto& i : m_gradientBuffer)
            {
                auto storageFormat = i.second->GetStorageFormat();
                if (storageFormat == StorageFormat::Dense)
                {
                    valuesToAggregate.push_back(i.second);
                }
                else
                {
                    if (storageFormat != StorageFormat::SparseBlockCol)
                        LogicError("Unsupported sparse gradient format");

                    // NOTE: CPU sparse block column stores block Ids in size_t and it's different from GPU SBC
                    // We should refactor the CPU SBC code to align with GPU in future
                    if (i.second->Device().Type() == DeviceKind::CPU)
                        LogicError("Unsupported CPU sparse block column aggregation");

                    sparseValuesToAggregate.push_back(i.second);
                }
            }
            valuesToAggregate.push_back(info.evalCriterionValue);
            valuesToAggregate.push_back(info.trainingLossValue);

            auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{}, DeviceDescriptor::CPUDevice());
            valuesToAggregate.push_back(value);

            m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());
            info.numberOfSamples = static_cast<size_t>(*valuesToAggregate.back()->WritableDataBuffer<double>());

            if (!sparseValuesToAggregate.empty())
            {
                m_communicator->AllReduceSparseBlockColumn(sparseValuesToAggregate);
            }
        }

#ifndef  CNTK_UWP
        auto profWeights = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainWeights);
#endif

        m_sampleCount += info.numberOfSamples;
        m_gradientBuffer.clear();

        if (info.IsEmpty())
            return false;

        return m_learner->Update(convertedGradientValues, info.numberOfSamples, info.atEndOfSweep);
    }
}
