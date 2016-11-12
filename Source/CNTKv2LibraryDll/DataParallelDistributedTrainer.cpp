//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedTrainer.h"
#include "DistributedCommunicator.h"

#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
#include "QuantizedDistributedCommunicator.h"
#include "QuantizedDataParallelDistributedTrainer.h"
#include "BlockMomentumDistributedTrainer.h"
#endif

namespace CNTK
{
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits)
    {
        return MakeSharedObject<QuantizedMPICommunicatorImpl>(zeroThresholdFor1Bit, useQuantizationForSelfStripe, numQuantizationBits);
    }

    DistributedTrainerPtr CreateQuantizedDataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedTrainer>(communicator, useAsyncBufferedParameterUpdate);
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr communicator,
        size_t blockSize,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedTrainer>(
            communicator,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate);
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr communicator,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedTrainer>(
            communicator,
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

    DistributedTrainerPtr CreateQuantizedDataParallelDistributedTrainer(DistributedCommunicatorPtr, bool)
    {
        LogicError("Quantized Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr /*communicator*/,
        size_t /*blockSize*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr /*communicator*/,
        size_t /*blockSize*/,
        double /*blockMomentumAsTimeConstant*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }
#endif

    DistributedTrainerPtr CreateDataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedTrainer>(communicator, useAsyncBufferedParameterUpdate);
    }

    DataParallelDistributedTrainer::DataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
        : DistributedTrainerBase(communicator)
    {
        if (useAsyncBufferedParameterUpdate)
            LogicError("Asynchronous parameter update is not yet supported.");
    }

    // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
    bool DataParallelDistributedTrainer::PreParameterUpdateCallback(const Trainer& /*trainer*/, std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info)
    {
        std::vector<NDArrayViewPtr> valuesToAggregate;
        for (const auto& i : gradientValues)
            valuesToAggregate.push_back(i.second);
        valuesToAggregate.push_back(info.evalCriterionValue);
        valuesToAggregate.push_back(info.trainingLossValue);

        auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{1}, DeviceDescriptor::CPUDevice());
        valuesToAggregate.push_back(value);

        m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());

        info.numberOfSamples = static_cast<size_t>(*valuesToAggregate.back()->WritableDataBuffer<double>());
        return info.numberOfSamples == 0;
    }
}
