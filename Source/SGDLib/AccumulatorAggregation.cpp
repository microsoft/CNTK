//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "AccumulatorAggregation.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<typename ElemType>
bool AggregateAccumulatorSums(
    MPIWrapperPtr &mpi,
    ComputationNetworkPtr &net,
    size_t &packThresholdSizeInBytes,
    std::vector<Matrix<ElemType> *> &accumulatorValues,
    std::shared_ptr<DistGradHeader> &gradHeader)
{
    bool samplesProcessed = false;

    // Prepare aggregator.
    std::shared_ptr<IDistGradAggregator<ElemType>> distGradAgg;
    if (Globals::UseV2Aggregator())
        distGradAgg = make_shared<V2SimpleDistGradAggregator<ElemType>>(
            mpi,
            false /*useAsyncAggregation*/,
            net->GetDeviceId(),
            0 /*syncStatsTrace*/,
            ::CNTK::MPICommunicator(packThresholdSizeInBytes));
    else
        distGradAgg = make_shared<SimpleDistGradAggregator<ElemType>>(
            mpi,
            false /*useAsyncAggregation*/,
            net->GetDeviceId(),
            0 /*syncStatsTrace*/,
            packThresholdSizeInBytes);

    // Aggregate accumulator sums.
    samplesProcessed = distGradAgg->AggregateGradients(accumulatorValues, gradHeader.get(), /*resetState =*/false);

    return samplesProcessed;
}

template<>
bool AggregateAccumulatorSums<half>(
    MPIWrapperPtr &mpi,
    ComputationNetworkPtr &net,
    size_t &packThresholdSizeInBytes,
    std::vector<Matrix<half> *> &accumulatorValues,
    std::shared_ptr<DistGradHeader> &gradHeader)
{
    bool samplesProcessed = false;

    // Prepare aggregator.
    std::shared_ptr<IDistGradAggregator<half>> distGradAgg;
    if (Globals::UseV2Aggregator())
        distGradAgg = make_shared<V2SimpleDistGradAggregator<half>>(
            mpi,
            false /*useAsyncAggregation*/,
            net->GetDeviceId(),
            0 /*syncStatsTrace*/,
            ::CNTK::MPICommunicator(packThresholdSizeInBytes));
    else
        RuntimeError("AggregateAccumulatorSums - half not supported if useV2Aggregator is false.");

    // Aggregate accumulator sums.
    samplesProcessed = distGradAgg->AggregateGradients(accumulatorValues, gradHeader.get(), /*resetState =*/false);

    return samplesProcessed;
}

template bool AggregateAccumulatorSums<float>(
    MPIWrapperPtr &mpi,
    ComputationNetworkPtr &net,
    size_t &packThresholdSizeInBytes,
    std::vector<Matrix<float> *> &accumulatorValues,
    std::shared_ptr<DistGradHeader> &gradHeader);
template bool AggregateAccumulatorSums<double>(
    MPIWrapperPtr &mpi,
    ComputationNetworkPtr &net,
    size_t &packThresholdSizeInBytes,
    std::vector<Matrix<double> *> &accumulatorValues,
    std::shared_ptr<DistGradHeader> &gradHeader);
}}}
