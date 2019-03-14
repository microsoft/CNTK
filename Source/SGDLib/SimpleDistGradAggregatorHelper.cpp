//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma warning(disable : 4267) // conversion from size_t to int or other types

#include "Basics.h"
#include "MPIWrapper.h"
#include "Matrix.h"
#include "SimpleDistGradAggregatorHelper.h"
#include "DistGradHeader.h"
#include "IDistGradAggregator.h"
#include "SimpleDistGradAggregator.h"
#include "V2SimpleDistGradAggregator.h"

namespace Microsoft { namespace MSR { namespace CNTK {


template <class ElemType>
std::shared_ptr<IDistGradAggregator<ElemType>> GetSimpleDistGradAggregator(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes,
    bool useFP16AllReduce)
{
    if (Globals::UseV2Aggregator())
        return std::make_shared<V2SimpleDistGradAggregator<ElemType>>(
            mpi,
            useAsyncAggregation,
            deviceId,
            syncStatsTrace,
            ::CNTK::MPICommunicator(packThresholdSizeInBytes, useFP16AllReduce));
    else
        return std::make_shared<SimpleDistGradAggregator<ElemType>>(
            mpi,
            useAsyncAggregation,
            deviceId,
            syncStatsTrace,
            packThresholdSizeInBytes);
}

template <>
std::shared_ptr<IDistGradAggregator<half>> GetSimpleDistGradAggregator<half>(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes,
    bool useFP16AllReduce)
{
    if (Globals::UseV2Aggregator())
        return std::make_shared<V2SimpleDistGradAggregator<half>>(
            mpi,
            useAsyncAggregation,
            deviceId,
            syncStatsTrace,
            ::CNTK::MPICommunicator(packThresholdSizeInBytes, useFP16AllReduce));
    else
        RuntimeError("SGD - half not supported when useV2Aggregator is false!");
}

template std::shared_ptr<IDistGradAggregator<float>> GetSimpleDistGradAggregator<float>(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes,
    bool useFP16AllReduce);

template std::shared_ptr<IDistGradAggregator<double>> GetSimpleDistGradAggregator<double>(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes,
    bool useFP16AllReduce);

}}}
