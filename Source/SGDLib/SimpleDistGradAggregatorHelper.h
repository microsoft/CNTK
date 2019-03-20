//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Constants.h"
#include "IDistGradAggregator.h"

namespace Microsoft { namespace MSR { namespace CNTK {


template <class ElemType>
std::shared_ptr<IDistGradAggregator<ElemType>> GetSimpleDistGradAggregator(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes = DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES,
    bool useFP16AllReduce = false);

}}}
