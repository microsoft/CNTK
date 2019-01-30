//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Matrix.h"
#include "SimpleEvaluator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
void SimpleEvaluator<ElemType>::InitDistGradAgg()
{
    if (Globals::UseV2Aggregator())
        m_distGradAgg = make_shared<V2SimpleDistGradAggregator<ElemType>>(m_mpi, false /*useAsyncAggregation*/, m_net->GetDeviceId(), 0 /*syncStatsTrace*/, ::CNTK::MPICommunicator());
    else
        m_distGradAgg = make_shared<SimpleDistGradAggregator<ElemType>>(m_mpi, false /*useAsyncAggregation*/, m_net->GetDeviceId(), 0 /*syncStatsTrace*/);
}

template<>
void SimpleEvaluator<half>::InitDistGradAgg()
{
    if (Globals::UseV2Aggregator())
        m_distGradAgg = make_shared<V2SimpleDistGradAggregator<half>>(m_mpi, false /*useAsyncAggregation*/, m_net->GetDeviceId(), 0 /*syncStatsTrace*/, ::CNTK::MPICommunicator());
    else
        RuntimeError("SimpleEvaluator - half not supported when useV2Aggregator is false.");
}

}}}
