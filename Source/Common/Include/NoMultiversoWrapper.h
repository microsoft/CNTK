#pragma once

#include "ASGDCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType = float>
class MultiversoHelper
{
public:
    MultiversoHelper(const std::list<ComputationNodeBasePtr> & learnableNodes,
        int nodeNumRanks,
        bool useAsyncBuffered = true,
        bool isSimModelAveragingSGD = false,
        AdjustLearningRateatBeginning adjusttype = AdjustLearningRateatBeginning::None,
        double adjustcoef = 0.2,
        size_t adjustnbmb = 600,
        int traceLevel = 0,
        int syncPerfStats = 0,
        const MPIWrapperPtr& pMPI = nullptr) { }

        ~MultiversoHelper() { }

        void InitModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }

        void PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNodes, size_t sampleSinceLastSynced = 0) { }

        void PushModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }

        void PullModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }

        void WaitAll() { }

        void WaitAsyncBuffer() { }
    };
}
}
}
