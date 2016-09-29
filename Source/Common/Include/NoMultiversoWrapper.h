#pragma once

#include "ASGDCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType = float>
class MultiversoHelper
{
public:
    MultiversoHelper(const std::list<ComputationNodeBasePtr> & learnableNodes,
        int localWorkerNumber,
        bool isPipeline = true,
        AdjustLearningRateatBeginning adjusttype = AdjustLearningRateatBeginning::None,
        double adjustcoef = 0.2,
        size_t adjustnbmb = 0) { }

        ~MultiversoHelper() { }

        void InitModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }

        void PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }
			
        void PushModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }

        void PullModel(const std::list<ComputationNodeBasePtr> & learnableNode) { }

        void WaitAll() { }

        void WaitAsyncBuffer() { }

};
}
}
}
