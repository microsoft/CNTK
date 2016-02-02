#pragma once

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			enum class AdjustLearningRateatBeginning : int
			{
				None = 0,
				Linearly = 1,
				Staircase = (1 << 1),
			};

			template<class ElemType = float>
			class MultiversoWrapper
			{
				MultiversoWrapper(const std::list<ComputationNodeBasePtr> & learnableNodes,
					int localWorkerNumber,
					bool isPipeline = true,
					AdjustLearningRateatBeginning adjusttype = AdjustLearningRateatBeginning::None,
					double adjustcoef = 0.2,
					size_t adjustnbmb = 0)
				{
					
				}

				~MultiversoWrapper()
				{

				}

				void InitModel(const std::list<ComputationNodeBasePtr> & learnableNode)
				{

				}

				void PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNode)
				{

				}
			
				void PushModel(const std::list<ComputationNodeBasePtr> & learnableNode)
				{

				}

				void PullModel(const std::list<ComputationNodeBasePtr> & learnableNode)
				{

				}

				void WaitAll()
				{

				}

			};
		}
	}
}