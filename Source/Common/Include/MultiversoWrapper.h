#pragma once

// This uses Multiverso.h which requires 
// the header files in ..\Multiverso\include
#include <multiverso/multiverso.h>
#include <multiverso/net.h>
#include <multiverso/util/log.h>
#include <multiverso/util/net_util.h>
#include <multiverso/table/array_table.h>
#pragma comment(lib, "IMultiverso.lib")

#ifndef CPUONLY
#include <cuda_runtime.h>
#pragma comment (lib, "cudart.lib")     // for cudaMemcpyAsync()
#endif


#include "MPIWrapper.h"
#include "ComputationNetwork.h"
#include "TimerUtility.h"

#include <functional>
#include <thread>
#include <unordered_map>
#include <numeric>

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

#ifndef CPUONLY
#define CudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
			inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
			{
				if (code != cudaSuccess)
				{
					fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
					if (abort) exit(code);
				}
			}
#endif

			enum class AdjustLearningRateatBeginning : int
			{
				None = 0,
				Linearly = 1,
				Staircase = (1 << 1),
			};

			template<class ElemType = float>
			class MultiversoWrapper
			{
				typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
			public:
				MultiversoWrapper(const std::list<ComputationNodeBasePtr> & learnableNodes,
					int MPINodeNum,
					bool isAsyncBuffered = true,
					AdjustLearningRateatBeginning adjusttype = AdjustLearningRateatBeginning::None,
					double adjustcoef = 0.2,
					size_t adjustnbmb = 600)
				{
					TestNet();
					m_modelSyncCount = 0;
					m_adjustLearningRateAtBeginningType = adjusttype;
					m_adjustCoefficient = adjustcoef;
					m_adjustMBNumber = adjustnbmb;

					m_totalClientNumber = MPINodeNum;

					//Pipeline releated variables
					m_isUseAsyncBuffered = isAsyncBuffered;
					m_localCacheNumber = m_isUseAsyncBuffered ? 2 : 1;
					m_cacheSwapIndex = new int[m_localCacheNumber];

					//CPU asynchronous buffer
					m_cpuAsyncBuffer = new ElemType*[m_localCacheNumber];
#ifndef CPUONLY
					//GPU asynchronous buffer
					m_gpuAsyncBuffer = new Matrix<ElemType>**[m_localCacheNumber];

					//creat an communication stream for the data tranfer between GPU and CPU
					CudaErrorCheck(cudaStreamCreate(&_commStream));
#endif
					m_bufferInUse = 0;
					for (int i = 0; i < m_localCacheNumber; i++)
						m_cacheSwapIndex[i] = (i + 1) % m_localCacheNumber;

					m_prefetchThread = nullptr;

					MultiversoInit(learnableNodes);
				}

				~MultiversoWrapper()
				{
					fprintf(stderr, "~MultiversoWrapper\n");
					fflush(stderr);

					if (m_isUseAsyncBuffered && m_prefetchThread != nullptr && m_prefetchThread->joinable())
						m_prefetchThread->join();

					delete m_cacheSwapIndex, m_deltaArray;

					for (size_t i = 0; i < m_localCacheNumber; i++)
					{
#ifndef CPUONLY
						CudaErrorCheck(cudaFreeHost(m_cpuAsyncBuffer[i]));
#else
						delete m_cpuAsyncBuffer[i];
#endif
					}
					delete m_cpuAsyncBuffer;
#ifndef CPUONLY
					CudaErrorCheck(cudaStreamDestroy(_commStream));
#endif
					multiverso::MV_ShutDown(false);
				}

				// upoload preCompute model to the parameter servers
				void InitModel(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					float factor = (float) 1.0 / m_totalClientNumber;

					int i = 0; // indicate the index of learnable nodes
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->Value();
#ifndef CPUONLY
						for (int j = 0; j < m_localCacheNumber; j++)
							m_gpuAsyncBuffer[j][i] = new Matrix<ElemType>(mat);
#endif

						ElemType* px = m_cpuAsyncBuffer[0] + m_tableIndex[i];
						mat.CopyToArray(px, m_tableLength[i]);
					}

					for (int i = 1; i < m_localCacheNumber; i++)
						memcpy(m_cpuAsyncBuffer[i], m_cpuAsyncBuffer[0], sizeof(ElemType) * m_totalModelSize);

					memcpy(m_deltaArray, m_cpuAsyncBuffer[0], sizeof(ElemType) * m_totalModelSize);
				
					std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));

					m_sharedArray->Add(m_deltaArray, m_totalModelSize);
					m_sharedArray->Get(m_deltaArray, m_totalModelSize);

				}

				//Todo: support auto adjust learning rate 
				void LearningrateSync(){ throw("not implement yet."); };

				//ASGD logic
				void PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					//Note: maybe overflow.
					m_modelSyncCount++;

					Timer timer;
					WaitAsyncBuffer();

					m_bufferInUse = m_cacheSwapIndex[m_bufferInUse];

					int i = 0; // indicate the index of learnable nodes
					if (m_isUseAsyncBuffered)
					{

						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();
#ifndef CPUONLY
							//CNTK model -> GPU buffer
							CudaErrorCheck(cudaMemcpy(m_gpuAsyncBuffer[m_bufferInUse][i]->BufferPointer(),
								mat.BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));

							//GPU buffer -> CNTK model
							CudaErrorCheck(cudaMemcpy(mat.BufferPointer(),
								m_gpuAsyncBuffer[m_cacheSwapIndex[m_bufferInUse]][i]->BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));
#else
							ElemType * px = m_cpuAsyncBuffer[m_bufferInUse] + m_tableIndex[i];

							mat.CopyToArray(px, m_tableLength[i]);

							ElemType * py = m_cpuAsyncBuffer[m_cacheSwapIndex[m_bufferInUse]] + m_tableIndex[i];

							mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), py);


							delete px;
#endif
						}
#ifndef CPUONLY
						m_prefetchThread = new thread([&](){
							float factor = DecayCoefficient();
							int t_cacheIdx = m_bufferInUse;
							int deviceId = m_gpuAsyncBuffer[t_cacheIdx][0]->GetDeviceId();

							CudaErrorCheck(cudaSetDevice(deviceId));

							for (int widx = 0; widx < m_tableCount; widx++)
							{
								ElemType * px = m_deltaArray + m_tableIndex[widx];
								//GPU buffer -> CPU buffer
								CudaErrorCheck(cudaMemcpyAsync(px,
									m_gpuAsyncBuffer[t_cacheIdx][widx]->BufferPointer(),
									m_gpuAsyncBuffer[t_cacheIdx][widx]->GetNumElements() * sizeof(ElemType),
									cudaMemcpyDeviceToHost,
									_commStream));
							}

							// waiting copy from GPU to CPU finished
							CudaErrorCheck(cudaStreamSynchronize(_commStream));

							// calculate delta
							std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_cpuAsyncBuffer[t_cacheIdx], m_deltaArray, std::minus<ElemType>());

							// lr decay
							std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));

							m_sharedArray->Add(m_deltaArray, m_totalModelSize);
							m_sharedArray->Get(m_cpuAsyncBuffer[t_cacheIdx], m_totalModelSize);

							// copy parameters from CPU buffer to GPU buffer
							for (int widx = 0; widx < m_tableCount; widx++)
							{
								ElemType * py = m_cpuAsyncBuffer[t_cacheIdx] + m_tableIndex[widx];

								CudaErrorCheck(cudaMemcpyAsync(m_gpuAsyncBuffer[t_cacheIdx][widx]->BufferPointer(),
									py,
									m_gpuAsyncBuffer[t_cacheIdx][widx]->GetNumElements() * sizeof(ElemType),
									cudaMemcpyHostToDevice,
									_commStream));
							}

							CudaErrorCheck(cudaStreamSynchronize(_commStream));

						});
#else
						m_prefetchThread = new thread([&](){
							float factor = DecayCoefficient();
							int t_cacheIdx = m_bufferInUse;

							std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_cpuAsyncBuffer[t_cacheIdx], m_deltaArray, std::minus<ElemType>());
							std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));
							m_sharedArray->Add(m_deltaArray, m_totalModelSize);
							m_sharedArray->Get(m_cpuAsyncBuffer[t_cacheIdx], m_totalModelSize);

						});
#endif
					}
					else
					{
						float factor = DecayCoefficient();
						i = 0;
						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();

							ElemType * px = m_deltaArray + m_tableIndex[i];
							mat.CopyToArray(px, m_tableLength[i]);
						}

						std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_cpuAsyncBuffer[0], m_deltaArray, std::minus<ElemType>());

						// lr decay
						std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));

						m_sharedArray->Add(m_deltaArray, m_totalModelSize);
						m_sharedArray->Get(m_cpuAsyncBuffer[0], m_totalModelSize);

						i = 0;
						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();

							ElemType * px = m_cpuAsyncBuffer[0] + m_tableIndex[i];
							mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px);
						}
					}
				}

				void PushModel(const std::list<ComputationNodeBasePtr> & learnableNode)
				{

				}

				void PullModel(const std::list<ComputationNodeBasePtr> & learnableNode)
				{

				}

				void WaitAll()
				{
					multiverso::MV_Barrier();
				}

				void WaitAsyncBuffer()
				{
					if (m_prefetchThread != nullptr && m_prefetchThread->joinable())
					{
						m_prefetchThread->join();
						delete m_prefetchThread;
						m_prefetchThread = nullptr;
					}
				}
			private:
				void MultiversoInit(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					assert(!m_isInitialized);
					m_isInitialized = true;

                    //multiverso::Log::ResetLogLevel(multiverso::LogLevel::Debug);
					multiverso::MV_Init();

					//weights
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->Value();
						size_t layerSize = mat.GetNumElements();

						m_tableLength.push_back(layerSize);
					}

					m_tableCount = m_tableLength.size();

					// cacluate total of learnable node's size
					m_totalModelSize = accumulate(m_tableLength.begin(), m_tableLength.end(), 0);
 
					m_sharedArray = new multiverso::ArrayWorker<ElemType>(m_totalModelSize);
					m_serverArray = new multiverso::ArrayServer<ElemType>(m_totalModelSize);
					
					multiverso::MV_Barrier();

					size_t idx = 0;
					for (size_t len : m_tableLength)
					{
						m_tableIndex.push_back(idx);
						idx += len;
					}

#ifndef CPUONLY
					for (int i = 0; i < m_localCacheNumber; i++)
						m_gpuAsyncBuffer[i] = new Matrix<ElemType>*[m_tableCount];

					//create pinned memory
					for (int i = 0; i < m_localCacheNumber; ++i)
						CudaErrorCheck(cudaMallocHost((void **)&m_cpuAsyncBuffer[i], sizeof(ElemType) * (m_totalModelSize), cudaHostAllocPortable));

					CudaErrorCheck(cudaMallocHost((void **)&m_deltaArray, sizeof(ElemType) * (m_totalModelSize), cudaHostAllocPortable));
#else
					for (int i = 0; i < m_localCacheNumber; i++)
						m_cpuAsyncBuffer[i] = new ElemType[m_totalModelSize];
#endif
				}

				float DecayCoefficient()
				{
					float f = 1.f;
					switch (m_adjustLearningRateAtBeginningType)
					{
					case AdjustLearningRateatBeginning::None:
						break;
					case AdjustLearningRateatBeginning::Linearly:
						f = min(f, max(0.f, (float)(m_adjustCoefficient + (1 - m_adjustCoefficient) / m_adjustMBNumber * m_modelSyncCount)));
						break;
					case AdjustLearningRateatBeginning::Staircase:
						f = min(f, max(0.f, (float)(m_adjustCoefficient * (m_modelSyncCount / m_adjustMBNumber + 1))));
						break;
					default:
						break;
					}
					return f;
				}
				void TestNet() {
					multiverso::NetInterface* net = multiverso::NetInterface::Get();
					net->Init();

					char* hi1 = "hello, world";
					char* hi2 = "hello, c++";
					char* hi3 = "hello, multiverso";
					if (net->rank() == 0) {
						multiverso::MessagePtr msg(new multiverso::Message());// = std::make_unique<Message>();
						msg->set_src(0);
						msg->set_dst(1);
						msg->Push(multiverso::Blob(hi1, 13));
						msg->Push(multiverso::Blob(hi2, 11));
						msg->Push(multiverso::Blob(hi3, 18));
						net->Send(msg);
						multiverso::Log::Info("rank 0 send\n");
						multiverso::Log::Info("Hi = %s\n", msg->data()[0].data());

						msg.reset(new multiverso::Message());
						while (net->Recv(&msg) == 0) {
							multiverso::Log::Info("recv return 0\n");
						}
						multiverso::Log::Info("rank 0 recv\n");
						// CHECK(strcmp(msg->data()[0].data(), hi) == 0);
						std::vector<multiverso::Blob> recv_data = msg->data();
						//CHECK(recv_data.size() == 3);
						for (int i = 0; i < msg->size(); ++i) {
							multiverso::Log::Info("%s\n", recv_data[i].data());
						};
					}
					else if (net->rank() == 1) {
						multiverso::MessagePtr msg(new multiverso::Message());// = std::make_unique<Message>();
						while (net->Recv(&msg) == 0) {
							multiverso::Log::Info("recv return 0\n");
						}
						multiverso::Log::Info("rank 1 recv\n");
						// CHECK(strcmp(msg->data()[0].data(), hi) == 0);
						std::vector<multiverso::Blob> recv_data = msg->data();
						//CHECK(recv_data.size() == 3);
						for (int i = 0; i < msg->size(); ++i) {
							multiverso::Log::Info("%s\n", recv_data[i].data());
						}

						msg.reset(new multiverso::Message());
						msg->set_src(1);
						msg->set_dst(0);
						msg->Push(multiverso::Blob(hi1, 13));
						msg->Push(multiverso::Blob(hi2, 11));
						msg->Push(multiverso::Blob(hi3, 18));
						net->Send(msg);
						multiverso::Log::Info("rank 0 send\n");
						multiverso::Log::Info("Hi = %s\n", msg->data()[0].data());
					}

					net->Finalize();
				}
				multiverso::ArrayWorker<ElemType>* m_sharedArray;
				multiverso::ArrayServer<ElemType>* m_serverArray;
				thread * m_prefetchThread;
				bool m_isInitialized;

				int m_totalClientNumber;

				bool m_isUseAsyncBuffered;
				int m_localCacheNumber;
				int * m_cacheSwapIndex;
				int m_bufferInUse;

				size_t m_modelSyncCount;

				AdjustLearningRateatBeginning m_adjustLearningRateAtBeginningType;
				double m_adjustCoefficient;
				size_t m_adjustMBNumber;

				vector<size_t> m_tableLength;
				size_t m_totalModelSize;
				vector<size_t> m_tableIndex;
				ElemType * m_deltaArray;
				ElemType ** m_cpuAsyncBuffer;

				//GPU double buffer
				Matrix<ElemType> *** m_gpuAsyncBuffer;
				int m_tableCount;
#ifndef CPUONLY
				cudaStream_t _commStream;
#endif
			};
		}
	}
}
