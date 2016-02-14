#pragma once

// This uses Multiverso.h which requires 
// the header files in ..\Multiverso\include
#include <multiverso/multiverso.h>
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
					int localWorkerNumber,
					bool isPipeline = true,
					AdjustLearningRateatBeginning adjusttype = AdjustLearningRateatBeginning::None,
					double adjustcoef = 0.2,
					size_t adjustnbmb = 600)
				{
					m_modelSyncCount = 0;
					m_adjustLearningRateAtBeginningType = adjusttype;
					m_adjustCoefficient = adjustcoef;
					m_adjustMBNumber = adjustnbmb;

					//m_multiversoAdaptor = false;

					m_totalClientNumber = localWorkerNumber;

					//Pipeline releated variables
					m_isPipelined = isPipeline;
					m_localCacheNumber = m_isPipelined ? 2 : 1;
					m_cacheSwapIndex = new int[m_localCacheNumber];

					//CPU double buffer
					m_cpuAsyncBuffer = new ElemType*[m_localCacheNumber];

#ifndef CPUONLY
					//GPU double buffer
					m_gpuAsyncBuffer = new Matrix<ElemType>**[m_localCacheNumber];

					//Communication Stream
					CudaErrorCheck(cudaStreamCreate(&_commStream));
#endif

					m_cacheIndex = 0;
					for (int i = 0; i < m_localCacheNumber; i++)
						m_cacheSwapIndex[i] = (i + 1) % m_localCacheNumber;

					m_prefetchThread = new thread();

					m_modelSizeOfEachServer = new size_t[m_totalClientNumber];
					m_indexOfEachServer = new size_t[m_totalClientNumber];
					MultiversoInit(learnableNodes, 1);
				}

				~MultiversoWrapper()
				{
					fprintf(stderr, "~MultiversoWrapper\n");
					fflush(stderr);

					if (m_isPipelined && m_prefetchThread != nullptr && m_prefetchThread->joinable())
						m_prefetchThread->join();

					delete m_cacheSwapIndex, m_deltaArray, m_modelSizeOfEachServer, m_indexOfEachServer;

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
					multiverso::MultiversoShutDown(false);
					//multiverso::FinishTrain();
					//multiverso::Close(false);
				}

				//  This function will upload parameters into Multiverso
				void InitModel(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					float factor = (float) 1.0 / m_totalClientNumber;

					//weights
					int i = 0;
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
					m_sharedArray->Get();
					//for (int row = 0; row < m_totalClientNumber; ++row)
					//	m_multiversoAdaptor->Add(table_id, row, m_deltaArray + m_indexOfEachServer[row], factor);
					//m_multiversoAdaptor->Barrier(); //should clock
					//m_multiversoAdaptor->BatchLoad(table_id, m_deltaArray, m_indexOfEachServer, m_modelSizeOfEachServer);

					memcpy(m_deltaArray, m_sharedArray->raw().data(), sizeof(ElemType) * m_totalModelSize);
				}

				//Todo: support auto adjust learning rate 
				void LearningrateSync(){ throw("not implement yet."); };

				//ASGD logic
				void PushAndPullModel(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					//Note: maybe overflow.
					m_modelSyncCount++;

					Timer timer;
					//if (m_isPipelined && m_prefetchThread->joinable())
					//	m_prefetchThread->join();
					WaitAsyncBuffer();

					m_cacheIndex = m_cacheSwapIndex[m_cacheIndex];

					int i = 0;
					if (m_isPipelined)
					{

						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();
#ifndef CPUONLY
							//CNTK model -> GPU buffer
							CudaErrorCheck(cudaMemcpy(m_gpuAsyncBuffer[m_cacheIndex][i]->BufferPointer(),
								mat.BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));

							//GPU buffer -> CNTK model
							CudaErrorCheck(cudaMemcpy(mat.BufferPointer(),
								m_gpuAsyncBuffer[m_cacheSwapIndex[m_cacheIndex]][i]->BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));
#else
							ElemType * px = m_cpuAsyncBuffer[m_cacheIndex] + m_tableIndex[i];

							mat.CopyToArray(px, m_tableLength[i]);

							ElemType * py = m_cpuAsyncBuffer[m_cacheSwapIndex[m_cacheIndex]] + m_tableIndex[i];

							mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), py);


							delete px;
#endif
						}
#ifndef CPUONLY
						m_prefetchThread = new thread([&](){
							float factor = DecayCoefficient();
							int t_cacheIdx = m_cacheIndex;
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

							//Sync for copy
							CudaErrorCheck(cudaStreamSynchronize(_commStream));

							//Calculate delta
							std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_cpuAsyncBuffer[t_cacheIdx], m_deltaArray, std::minus<ElemType>());

							// lr decay
							std::transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_deltaArray, std::bind1st(std::multiplies<ElemType>(), factor));

							m_sharedArray->Add(m_deltaArray, m_totalModelSize);
							m_sharedArray->Get();
							memcpy(m_cpuAsyncBuffer[t_cacheIdx], m_sharedArray->raw().data(), m_totalModelSize);
							//////Communication
							//for (int row = 0; row < m_totalClientNumber; row++)
							//	m_multiversoAdaptor->Add(table_id, row, m_deltaArray + m_indexOfEachServer[row], factor);
							//m_multiversoAdaptor->BatchLoad(table_id, m_cpuAsyncBuffer[t_cacheIdx], m_indexOfEachServer, m_modelSizeOfEachServer);

							//CPU buffer -> GPU buffer
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
							float factor = getUpdateCoefficient();
							int table_id = 0, t_cacheIdx = m_cacheIndex;

							transform(m_deltaArray, m_deltaArray + m_totalModelSize, m_cpuAsyncBuffer[t_cacheIdx], m_deltaArray, std::minus<ElemType>());
							for (int row = 0; row < g_mpi->NumNodesInUse(); row++)
								m_multiversoAdaptor->Add(table_id, row, m_deltaArray + m_indexOfEachServer[row], factor);


							m_multiversoAdaptor->BatchLoad(table_id, m_cpuAsyncBuffer[t_cacheIdx], m_indexOfEachServer, m_modelSizeOfEachServer);

						});
#endif
					}
					else
					{
						float factor = DecayCoefficient();
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
						m_sharedArray->Get();
						memcpy(m_cpuAsyncBuffer[0], m_sharedArray->raw().data(), m_totalModelSize);

						//for (int row = 0; row < m_totalClientNumber; row++)
						//	m_multiversoAdaptor->Add(table_id, row, m_deltaArray + m_indexOfEachServer[row], factor);

						//m_multiversoAdaptor->BatchLoad(table_id, m_cpuAsyncBuffer[0], m_indexOfEachServer, m_modelSizeOfEachServer);

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
					multiverso::MultiversoBarrier();
				}

				void WaitAsyncBuffer()
				{
					if (m_prefetchThread != nullptr && m_prefetchThread->joinable())
						m_prefetchThread->join();
				}
			private:
				void MultiversoInit(const std::list<ComputationNodeBasePtr> & learnableNodes, int localWorkerNumber)
				{
					assert(!m_isInitialized);
					m_isInitialized = true;

					multiverso::MultiversoInit();

					//weights
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->Value();
						size_t layerSize = mat.GetNumElements();

						m_tableLength.push_back(layerSize);
					}

					m_tableCount = m_tableLength.size();

					//init cache space.
					m_totalModelSize = accumulate(m_tableLength.begin(), m_tableLength.end(), 0);
					size_t idx = 0;
					for (int i = 0; i < m_totalClientNumber; i++)
					{
						m_indexOfEachServer[i] = idx;
						m_modelSizeOfEachServer[i] = i < m_totalModelSize % m_totalClientNumber ? m_totalModelSize / m_totalClientNumber + 1 : m_totalModelSize / m_totalClientNumber;
						idx += m_modelSizeOfEachServer[i];
					}
					m_sharedArray = new multiverso::ArrayWorker<ElemType>(m_totalModelSize);
					m_serverArray = new multiverso::ArrayServer<ElemType>(m_totalModelSize);
					
					multiverso::MultiversoBarrier();
					//multiverso::SetTable(table_id, m_totalClientNumber, ((size_t)(m_totalModelSize / m_totalClientNumber)) + 1, sizeof(ElemType) == 4 ? "float" : "double");
					idx = 0;
					for (size_t len : m_tableLength)
					{
						m_tableIndex.push_back(idx);
						idx += len;
					}

#ifndef CPUONLY
					//pinned memory
					for (int i = 0; i < m_localCacheNumber; ++i)
						CudaErrorCheck(cudaMallocHost((void **)&m_cpuAsyncBuffer[i], sizeof(ElemType) * (m_totalModelSize + 1), cudaHostAllocPortable));

					CudaErrorCheck(cudaMallocHost((void **)&m_deltaArray, sizeof(ElemType) * (m_totalModelSize + 1), cudaHostAllocPortable));

					//GPU memory cache
					for (int i = 0; i < m_localCacheNumber; i++)
						m_gpuAsyncBuffer[i] = new Matrix<ElemType>*[m_tableCount];
#else
					for (int i = 0; i < m_localCacheNumber; i++)
						m_cpuAsyncBuffer[i] = new ElemType[m_totalModelSize + 1];
#endif

					//multiverso::Init(localWorkerNumber);

					//printf("%s@rank %d/%d: Initialized multiverso.\n",
					//	getenv("COMPUTERNAME"), multiverso::GetMPIRank(), multiverso::GetMPISize());
					//fflush(stdout);

					//int adaptor_id = g_mpi->CurrentNodeRank();

					////m_multiversoAdaptor = new multiverso::Adaptor(adaptor_id, 0);
					//printf("%s@rank %d/%d: Initialized Adaptor.\n",
					//	getenv("COMPUTERNAME"), multiverso::GetMPIRank(), multiverso::GetMPISize());
					fflush(stdout);
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
				multiverso::ArrayWorker<ElemType>* m_sharedArray;
				multiverso::ArrayServer<ElemType>* m_serverArray;
				//multiverso::Adaptor * m_multiversoAdaptor;
				thread * m_prefetchThread;
				bool m_isInitialized;

				int m_totalClientNumber;

				bool m_isPipelined;
				int m_localCacheNumber;
				int * m_cacheSwapIndex;
				int m_cacheIndex;

				size_t m_modelSyncCount;

				AdjustLearningRateatBeginning m_adjustLearningRateAtBeginningType;
				double m_adjustCoefficient;
				size_t m_adjustMBNumber;

				vector<size_t> m_tableLength;
				size_t m_totalModelSize;
				vector<size_t> m_tableIndex;
				ElemType * m_deltaArray;
				ElemType ** m_cpuAsyncBuffer;

				size_t * m_modelSizeOfEachServer;
				size_t * m_indexOfEachServer;

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