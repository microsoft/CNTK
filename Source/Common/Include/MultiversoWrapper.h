#pragma once

// This uses Multiverso.h which requires 
// the header files in ..\Multiverso\include
#include "multiverso.h"
#pragma comment(lib, "Multiverso.lib")

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
					_commCnt = 0;
					_adjustlearningrateatbeginningtype = adjusttype;
					_adjustcoefficient = adjustcoef;
					_adjustnbmb = adjustnbmb;

					m_multiversoAdaptor = false;

					m_totalClientNumber = localWorkerNumber;

					//Pipeline releated variables
					m_isPipelined = isPipeline;
					_nLocalCache = m_isPipelined ? 2 : 1;
					_pCacheState = new int[_nLocalCache];

					//CPU double buffer
					_pPCache = new ElemType*[_nLocalCache];

#ifndef CPUONLY
					//GPU double buffer
					_pPMatrixCache = new Matrix<ElemType>**[_nLocalCache];

					//Communication Stream
					CudaErrorCheck(cudaStreamCreate(&_commStream));
#endif

					_nCacheIdx = 0;
					for (int i = 0; i < _nLocalCache; i++)
						_pCacheState[i] = (i + 1) % _nLocalCache;

					m_prefetchThread = new thread();

					_pSizeEachServer = new size_t[m_totalClientNumber];
					_pIdxEachServer = new size_t[m_totalClientNumber];
					Init(learnableNodes, 1);
				}

				~MultiversoWrapper()
				{
					fprintf(stderr, "~MultiversoWrapper\n");
					fflush(stderr);

					if (m_isPipelined && m_prefetchThread != nullptr && m_prefetchThread->joinable())
						m_prefetchThread->join();

					delete _pCacheState, _pDelta, _pSizeEachServer, _pIdxEachServer;

					for (size_t i = 0; i < _nLocalCache; i++)
					{
#ifndef CPUONLY
						CudaErrorCheck(cudaFreeHost(_pPCache[i]));
#else
						delete _pPCache[i];
#endif
					}
					delete _pPCache;
#ifndef CPUONLY
					CudaErrorCheck(cudaStreamDestroy(_commStream));
#endif

					multiverso::FinishTrain();
					multiverso::Close(false);
				}

				//  This function will upload parameters into Multiverso
				void ModelInit(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					float factor = (float) 1.0 / m_totalClientNumber;

					int table_id = 0;

					//weights
					int i = 0;
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->Value();
#ifndef CPUONLY
						for (int j = 0; j < _nLocalCache; j++)
							_pPMatrixCache[j][i] = new Matrix<ElemType>(mat);
#endif

						ElemType* px = _pPCache[0] + _vTableIdx[i];
						mat.CopyToArray(px, _vTableLength[i]);
					}

					for (int i = 1; i < _nLocalCache; i++)
						memcpy(_pPCache[i], _pPCache[0], sizeof(ElemType) * _lTotalLength);

					memcpy(_pDelta, _pPCache[0], sizeof(ElemType) * _lTotalLength);

					for (int row = 0; row < m_totalClientNumber; ++row)
						m_multiversoAdaptor->Add(table_id, row, _pDelta + _pIdxEachServer[row], factor);
					m_multiversoAdaptor->Barrier(); //should clock
					m_multiversoAdaptor->BatchLoad(table_id, _pDelta, _pIdxEachServer, _pSizeEachServer);

					memcpy(_pDelta, _pPCache[0], sizeof(ElemType) * _lTotalLength);
				}

				//Todo: support auto adjust learning rate 
				void LearningrateSync(){ throw("not implement yet."); };

				//ASGD logic
				void ModelSync(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					//Note: maybe overflow.
					_commCnt++;

					Timer timer;
					int table_id = 0;
					if (m_isPipelined && m_prefetchThread->joinable())
						m_prefetchThread->join();

					_nCacheIdx = _pCacheState[_nCacheIdx];

					int i = 0;
					if (m_isPipelined)
					{

						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();
#ifndef CPUONLY
							//CNTK model -> GPU buffer
							CudaErrorCheck(cudaMemcpy(_pPMatrixCache[_nCacheIdx][i]->BufferPointer(),
								mat.BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));

							//GPU buffer -> CNTK model
							CudaErrorCheck(cudaMemcpy(mat.BufferPointer(),
								_pPMatrixCache[_pCacheState[_nCacheIdx]][i]->BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));
#else
							ElemType * px = _pPCache[_nCacheIdx] + _vTableIdx[i];

							mat.CopyToArray(px, _vTableLength[i]);

							ElemType * py = _pPCache[_pCacheState[_nCacheIdx]] + _vTableIdx[i];

							mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), py);


							delete px;
#endif
						}
#ifndef CPUONLY
						m_prefetchThread = new thread([&](){
							float factor = getUpdateCoefficient();
							int table_id = 0, t_cacheIdx = _nCacheIdx;
							int deviceId = _pPMatrixCache[t_cacheIdx][0]->GetDeviceId();

							CudaErrorCheck(cudaSetDevice(deviceId));

							for (int widx = 0; widx < _nTableCnt; widx++)
							{
								ElemType * px = _pDelta + _vTableIdx[widx];
								//GPU buffer -> CPU buffer
								CudaErrorCheck(cudaMemcpyAsync(px,
									_pPMatrixCache[t_cacheIdx][widx]->BufferPointer(),
									_pPMatrixCache[t_cacheIdx][widx]->GetNumElements() * sizeof(ElemType),
									cudaMemcpyDeviceToHost,
									_commStream));
							}

							//Sync for copy
							CudaErrorCheck(cudaStreamSynchronize(_commStream));

							//Calculate delta
							transform(_pDelta, _pDelta + _lTotalLength, _pPCache[t_cacheIdx], _pDelta, std::minus<ElemType>());

							//////Communication
							for (int row = 0; row < m_totalClientNumber; row++)
								m_multiversoAdaptor->Add(table_id, row, _pDelta + _pIdxEachServer[row], factor);
							m_multiversoAdaptor->BatchLoad(table_id, _pPCache[t_cacheIdx], _pIdxEachServer, _pSizeEachServer);

							//CPU buffer -> GPU buffer
							for (int widx = 0; widx < _nTableCnt; widx++)
							{
								ElemType * py = _pPCache[t_cacheIdx] + _vTableIdx[widx];

								CudaErrorCheck(cudaMemcpyAsync(_pPMatrixCache[t_cacheIdx][widx]->BufferPointer(),
									py,
									_pPMatrixCache[t_cacheIdx][widx]->GetNumElements() * sizeof(ElemType),
									cudaMemcpyHostToDevice,
									_commStream));
							}

							CudaErrorCheck(cudaStreamSynchronize(_commStream));

						});
#else
						m_prefetchThread = new thread([&](){
							float factor = getUpdateCoefficient();
							int table_id = 0, t_cacheIdx = _nCacheIdx;

							transform(_pDelta, _pDelta + _lTotalLength, _pPCache[t_cacheIdx], _pDelta, std::minus<ElemType>());
							for (int row = 0; row < g_mpi->NumNodesInUse(); row++)
								m_multiversoAdaptor->Add(table_id, row, _pDelta + _pIdxEachServer[row], factor);


							m_multiversoAdaptor->BatchLoad(table_id, _pPCache[t_cacheIdx], _pIdxEachServer, _pSizeEachServer);

						});
#endif
					}
					else
					{
						float factor = getUpdateCoefficient();
						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();

							ElemType * px = _pDelta + _vTableIdx[i];
							mat.CopyToArray(px, _vTableLength[i]);
						}

						transform(_pDelta, _pDelta + _lTotalLength, _pPCache[0], _pDelta, std::minus<ElemType>());

						for (int row = 0; row < m_totalClientNumber; row++)
							m_multiversoAdaptor->Add(table_id, row, _pDelta + _pIdxEachServer[row], factor);

						m_multiversoAdaptor->BatchLoad(table_id, _pPCache[0], _pIdxEachServer, _pSizeEachServer);

						i = 0;

						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->Value();

							ElemType * px = _pPCache[0] + _vTableIdx[i];

							mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px);
						}
					}
				}

			private:
				void Init(const std::list<ComputationNodeBasePtr> & learnableNodes, int localWorkerNumber)
				{
					assert(!m_isInitialized);
					m_isInitialized = true;

					multiverso::SetCommType("p2p");
					multiverso::SetSyncType("async");
					multiverso::SetLog(true);

					int table_id = 0;
					//weights
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->Value();
						size_t layerSize = mat.GetNumElements();

						_vTableLength.push_back(layerSize);
					}

					_nTableCnt = _vTableLength.size();

					//init cache space.
					_lTotalLength = accumulate(_vTableLength.begin(), _vTableLength.end(), 0);
					size_t idx = 0;
					for (int i = 0; i < m_totalClientNumber; i++)
					{
						_pIdxEachServer[i] = idx;
						_pSizeEachServer[i] = i < _lTotalLength % m_totalClientNumber ? _lTotalLength / m_totalClientNumber + 1 : _lTotalLength / m_totalClientNumber;
						idx += _pSizeEachServer[i];
					}
					multiverso::SetTable(table_id, m_totalClientNumber, ((size_t)(_lTotalLength / m_totalClientNumber)) + 1, sizeof(ElemType) == 4 ? "float" : "double");
					idx = 0;
					for (size_t len : _vTableLength)
					{
						_vTableIdx.push_back(idx);
						idx += len;
					}

#ifndef CPUONLY
					//pinned memory
					for (int i = 0; i < _nLocalCache; ++i)
						CudaErrorCheck(cudaMallocHost((void **)&_pPCache[i], sizeof(ElemType) * (_lTotalLength + 1), cudaHostAllocPortable));

					CudaErrorCheck(cudaMallocHost((void **)&_pDelta, sizeof(ElemType) * (_lTotalLength + 1), cudaHostAllocPortable));

					//GPU memory cache
					for (int i = 0; i < _nLocalCache; i++)
						_pPMatrixCache[i] = new Matrix<ElemType>*[_nTableCnt];
#else
					for (int i = 0; i < _nLocalCache; i++)
						_pPCache[i] = new ElemType[_lTotalLength + 1];
#endif

					multiverso::Init(localWorkerNumber);

					printf("%s@rank %d/%d: Initialized multiverso.\n",
						getenv("COMPUTERNAME"), multiverso::GetMPIRank(), multiverso::GetMPISize());
					fflush(stdout);

					int adaptor_id = g_mpi->CurrentNodeRank();

					m_multiversoAdaptor = new multiverso::Adaptor(adaptor_id, 0);
					printf("%s@rank %d/%d: Initialized Adaptor.\n",
						getenv("COMPUTERNAME"), multiverso::GetMPIRank(), multiverso::GetMPISize());
					fflush(stdout);
				}

				float getUpdateCoefficient()
				{
					float f = 1.f;
					switch (_adjustlearningrateatbeginningtype)
					{
					case AdjustLearningRateatBeginning::None:
						break;
					case AdjustLearningRateatBeginning::Linearly:
						f = min(f, max(0.f, (float)(_adjustcoefficient + (1 - _adjustcoefficient) / _adjustnbmb * _commCnt)));
						break;
					case AdjustLearningRateatBeginning::Staircase:
						f = min(f, max(0.f, (float)(_adjustcoefficient * (_commCnt / _adjustnbmb + 1))));
						break;
					default:
						break;
					}
					return f;
				}

				multiverso::Adaptor * m_multiversoAdaptor;
				thread * m_prefetchThread;
				bool m_isInitialized;

				int m_totalClientNumber;

				bool m_isPipelined;
				int _nLocalCache;
				int * _pCacheState;
				int _nCacheIdx;

				size_t _commCnt;

				AdjustLearningRateatBeginning _adjustlearningrateatbeginningtype;
				double _adjustcoefficient;
				size_t _adjustnbmb;

				vector<size_t> _vTableLength;
				size_t _lTotalLength;
				vector<size_t> _vTableIdx;
				ElemType * _pDelta;
				ElemType ** _pPCache;

				size_t * _pSizeEachServer;
				size_t * _pIdxEachServer;

				//GPU double buffer
				Matrix<ElemType> *** _pPMatrixCache;
				int _nTableCnt;
#ifndef CPUONLY
				cudaStream_t _commStream;
#endif
			};
		}
	}
}