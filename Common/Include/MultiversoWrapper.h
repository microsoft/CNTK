#pragma once

// This uses Multiverso.h which requires 
// the header files in ..\..\Multiverso\include
// and the lib files in ..\..\Multiverso\x64
#include "multiverso.h"
#pragma comment(lib, "Multiverso.lib")

#include <cuda_runtime.h>
#pragma comment (lib, "cudart.lib")     // for cudaMemcpyAsync()


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

#define CudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
			inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
			{
				if (code != cudaSuccess)
				{
					fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
					if (abort) exit(code);
				}
			}

			template<class ElemType = float>
			class MultiversoWrapper
			{
				typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
			public:
				//temp declare in public for barrier()
				//TODO: move to private
				multiverso::Adaptor * _adaptor;
				thread * _pThread;

				MultiversoWrapper(const std::list<ComputationNodeBasePtr> & learnableNodes, int localWorkerNumber, bool isPipeline = true)
				{
					_isInitialized = false;

					_nClients = localWorkerNumber;
					//Pipeline releated variables
					_isPipeline = isPipeline;
					_nLocalCache = _isPipeline ? 2 : 1;
					_pCacheState = new int[_nLocalCache];

					//CPU double buffer
					_pPCache = new ElemType*[_nLocalCache];

					//GPU double buffer
					_pPMatrixCache = new Matrix<ElemType>**[_nLocalCache];

					//Communication Stream
					CudaErrorCheck(cudaStreamCreate(&_commStream));

					_nCacheIdx = 0;
					for (int i = 0; i < _nLocalCache; i++)
						_pCacheState[i] = (i + 1) % _nLocalCache;

					_pThread = new thread();

					_pSizeEachServer = new size_t[_nClients];
					_pIdxEachServer = new size_t[_nClients];
					Init(learnableNodes, 1);
				}

				~MultiversoWrapper()
				{
					if (_isPipeline && _pThread->joinable())
						_pThread->join();

					delete _pCacheState, /*_pPCache, */_pDelta, _pSizeEachServer, _pIdxEachServer;

					for (size_t i = 0; i < _nLocalCache; i++)
					{
						CudaErrorCheck(cudaFreeHost(_pPCache[i]));
					}
					//CudaErrorCheck(cudaFreeHost(_pCache));
					delete _pPCache;
					CudaErrorCheck(cudaStreamDestroy(_commStream));
					//todo: delete _pMatri

					multiverso::FinishTrain();
				}

				//As for Multiverso, parameters store in server.
				void ModelInit(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					float factor = (float) 1.0 / _nClients;

					int table_id = 0;

					//weights
					int i = 0;
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->FunctionValues();

						for (int j = 0; j < _nLocalCache; j++)
							_pPMatrixCache[j][i] = new Matrix<ElemType>(mat);

						ElemType* px = _pPCache[0] + _vTableIdx[i];
						//ElemType* px = _pCache + _vTableIdx[i];
						mat.CopyToArray(px, _vTableLength[i]);
					}
					
					for (int i = 1; i < _nLocalCache; i++)
						memcpy(_pPCache[i], _pPCache[0], sizeof(ElemType) * _lTotalLength);

					//////////////////////////////////////////////////
					for (int row = 0; row < _nClients; ++row)
						_adaptor->Add(table_id, row, _pPCache[0] + _pIdxEachServer[row], factor);
					//_adaptor->Add(table_id, 0, _pPCache[0], factor);
					//////////////////////////////////////////////////
					_adaptor->Clock();
					_adaptor->BatchLoad(table_id);
				}

				//Todo: support auto adjust learning rate 
				void LearningrateSync(){ throw("not implement yet."); };

				//ASGD logic
				void ModelSync(const std::list<ComputationNodeBasePtr> & learnableNodes)
				{
					Timer timer;
					//float factor = (float) 1.0 / _nClients;
					int table_id = 0;
					if (_isPipeline && _pThread->joinable())
						_pThread->join();

					_nCacheIdx = _pCacheState[_nCacheIdx];

					int i = 0;
					if (_isPipeline)
					{

						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->FunctionValues();

							//CNTK model -> GPU buffer
							//*_pPMatrixCache[_nCacheIdx][i] = mat;
							CudaErrorCheck(cudaMemcpy(_pPMatrixCache[_nCacheIdx][i]->BufferPointer(),
								mat.BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));

							//GPU buffer -> CNTK model
							//mat = *_pPMatrixCache[_pCacheState[_nCacheIdx]][i];
							CudaErrorCheck(cudaMemcpy(mat.BufferPointer(),
								_pPMatrixCache[_pCacheState[_nCacheIdx]][i]->BufferPointer(),
								mat.GetNumElements() * sizeof(ElemType),
								cudaMemcpyDeviceToDevice));
						}

						_pThread = new thread([&](){
							//float t_factor = (float) 1.0 / _nClients;
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
							//transform(_pDelta, _pDelta + _lTotalLength, _pCache, _pDelta, std::minus<ElemType>());

							//Communication
							///////////////////////////////////////////////////////
							for (int row = 0; row < _nClients; row++)
								_adaptor->Add(table_id, row, _pDelta + _pIdxEachServer[row]/*, t_factor*/);

							_adaptor->Clock();

							_adaptor->BatchLoad(table_id);


							for (int row = 0; row < _nClients; row++)
							{
								ElemType *py = static_cast<ElemType*>(_adaptor->Get(table_id, row));
								memcpy(_pPCache[t_cacheIdx] + _pIdxEachServer[row], py, sizeof(ElemType) * _pSizeEachServer[row]);
								//memcpy(_pCache + _pIdxEachServer[row], py, sizeof(ElemType) * _pSizeEachServer[row]);
							}

							//_adaptor->Add(table_id, 0, _pDelta/*, t_factor*/);

							//_adaptor->Clock();

							//_adaptor->BatchLoad(table_id);

							//ElemType *py = static_cast<ElemType*>(_adaptor->Get(table_id, 0));
							//memcpy(_pPCache[t_cacheIdx], py, sizeof(ElemType) * _lTotalLength);
							///////////////////////////////////////////////////////


							//CPU buffer -> GPU buffer
							for (int widx = 0; widx < _nTableCnt; widx++)
							{
								ElemType * py = _pPCache[t_cacheIdx] + _vTableIdx[widx];
								//ElemType * py = _pCache + _vTableIdx[widx];

								CudaErrorCheck(cudaMemcpyAsync(_pPMatrixCache[t_cacheIdx][widx]->BufferPointer(),
									py,
									_pPMatrixCache[t_cacheIdx][widx]->GetNumElements() * sizeof(ElemType),
									cudaMemcpyHostToDevice,
									_commStream));
							}

							CudaErrorCheck(cudaStreamSynchronize(_commStream));

						});
					}
					else
					{
						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->FunctionValues();

							ElemType * px = _pDelta + _vTableIdx[i];
							mat.CopyToArray(px, _vTableLength[i]);
						}

						transform(_pDelta, _pDelta + _lTotalLength, _pPCache[0], _pDelta, std::minus<ElemType>());
						//transform(_pDelta, _pDelta + _lTotalLength, _pCache, _pDelta, std::minus<ElemType>());

						////////////////////////////////////////////////////////
						for (int row = 0; row < _nClients; row++)
							_adaptor->Add(table_id, row, _pDelta + _pIdxEachServer[row]/*, factor*/);

						_adaptor->Clock();
						//_adaptor->Barrier();
						_adaptor->BatchLoad(table_id);

						for (int row = 0; row < _nClients; row++)
						{
							ElemType *py = static_cast<ElemType*>(_adaptor->Get(table_id, row));
							memcpy(_pPCache[0] + _pIdxEachServer[row], py, sizeof(ElemType) * _pSizeEachServer[row]);
							//memcpy(_pCache + _pIdxEachServer[row], py, sizeof(ElemType) * _pSizeEachServer[row]);
						}

						//_adaptor->Add(table_id, 0, _pDelta/*, factor*/);
						//_adaptor->Clock();
						//_adaptor->BatchLoad(table_id);
						//ElemType *py = static_cast<ElemType*>(_adaptor->Get(table_id, 0));
						//memcpy(_pPCache[0], py, sizeof(ElemType) * _lTotalLength);
						////////////////////////////////////////////////////////

						i = 0;

						for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, i++)
						{
							ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
							Microsoft::MSR::CNTK::Matrix<ElemType> &mat = node->FunctionValues();

							ElemType * px = _pPCache[0] + _vTableIdx[i];
							//ElemType * px = _pCache + _vTableIdx[i];

							mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px);
						}
					}
				}

			private:
				void Init(const std::list<ComputationNodeBasePtr> & learnableNodes, int localWorkerNumber)
				{
					assert(!_isInitialized);
					_isInitialized = true;

					multiverso::SetCommType("p2p");
					multiverso::SetSyncType("async");

					int table_id = 0;
					//weights
					for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++)
					{
						ComputationNodePtr node = dynamic_pointer_cast<ComputationNode<ElemType>>(*nodeIter);
						Matrix<ElemType> &mat = node->FunctionValues();
						size_t layerSize = mat.GetNumElements();

						_vTableLength.push_back(layerSize);
					}

					_nTableCnt = _vTableLength.size();

					//init cache space.
					_lTotalLength = accumulate(_vTableLength.begin(), _vTableLength.end(), 0);

					size_t idx = 0;
					for (int i = 0; i < _nClients; i++)
					{
						_pIdxEachServer[i] = idx;
						_pSizeEachServer[i] = i < _lTotalLength % _nClients ? _lTotalLength / _nClients + 1 : _lTotalLength / _nClients;
						idx += _pSizeEachServer[i];
					}

					//////////////////////////////////////////////////////////////////
					multiverso::SetTable(table_id, _nClients, _lTotalLength / _nClients + 1, sizeof(ElemType) == 4 ? "float" : "double");
					//multiverso::SetTable(table_id, 1, _lTotalLength , sizeof(ElemType) == 4 ? "float" : "double");
					//////////////////////////////////////////////////////////////////
					idx = 0;
					for (size_t len : _vTableLength)
					{
						_vTableIdx.push_back(idx);
						idx += len;
					}

					//pinned memory
					//CudaErrorCheck(cudaMallocHost((void **)&_pCache, sizeof(ElemType) * _lTotalLength, cudaHostAllocPortable));
					for (int i = 0; i < _nLocalCache; ++i)
						CudaErrorCheck(cudaMallocHost((void **)&_pPCache[i], sizeof(ElemType) * (_lTotalLength + 1), cudaHostAllocPortable));

					CudaErrorCheck(cudaMallocHost((void **)&_pDelta, sizeof(ElemType) * (_lTotalLength + 1), cudaHostAllocPortable));

					//GPU memory cache
					for (int i = 0; i < _nLocalCache; i++)
						_pPMatrixCache[i] = new Matrix<ElemType>*[_nTableCnt];

					multiverso::Init(localWorkerNumber);

					printf("%s@rank %d/%d: Initialized multiverso.\n",
						getenv("COMPUTERNAME"), multiverso::GetMPIRank(), multiverso::GetMPISize());
					fflush(stdout);

					int adaptor_id = g_mpi->CurrentNodeRank();

					_adaptor = new multiverso::Adaptor(adaptor_id, 0);
					printf("%s@rank %d/%d: Initialized Adaptor.\n",
						getenv("COMPUTERNAME"), multiverso::GetMPIRank(), multiverso::GetMPISize());
					fflush(stdout);
				}


				bool _isInitialized;

				int _nClients;

				bool _isPipeline;
				int _nLocalCache;
				int * _pCacheState;
				int _nCacheIdx;

				vector<size_t> _vTableLength;
				size_t _lTotalLength;
				vector<size_t> _vTableIdx;
				ElemType * _pDelta;
				ElemType ** _pPCache;
				//ElemType *_pCache;

				size_t * _pSizeEachServer;
				size_t * _pIdxEachServer;

				//GPU double buffer
				Matrix<ElemType> *** _pPMatrixCache;
				int _nTableCnt;
				cudaStream_t _commStream;

			};
		}
	}
}