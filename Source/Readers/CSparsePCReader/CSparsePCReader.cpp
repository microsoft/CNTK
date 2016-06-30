//
// <copyright file="CSparsePCReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// CSparsePCReader.cpp : Defines the Sparse Parallel Corpus reader.
//

#include "stdafx.h"
#include <cstdint>
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "CSparsePCReader.h"
#include "zlib.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			DWORD HIDWORD(size_t size) { return size >> 32; }
			DWORD LODWORD(size_t size) { return size & 0xFFFFFFFF; }




			template<class ElemType>
			void SparseDenseMemory<ElemType>::Clear()
			{
				this->m_denseIndex = 0;
				this->m_sparseValueIndex = 0;
				this->m_sparseRowIndex = 0;
				this->m_sparseColIndex = 0;
				this->m_sampleNumber = 0;
				this->m_nnZ = 0;

				if (this->m_bIsSparse)
				{
					this->m_colIndices[0] = 0;
				}

			}

			template<class ElemType>
			void SparseDenseMemory<ElemType>::AddDenseData(void* pSource, size_t dim)
			{
				this->m_sampleNumber++;
				size_t increase = dim*sizeof(ElemType);

				memcpy((char*)this->m_values + this->m_denseIndex, pSource, increase);
				this->m_denseIndex += increase;

			}

			template<class ElemType>
			void SparseDenseMemory<ElemType>::AddSparseData(void* nnzValue, void* nnzIndex, int32_t length)
			{
				size_t increaseOfnnzValue = length* sizeof(ElemType);
				size_t increaseofnnzIndex = length*sizeof(int32_t);

				if (this->m_sampleNumber == 0)
				{
					this->m_colIndices[0] = 0;
				}

				this->m_sampleNumber++;
				memcpy((char*)this->m_values + this->m_sparseValueIndex*sizeof(ElemType), nnzValue, increaseOfnnzValue);
				this->m_sparseValueIndex += length;

				this->m_nnZ += length;

				memcpy((char*)this->m_rowIndices + this->m_sparseRowIndex*sizeof(int32_t), nnzIndex, increaseofnnzIndex);

				this->m_sparseRowIndex += length;

				this->m_colIndices[m_sampleNumber] = m_colIndices[m_sampleNumber - 1] + length;
				this->m_sparseColIndex++;


			}


			template<class ElemType>
			void SparseDenseMemory<ElemType>::FillDenseMatrix(Matrix<ElemType>* matrix)
			{

				matrix->Resize(m_Dim, this->m_sampleNumber);
				matrix->SetValue((ElemType)0);

				matrix->SetValue(this->m_Dim, this->m_sampleNumber, matrix->GetDeviceId(), this->m_values, matrixFlagNormal);
			}
			template<class ElemType>
			void SparseDenseMemory<ElemType>::FillSparseMatrix(Matrix<ElemType>* matrix)
			{
				if (matrix->GetFormat() != MatrixFormat::matrixFormatSparseCSC)
					matrix->SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);

				matrix->SetMatrixFromCSCFormat(m_colIndices, m_rowIndices, m_values, m_nnZ, m_Dim, m_sampleNumber);
			}




			template<class ElemType>
			CSparsePCReader<ElemType>::~CSparsePCReader()
			{

			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::Destroy()
			{
				delete this;
			}

			// Init - Reader Initialize for multiple data sets
			// config - [in] configuration parameters for the datareader
			template<class ElemType>
			template<class ConfigRecordType>
			void CSparsePCReader<ElemType>::InitFromConfig(const ConfigRecordType & readerConfig)
			{
				// Sparse PC reader considers every consecutive N rows to be part of a single block.
				// This is used later to compute the corss-entropy with softmax per block.
				// Default value is 1 to indicate all rows are independent.
				m_microBatchSize = readerConfig(L"microbatchSize", (size_t)1);

				m_miniBatchSize = 0;
				m_traceLevel = readerConfig(L"traceLevel", 0);
				m_maxReadData = readerConfig(L"maxReadData", (size_t)0);
				m_doGradientCheck = readerConfig(L"gradientCheck", false);
				m_returnDense = readerConfig(L"returnDense", false);
				m_sparsenessFactor = readerConfig(L"sparsenessFactor", (size_t)50); // We don't expect more than one in 50 input positions to have non-zero values
				m_verificationCode = (int32_t)readerConfig(L"verificationCode", (size_t)0);
				m_reshapeInputToRowSize = readerConfig(L"reshapeInputToRowSize", (size_t)0);

				std::vector<std::wstring> featureNames;
				std::vector<std::wstring> labelNames;

				m_file = (const wstring &)readerConfig(L"file");

				m_mapReaderOrder2FeatureName = std::map<int32_t, std::wstring>();



				// Determine the names of the features and lables sections in the config file.
				// features - [in,out] a vector of feature name strings
				// labels - [in,out] a vector of label name strings
				for (const auto & id : readerConfig.GetMemberIds())
				{
					if (!readerConfig.CanBeConfigRecord(id))
						continue;
					const ConfigRecordType & temp = readerConfig(id);
					// see if we have a config parameters that contains a "dim" element, it's a sub key, use it
					if (temp.ExistsCurrent(L"dim"))
					{
						bool bIsSparse = temp(L"isSparse", true);
						int32_t featureDim = temp(L"dim", 0);
						int32_t featureSparseFactor = temp(L"sparsenessFactor", 0);
						int32_t readerOrder = temp(L"readerOrder", 0);

						if (featureSparseFactor == 0)
						{
							featureSparseFactor = (int32_t)m_sparsenessFactor;
						}

						SparseDenseFeatureInfo* featureInfo = new SparseDenseFeatureInfo();
						featureInfo->m_bIsSparse = bIsSparse;
						featureInfo->m_Dim = featureDim;
						featureInfo->m_SparseFactor = featureSparseFactor;
						wstring wname = id;
						m_mapSparseDenseInfo[wname] = *featureInfo;
						m_mapReaderOrder2FeatureName[readerOrder] = wname;

					}
				}

				this->m_bSparseDenseInfoInitialized = false;


				m_inFile.open(m_file, ifstream::binary | ifstream::in);

				m_inFile.seekg(0, ios::end);
				this->m_filePositionMax = (int64_t)m_inFile.tellg();

				if (m_filePositionMax < 0)
				{
					RuntimeError("Your Data file Does not exists, Check your Path");
				}

				//Compress
				auto start = clock();
				printf("get zipped file info start\n");
				GetZippedFileInfo();
				auto sec = (clock() - start) / CLOCKS_PER_SEC;
				printf("get zipped file info finished: %ds\n", sec);
				

				m_dThreadCnt = readerConfig(L"dThreadCnt", (int32_t)3);
				m_processedBlockCntPerThread = (size_t *)malloc(sizeof(size_t) * m_dThreadCnt);
				for (int i = 0; i < m_dThreadCnt; i++)
					m_processedBlockCntPerThread[i] = 0;

				m_numBatches = m_numRows / m_microBatchSize;
				if (m_numRows% m_microBatchSize)
					m_numBatches += 1;
				//Compress END
			}

			//Compress
			template<class ElemType>
			void CSparsePCReader<ElemType>::GetZippedFileInfo()
			{
				m_inFile.seekg(0, ios::end);
				m_fileSize = (size_t)m_inFile.tellg();
				if (m_fileSize <= 0)
					RuntimeError("Input Data file Does Not Exists Or Empty");

				int32_t block_size = 0;
				int32_t block_sample_cnt = 0;

				m_inFile.seekg(0, ios::beg);
				m_inFile.read((char *)&block_size, 4);
				m_inFile.read((char *)&block_sample_cnt, 4);
				m_blockSizeInByte.reserve((size_t)((m_fileSize / block_size) * 1.1));
				m_sampleCntInBlock.reserve(m_blockSizeInByte.capacity());
				m_blockOffset.reserve(m_blockSizeInByte.capacity());

				m_inFile.seekg(0, ios::beg);
				size_t pos = 0;
				m_numRows = 0;
				m_numBlocks = 0;
				m_blockSize = 0;
				m_blockSampleCnt = 0;
				int progress = -1;
				while (pos < m_fileSize)
				{
					int curProgress = pos * 100 / m_fileSize;
					if (curProgress != progress) {
						progress = curProgress;
						printf("progress %d%%\n", progress);
					}


					m_inFile.read((char *)&block_size, 4);
					m_inFile.read((char *)&block_sample_cnt, 4);

					m_blockSizeInByte.push_back(block_size + 4 + 4);
					m_sampleCntInBlock.push_back(block_sample_cnt);
					m_blockOffset.push_back(pos);

					m_numRows += block_sample_cnt;
					m_numBlocks++;

					pos += block_size + 4 + 4;
					m_inFile.seekg(pos, ios::beg);

					m_blockSize = max(m_blockSize, block_size + 8);
					m_blockSampleCnt = max(m_blockSampleCnt, block_sample_cnt);
				}
			}

			//StartMinibatchLoop - Startup a minibatch loop 
			// mbSize - [in] size of the minibatch (number of Samples, etc.)
			// epoch - [in] epoch number for this loop --ignored
			// requestedEpochSamples - [in] number of samples to randomize --ignored
			template<class ElemType>
			void CSparsePCReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t /*epoch*/, size_t /*requestedEpochSamples*/)
			{
				//Set file offset to zero
				this->m_currOffset = 0;

				this->m_inFile.seekg(ios_base::beg);
				if (m_miniBatchSize != mbSize)
				{
					m_miniBatchSize = mbSize;
				}

				if (!this->m_bSparseDenseInfoInitialized)
				{

					size_t maxMBSize = 0;
					size_t maxMem = 1 * 1024 * 1024 * 1024; // 1GB

					for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
					{
						auto featureName = m_mapReaderOrder2FeatureName[i];

						SparseDenseFeatureInfo featureInfo = this->m_mapSparseDenseInfo[featureName];

						SparseDenseMemory<ElemType>* pieceMemory = new SparseDenseMemory<ElemType>();

						pieceMemory->m_bIsSparse = featureInfo.m_bIsSparse;
						pieceMemory->m_Dim = featureInfo.m_Dim;


						if (featureInfo.m_bIsSparse)
						{
							size_t valuesSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor;
							pieceMemory->m_values = (ElemType*)malloc(valuesSize);

							size_t rowIdSize = sizeof(int32_t)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor;

							pieceMemory->m_rowIndices = (int32_t*)malloc(sizeof(int32_t)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor);
							size_t columnIdSize = sizeof(int32_t)* (m_miniBatchSize + 1);
							pieceMemory->m_colIndices = (int32_t*)malloc(sizeof(int32_t)* (m_miniBatchSize + 1));

							maxMBSize += valuesSize;
							maxMBSize += rowIdSize;
							maxMBSize += columnIdSize;
						}
						else
						{
							size_t denseValueSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize;
							pieceMemory->m_values = (ElemType*)malloc(denseValueSize);
							maxMBSize += denseValueSize;
						}

						this->m_mapWorkingMemory[featureName] = pieceMemory;
					}

					size_t maxPointers = maxMem / maxMBSize;
					m_SparceDenseFeatureSize = maxMBSize;

					for (int i = 0; i < maxPointers; i++)
					{
						std::map<wstring, SparseDenseMemory<ElemType>*> perMap;

						for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
						{
							auto featureName = m_mapReaderOrder2FeatureName[i];

							SparseDenseFeatureInfo featureInfo = this->m_mapSparseDenseInfo[featureName];

							SparseDenseMemory<ElemType>* pieceMemory = new SparseDenseMemory<ElemType>();

							pieceMemory->m_bIsSparse = featureInfo.m_bIsSparse;
							pieceMemory->m_Dim = featureInfo.m_Dim;


							if (featureInfo.m_bIsSparse)
							{
								size_t valuesSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor;
								pieceMemory->m_values = (ElemType*)malloc(valuesSize);
								pieceMemory->m_rowIndices = (int32_t*)malloc(sizeof(int32_t)* featureInfo.m_Dim * m_miniBatchSize / featureInfo.m_SparseFactor);
								pieceMemory->m_colIndices = (int32_t*)malloc(sizeof(int32_t)* (m_miniBatchSize + 1));

							}
							else
							{
								size_t denseValueSize = sizeof(ElemType)* featureInfo.m_Dim * m_miniBatchSize;
								pieceMemory->m_values = (ElemType*)malloc(denseValueSize);
							}

							perMap[featureName] = pieceMemory;


						}
						this->m_dataToProduce.push(perMap);

					}

					//Compress
					size_t unzipQueueLen = 2 * maxMem / (2 * maxMBSize * m_blockSampleCnt / m_miniBatchSize);
					if (unzipQueueLen == 0) unzipQueueLen = 1;
					if (unzipQueueLen == 0) {
						fprintf(stderr, "exceeds maxMem\n");
						exit(-1);
					}

					for (int i = 0; i < unzipQueueLen; i++)
					{
						void* unzipDataBuffer = malloc(2 * maxMBSize * m_blockSampleCnt / m_miniBatchSize);
						m_unzipedDataToProduce.push(unzipDataBuffer);
					}
					size_t zipQueueLen = maxMem / m_blockSize;
					for (size_t c = 0; c < zipQueueLen; c++)
					{
						void* zipDataBuffer = malloc(m_blockSize);
						m_zipedDataToProduce.push(zipDataBuffer);
					}

					m_unzippedBuffer = malloc(2 * maxMBSize*m_blockSampleCnt / m_miniBatchSize);

					m_nextMB = 0;
					m_mbSize = m_miniBatchSize;

					size_t subsetNum = 0; //the index of this subset
					size_t numSubsets = 1; // the total count of subset

					size_t blockCnt = m_blockOffset.size() / numSubsets;
					size_t startBlock = blockCnt * subsetNum;
					size_t endBlock = blockCnt * (subsetNum + 1);
					size_t remainder = m_numBatches % numSubsets;

					size_t lb = min(remainder, subsetNum);
					size_t ub = min(remainder, subsetNum + 1);

					startBlock += lb;
					endBlock += ub;

					size_t sampleCnt = 0;
					for (size_t i = startBlock; i < endBlock; i++)
						sampleCnt += m_sampleCntInBlock[i];

					m_epochSize = sampleCnt / m_mbSize;
					if (sampleCnt % m_mbSize != 0)
						m_epochSize++;


					m_windowSize = endBlock - startBlock;

					if (m_windowSize != m_readOrderLength) {
						FillReadOrder(m_windowSize);
						m_readOrderLength = m_windowSize;
					}
					//Compress END

					m_bSparseDenseInfoInitialized = true;
				}

				// reset the next read sample
				/*ReadZipData(m_readOrder, m_readOrderLength);
				UnzipData(1, m_readOrderLength);
				ReadMinibatches(m_readOrder, m_readOrderLength);*/

				std::thread readZipData([this] { this->ReadZipData(m_readOrder, m_readOrderLength); });
				readZipData.detach();
				
				for (m_dIndex = 0; m_dIndex < m_dThreadCnt; m_dIndex++)
				{
					m_unzipThreads[m_dIndex] = std::thread([this] { this->UnzipData(m_dIndex, m_readOrderLength); });
					m_unzipThreads[m_dIndex].detach();
				}
				
				std::thread processData([this] { this->ReadMinibatches(m_readOrder, m_readOrderLength); });
				processData.detach();
			}
			//Compress END

			template<class ElemType>
			void CSparsePCReader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples)
			{


			}

			// GetMinibatch - Get the next minibatch (features and labels)
			// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
			//             [out] each matrix resized if necessary containing data. 
			// returns - true if there are more minibatches, false if no more minibatchs remain
			template<class ElemType>
			bool CSparsePCReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
			{

				// get out if they didn't call StartMinibatchLoop() first
				if (m_miniBatchSize == 0)
					return false;

				std::map<wstring, SparseDenseMemory<ElemType>*> dataToConsume = this->m_dataToConsume.pop();

				//First test if there is not any data left, when end of data is met, it will send out an empty data.

				wstring firstFeatureName = this->m_mapReaderOrder2FeatureName[0];
				if (dataToConsume[firstFeatureName]->m_sampleNumber == 0)
				{
					for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
					{
						std::wstring featureName = m_mapReaderOrder2FeatureName[i];
						dataToConsume[featureName]->Clear();
					}
					this->m_dataToProduce.push(dataToConsume);
					return false;
				}


				int32_t sampleSize = 0;

				for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
				{
					std::wstring featureName = m_mapReaderOrder2FeatureName[i];

					SparseDenseMemory<ElemType>* pMemoryInfo = dataToConsume[featureName];
					if (!matrices.HasInput(featureName))
					{
						pMemoryInfo->Clear();
						//this matrix is not needed in the NDL file, skip it 
						continue;
					}

					Matrix<ElemType>* pMatrix = &matrices.GetInputMatrix<ElemType>(featureName);
					SparseDenseFeatureInfo info = this->m_mapSparseDenseInfo[featureName];

					sampleSize = pMemoryInfo->m_sampleNumber;

					if (info.m_bIsSparse)
					{
						pMemoryInfo->FillSparseMatrix(pMatrix);
						
						//This only happens in sparse matrix

						if (m_reshapeInputToRowSize != 0)
						{
							pMatrix->Reshape(m_reshapeInputToRowSize, info.m_Dim * sampleSize / m_reshapeInputToRowSize);
						}

						if (m_returnDense || m_doGradientCheck)
						{
							pMatrix->SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
						}
					}
					else
					{
						pMemoryInfo->FillDenseMatrix(pMatrix);
					}

					pMemoryInfo->Clear();

				}

				this->m_dataToProduce.push(dataToConsume);
				// TODO: They are not really temporally ordered, so a better way would be to use tensors, once that is ready.
				m_pMBLayout->InitAsFrameMode(sampleSize);

				return true;
			}

			//Compress
			//ReadData, read data into a shared memory, if no data is read, then just break

			template<class ElemType>
			void CSparsePCReader<ElemType>::ReadZipData(size_t* read_order, size_t numToRead)
			{
				for (int i = 0; i < numToRead; i++)
				{
					size_t readSize = m_blockSizeInByte[read_order[i]];
					m_inFile.seekg(m_blockOffset[read_order[i]], ios::beg);

					//printf("read zip data - pop\n");
					void * zipDataBuffer = m_zipedDataToProduce.pop();

					//printf("read zip data - read\n");
					m_inFile.read((char*)zipDataBuffer, readSize);

					//printf("read zip data - end\n");

					m_zipedDataToConsume.push(zipDataBuffer);
					m_blockCntBeenRead++;
				}
			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::ClearUnzipBufferStatus()
			{
				m_blockCntBeenRead = 0;
				m_sampleCntInUnzippedBuffer = 0;
				m_lastValidPosOfUnzippedBuffer = -1;
				m_firstValidPosOfUnzippedBuffer = 0;


				m_blockCntBeenCopied = 0;
				m_batchCntBeenCopied = 0;
				for (int i = 0; i < m_dThreadCnt; i++)
					m_processedBlockCntPerThread[i] = 0;
			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::CompactUnzipBuffer()
			{
				int cnt = 0;
				for (int i = m_firstValidPosOfUnzippedBuffer; i <= m_lastValidPosOfUnzippedBuffer; i++)
				{
					((char *)m_unzippedBuffer)[cnt] = ((char *)m_unzippedBuffer)[i];
					cnt++;
				}

				m_firstValidPosOfUnzippedBuffer = 0;
				m_lastValidPosOfUnzippedBuffer = cnt - 1;
			}

			template<class ElemType>
			bool CSparsePCReader<ElemType>::Copy2Buffer(size_t numToRead)
			{
				std::map<wstring, SparseDenseMemory<ElemType>*> workingMemory;

				//first, fill the source buffer
				while (m_sampleCntInUnzippedBuffer < m_microBatchSize && m_blockCntBeenCopied < numToRead)
				{
					CompactUnzipBuffer();
					void * unzipBuffer = m_unzipedDataToConsume.pop();
					int sampleCnt = 0;
					memcpy(&sampleCnt, unzipBuffer, 4);

					int byteCnt = 0;
					memcpy(&byteCnt, (char *)unzipBuffer + 4, 4);

					memcpy((char *)m_unzippedBuffer + m_lastValidPosOfUnzippedBuffer + 1, (char *)unzipBuffer + 8, byteCnt);

					m_sampleCntInUnzippedBuffer += sampleCnt;
					m_lastValidPosOfUnzippedBuffer += byteCnt;

					m_blockCntBeenCopied++;

					m_unzipedDataToProduce.push(unzipBuffer);
				}

				size_t currentSampleNumber = 0;

				for (int32_t sampleI = 0; sampleI < min(m_microBatchSize, m_sampleCntInUnzippedBuffer); ++sampleI)
				{
					//Now a new batch
					if (currentSampleNumber == 0)
					{
						workingMemory = this->m_dataToProduce.pop();
					}

					for (int32_t i = 0; i < m_mapReaderOrder2FeatureName.size(); i++)
					{
						std::wstring featureName = m_mapReaderOrder2FeatureName[i];
						auto& featureInfo = this->m_mapSparseDenseInfo[featureName];

						SparseDenseMemory<ElemType>* pmemory = this->m_mapWorkingMemory[featureName];
						SparseDenseMemory<ElemType>* pConsumeMemory = workingMemory[featureName];


						if (featureInfo.m_bIsSparse)
						{
							int32_t nnZ;

							memcpy((char *)&nnZ, (char *)m_unzippedBuffer + m_firstValidPosOfUnzippedBuffer, 4);
							m_firstValidPosOfUnzippedBuffer += 4;
							memcpy((char*)pmemory->m_values, (char *)m_unzippedBuffer + m_firstValidPosOfUnzippedBuffer, sizeof(ElemType)*nnZ);
							m_firstValidPosOfUnzippedBuffer += sizeof(ElemType)*nnZ;
							memcpy((char*)pmemory->m_rowIndices, (char *)m_unzippedBuffer + m_firstValidPosOfUnzippedBuffer, sizeof(int32_t)*nnZ);
							m_firstValidPosOfUnzippedBuffer += sizeof(int32_t)*nnZ;
							pConsumeMemory->AddSparseData(pmemory->m_values, pmemory->m_rowIndices, nnZ);
						}
						else
						{
							memcpy((char*)pmemory->m_values, (char *)m_unzippedBuffer + m_firstValidPosOfUnzippedBuffer, sizeof(ElemType)*featureInfo.m_Dim);
							m_firstValidPosOfUnzippedBuffer += sizeof(ElemType)*featureInfo.m_Dim;
							pConsumeMemory->AddDenseData(pmemory->m_values, featureInfo.m_Dim);
						}
					}

					int32_t verificationCode;
					memcpy((char*)&verificationCode, (char *)m_unzippedBuffer + m_firstValidPosOfUnzippedBuffer, sizeof(int32_t));
					m_firstValidPosOfUnzippedBuffer += sizeof(int32_t);

					if (verificationCode != this->m_verificationCode)
					{
						RuntimeError("Verification code did not match (expected %d) - error in reading data", m_verificationCode);
						return false;
					}

					//Reading one End
					currentSampleNumber++;
				}

				this->m_dataToConsume.push(workingMemory);
				m_sampleCntInUnzippedBuffer -= currentSampleNumber;

				return true;
			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::ReadMinibatches(size_t* read_order, size_t numToRead)
			{
				do{
					if (m_sampleCntInUnzippedBuffer <= 0 && m_blockCntBeenCopied >= numToRead)
					{
						ClearUnzipBufferStatus();
						break;
					}

					if (!Copy2Buffer(numToRead))
					{
						break;
					}
				} while (true);

				std::map<wstring, SparseDenseMemory<ElemType>*> workingMemory;
				workingMemory = this->m_dataToProduce.pop();
				wstring firstFeature = this->m_mapReaderOrder2FeatureName[0];
				workingMemory[firstFeature]->Clear();
				this->m_dataToConsume.push(workingMemory);
			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::UnzipData(int threadIndex, size_t numToRead)
			{
				
				while (true)
				{
					size_t processedBlockCnt = 0;
					for (int i = 0; i < m_dThreadCnt; i++)
						processedBlockCnt += m_processedBlockCntPerThread[i];
					if (processedBlockCnt >= numToRead)
						return;

					m_processedBlockCntPerThread[threadIndex]++;
					
					//printf("unzip data - pop1\n");


					void * zipData = m_zipedDataToConsume.pop();

					//printf("unzip data - pop2\n");
					void * unzipData = m_unzipedDataToProduce.pop();

					int zipBytesCnt = 0;
					int sampleCnt = 0;
					memcpy(&zipBytesCnt, zipData, 4);
					memcpy(&sampleCnt, (char *)zipData + 4, 4);
					unsigned int unzipBytesCnt = (unsigned int)2 * sampleCnt * m_SparceDenseFeatureSize / m_miniBatchSize;

					//printf("unzip data - unzip\n");
					Unzip((char *)zipData + 8, (char *)unzipData + 8, (unsigned int)zipBytesCnt, unzipBytesCnt);
					//printf("unzip data - end\n");

					memcpy(unzipData, &sampleCnt, 4);
					memcpy((char*)unzipData + 4, &unzipBytesCnt, 4);

					m_unzipedDataToConsume.push(unzipData);
					m_zipedDataToProduce.push(zipData);
				}
			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::Unzip(void * input, void * output, unsigned int inputSize, unsigned int& outputSize)
			{
				z_stream infstream;
				infstream.zalloc = Z_NULL;
				infstream.zfree = Z_NULL;
				infstream.opaque = Z_NULL;

				infstream.avail_in = (unsigned int)inputSize;
				infstream.next_in = (Bytef *)input;
				infstream.avail_out = (unsigned int)outputSize;
				infstream.next_out = (Bytef *)output;

				// the actual DE-compression work.
				inflateInit2(&infstream, MAX_WBITS | 16);
				inflate(&infstream, Z_NO_FLUSH);
				inflateEnd(&infstream);

				outputSize = infstream.total_out;
			}

			template<class ElemType>
			void CSparsePCReader<ElemType>::FillReadOrder(size_t windowSize)
			{
				if (m_readOrder != nullptr)
				{
					free(m_readOrder);
				}
				m_readOrder = (size_t*)malloc(sizeof(size_t)*windowSize);
				for (size_t c = 0; c < windowSize; c++)
				{
					m_readOrder[c] = c;
				}
			}
			//Compress END

			template <class ElemType>
			bool CSparsePCReader<ElemType>::DataEnd() { return true; }

			// GetLabelMapping - Gets the label mapping from integer index to label type 
			// returns - a map from numeric datatype to native label type 
			template <class ElemType>
			const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& CSparsePCReader<ElemType>::GetLabelMapping(const std::wstring& /*sectionName*/)
			{
				return m_mapIdToLabel;
			}


			// SetLabelMapping - Sets the label mapping from integer index to label 
			// labelMapping - mapping table from label values to IDs (must be 0-n)
			// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
			// SetLabelMapping - Sets the label mapping from integer index to label 
			// labelMapping - mapping table from label values to IDs (must be 0-n)
			// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
			template <class ElemType>
			void CSparsePCReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<IDataReader::LabelIdType, LabelType>& labelMapping)
			{
				m_mapIdToLabel = labelMapping;
				m_mapLabelToId.clear();
				for (std::pair<unsigned, LabelType> var : labelMapping)
				{
					m_mapLabelToId[var.second] = var.first;
				}
			}

			// instantiate all the combinations we expect to be used
			template class CSparsePCReader<double>;
			template class CSparsePCReader<float>;
			template class SparseDenseMemory<double>;
			template class SparseDenseMemory<float>;
		}
	}
}
