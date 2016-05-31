//
// <copyright file="CDensereader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// CDensereader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "CDenseReader.h"
#include "fileutil.h"   // for fexists()
#include "TimerUtility.h"
#include <random>
#include <map>
#include <mutex>
#include <ctime>
#include <time.h>
#include "CUDAPageLockedMemAllocator.h"
#include <chrono>
#include <thread>
#include "LzmaDec.h"
#include "zlib.h"

#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif




namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			DWORD HIDWORD(size_t size) { return size >> 32; }
			DWORD LODWORD(size_t size) { return size & 0xFFFFFFFF; }

			template <class ElemType>
			SDenseBinaryMatrix<ElemType>::SDenseBinaryMatrix(wstring name, int deviceID, size_t numRows, size_t numCols) : BDenseBinaryMatrix<ElemType>(name, deviceID, numRows, numCols) {
				//this->m_values = (ElemType*)malloc(sizeof(ElemType)*numRows*numCols);
				this->m_values = (ElemType*)CUDAPageLockedMemAllocator::Malloc(sizeof(ElemType)*numRows*numCols, deviceID);
				//			if (this->m_values == nullptr)
				//			{
				//cpu only
				//				this->m_values = (ElemType*)malloc(sizeof(ElemType)*numRows*numCols);
				//			}
			}

			template <class ElemType>
			void SDenseBinaryMatrix<ElemType>::Clear() {
				this->m_numRows = 0;
			}

			template <class ElemType>
			void SDenseBinaryMatrix<ElemType>::Dispose() {
				if (this->m_values != nullptr) {
					//free(this->m_values);
					CUDAPageLockedMemAllocator::Free(this->m_values, this->m_deviceID);
				}

				//			if (this->m_values != nullptr) {
				//cpu only
				//				free(this->m_values);
				//			}

				this->m_values = nullptr;
			}


			template <class ElemType>
			void SDenseBinaryMatrix<ElemType>::Fill(Matrix<ElemType>* matrix) {
				matrix->SetValue(this->m_maxNumCols, this->m_numRows, matrix->GetDeviceId(), this->m_values, matrixFlagNormal);
#if DEBUG
				matrix->Print("testname");
#endif
			}

			template <class ElemType>
			void SDenseBinaryMatrix<ElemType>::SetMaxRows(size_t maxRows) {
				if (maxRows > this->m_maxNumRows) {
					//ElemType* values = (ElemType*)malloc(sizeof(ElemType)*this->m_maxNumCols*maxRows);
					ElemType* values = (ElemType*)CUDAPageLockedMemAllocator::Malloc(sizeof(ElemType)*this->m_maxNumCols*maxRows, this->m_deviceID);
					if (this->m_values != nullptr) {
						if (this->m_numRows > 0) {
							memcpy(values, this->m_values, sizeof(ElemType)*this->m_numRows*this->m_maxNumCols);
						}
						//free(this->m_values);
						CUDAPageLockedMemAllocator::Free(this->m_values, this->m_deviceID);
					}
					this->m_values = values;
					this->m_maxNumRows = maxRows;
				}
			}

			template <class ElemType>
			void SDenseBinaryMatrix<ElemType>::AddValues(void* values, size_t numRows) {
				memcpy(this->m_values + this->m_numRows*this->m_maxNumCols, values, sizeof(ElemType)*numRows*this->m_maxNumCols);
				this->m_numRows += numRows;
			}

			template<class ElemType>
			DenseBinaryInput<ElemType>::DenseBinaryInput(std::wstring fileName) : m_fileName(fileName), m_readOrder(nullptr), m_readOrderLength(0), m_randomize(false),
				m_startBlock(0), m_unzippedBuffer(nullptr), m_zippedFileBlockBuffer(nullptr), m_unzippedBufferLen(0), m_sampleCntInUnzippedBuffer(0), m_lastValidPosOfUnzippedBuffer(-1), m_firstValidPosOfUnzippedBuffer(0), m_blockCntBeenRead(0),
				m_blockCntBeenCopied(0), m_dThreadCnt(0), m_batchCntBeenCopied(0), m_processedBlockCnt(0){
				std::string name = msra::strfun::utf8(m_fileName);
				m_inFile.open(name, ifstream::binary | ifstream::in);
			}

			template<class ElemType>
			DenseBinaryInput<ElemType>::~DenseBinaryInput() {

			}

			template<class ElemType>
			template<class ConfigRecordType>
			void DenseBinaryInput<ElemType>::Init(std::map<std::wstring, std::wstring> rename, const ConfigRecordType & config) {

				GetZippedFileInfo();

				m_dThreadCnt = config(L"dThreadCnt", (int32_t)4);
				m_cAlgo = config(L"cAlgo", "7z");  //accapt value, "7z" or "gz"

				m_totalDim = config(L"totalDim", (int32_t)0);
				m_microBatchSize = config(L"microBatchSize", (int32_t)(1024));
				m_mbSize = (size_t)m_microBatchSize;

				//cache is disabled by default
				m_maxCacheSize = ((size_t)config(L"maxCacheSize", (size_t)(0))) * 1024 * 1024;

				m_blockSizeOfUnzippedBuffer = 4;
				size_t maxSampleNum = std::max(m_blockSampleCnt, m_microBatchSize);
				m_unzippedBufferLen = sizeof(int32_t) * m_totalDim * maxSampleNum * m_blockSizeOfUnzippedBuffer;
				if (m_unzippedBuffer != nullptr)
					free(m_unzippedBuffer);
				m_unzippedBuffer = malloc(m_unzippedBufferLen);

				if (m_zippedFileBlockBuffer != nullptr)
					free(m_zippedFileBlockBuffer);
				m_zippedFileBlockBuffer = malloc(m_blockSize);

				m_numBatches = m_numRows / m_microBatchSize;
				if (m_numRows% m_microBatchSize)
					m_numBatches += 1;

				m_microbatchFileSize = (sizeof(ElemType)) * m_totalDim * m_microBatchSize;
				for (const auto & id : config.GetMemberIds())
				{
					if (!config.CanBeConfigRecord(id))
						continue;
					const ConfigRecordType & temp = config(id);
					// see if we have a config parameters that contains a "dim" element, it's a sub key, use it
					if (temp.ExistsCurrent(L"dim"))
					{
						wstring wname = id;
						int32_t singlefeature_dim = temp(L"dim", (int32_t)(0));
						int32_t singlefeature_startIndex = temp(L"startIndex", (int32_t)(0));
						if (rename.find(wname) == rename.end())
						{
							m_labels.emplace_back(wname);
						}
						else
						{
							m_labels.emplace_back(rename[wname]);
						}
						m_mappedNumCols[m_labels.back()] = singlefeature_dim;
						m_mappedStartIndex[m_labels.back()] = singlefeature_startIndex;

						m_mappedBuffer[m_labels.back()] = malloc(m_microbatchFileSize);
						m_mappedBufferForConsumption[m_labels.back()] = malloc(m_microbatchFileSize);
					}
				}

				m_bQueueBufferAllocated = false;
			}

			template<class ElemType>
			bool DenseBinaryInput<ElemType>::Randomize()
			{
				return false;
				/*
				if (m_randomize > 0)
				{
				return true;
				}
				return false;
				*/
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::GetZippedFileInfo()
			{
				cout << "GetZippedFileInfo Begin" << endl;

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
				size_t totalSampel = 0;

				int progress = -1;
				while (pos < m_fileSize)
				{
					int newProgress = pos * 10 / m_fileSize;
					if (newProgress != progress) {
						cout << "GetZippedFileInfo: " << newProgress * 10 << "%" << endl;
						progress = newProgress;
					}

					m_inFile.read((char *)&block_size, 4);
					m_inFile.read((char *)&block_sample_cnt, 4);

					totalSampel += block_sample_cnt;

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

				cout << "GetZippedFileInfo End" << endl;
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::FillReadOrder()
			{
				if (m_readOrder != nullptr)
					free(m_readOrder);

				m_readOrder = (size_t*)malloc(sizeof(size_t)*m_windowSize);
				for (size_t c = 0; c < m_windowSize; c++)
					m_readOrder[c] = c + m_startBlock;
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::Shuffle()
			{
				if (Randomize())
				{
					shuffle(&(m_readOrder[0]), &(m_readOrder[m_readOrderLength - 1]), m_randomEngine);
				}
			}

			template<class ElemType>
			int DenseBinaryInput<ElemType>::GetMinimumEpochSizeCrossAllWorker(size_t mbSize, size_t subsetNum, size_t numSubsets){

				int * tmpEpochSize = (int*)malloc(numSubsets * sizeof(int));
				size_t minEpochSize = 1e10;
				size_t blockNum = m_blockOffset.size();
				size_t local_startBlock = 0;
				size_t local_endBlock = 0;
				int blockCnt = m_blockOffset.size() / numSubsets;
				for (size_t i = 0; i < numSubsets; i++){
					size_t startBlock = blockCnt * i;
					size_t endBlock = blockCnt * (i + 1);
					size_t remainder = blockNum % numSubsets;

					size_t lb = min(remainder, i);
					size_t ub = min(remainder, i + 1);

					startBlock += lb;
					endBlock += ub;

					size_t sampleCnt = 0;
					for (int j = startBlock; j < endBlock; j++)
						sampleCnt += m_sampleCntInBlock[j];

					size_t iEpochSize = sampleCnt / mbSize;
					if (sampleCnt % mbSize != 0)
						iEpochSize++;

					tmpEpochSize[i] = iEpochSize;
					minEpochSize = min(iEpochSize, minEpochSize);
					if (i == subsetNum){
						local_startBlock = startBlock;
						local_endBlock = endBlock;
					}
				}

				size_t sampleNum = 0;
				size_t totalSampleNum = minEpochSize * mbSize;
				size_t blockId = local_startBlock;
				while ((sampleNum < totalSampleNum) && (blockId < local_endBlock)){
					sampleNum += m_sampleCntInBlock[blockId];
					blockId += 1;
				}
				m_windowSize = blockId - local_startBlock;
				m_startBlock = local_startBlock;

				for (int i = 0; i < numSubsets; i++){
					fprintf(stderr, "iEpochSize %d, minEpochSize %d, startBlock %d, windowSize %d, subsetNum %d, numSubsets %d\n", tmpEpochSize[i], (int)minEpochSize, (int)m_startBlock, (int)m_windowSize, (int)subsetNum, (int)numSubsets);
				}

				free(tmpEpochSize);

				return minEpochSize;
			}

			void InitCacheDir(const char* dirPath, int maxFileIndex = 100) {
				//std C++ has no protable code to iterate a dir
#ifdef _WIN32
				int flag = _mkdir(dirPath);
#else
				int flag = mkdir(dirPath, 0755);
#endif
				if (flag != 0) {
					char buf[255];
					for (int i = 0; i < maxFileIndex; ++i) {
						sprintf(buf, "%s/%d", dirPath, i);
						remove(buf);
					}
				}
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t subsetNum, size_t numSubsets) {
				m_nextMB = 0;
				m_mbSize = mbSize;

				m_epochSize = GetMinimumEpochSizeCrossAllWorker(mbSize, subsetNum, numSubsets);

				if (m_windowSize != m_readOrderLength) {
					FillReadOrder();
					m_readOrderLength = m_windowSize;
				}

				//Shuffle();
				size_t G1 = 1024 * 1024 * 1024 * 1.5;

				size_t maxPointers = G1 / m_microbatchFileSize;
				size_t zipQueueLen = G1 * 5 / m_blockSize;
				size_t unzipQueueLen = G1 * 2 / (sizeof(int32_t) * m_totalDim * m_blockSampleCnt * 2);

				bool firstEpoch = !m_bQueueBufferAllocated;

				if (!m_bQueueBufferAllocated) {
					for (size_t c = 0; c < maxPointers; c++) {
						void* dataBuffer = malloc(m_microbatchFileSize + sizeof(int32_t)); //sample cnt
						m_dataToProduce.push(dataBuffer);
					}

					for (size_t c = 0; c < unzipQueueLen; c++) {
						void* unzipDataBuffer = malloc(sizeof(int32_t) * m_totalDim * m_blockSampleCnt * 2 + sizeof(int32_t)); //sample cnt
						m_unzipedDataToProduce.push(unzipDataBuffer);
					}

					for (size_t c = 0; c < zipQueueLen; c++){
						void* zipDataBuffer = malloc(m_blockSize);
						m_zipedDataToProduce.push(zipDataBuffer);
					}
					m_bQueueBufferAllocated = true;
				}

				bool enableCache = (this->m_maxCacheSize != 0);

				if (firstEpoch) {
					if (enableCache){
						//init cache dir
#ifdef _WIN32
						const char dirName[] = "./cache";
#else
						const char dirName[] = "/tmp/cachefromcdensereader";
#endif
						InitCacheDir(dirName);

						//open cache file
						char buf[255];
						for (int i = 0; i < 100; ++i) {
							sprintf(buf, "%s/%d", dirName, i);
							this->m_cacheFile.open(buf, ios::in | ios::out | ios::binary | ios::trunc);

							if (this->m_cacheFile) {
								break;
							}
						}

						if (!this->m_cacheFile) {
							cerr << "allocate cache file failed." << endl;
							exit(-1);
						}
					}
					this->m_cachedBlockNum = 0;
				}


				//read thread
				bool writeCache = firstEpoch && enableCache;
				std::thread readZipData([this](bool writeCache)
				{
					size_t writedBlockNum = this->ReadZipData(m_readOrder,
						m_readOrderLength, this->m_maxCacheSize,
						this->m_cachedBlockNum, writeCache);

					if (writeCache) {
						this->m_cachedBlockNum = writedBlockNum;
					}

				}, writeCache);
				readZipData.detach();

				//cache thread
				if (!firstEpoch && enableCache) {
					std::thread readCacheData([this]
					{
						this->ReadCachedZipData(this->m_readOrder, this->m_cachedBlockNum);
					});
					readCacheData.detach();
				}


				for (m_dIndex = 0; m_dIndex < m_dThreadCnt; m_dIndex++){
					m_unzipThreads[m_dIndex] = std::thread([this](int idx) { this->UnzipData(idx, m_numBlocks); }, m_dIndex);
					m_unzipThreads[m_dIndex].detach();
				}

				std::thread processData([this] { this->ReadMinibatches(m_readOrder, m_readOrderLength); });
				processData.detach();

			}

			template<class ElemType>
			shared_ptr<BDenseBinaryMatrix<ElemType>> DenseBinaryInput<ElemType>::CreateMatrix(std::wstring matName, int deviceId) {
				shared_ptr<BDenseBinaryMatrix<ElemType>> retVal;// = nullptr;
				if (std::find(m_labels.begin(), m_labels.end(), matName) != m_labels.end()) {
					retVal = make_shared<SDenseBinaryMatrix<ElemType>>(matName, deviceId, m_mbSize, m_mappedNumCols[matName]);
				}

				return retVal;
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::Print(void * buffer, int start, int end){
				int tmpcnt = 0;
				for (int i = start; i < end; i += 4){
					float d = 0;
					memcpy(&d, (char *)buffer + i, 4);
					printf("%f ", d);
					tmpcnt++;
					if (tmpcnt % m_totalDim == 0)
						printf("\n");
				}
				printf("\n****\n");
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::ClearUnzipBufferStatus(){
				m_blockCntBeenRead = 0;
				m_sampleCntInUnzippedBuffer = 0;
				m_lastValidPosOfUnzippedBuffer = -1;
				m_firstValidPosOfUnzippedBuffer = 0;


				m_blockCntBeenCopied = 0;
				m_batchCntBeenCopied = 0;
				m_processedBlockCnt = 0;
			}

			template<class ElemType>
			int DenseBinaryInput<ElemType>::UnzipGz(void * input, void * output, int inputSize, int outputSize)
			{
				z_stream infstream;
				infstream.zalloc = Z_NULL;
				infstream.zfree = Z_NULL;
				infstream.opaque = Z_NULL;

				infstream.avail_in = inputSize;
				infstream.next_in = (Bytef *)input;
				infstream.avail_out = outputSize;
				infstream.next_out = (Bytef *)output;

				// the actual DE-compression work.
				inflateInit2(&infstream, MAX_WBITS | 16);
				int res = inflate(&infstream, Z_NO_FLUSH);
				inflateEnd(&infstream);

				outputSize = infstream.avail_out;

				return res;
			}

			template<class ElemType>
			int DenseBinaryInput<ElemType>::Unzip7z(void * input, void * output, int inputSize, int outputSize)
			{
				size_t os = outputSize;
				return DeCompressMem((Byte *)input, (Byte *)output, inputSize, &os);
			}

			template<class ElemType>
			int DenseBinaryInput<ElemType>::Unzip(void * input, void * output, int inputSize, int outputSize)
			{
				if (m_cAlgo.compare("7z") == 0)
					return Unzip7z(input, output, inputSize, outputSize);
				else
					return UnzipGz(input, output, inputSize, outputSize);
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::CompactUnzipBuffer(){
				int cnt = 0;
				for (int i = m_firstValidPosOfUnzippedBuffer; i <= m_lastValidPosOfUnzippedBuffer; i++){
					((char *)m_unzippedBuffer)[cnt] = ((char *)m_unzippedBuffer)[i];
					cnt++;
				}

				m_firstValidPosOfUnzippedBuffer = 0;
				m_lastValidPosOfUnzippedBuffer = cnt - 1;
			}

			template<class ElemType>
			void DenseBinaryInput<ElemType>::UnzipData(int threadIndex, size_t numToRead){
				while (true){

					m_unzipLocker.lock();
					if (m_processedBlockCnt >= numToRead) {
						m_unzipLocker.unlock();
						return;
					}
					else{
						m_processedBlockCnt++;
					}
					m_unzipLocker.unlock();


					void * zipData = m_zipedDataToConsume.pop();
					void * unzipData = m_unzipedDataToProduce.pop();

					int zipBytesCnt = 0;
					int sampleCnt = 0;
					memcpy(&zipBytesCnt, zipData, 4);
					memcpy(&sampleCnt, (char *)zipData + 4, 4);
					int unzipBytesCnt = sampleCnt * m_totalDim * 4;
					Unzip((char *)zipData + 8, (char *)unzipData + 4, zipBytesCnt, unzipBytesCnt * 2);

					memcpy(unzipData, &sampleCnt, 4);

					m_unzipedDataToConsume.push(unzipData);
					m_zipedDataToProduce.push(zipData);

				}
			}


			template<class ElemType>
			void DenseBinaryInput<ElemType>::ReadCachedZipData(size_t* read_order, size_t numToRead) {

				size_t limit = (this->m_zipedDataToConsume.size() + this->m_zipedDataToProduce.size()) / 2;

				this->m_cacheFile.seekg(0, ios::beg);

				time_t start = time(0);

				for (int i = 0; i < numToRead; ++i) {
					void* zipDataBuffer = this->m_zipedDataToProduce.pop(limit);
					cerr << "read cached data:"
						<< this->m_zipedDataToProduce.size()
						<< ","
						<< this->m_zipedDataToConsume.size()
						<< endl;

					size_t readSize = m_blockSizeInByte[read_order[i]];
					this->m_cacheFile.read((char*)zipDataBuffer, readSize);

					m_zipedDataToConsume.push(zipDataBuffer);

					m_blockCntLocker.lock();
					m_blockCntBeenRead += 1;
					m_blockCntLocker.unlock();
				}

				cerr << "read cached data finished in " << time(0) - start << "s" << endl;
			}

			template<class ElemType>
			size_t DenseBinaryInput<ElemType>::ReadZipData(size_t* read_order,
				size_t numToRead, size_t maxCacheSize, size_t skipBlockNum, bool writeToCache)
			{
				size_t cachedNum = 0;

				time_t start = time(0);

				for (int i = skipBlockNum; i < numToRead; i++) {
					void * zipDataBuffer = this->m_zipedDataToProduce.pop();
					cerr << "read zip data:"
						<< this->m_zipedDataToProduce.size()
						<< ","
						<< this->m_zipedDataToConsume.size()
						<< endl;

					size_t readSize = m_blockSizeInByte[read_order[i]];
					this->m_inFile.seekg(m_blockOffset[read_order[i]], ios::beg);

					this->m_inFile.read((char*)zipDataBuffer, readSize);
					m_zipedDataToConsume.push(zipDataBuffer);

					this->m_blockCntLocker.lock();
					m_blockCntBeenRead += 1;
					this->m_blockCntLocker.unlock();


					if (writeToCache && maxCacheSize >= readSize) {
						this->m_cacheFile.write((char*)zipDataBuffer, readSize);
						cachedNum += 1;
						maxCacheSize -= readSize;
					}
					else {
						//stop caching
						writeToCache = false;
					}
				}

				cerr << "read zip finished in " << time(0) - start << "s" << endl;

				return cachedNum;
			}


			template<class ElemType>
			int32_t DenseBinaryInput<ElemType>::Copy2Buffer(void *bufferInProduce, size_t numToRead){

				//first, fill the source buffer
				while (m_sampleCntInUnzippedBuffer < m_microBatchSize && m_blockCntBeenCopied < numToRead){
					CompactUnzipBuffer();
					void * unzipBuffer = m_unzipedDataToConsume.pop();
					int sampleCnt = 0;
					memcpy(&sampleCnt, unzipBuffer, 4);
					int byteCnt = sampleCnt * m_totalDim * sizeof(int32_t);
					memcpy((char *)m_unzippedBuffer + m_lastValidPosOfUnzippedBuffer + 1, (char *)unzipBuffer + 4, byteCnt);

					m_sampleCntInUnzippedBuffer += sampleCnt;
					m_lastValidPosOfUnzippedBuffer += byteCnt;

					m_blockCntBeenCopied++;

					m_unzipedDataToProduce.push(unzipBuffer);
				}

				size_t sizeToBeCopied = m_microBatchSize * m_totalDim * 4;
				int32_t sampleCntToBeCopied = m_microBatchSize;
				if (m_sampleCntInUnzippedBuffer < m_microBatchSize){
					sizeToBeCopied = m_sampleCntInUnzippedBuffer * m_totalDim * 4;
					sampleCntToBeCopied = m_sampleCntInUnzippedBuffer;
				}

				memcpy(bufferInProduce, (char *)m_unzippedBuffer + m_firstValidPosOfUnzippedBuffer, sizeToBeCopied);

				m_firstValidPosOfUnzippedBuffer += sizeToBeCopied;
				m_sampleCntInUnzippedBuffer -= sampleCntToBeCopied;

				return sampleCntToBeCopied;
			}


			template<class ElemType>
			void DenseBinaryInput<ElemType>::ReadMinibatches(size_t* read_order, size_t numToRead) {
				do{
					if ((m_sampleCntInUnzippedBuffer <= 0 && m_blockCntBeenCopied >= numToRead) || (m_nextMB == m_epochSize)){
						ClearUnzipBufferStatus();
						break;
					}

					void* data_buffer = m_dataToProduce.pop();
					int32_t sampleNumber = Copy2Buffer((char *)data_buffer + 4, numToRead);
					memcpy(data_buffer, &sampleNumber, 4);
					m_batchCntBeenCopied++;

					for (int32_t c = 0; c < m_labels.size(); c++)
					{
						void* labelBuffer = m_mappedBuffer[m_labels[c]];
						int32_t labelStartIndex = m_mappedStartIndex[m_labels[c]];
						int32_t labelDim = m_mappedNumCols[m_labels[c]];
						void* pSource = (char*)data_buffer + sizeof(int32_t) + labelStartIndex*sizeof(ElemType);
						void* pDest = (char*)labelBuffer;

						for (int32_t s = 0; s < sampleNumber; s++)
						{
							memcpy(pDest, pSource, labelDim*sizeof(ElemType));
							pSource = (char*)pSource + m_totalDim*sizeof(ElemType);
							pDest = (char*)pDest + labelDim*sizeof(ElemType);
						}
					}

					void* pDest = (char*)data_buffer + sizeof(int32_t);
					for (int32_t c = 0; c < m_labels.size(); c++)
					{
						void* labelBuffer = m_mappedBuffer[m_labels[c]];
						int32_t labelDim = m_mappedNumCols[m_labels[c]];
						memcpy(pDest, labelBuffer, sampleNumber* sizeof(ElemType)*labelDim);
						pDest = (char*)pDest + sampleNumber* sizeof(ElemType)*labelDim;

					}

					m_dataToConsume.push(data_buffer);
				} while (true);

			}

			template<class ElemType>
			size_t DenseBinaryInput<ElemType>::ReadMinibatch(void* data_buffer, std::map<std::wstring, shared_ptr<BDenseBinaryMatrix<ElemType>>>& matrices) {

				int32_t curMBSize;

				int64_t buffer_offset = 0;

				curMBSize = *(int32_t*)((char*)data_buffer + buffer_offset);
				buffer_offset += sizeof(int32_t);

				void* pSource = (char*)data_buffer + sizeof(int32_t);


				for (int32_t c = 0; c < m_labels.size(); c++)
				{
					int32_t labelDim = m_mappedNumCols[m_labels[c]];

					auto findMat = matrices.find(m_labels[c]);
					if (findMat != matrices.end())
					{
						auto mat = findMat->second;
						mat->AddValues(pSource, curMBSize);
					}

					pSource = (char*)pSource + sizeof(ElemType)* curMBSize*labelDim;
				}
				return (size_t)curMBSize;
			}

			template<class ElemType>
			size_t DenseBinaryInput<ElemType>::FillMatrices(std::map<std::wstring, shared_ptr<BDenseBinaryMatrix<ElemType>>>& matrices) {
				size_t curSize = 0;
				for (auto mat : matrices) {
					mat.second->SetMaxRows(m_mbSize);
					mat.second->Clear();
				}
				void* data_buffer;
				while (curSize + m_microBatchSize <= m_mbSize && m_nextMB < m_epochSize) {
					data_buffer = m_dataToConsume.pop();
					curSize += ReadMinibatch(data_buffer, matrices);
					m_nextMB++;
					m_dataToProduce.push(data_buffer);
				}
				//Debug
				if (curSize == 0){

					while (!m_dataToConsume.empty())
					{
						m_dataToProduce.push(m_dataToConsume.pop());
					}

					while (!m_zipedDataToConsume.empty())
					{
						m_zipedDataToProduce.push(m_zipedDataToConsume.pop());
					}

					while (!m_unzipedDataToConsume.empty())
					{
						m_unzipedDataToProduce.push(m_unzipedDataToConsume.pop());
					}

				}
				return curSize;
			}

			template<class ElemType>
			template<class ConfigRecordType>
			void CDensereader<ElemType>::InitFromConfig(const ConfigRecordType & readerConfig) {

				std::map<std::wstring, std::wstring> rename;
				RenamedMatrices(readerConfig, rename);

				if (readerConfig.Exists(L"randomize"))
				{
					string randomizeString = readerConfig(L"randomize");
					if (randomizeString == "None")
					{
						m_randomize = 0L;
					}
					else if (randomizeString == "Auto")
					{
						time_t rawtime;
						struct tm* timeinfo;
						time(&rawtime);
						timeinfo = localtime(&rawtime);
						m_randomize = (unsigned long)(timeinfo->tm_sec + timeinfo->tm_min * 60 + timeinfo->tm_hour * 60 * 60);
					}
					else
					{
						m_randomize = readerConfig(L"randomize", 0);
					}
				}
				else
				{
					m_randomize = 0L;
				}

				m_partialMinibatch = true;
				std::string minibatchMode(readerConfig(L"minibatchMode", "Partial"));
				m_partialMinibatch = !_stricmp(minibatchMode.c_str(), "Partial");

				std::wstring file = readerConfig(L"file", L"");

				m_dataInput = make_shared<DenseBinaryInput<ElemType>>(file);
				m_dataInput->Init(rename, readerConfig);

				m_mbSize = (size_t)readerConfig(L"microBatchSize", 0);
				if (m_mbSize > 0)
				{
					if (m_dataInput->GetMBSize() != m_mbSize)
					{
						RuntimeError("Data file and config file have mismatched minibatch sizes.\n");
						return;
					}
				}
				else
				{
					m_mbSize = m_dataInput->GetMBSize();
				}

				m_prefetchEnabled = true;
			}

			template<class ElemType>
			void CDensereader<ElemType>::Destroy() {

			}

			template<class ElemType>
			CDensereader<ElemType>::~CDensereader()
			{
				Destroy();
			}

			template<class ElemType>
			void CDensereader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples) {
				return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
			}

			template<class ElemType>
			void CDensereader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t /*requestedEpochSamples*/) {
				m_epoch = epoch;
				m_mbSize = mbSize;
#if DEBUG
				if (reader_series != NULL) {
					delete reader_series;
				}
				reader_series = new marker_series(L"Base Reader");
				cur_read = 0;
#endif
				m_dataInput->StartDistributedMinibatchLoop(mbSize, subsetNum, numSubsets);

			}

			template<class ElemType>
			void CDensereader<ElemType>::CheckDataMatrices(StreamMinibatchInputs& matrices) {
				if (m_dataMatrices.empty())
				{
					for (auto inmat : matrices) {
						shared_ptr<BDenseBinaryMatrix<ElemType>> mat = m_dataInput->CreateMatrix(inmat.first, inmat.second.matrix->GetDeviceId());
						if (mat != nullptr) {
							m_dataMatrices[inmat.first] = mat;
						}
					}
				}
			}
			template<class ElemType>
			void CDensereader<ElemType>::DoDSSMMatrix(Matrix<ElemType>& mat, size_t actualMBSize) {
				size_t numRows = mat.GetNumRows();
				if (DSSMCols < actualMBSize) {
					if (DSSMLabels != nullptr) {
						//free(DSSMLabels);
						CUDAPageLockedMemAllocator::Free(DSSMLabels, mat.GetDeviceId());
					}
					DSSMCols = actualMBSize;
					//DSSMLabels = (ElemType*)malloc(sizeof(ElemType)*numRows*actualMBSize);
					DSSMLabels = (ElemType*)CUDAPageLockedMemAllocator::Malloc(sizeof(ElemType)*numRows*actualMBSize, mat.GetDeviceId());
					memset(DSSMLabels, 0, sizeof(ElemType)*numRows*actualMBSize);
					for (size_t c = 0; c < numRows*actualMBSize; c += numRows) {
						DSSMLabels[c] = 1;
					}
				}
				if (mat.GetNumCols() != actualMBSize) {
					mat.SetValue(numRows, actualMBSize, mat.GetDeviceId(), DSSMLabels, matrixFlagNormal);
				}
			}


			template<class ElemType>
			bool CDensereader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices) {
#if DEBUG
				span minibatch_span(*reader_series, 1, L"Get Minibatch: %ld", cur_read);
#endif
				size_t actualMBSize = 0;
				if (m_prefetchEnabled)
				{
					if (!m_pendingAsyncGetMinibatch.valid()) {
						//fprintf(stderr, "not valid\n");
						CheckDataMatrices(matrices);
						m_pendingAsyncGetMinibatch = std::async(std::launch::async, [this]()
						{
							return m_dataInput->FillMatrices(m_dataMatrices);
						});
					}

					//fprintf(stderr, "before get.\n");
					//timer = clock();
#if DEBUG
					reader_series->write_flag(_T("before get."));
#endif
					actualMBSize = m_pendingAsyncGetMinibatch.get();
#if DEBUG
					reader_series->write_flag(_T("after get."));
#endif
					//timer = clock() - timer;
					//fprintf(stderr, "done get\tIt took me %d clicks (%f seconds).\n", timer, ((float)timer) / CLOCKS_PER_SEC);

					if (actualMBSize == 0) {
						return false;
					}

					m_pMBLayout->InitAsFrameMode(actualMBSize);
#if DEBUG
					reader_series->write_flag(_T("starting fill."));
#endif
					for (auto matrix : m_dataMatrices) {
						if (matrices.HasInput(matrix.first))
							matrix.second->Fill(&matrices.GetInputMatrix<ElemType>(matrix.first));
					}
#if DEBUG
					reader_series->write_flag(_T("done fill."));
#endif
					if (matrices.HasInput((L"DSSMLabel")))
						DoDSSMMatrix(matrices.GetInputMatrix<ElemType>(L"DSSMLabel"), actualMBSize);

					m_pendingAsyncGetMinibatch = std::async(std::launch::async, [this]()
					{
						//CheckDataMatrices(matrices);
						return m_dataInput->FillMatrices(m_dataMatrices);
					});
				}
#if DEBUG
				cur_read++;
#endif

				m_pMBLayout->InitAsFrameMode(actualMBSize);

				return true;
			}


			template<class ElemType>
			template<class ConfigRecordType>
			void CDensereader<ElemType>::RenamedMatrices(const ConfigRecordType& config, std::map<std::wstring, std::wstring>& rename) {
				for (const auto & id : config.GetMemberIds())
				{
					if (!config.CanBeConfigRecord(id))
						continue;
					const ConfigRecordType & temp = config(id);
					// see if we have a config parameters that contains a "dim" element, it's a sub key, use it
					if (temp.ExistsCurrent(L"rename"))
					{

						std::wstring ren = temp(L"rename");
						rename.emplace(msra::strfun::utf16(id), msra::strfun::utf16(ren));
					}
				}
			}

			template<class ElemType>
			bool CDensereader<ElemType>::DataEnd()
			{
				return m_dataInput->DataEnd();
			}

			template<class ElemType>
			bool DenseBinaryInput<ElemType>::DataEnd()
			{
				return true;
			}

			// instantiate all the combinations we expect to be used
			template class CDensereader<double>;
			template class CDensereader<float>;
		}
	}

}
