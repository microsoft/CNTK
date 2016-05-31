#pragma once
#include "stdafx.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>
#include <random>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#if DEBUG
#include <cvmarkersobj.h>
using namespace Concurrency::diagnostic;
#endif

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			template <typename T>
			class DenseBlockingQueue
			{
			private:
				std::mutex              d_mutex;
				std::condition_variable d_condition;
				std::deque<T>           d_queue;
			public:
				void push(T const& value) {
					{
						std::unique_lock<std::mutex> lock(this->d_mutex);
						d_queue.push_front(value);
					}
					this->d_condition.notify_one();
				}

				T pop() {
					return pop(1);
				}

				T pop(size_t limit) {
					if (limit < 1) limit = 1;
					std::unique_lock<std::mutex> lock(this->d_mutex);
					this->d_condition.wait(lock, [=]{ return this->d_queue.size() >= limit; });
					T rc(std::move(this->d_queue.back()));
					this->d_queue.pop_back();
					return rc;
				}

				bool empty() {
					std::lock_guard<std::mutex> lock(this->d_mutex);
					return this->d_queue.empty();
				}

				size_t size() {
					std::lock_guard<std::mutex> lock(this->d_mutex);
					return this->d_queue.size();
				}
			};


			template<class ElemType>
			class BDenseBinaryMatrix {
			public:
				BDenseBinaryMatrix(wstring name, int deviceID, size_t numRows, size_t numCols) : m_matrixName(name), m_deviceID(deviceID), m_maxNumRows(numRows), m_numRows(0), m_maxNumCols(numCols), m_values(nullptr) {};
				//BinaryMatrix(wstring name, size_t numRows, size_t numCols) : m_matrixName(name), m_maxNumRows(numRows), m_numRows(0), m_maxNumCols(numCols), m_values(nullptr) {};
				virtual void Clear() = 0;
				virtual void Dispose() = 0;
				virtual void Fill(Matrix<ElemType>*) = 0;
				virtual void AddValues(void*, size_t) = 0;
				virtual void AddColIndices(void*, size_t) = 0;
				virtual void AddRowIndices(void*, size_t) = 0;
				virtual void UpdateNNz(size_t) = 0;
				virtual void UpdateCurMB(size_t mb) { m_numRows += mb; }
				virtual void ResizeArrays(size_t) = 0;
				virtual void SetMaxRows(size_t maxRows) = 0;
			protected:
				wstring m_matrixName;
				int m_deviceID;
				ElemType* m_values;
				size_t m_maxNumRows;
				size_t m_maxNumCols;

				size_t m_numRows;

			};

			template<class ElemType>
			class SDenseBinaryMatrix : public BDenseBinaryMatrix<ElemType> {
			public:
				SDenseBinaryMatrix(wstring name, int deviceID, size_t numRows, size_t numCols);
				//DenseBinaryMatrix(wstring name, size_t numRows, size_t numCols);
				virtual void Clear();
				virtual void Dispose();
				virtual void Fill(Matrix<ElemType>* matrix) override;
				virtual void AddValues(void* values, size_t numRows) override;
				virtual void AddColIndices(void* /*colIndices*/, size_t /*numCols*/) override { NOT_IMPLEMENTED }
				virtual void AddRowIndices(void* /*rowIndices*/, size_t /*nnz*/) override { NOT_IMPLEMENTED }
				virtual void UpdateNNz(size_t /*nnz*/) override { NOT_IMPLEMENTED }
				virtual void ResizeArrays(size_t) { NOT_IMPLEMENTED }
				virtual void SetMaxRows(size_t maxRows) override;
			protected:
			};

			template<class ElemType>
			class DenseBinaryInput {
			public:
				DenseBinaryInput(std::wstring fileName);
				~DenseBinaryInput();
				template<class ConfigRecordType>
				void Init(std::map<std::wstring, std::wstring> rename, const ConfigRecordType & config);
				void StartDistributedMinibatchLoop(size_t mbSize, size_t subsetNum, size_t numSubsets);
				void ReadMinibatches(size_t* read_order, size_t numToRead);
				size_t ReadMinibatch(void* data_buffer, std::map<std::wstring, shared_ptr<BDenseBinaryMatrix<ElemType>>>& matrices);
				size_t FillMatrices(std::map<std::wstring, shared_ptr<BDenseBinaryMatrix<ElemType>>>& matrices);
				size_t GetMBSize() { return m_mbSize; }
				size_t GetNumMB() { return m_numBatches / (m_mbSize / m_microBatchSize); }
				void Shuffle();
				shared_ptr<BDenseBinaryMatrix<ElemType>> CreateMatrix(std::wstring matName, int deviceId);
				virtual bool DataEnd();


			private:
				void FillReadOrder();
				bool Randomize();
				void GetZippedFileInfo();
				int Unzip(void * input, void * output, int inputSize, int outputSize);
				int Unzip7z(void * input, void * output, int inputSize, int outputSize);
				int UnzipGz(void * input, void * output, int inputSize, int outputSize);
				void CompactUnzipBuffer();
				void UnzipData(int threadIndex, size_t numToRead);
				void Print(void * buffer, int start, int end);
				void ClearUnzipBufferStatus();
				int GetMinimumEpochSizeCrossAllWorker(size_t mbSize, size_t subsetNum, size_t numSubsets);
				int32_t Copy2Buffer(void *bufferInProduce, size_t numToRead);

				size_t ReadZipData(size_t* read_order, size_t numToRead, size_t maxCacheSize, bool writeToCache); //return: cached block num

				void ReadCachedZipData(size_t* read_order, size_t numToTread);

				ifstream m_inFile;

				fstream m_cacheFile;
				size_t m_maxCacheSize; //MB
				size_t m_cachedBlockNum;

				std::wstring m_fileName;
				size_t m_fileSize;

				size_t m_nextMB; // starting sample # of the next minibatch
				size_t m_epochSize; // size of an epoch

				size_t m_numRows;    // size of minibatch requested
				size_t m_numBatches;    // size of minibatch requested

				int32_t m_microBatchSize;
				size_t m_mbSize;

				size_t m_windowSize;
				size_t m_startBlock;

				bool m_randomize;
				size_t* m_readOrder; // array to shuffle to reorder the dataset
				size_t m_readOrderLength;
				size_t m_maxMBSize;

				size_t m_totalDim;

				size_t m_microbatchFileSize;
				bool m_bQueueBufferAllocated;


				std::vector<std::wstring> m_features;
				std::vector<std::wstring> m_labels;
				std::map<std::wstring, int32_t> m_mappedNumCols;
				std::map<std::wstring, int32_t> m_mappedStartIndex;

				std::map<std::wstring, void*> m_mappedBuffer;
				std::map<std::wstring, void*> m_mappedBufferForConsumption;

				std::vector<size_t> m_blockSizeInByte;
				std::vector<size_t> m_sampleCntInBlock;
				std::vector<size_t> m_blockOffset;
				size_t m_numBlocks;
				int32_t m_blockSampleCnt;
				int32_t m_blockSize;
				int32_t m_blockSizeOfUnzippedBuffer;

				void * m_unzippedBuffer;
				void * m_zippedFileBlockBuffer;
				size_t m_unzippedBufferLen;
				size_t m_sampleCntInUnzippedBuffer;
				size_t m_blockCntBeenRead;
				long m_lastValidPosOfUnzippedBuffer;
				long m_firstValidPosOfUnzippedBuffer;

				//muti-thread to decompress
				std::mutex m_unzipLocker;
				std::mutex m_blockCntLocker;
				string m_cAlgo;
				size_t m_processedBlockCnt;
				int32_t m_dThreadCnt;
				int32_t m_dIndex;
				size_t m_blockCntBeenCopied;
				size_t m_batchCntBeenCopied;
				std::thread m_unzipThreads[1000];

				DenseBlockingQueue<void*> m_zipedDataToProduce; //read zip data to this queue
				DenseBlockingQueue<void*> m_zipedDataToConsume; //read zip data to this queue

				DenseBlockingQueue<void*> m_unzipedDataToProduce; //read zip data to this queue
				DenseBlockingQueue<void*> m_unzipedDataToConsume; //read zip data to this queue


				RandomOrdering m_randomordering;   // randomizing class
				std::mt19937_64 m_randomEngine;
#ifdef _WIN32
				DWORD sysGran;
#else
				int32_t sysGran;
#endif
				DenseBlockingQueue<void*> m_dataToProduce;
				DenseBlockingQueue<void*> m_dataToConsume;
			};

			template <class ElemType>
			class CDensereader : public DataReaderBase
			{
			public:

				virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
				virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }

				template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);

				virtual void Destroy();

				CDensereader() : DSSMLabels(nullptr), DSSMCols(0) {
					m_pMBLayout = make_shared<MBLayout>();
					m_pMBLayout->SetUniqueAxisName(L"CDenseReader");
				};

				virtual ~CDensereader();

				virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
				virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples) override;
				virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

				virtual bool SupportsDistributedMBRead() const override { return true; }

				template<class ConfigRecordType> void RenamedMatrices(const ConfigRecordType& readerConfig, std::map<std::wstring, std::wstring>& rename);
				virtual void SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<LabelIdType, LabelType>& /*labelMapping*/) { NOT_IMPLEMENTED };
				virtual bool GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart = 0*/) { NOT_IMPLEMENTED };
				virtual bool DataEnd();

				size_t GetNumParallelSequencesForFixingBPTTMode()
				{
					return m_pMBLayout->GetNumParallelSequences();
				}

				size_t GetNumParallelSequences() { return m_pMBLayout->GetNumParallelSequences(); }
				void CopyMBLayoutTo(MBLayoutPtr pMBLayout) { pMBLayout->CopyFrom(m_pMBLayout); };

				//virtual bool DataEnd(EndDataType endDataType);

				size_t NumberSlicesInEachRecurrentIter() { return 1; }
				void SetNbrSlicesEachRecurrentIter(const size_t) { };
				void SetSentenceEndInBatch(std::vector<size_t> &/*sentenceEnd*/){};

			private:
#if DEBUG
				marker_series* reader_series;
				size_t cur_read;
#endif
				clock_t timer;
				void DoDSSMMatrix(Matrix<ElemType>& mat, size_t actualMBSize);

				void CheckDataMatrices(StreamMinibatchInputs& matrices);
				MBLayoutPtr m_pMBLayout;
				ConfigParameters m_readerConfig;

				std::shared_ptr<DenseBinaryInput<ElemType>> m_dataInput;

				std::map<std::wstring, shared_ptr<BDenseBinaryMatrix<ElemType>>> m_dataMatrices;

				unsigned long m_randomize; // randomization range

				ElemType* DSSMLabels;
				size_t DSSMCols;


				size_t m_mbSize;    // size of minibatch requested

				size_t m_requestedEpochSize; // size of an epoch

				size_t m_epoch; // which epoch are we on

				bool m_partialMinibatch;    // a partial minibatch is allowed

				bool m_prefetchEnabled;
				std::future<size_t> m_pendingAsyncGetMinibatch;

			};
		}
	}
}
