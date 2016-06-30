//
// <copyright file="CSparsePCReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// CSparsePCReader.h - Include file for the Sparse Parallel Corpus reader.
#pragma once
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>

namespace Microsoft { namespace MSR { namespace CNTK {



	template <typename T>
	class DenseSparseBlockingQueue
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
			std::unique_lock<std::mutex> lock(this->d_mutex);
			this->d_condition.wait(lock, [=]{ return !this->d_queue.empty(); });
			T rc(std::move(this->d_queue.back()));
			this->d_queue.pop_back();
			return rc;
		}
	};

	template<class ElemType>
	class SparseDenseMemory
	{
	public:
		bool m_bIsSparse;

		int32_t m_sampleNumber;
		int32_t m_Dim;
		size_t m_nnZ;
		int32_t* m_rowIndices;
		int32_t* m_colIndices;
		ElemType* m_values;

		size_t m_denseIndex;

		size_t m_sparseValueIndex;
		size_t m_sparseRowIndex;
		size_t m_sparseColIndex;

		void Clear();
		void AddDenseData(void* pSource, size_t dim);
		void AddSparseData(void* nnzValue, void* nnzIndex, int32_t length);

		void FillDenseMatrix(Matrix<ElemType>* pMatrix);
		void FillSparseMatrix(Matrix<ElemType>* pMatrix);

		SparseDenseMemory()
		{
			m_bIsSparse = true;
			m_Dim = 0;
			m_nnZ = 0;
			m_rowIndices = nullptr;
			m_colIndices = nullptr;
			m_values = nullptr;
			m_denseIndex = 0;
			m_sparseValueIndex = 0;
			m_sparseRowIndex = 0;
			m_sparseColIndex = 0;
		}
	};



class SparseDenseFeatureInfo
{
public:
	bool m_bIsSparse;
	int32_t m_Dim;

	int32_t m_SparseFactor;
};


template<class ElemType>
class CSparsePCReader : public DataReaderBase
{
private:
    ConfigParameters m_readerConfig;
    std::wstring m_file;
    size_t m_miniBatchSize;
    size_t m_microBatchSize;
    int64_t m_maxReadData; // For early exit during debugging
    bool m_doGradientCheck;
    bool m_returnDense;
    size_t m_sparsenessFactor;
    int32_t m_verificationCode;
    size_t m_reshapeInputToRowSize;
    MBLayoutPtr m_pMBLayout;

    int64_t m_filePositionMax;
    int64_t m_currOffset;
    int m_traceLevel;

    std::map<LabelIdType, LabelType> m_mapIdToLabel;
    std::map<LabelType, LabelIdType> m_mapLabelToId;

	//Below 

	DenseSparseBlockingQueue<std::map<std::wstring, SparseDenseMemory<ElemType>*>> m_dataToProduce;
	DenseSparseBlockingQueue<std::map<std::wstring, SparseDenseMemory<ElemType>*>> m_dataToConsume;

	//This buffer is used to 
	std::map<std::wstring, void*> m_mappedBuffer;


	std::map <std::wstring,  SparseDenseMemory<ElemType>*> m_mapWorkingMemory;

	std::map<std::wstring, SparseDenseFeatureInfo> m_mapSparseDenseInfo;
	bool m_bSparseDenseInfoInitialized;

	std::map<int32_t, std::wstring> m_mapReaderOrder2FeatureName;


	ifstream m_inFile;

	//Compress
	size_t m_fileSize;
	std::vector<size_t> m_blockSizeInByte;
	std::vector<size_t> m_sampleCntInBlock;
	std::vector<size_t> m_blockOffset;
	size_t m_numRows;
	size_t m_numBatches;
	size_t m_numBlocks;
	int32_t m_blockSize;
	int32_t m_blockSampleCnt;
	int32_t m_blockSizeOfUnzippedBuffer;

	size_t m_mbSize;
	size_t m_nextMB;
	size_t m_windowSize;
	size_t m_epochSize;
	size_t* m_readOrder;
	size_t m_readOrderLength = 0;
	size_t m_blockCntBeenRead = 0;
	size_t m_sampleCntInUnzippedBuffer = 0;
	long m_lastValidPosOfUnzippedBuffer = -1;
	long m_firstValidPosOfUnzippedBuffer = 0;

	size_t m_SparceDenseFeatureSize = 0;

	void * m_unzippedBuffer;
	void * m_zippedFileBlockBuffer;
	size_t m_unzippedBufferLen;

	//muti-thread to decompress
	int32_t m_dThreadCnt;
	int32_t m_dIndex;
	size_t * m_processedBlockCntPerThread;
	size_t m_blockCntBeenCopied = 0;
	size_t m_batchCntBeenCopied;
	std::thread m_unzipThreads[20];
	DenseSparseBlockingQueue<void*> m_zipedDataToProduce; //read zip data to this queue
	DenseSparseBlockingQueue<void*> m_zipedDataToConsume; //read zip data to this queue

	DenseSparseBlockingQueue<void*> m_unzipedDataToProduce; //read zip data to this queue
	DenseSparseBlockingQueue<void*> m_unzipedDataToConsume; //read zip data to this queue
	//Compress END

public:
    CSparsePCReader() : m_pMBLayout(make_shared<MBLayout>()) {};
    virtual ~CSparsePCReader();
    virtual void Destroy();
    template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
    virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
	virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples) override;
	//virtual bool GetMinibatch(StreamMinibatchInputs& matrices);
	virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

    size_t GetNumParallelSequences() { return m_pMBLayout->GetNumParallelSequences(); }
    void SetNumParallelSequences(const size_t) { };
    void CopyMBLayoutTo(MBLayoutPtr pMBLayout) { pMBLayout->CopyFrom(m_pMBLayout); }
	virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
	virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
	virtual bool GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart*/)
	{
		RuntimeError("GetData not supported in SparsePCReader");
	};
	virtual bool DataEnd();

	size_t GetNumParallelSequencesForFixingBPTTMode()
	{
		return m_pMBLayout->GetNumParallelSequences();
	}

	//Compress
	//void StartMinibatchLoopOldVersion(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
	void GetZippedFileInfo();
	void UnzipData(int threadIndex, size_t numToRead);
	void Unzip(void * input, void * output, unsigned int inputSize, unsigned int &outputSize);
	void ReadZipData(size_t* read_order, size_t numToRead);
	void FillReadOrder(size_t windowSize);
	void ReadMinibatches(size_t* read_order, size_t numToRead);
	void CompactUnzipBuffer();
	bool Copy2Buffer(size_t numToRead);
	void ClearUnzipBufferStatus();
	//Comress END
};
}}}
